# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Cloud engine core: overrides the socket IO thread entry; the PRE_OUT loop
runs on its own thread, and edge notifies and step metadata reach the native
dispatch through input_queue. Enabled for the prefill_only cloud role only."""

from __future__ import annotations

import threading
import time

import torch

from vllm.lwd_logger import lwd_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler import (
    LwdCloudScheduler,
)
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LwdControlPublisher,
)
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LWD_RECV_TIMEOUT_MS,
    LwdControlSubscriber,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LWD_SAMPLING_DEFAULTS,
    LwdHelloNotify,
    LwdRangeNotify,
    LwdRequestNotify,
    lwd_encode_cloud_notify,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import (
    LwdBaseEngineCore,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler import (
    lwd_spec_factor,
)
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.request import Request

# Sleep between retries of a full outbound queue: step metadata must not be
# lost, the edge posts its receive of the exact size from it.
_LWD_C2E_SEND_RETRY_SLEEP_S = 0.05


class LwdCloudEngineCore(LwdBaseEngineCore):
    """Cloud PO engine: PRE_OUT loop on its own thread, the rest is native."""

    lwd_scheduler_cls = LwdCloudScheduler
    scheduler: LwdCloudScheduler

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # PRE_OUT 接收线程构造期即起,与父线程 socket IO 共用 input_queue
        # (部署校验在 _lwd_setup_zmq 头部,失败走 EXECUTOR_FAILED 升级)
        threading.Thread(
            target=self._lwd_pre_out_loop, name="lwd-pre-out", daemon=True
        ).start()

    def _lwd_setup_zmq(self) -> None:
        """介入 ZMQ 双面:PRE_OUT bind 收边;POST_OUT connect 边承载 HELLO
        与步元数据;注册 pre-recv hint 发送通道(经 executor 旁路 MQ,不走
        busy_loop 消费的 rpc_broadcast_mq)。建站失败走 EXECUTOR_FAILED 升级。"""
        config = self.lwd_config
        master_addr = self.vllm_config.parallel_config.master_addr
        # 部署校验(fail-fast):master_addr 空则 zmq connect 报天书,这里
        # 给人话;pre_out_host 是随 HELLO 通告给边的唯一事实源,0.0.0.0
        # 可 bind 但不可通告——不拦会把失败变成边侧 600s HELLO 超时
        if not master_addr:
            raise ValueError(
                "[LWD] prefill_only cloud requires --master-addr (POST_OUT connect)"
            )
        if config.pre_out_host == "0.0.0.0":
            raise ValueError(
                "[LWD] prefill_only cloud pre_out_host=0.0.0.0 is not "
                "announceable; set a routable IP "
                "(VLLM_ASCEND_LWD_PRE_OUT_HOST)"
            )
        self._lwd_subscriber = LwdControlSubscriber(
            config.lwd_pre_out_endpoint(), bind=True
        )
        self._lwd_publisher = LwdControlPublisher(
            f"tcp://{master_addr}:{config.post_out_port}",
            bind=False,
            encoder=lwd_encode_cloud_notify,
        )
        sched_config = self.vllm_config.scheduler_config
        self._lwd_hello = LwdHelloNotify(
            pre_out_host=config.pre_out_host,
            pre_out_port=config.pre_out_port,
            # Scheduling triple the edge cross-checks (same inputs of the
            # cap/preflight formulas on both sides).
            max_num_batched_tokens=sched_config.max_num_batched_tokens,
            max_num_seqs=sched_config.max_num_seqs,
            num_speculative_tokens=lwd_spec_factor(self.vllm_config) - 1,
        )
        # 首拍即通告(边侧可能已 bind 等待)
        self._lwd_announce()
        self._lwd_register_recv_hint()
        lwd_logger.info(
            "[cloud] engine assembled: PRE_OUT bind %s, POST_OUT announce -> "
            "%s:%s via master %s",
            config.lwd_pre_out_endpoint(),
            config.pre_out_host,
            config.pre_out_port,
            master_addr,
        )

    def _lwd_register_recv_hint(self) -> None:
        """[Lwd][pre-recv] 提前收:提示下发通道 = 引擎 -> 云端点 worker
        通信线程。走 executor 上的旁路 MQ(不是 rpc_broadcast_mq:那条由
        busy_loop 单线程消费,会被几百毫秒的 prefill 前向堵住,提示就
        "早"不起来了)。引擎在 super().__init__ 里就把 executor 建好了,
        故此处直接取。"""
        hint_mq = getattr(self.model_executor, "lwd_recv_hint_mq", None)
        if hint_mq is not None:
            from vllm_ascend.distributed.lwd_comm.pre_recv import (
                register_hint_sender,
            )

            # 阻塞投递:闸门开启时提示是正确性依赖,丢一条就没人提前挂该 recv,
            # 水位不前进,闸门后的批永不下发。
            register_hint_sender(lambda hint, mq=hint_mq: mq.enqueue(hint))
            lwd_logger.info("[pre-recv] hint sender registered (PD_mix -> hint MQ)")
        else:
            # PD_mix 是唯一调度形态:hint MQ 缺失 = 提示发不出去,闸门永不
            # 放行,必须显式可见
            lwd_logger.warning(
                "[pre-recv] hint MQ missing: pre-recv gate will never open"
            )

    def _lwd_pre_out_loop(self) -> None:
        """PRE_OUT 接收循环:socket 与门状态在本线程内先建后用(zmq 单线程
        亲和);超时拍仅作关停响应上限。"""
        try:
            self._lwd_setup_zmq()
        except Exception:
            lwd_logger.exception("[cloud] PRE_OUT setup failed")
            self.input_queue.put_nowait((EngineCoreRequestType.EXECUTOR_FAILED, b""))
            return
        while True:
            msg = self._lwd_subscriber.recv(timeout_ms=LWD_RECV_TIMEOUT_MS)
            if msg is None:
                if self._lwd_subscriber.closed:
                    break
                continue
            self._lwd_on_message(msg)

    def _lwd_announce(self) -> None:
        """首拍 HELLO 通告一次;队满不重试,由边侧等待超时 fail-fast 兜底。"""
        self._lwd_publisher.publish(self._lwd_hello)
        lwd_logger.info(
            "[cloud] HELLO announced: pre_out=%s:%s",
            self._lwd_hello.pre_out_host,
            self._lwd_hello.pre_out_port,
        )

    def _lwd_dispatch_message(self, msg) -> bool:
        """PRE_OUT dispatch of two kinds (on this IO thread): range notify ->
        notify queue + pre-recv hint; request notify -> native ADD. Abort is
        not wired yet: it arrives as an unknown frame, dropped with a warning
        by the common entry."""
        if isinstance(msg, LwdRangeNotify):
            # 范围预告:整条入通知队列(重复预告即重复点名,调度器分类
            # 幂等剔除,无副作用)。预告自带 seqno,出批时作 UP 链配对号。
            items = msg.items
            self.scheduler.prefill_notify_queue.append(msg)
            # [Lwd][pre-recv] 提前收①:预告到达即给云 worker 下发提示(seqno +
            # flat token 数),worker 的通信线程收到就挂精确尺寸的
            # UP irecv(主帧 hash 载荷) —— 不再等这个批的
            # SchedulerOutput 下发。num_tokens 取 items 逐条目之和
            # (mixed 恒填 items;顶层字段只描述首条目)。mrope positions
            # 已随 RangeNotify 走控制面(wire_version>=2),UP 数据面只有
            # 主帧,aux_rows 恒 0——若残留非零,worker 会预挂一条永远收
            # 不到数据的 aux irecv,水位停摆、闸门后的批永不下发。
            from vllm_ascend.distributed.lwd_comm.pre_recv import (
                make_recv_hint,
                post_irecv_hint,
            )

            num_tokens = sum(item.num_tokens for item in items)
            aux_rows = 0
            # 先打点再入队:worker 通信线程 0.5ms 就能取走,若打点在入队之后,
            # 跨进程写日志的次序会让 hint2sub 出现负值,看着像时序错乱。
            post_irecv_hint(make_recv_hint(msg.seqno, num_tokens, aux_rows))
            return True
        if isinstance(msg, LwdRequestNotify):
            request = self._lwd_build_request(msg)
            self.input_queue.put_nowait((EngineCoreRequestType.ADD, (request, 0)))
            return True
        return False

    def _lwd_build_request(self, req_notify: LwdRequestNotify) -> Request:
        """请求构建(唯一建请求点):采样参数自请求预告重建 + 占位零值
        embeds(prompt 长度即行数,UP 注入直接写该缓冲窗口)。前缀缓存
        哈希链未随控制面移植(功能另上,届时再挂 block_hasher)。"""
        sampling_params = self._lwd_build_sampling_params(req_notify)
        # 不传真实 ids 也不造占位:按原生 prompt-embeds 语义挂零缓冲,
        # 行数即 prompt 长度;ids=None 时 input_batch 自动把 prompt 段
        # is_token_ids 置 False,M-RoPE 走纯文本直通构造(与扫描结果
        # 逐值一致)。
        prompt_embeds = torch.zeros(
            req_notify.num_prompt_tokens,
            self.vllm_config.model_config.get_hidden_size(),
            dtype=self.vllm_config.model_config.dtype,
        )
        request = Request(
            request_id=req_notify.request_id,
            prompt_token_ids=None,
            prompt_embeds=prompt_embeds,
            sampling_params=sampling_params,
            pooling_params=None,
        )
        # 占位 embeds 不随 SO 上传运输层(NewRequestData 只带形状,
        # worker 本地分配):35MB 零 buffer 走 MQ overflow 通道实测
        # 单程 240ms+,是 prefill SO 开工延迟的主因。
        request.lwd_embeds_placeholder = True
        return request

    @staticmethod
    def _lwd_build_sampling_params(req_notify: LwdRequestNotify) -> SamplingParams:
        """采样参数自请求预告重建(字段表驱动);eos_token_id 走原生回填
        入口(设 _eos_token_id 并计入 _all_stop_token_ids 供 min_tokens
        判定;云侧无客户端 generation_config,传空)。"""
        values = {f: getattr(req_notify, f) for f in LWD_SAMPLING_DEFAULTS}
        values["stop_token_ids"] = list(values["stop_token_ids"])
        sampling_params = SamplingParams(**values)
        sampling_params.update_from_generation_config({}, req_notify.eos_token_id)
        LwdControlDiag.cloud_request_admitted(req_notify, sampling_params)
        return sampling_params

    def step_with_batch_queue(self):
        """步首排空提前收水位(done MQ -> 闸门就绪判据),其余走原生。"""
        self._lwd_drain_irecv_completions()
        out = super().step_with_batch_queue()
        return out

    def step(self):
        """同步步路径同款:步首排水位,其余走原生。"""
        self._lwd_drain_irecv_completions()
        out = super().step()
        return out

    def _lwd_drain_irecv_completions(self) -> None:
        """每步排空 done MQ,把 worker 收完的 UP recv 并进提前收水位(闸门就绪判据)。"""
        done_mq = getattr(self.model_executor, "lwd_irecv_done_mq", None)
        if done_mq is None:
            return
        from vllm_ascend.distributed.lwd_comm.pre_recv import (
            record_irecv_completions,
        )

        items = []
        while True:
            try:
                items.append(done_mq.dequeue(timeout=0))
            except TimeoutError:
                break
            except Exception:
                # 关停时 ring 被取消:记一条即可,水位停在原处不影响收尾
                lwd_logger.exception("[pre-recv] done MQ drain failed")
                break
        if not items:
            return
        record_irecv_completions(items)

    def post_step(self, model_executed: bool) -> None:
        """Flush pending c2e to the edge at the end of the step (the
        scheduler only queues them; both native step paths reach this hook).
        A full queue is retried after a short sleep until shutdown."""
        for notify in self.scheduler.pending_c2e:
            while not self._lwd_publisher.closed:
                if self._lwd_publisher.publish(notify):
                    break
                time.sleep(_LWD_C2E_SEND_RETRY_SLEEP_S)
        self.scheduler.pending_c2e.clear()
        super().post_step(model_executed)
