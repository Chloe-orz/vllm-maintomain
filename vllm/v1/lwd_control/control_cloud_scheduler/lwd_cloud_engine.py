# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""云侧 EngineCore 子类:覆写 socket IO 线程入口,PRE_OUT 循环独立成线程,
边侧预告与步内元数据经 input_queue 走原生分发;仅 prefill_only 云角色启用。"""

from __future__ import annotations

import threading
import time

import torch

from vllm.config.lwd import lwd_spec_factor
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import BlockHash, resolve_kv_cache_block_sizes
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
    LwdAbortNotify,
    LwdHelloNotify,
    LwdRangeNotify,
    LwdRequestNotify,
    lwd_encode_cloud_notify,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import (
    LwdBaseEngineCore,
)
from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
from vllm.v1.request import Request

logger = init_logger(__name__)


# 步元数据队满重试小睡:元数据不可丢(边侧据此预挂精确尺寸 recv)
_LWD_C2E_SEND_RETRY_SLEEP_S = 0.05


class LwdCloudEngineCore(LwdBaseEngineCore):
    """云 PO 引擎:PRE_OUT 循环独立成线程,其余全走原生。"""

    lwd_scheduler_cls = LwdCloudScheduler
    scheduler: LwdCloudScheduler
    _lwd_publisher: LwdControlPublisher
    _lwd_subscriber: LwdControlSubscriber

    def __init__(self, *args, **kwargs) -> None:
        self._lwd_recv_hint_mq = None
        self._lwd_irecv_done_mq = None
        super().__init__(*args, **kwargs)
        self._lwd_register_recv_hint()
        # PRE_OUT 接收线程构造期即起,与父线程 socket IO 共用 input_queue
        # (部署校验在 _lwd_setup_zmq 头部,失败走 EXECUTOR_FAILED 升级)
        threading.Thread(
            target=self._lwd_pre_out_loop, name="lwd-pre-out", daemon=True
        ).start()

    def _lwd_setup_zmq(self) -> None:
        """Bind requests, connect responses and register the independent recv queue."""
        config = self.lwd_config
        transport = config.transport
        connection = config.runtime_connection
        endpoints = transport.endpoints(connection)
        cloud_host = transport.pre_out_host or connection.cloud.addr
        self._lwd_subscriber = LwdControlSubscriber(endpoints["pre_out"], bind=True)
        self._lwd_publisher = LwdControlPublisher(
            endpoints["post_out"],
            bind=False,
            encoder=lwd_encode_cloud_notify,
        )
        sched_config = self.vllm_config.scheduler_config
        self._lwd_hello = LwdHelloNotify(
            pre_out_host=cloud_host,
            pre_out_port=transport.pre_out_port,
            # 调度三元组供边侧互校(cap/preflight 公式的同源输入)
            max_num_batched_tokens=sched_config.max_num_batched_tokens,
            max_num_seqs=sched_config.max_num_seqs,
            num_speculative_tokens=lwd_spec_factor(self.vllm_config) - 1,
        )
        # 首拍即通告(边侧可能已 bind 等待)
        self._lwd_announce()
        logger.info(
            "[LWD] cloud engine assembled: PRE_OUT bind %s, POST_OUT connect %s",
            endpoints["pre_out"],
            endpoints["post_out"],
        )

    def _lwd_register_recv_hint(self) -> None:
        if not self.lwd_config.topology.feature_ctrl.enable_early_recv:
            return
        from vllm_ascend.distributed.lwd_comm.pre_recv import (
            connect_recv_queues,
            register_hint_sender,
        )

        # Only queue setup uses RPC; hints bypass the model execution loop.
        hints, done = connect_recv_queues(self.model_executor)
        self._lwd_recv_hint_mq, self._lwd_irecv_done_mq = hints, done
        register_hint_sender(hints.enqueue)
        logger.info("[LWD] cloud early receive enabled")

    def shutdown(self) -> None:
        self._lwd_shutdown_comm()
        for queue in (self._lwd_recv_hint_mq, self._lwd_irecv_done_mq):
            if queue is not None:
                queue.shutdown()
        super().shutdown()

    def _lwd_pre_out_loop(self) -> None:
        """Create and consume the socket on its owning IO thread."""
        try:
            self._lwd_setup_zmq()
        except Exception:
            logger.exception("[Lwd] cloud PRE_OUT setup failed")
            self.input_queue.put_nowait((EngineCoreRequestType.EXECUTOR_FAILED, b""))
            return
        while True:
            msg = self._lwd_subscriber.recv(timeout_ms=LWD_RECV_TIMEOUT_MS)
            if msg is None:
                if self._lwd_subscriber.closed:
                    break
                continue
            try:
                self._lwd_dispatch(msg)
            except Exception:
                logger.exception("[LWD] cloud control dispatch failed")
                self.input_queue.put_nowait(
                    (EngineCoreRequestType.EXECUTOR_FAILED, b"")
                )
                return

    def _lwd_announce(self) -> None:
        """首拍 HELLO 通告一次;队满不重试,由边侧等待超时 fail-fast 兜底。"""
        self._lwd_publisher.publish(self._lwd_hello)
        logger.info(
            "[Lwd][cloud] HELLO announced: pre_out=%s:%s",
            self._lwd_hello.pre_out_host,
            self._lwd_hello.pre_out_port,
        )

    def _lwd_dispatch(self, msg) -> None:
        """Dispatch request, abort and chunk notifications on the IO thread."""
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
            if self.lwd_config.topology.feature_ctrl.enable_early_recv:
                post_irecv_hint(make_recv_hint(msg.seqno, num_tokens, aux_rows))
            return
        if isinstance(msg, LwdAbortNotify):
            # abort 清扫:摘除通知队列里该请求的条目——mixed 调度对
            # "ADD 未落地"的预告是留队首等落地, aborted 请求永不落地,
            # 不清扫会把队首永久堵住(条目掏空则整条移除)
            removed = self.scheduler.prefill_notify_queue.purge(msg.request_id)
            if removed:
                logger.info(
                    "[Lwd][cloud-ctrl] purged %d pending notify item(s) for "
                    "aborted req=%s",
                    removed,
                    msg.request_id,
                )
            # 双队列与原生 ABORT 同款:eager 处理 + 保持 input_queue 次序
            self.aborts_queue.put_nowait([msg.request_id])
            self.input_queue.put_nowait((EngineCoreRequestType.ABORT, [msg.request_id]))
            return
        request = self._lwd_build_request(msg)
        self.input_queue.put_nowait((EngineCoreRequestType.ADD, (request, 0)))

    def _lwd_build_request(self, req_notify: LwdRequestNotify) -> Request:
        """请求构建(唯一建请求点):采样参数自请求预告重建 + 占位零值
        embeds(prompt 长度即行数,UP 注入直接写该缓冲窗口);哈希链
        按缓存使能分叉。"""
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
        local_hasher = self.request_block_hasher
        if local_hasher is None:
            request = Request(
                request_id=req_notify.request_id,
                prompt_token_ids=None,
                prompt_embeds=prompt_embeds,
                sampling_params=sampling_params,
                pooling_params=None,
            )
        else:
            request = self._lwd_build_hashed_request(
                req_notify, sampling_params, prompt_embeds, local_hasher
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

    def _lwd_build_hashed_request(
        self,
        req_notify: LwdRequestNotify,
        sampling_params: SamplingParams,
        prompt_embeds,
        local_hasher,
    ) -> Request:
        """带边侧哈希链的请求体:prompt 首建用边侧预告链,续算/缺链/
        长度不符回退本地(fail-open)——云侧占位 prompt 算不出真实
        内容哈希,前缀缓存命中只能靠边侧按真实内容算出的链。"""
        hash_block_size = resolve_kv_cache_block_sizes(
            self.scheduler.kv_cache_config, self.vllm_config
        )[1]

        def block_hasher(request: Request) -> list[BlockHash]:
            if len(request.block_hashes) == 0 and request.num_output_tokens == 0:
                expected = request.num_prompt_tokens // hash_block_size
                if len(req_notify.block_hashes) == expected:
                    return [BlockHash(value) for value in req_notify.block_hashes]
            return local_hasher(request)

        return Request(
            request_id=req_notify.request_id,
            prompt_token_ids=None,
            prompt_embeds=prompt_embeds,
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=block_hasher,
        )

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
        done_mq = self._lwd_irecv_done_mq
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
        if not items:
            return
        record_irecv_completions(items)

    def post_step(self, model_executed: bool) -> None:
        """步末冲刷待发 c2e(调度器只入列,发布归引擎层;两条步路径的
        原生尾部都会走到本钩子)。队满小睡重试,关停退出。"""
        for notify in self.scheduler.pending_c2e:
            while not self._lwd_publisher.closed:
                if self._lwd_publisher.publish(notify):
                    break
                time.sleep(_LWD_C2E_SEND_RETRY_SLEEP_S)
        self.scheduler.pending_c2e.clear()
        super().post_step(model_executed)
