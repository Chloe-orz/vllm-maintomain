# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""边侧引擎子类:通信面装配(POST_OUT bind/PRE_OUT 延迟连接)、
HELLO 发现与互校、请求入口覆写(边准入 + 云预告,通信归引擎)。
步进零覆写,直用原生 step_with_batch_queue(批队列),LWD 契约经三项
构造期适配满足:深度下限 2、is_pooling_model=True(禁原生 sample_tokens
补发)、executor 代理(底层正常完成返回 None 的默认形态适配为占位)。

构造约束(注入时序与两面生命周期见基类):super() 建好引擎主体后
bind POST_OUT、起接收线程、阻塞等云首拍 HELLO(云全量初始化后才发,
超时 fail-fast,无重发自愈——重启即整组重拉),三元组互校通过后
回填 publisher 给调度器。"""

from __future__ import annotations

import threading
import time
from collections import deque

from vllm.lwd_logger import lwd_logger
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LwdControlPublisher,
)
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LwdControlSubscriber,
)
from vllm.v1.lwd_control.control_communication.lwd_notify import (
    LWD_SAMPLING_DEFAULTS,
    LwdC2eNotify,
    LwdHelloNotify,
    LwdRequestNotify,
    lwd_decode_cloud_notify,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_executor_proxy import (
    LwdEdgeExecutorProxy,
)
from vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler import (
    LwdEdgeScheduler,
)
from vllm.v1.lwd_control.control_scheduler.lwd_base_engine import (
    LwdBaseEngineCore,
)
from vllm.v1.request import Request

# 原生批队列深度:派发(embed+unembed 合计)不收割的批数上限——
# 深度 2 已维持流水对齐(参照仓经验 4 属保守值,先 2 验证性能)
LWD_EDGE_BATCH_QUEUE_DEPTH = 2

# add 预告发布重试:次数 x 递增间隔(共约 3s),耗尽即请求级报错
_LWD_ADD_RETRY_STEPS = 5
_LWD_ADD_RETRY_INTERVAL_S = 0.2

# 运行时旋钮固定值(评审定案:不接收配置/env 注入):HELLO 等待 600s、
# 发布队列深 1000(与通信层 LWD_PUBLISH_QUEUE_MAX 同值)。诊断层日志
# 默认关——开关归 lwd_debug 自带的 env 通道,随其移植生效,引擎不再
# 传播 config 段开关。
_LWD_HELLO_TIMEOUT_S = 600.0
_LWD_PUBLISH_QUEUE_MAX = 1000


class LwdEdgeEngineCore(LwdBaseEngineCore):
    """边 PO 引擎:通信面装配 + HELLO 互校 + 原生步进适配。"""

    lwd_scheduler_cls = LwdEdgeScheduler

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = self.lwd_config
        # 通信面:bind POST_OUT 订阅面 + 延迟连接的 PRE_OUT 发布面;
        # 云端点由 HELLO 通告决定(边不预知云地址)
        self._lwd_subscriber = LwdControlSubscriber(
            config.lwd_post_out_bind_endpoint(),
            bind=True,
            decoder=lwd_decode_cloud_notify,
        )
        self._lwd_publisher = LwdControlPublisher(
            None, bind=False, queue_max=_LWD_PUBLISH_QUEUE_MAX
        )
        self._lwd_hello_event = threading.Event()
        self._lwd_subscriber.start(self._lwd_on_message, name="lwd-post-in")
        if not self._lwd_hello_event.wait(_LWD_HELLO_TIMEOUT_S):
            self._lwd_shutdown_comm()
            raise RuntimeError(
                f"[LWD] edge engine init failed: no cloud HELLO within "
                f"{_LWD_HELLO_TIMEOUT_S}s on POST_OUT "
                f"(bind {config.lwd_post_out_bind_endpoint()}; check cloud "
                f"master_addr connectivity and POST_OUT port)"
            )
        self._lwd_cross_check_hello()
        self.scheduler.lwd_edge_publisher = self._lwd_publisher
        # 原生批队列步进适配(step/step_with_batch_queue 零覆写):
        # ① 深度 2(size 与 maxlen 同步重建,防 appendleft 在满队
        #    时静默挤掉在飞批);② 边侧采样在 execute 内完成,置
        #    is_pooling_model 禁原生 sample_tokens 补发;③ executor 代理。
        self.is_pooling_model = True
        # 批队列深度(size 与 maxlen 同步,防 appendleft 静默挤掉在飞批)
        self.batch_queue_size = LWD_EDGE_BATCH_QUEUE_DEPTH
        self.batch_queue = deque(maxlen=self.batch_queue_size)
        self.step_fn = self.step_with_batch_queue
        self.model_executor = LwdEdgeExecutorProxy(self.model_executor)
        lwd_logger.info(
            "[edge] engine assembled: POST_OUT bind %s, PRE_OUT "
            "discovered, native batch-queue stepping depth=%d",
            config.lwd_post_out_bind_endpoint(),
            self.batch_queue_size,
        )

    def _lwd_dispatch_message(self, msg) -> bool:
        """POST_OUT 消息分派(接收线程,入口在基类):HELLO -> retarget
        PRE_OUT(云端点唯一事实源,队满自旋重试);C2e -> 调度器通告队列
        + WAKEUP 敲门;未知帧返回 False 由公共入口告警丢弃。"""
        if isinstance(msg, LwdHelloNotify):
            endpoint = f"tcp://{msg.pre_out_host}:{msg.pre_out_port}"
            lwd_logger.info("[hello] cloud discovered: PRE_OUT -> %s", endpoint)
            self._lwd_peer_hello = msg
            while not self._lwd_publisher.retarget(endpoint):
                if self._lwd_subscriber.closed:
                    break
                lwd_logger.warning(
                    "[hello] PRE_OUT retarget deferred (queue full), retrying"
                )
                threading.Event().wait(0.05)
            self._lwd_hello_event.set()
            return True
        if isinstance(msg, LwdC2eNotify):
            self.scheduler.unembed_notify_queue.append(msg)
            self.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))
            return True
        return False

    def _lwd_cross_check_hello(self) -> None:
        """HELLO 互校:max_num_batched_tokens/max_num_seqs 与云不一致即关停
        fail-fast(不一致不报错,只会 preflight 恒 hold 静默串行);spec 系数
        以云为准采纳并复核活性;mrope 模型遇旧 wire 版本关停。"""
        hello = self._lwd_peer_hello
        sched = self.vllm_config.scheduler_config
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        local = (
            sched.max_num_batched_tokens,
            sched.max_num_seqs,
            getattr(spec_config, "num_speculative_tokens", None) or 0,
        )
        remote = (
            hello.max_num_batched_tokens,
            hello.max_num_seqs,
            hello.num_speculative_tokens,
        )
        if local[:2] != remote[:2]:
            self._lwd_shutdown_comm()
            raise RuntimeError(
                f"[LWD] edge/cloud scheduling config mismatch: "
                f"(max_num_batched_tokens, max_num_seqs) "
                f"edge={local[:2]} != cloud={remote[:2]}; "
                f"cap/preflight 公式两侧同源输入不一致会让 preflight 恒 "
                f"hold、prefill 被 decode 静默串行——两侧拉起配置须同值"
            )
        if local[2] != remote[2]:
            # spec 系数以云为准:它预留的是云 decode 步的草稿预算,边侧
            # 本地取不到合法来源(边实例带 spec 配置会拖起 draft机制)。
            # 采纳对端值而非 fail-fast——部署上边侧无需再配 spec。
            lwd_logger.warning(
                "[hello] adopt cloud spec_factor=%d over edge local %d "
                "(edge has no legitimate local source for it)",
                remote[2] + 1,
                local[2] + 1,
            )
            self.scheduler._lwd_spec_factor = remote[2] + 1
            # 采纳后复核活性下界(构造期检查用的是本地 factor)
            if (
                self.scheduler.max_num_scheduled_tokens
                - self.scheduler.max_num_running_reqs * (remote[2] + 1)
                < 1
            ):
                self._lwd_shutdown_comm()
                raise RuntimeError(
                    f"[LWD] 死锁配置: 采纳云侧 spec_factor={remote[2] + 1} "
                    f"后 cap 恒 < 1(预算 {self.scheduler.max_num_scheduled_tokens}"
                    f" − 满员 {self.scheduler.max_num_running_reqs} × "
                    f"{remote[2] + 1});请增大 max_num_batched_tokens "
                    f"或减小 max_num_seqs"
                )
        lwd_logger.info("[hello] scheduling triple verified: %s", local)

    # ------------------------------------------------------------------ #
    # 引擎接口覆写                                                        #
    # ------------------------------------------------------------------ #
    def add_request(self, request: Request, _request_wave: int = 0) -> None:
        """边准入(校验+mrope,调度器本地账) -> 云请求预告(通信归引擎)
        -> 原生引擎入口(原生校验 + 入队 + abort_immediately 原生处理)。"""
        self.scheduler.lwd_admit_request(request)
        self._lwd_notify_request(request)
        super().add_request(request)

    def abort_requests(self, request_ids: list[str]) -> None:
        """pending 台账摘除(不摘会顶高 prefill 闸门),终结走原生;
        云侧 abort 通告功能未上,随功能一并补齐。"""
        self.scheduler.lwd_edge_abort(request_ids)
        super().abort_requests(request_ids)

    def _lwd_notify_request(self, request: Request) -> None:
        """发请求元数据预告(唯一通信点,归引擎):采样参数字段表驱动
        组包,队满短退避重试,耗尽抛 RuntimeError(沿 add 链回前端)。"""
        values = dict(LWD_SAMPLING_DEFAULTS)
        sampling_params = request.sampling_params
        values.update({f: getattr(sampling_params, f) for f in LWD_SAMPLING_DEFAULTS})
        if values["max_tokens"] is None:
            values["max_tokens"] = 16
        values["stop_token_ids"] = list(values["stop_token_ids"] or [])
        message = LwdRequestNotify(
            request_id=request.request_id,
            num_prompt_tokens=len(request.prompt_token_ids),
            block_hashes=list(request.block_hashes),
            eos_token_id=sampling_params.eos_token_id,
            **values,
        )
        for attempt in range(_LWD_ADD_RETRY_STEPS):
            if self._lwd_publisher.publish(message):
                return
            time.sleep(_LWD_ADD_RETRY_INTERVAL_S * (attempt + 1))
        raise RuntimeError(
            f"[LWD] add-request notify for {request.request_id} dropped: "
            f"publish queue full after {_LWD_ADD_RETRY_STEPS} retries "
            f"(cloud PRE_OUT consumption stalled?)"
        )
