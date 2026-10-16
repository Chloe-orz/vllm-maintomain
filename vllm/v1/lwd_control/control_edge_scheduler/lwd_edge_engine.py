# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Edge engine core: communication setup (POST_OUT bind, PRE_OUT connected
lazily), HELLO discovery and cross-check, and the request entry (edge
admission + announcement to the cloud; communication stays in the engine).

Stepping is the native step_with_batch_queue, adapted at construction: depth
2, is_pooling_model=True (no native sample_tokens) and an executor proxy (a
normally completed batch returning None becomes a placeholder).

Construction (injection order and the lifecycle of both channels: see the
base class): once super() has built the engine, bind POST_OUT, start the
receiver thread and block until the cloud's first HELLO, which the cloud sends
once fully initialized; a timeout fails fast. There is no self-healing:
restart the whole group. After the triple cross-check, the publisher is handed
to the scheduler.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

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

# Native batch queue depth: dispatched (embed + unembed) batches not yet
# reaped. 2 keeps the pipeline aligned; the reference branch's 4 is
# conservative, validate performance with 2 first.
LWD_EDGE_BATCH_QUEUE_DEPTH = 2

# A full publish queue is retried for the add-request notify with growing
# intervals (about 3 s in total); running out fails the request.
_LWD_ADD_RETRY_STEPS = 5
_LWD_ADD_RETRY_INTERVAL_S = 0.2

# 运行时旋钮固定值(评审定案:不接收配置/env 注入):HELLO 等待 600s、
# 发布队列深 1000(与通信层 LWD_PUBLISH_QUEUE_MAX 同值)。诊断层日志
# 默认关——开关归 lwd_debug 自带的 env 通道,随其移植生效,引擎不再
# 传播 config 段开关。
_LWD_HELLO_TIMEOUT_S = 600.0
_LWD_PUBLISH_QUEUE_MAX = 1000


class LwdEdgeEngineCore(LwdBaseEngineCore):
    """Edge PO engine: communication setup + HELLO cross-check + native
    stepping."""

    lwd_scheduler_cls = LwdEdgeScheduler
    scheduler: LwdEdgeScheduler

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
        # Native batch queue stepping (step/step_with_batch_queue untouched):
        # 1. depth 2, rebuilding the deque so its maxlen matches and
        #    appendleft on a full queue cannot silently evict a batch in
        #    flight;
        # 2. the edge samples inside execute, so is_pooling_model stops the
        #    native sample_tokens call;
        # 3. the executor proxy.
        self.is_pooling_model = True
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
        """POST_OUT dispatch (receiver thread, entry in the base class): HELLO
        -> retarget PRE_OUT (the only source of the cloud endpoint; retried
        while the queue is full); c2e -> scheduler notify queue + WAKEUP;
        anything else is unknown and dropped with a warning by the entry."""
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
        """HELLO cross-check. (max_num_batched_tokens, max_num_seqs) must match
        the cloud: both feed the cap/preflight formulas, and a mismatch makes
        the cloud hold prefills behind decodes forever without any error, so
        it shuts down and fails fast. The speculative factor is adopted from
        the cloud and liveness re-checked with it; mrope models shut down on an
        old wire version."""
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
                f"(max_num_batched_tokens, max_num_seqs) edge={local[:2]} != "
                f"cloud={remote[:2]}; a mismatch makes the cloud hold prefills "
                f"behind decodes forever, so start both sides with the same values"
            )
        if local[2] != remote[2]:
            # The factor reserves draft budget of cloud decode steps; the edge
            # has no legitimate local source (a spec config on the edge would
            # start the draft machinery), so it adopts the cloud's.
            spec_factor = remote[2] + 1
            lwd_logger.warning(
                "[hello] adopt cloud spec_factor=%d over edge local %d",
                spec_factor,
                local[2] + 1,
            )
            self.scheduler._lwd_spec_factor = spec_factor
            # The scheduler checked liveness with the local factor.
            if (
                self.scheduler.max_num_scheduled_tokens
                - self.scheduler.max_num_running_reqs * spec_factor
                < 1
            ):
                self._lwd_shutdown_comm()
                raise RuntimeError(
                    f"[LWD] deadlock config: with the cloud spec_factor="
                    f"{spec_factor} the cap is always < 1 (budget "
                    f"{self.scheduler.max_num_scheduled_tokens} - max_num_seqs "
                    f"{self.scheduler.max_num_running_reqs} x {spec_factor}); "
                    f"increase max_num_batched_tokens or decrease max_num_seqs"
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
        """Announce the request to the cloud (the engine's only communication
        point), packing the sampling fields from the shared table; a full queue
        is retried with short backoffs, then RuntimeError is raised (back to
        the frontend through add_request)."""
        sampling_params = request.sampling_params
        assert sampling_params is not None, "admission rejects pooling requests"
        values: dict[str, Any] = {
            f: getattr(sampling_params, f) for f in LWD_SAMPLING_DEFAULTS
        }
        if values["max_tokens"] is None:
            values["max_tokens"] = LWD_SAMPLING_DEFAULTS["max_tokens"]
        values["stop_token_ids"] = list(values["stop_token_ids"] or [])
        message = LwdRequestNotify(
            request_id=request.request_id,
            num_prompt_tokens=request.num_prompt_tokens,
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
