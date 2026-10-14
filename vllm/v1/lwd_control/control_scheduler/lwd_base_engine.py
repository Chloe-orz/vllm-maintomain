# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared edge/cloud engine initialization and control transport ownership."""

from __future__ import annotations

from vllm.config.lwd import LwdConfig
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.lwd_control.control_communication.lwd_control_publisher import (
    LwdControlPublisher,
)
from vllm.v1.lwd_control.control_communication.lwd_control_subscriber import (
    LwdControlSubscriber,
)


class LwdBaseEngineCore(EngineCoreProc):
    """边/云 EngineCore 公共基类。

    子类经 lwd_scheduler_cls 声明调度器类;注入必须发生在 super() 之前
    (super 构建 scheduler 时一次性消费 scheduler_cls,后设无效)。
    收发两面构造前为 None,关停容忍未建。上游 EngineCore 会对无 KV
    cache 组的部署禁用 chunked prefill,而 LWD 边侧 chunk 是纯调度
    记账(执行层只 embed 切片),构造后恢复使开关仅作用于 LWD 引擎。"""

    lwd_scheduler_cls: type
    lwd_config: LwdConfig
    _lwd_publisher: LwdControlPublisher | None
    _lwd_subscriber: LwdControlSubscriber | None

    def __init__(self, *args, **kwargs) -> None:
        self._lwd_publisher = None
        self._lwd_subscriber = None
        vllm_config = kwargs["vllm_config"]
        self.lwd_config = vllm_config.lwd_config
        # Reject unsupported runtime layouts before allocating any model workers.
        _ = self.lwd_config.runtime_connection
        scheduler_config = vllm_config.scheduler_config
        scheduler_config.scheduler_cls = self.lwd_scheduler_cls
        chunked_prefill_wanted = scheduler_config.enable_chunked_prefill
        super().__init__(*args, **kwargs)
        if not scheduler_config.enable_chunked_prefill and chunked_prefill_wanted:
            scheduler_config.enable_chunked_prefill = True
        self.lwd_config = vllm_config.lwd_config

    def _lwd_shutdown_comm(self) -> None:
        """两面关停(幂等,容忍未建):subscriber 先关断输入。"""
        if self._lwd_subscriber is not None:
            self._lwd_subscriber.shutdown()
        if self._lwd_publisher is not None:
            self._lwd_publisher.shutdown()

    def shutdown(self) -> None:
        self._lwd_shutdown_comm()
        super().shutdown()
