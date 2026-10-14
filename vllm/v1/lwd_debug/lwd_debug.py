# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Request diagnostics using the existing control-plane callback names."""

from typing import TYPE_CHECKING

from vllm.v1.lwd_debug.log_base import LwdLogBase

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.lwd.types import LwdC2eMeta
    from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler import (
        LwdCloudScheduler,
    )
    from vllm.v1.lwd_control.control_communication.lwd_notify import LwdRequestNotify


class LwdDebug(LwdLogBase):
    LAYER = "debug"

    @classmethod
    def cloud_embeds_injected(
        cls, req_id: str, idx: int, start: int, n: int, buf: "torch.Tensor"
    ) -> None:
        cls.event(
            "inject",
            "req=%s index=%d offset=%d tokens=%d shape=%s",
            req_id,
            idx,
            start,
            n,
            tuple(buf.shape),
        )

    @classmethod
    def cloud_fill_window(
        cls, req_id: str, start_pos: int, num_sched: int, window: "torch.Tensor"
    ) -> None:
        cls.event(
            "fill",
            "req=%s offset=%d tokens=%d shape=%s",
            req_id,
            start_pos,
            num_sched,
            tuple(window.shape),
        )

    @classmethod
    def cloud_request_admitted(
        cls, wire: "LwdRequestNotify", sampling_params: "SamplingParams"
    ) -> None:
        cls.event(
            "admit",
            "req=%s max_tokens=%s eos=%s ignore_eos=%s",
            wire.request_id,
            sampling_params.max_tokens,
            sampling_params.eos_token_id,
            sampling_params.ignore_eos,
        )

    @classmethod
    def cloud_step(
        cls,
        scheduler: "LwdCloudScheduler",
        meta: "LwdC2eMeta",
        engine_core_outputs: dict[int, "EngineCoreOutputs"],
    ) -> None:
        if not cls.DEBUG:
            return
        # The refactored control PR owns finish-reason extraction in the scheduler.
        reasons = scheduler._lwd_c2e_finish_reasons(meta.req_ids, engine_core_outputs)
        for req_id, finish in zip(meta.req_ids, reasons):
            cls.event("cloud-step", "req=%s finish=%s", req_id, finish)

    @classmethod
    def edge_tokens_delivered(
        cls,
        request_id: str,
        sampled_token_ids: list[int],
        finish_reason: int,
        vllm_config: "VllmConfig",
    ) -> None:
        cls.event(
            "deliver",
            "req=%s tokens=%s finish=%s",
            request_id,
            sampled_token_ids,
            finish_reason,
        )
