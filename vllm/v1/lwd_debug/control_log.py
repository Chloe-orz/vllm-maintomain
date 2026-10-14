# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Control-plane progress logging, migrated from 09ed952."""

from vllm.logger import init_logger
from vllm.v1.lwd_debug.log_base import LwdLogBase

logger = init_logger(__name__)
LWD_FLIGHT_LOG_INTERVAL_S = 5.0


class LwdControlLog(LwdLogBase):
    LAYER = "control"

    @classmethod
    def flight(cls, running: int, awaiting: int) -> None:
        if cls._rate_pass("flight", LWD_FLIGHT_LOG_INTERVAL_S):
            logger.info(
                "[LWD][control-flight] running=%d awaiting=%d cloud_active=%d",
                running,
                awaiting,
                running + awaiting,
            )
