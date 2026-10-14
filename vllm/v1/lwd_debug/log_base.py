# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared LWD log gates and rate limiting, migrated from 0e59528."""

import logging
import time
from typing import ClassVar

from vllm.logger import init_logger

logger = init_logger(__name__)


class LwdLogBase:
    LAYER = ""
    DEBUG = logger.isEnabledFor(logging.DEBUG)
    _rate_last: ClassVar[dict[tuple[str, str], float]] = {}

    @classmethod
    def set_debug(cls, enabled: bool) -> None:
        cls.DEBUG = enabled

    @classmethod
    def event(cls, tag: str, message: str, *args: object) -> None:
        if cls.DEBUG:
            logger.info(
                "[LWD][%s-%s] %s", cls.LAYER, tag, message % args if args else message
            )

    @classmethod
    def phase(cls, message: str, *args: object) -> None:
        if cls.DEBUG:
            logger.info("[LWD][%s] %s", cls.LAYER, message % args if args else message)

    @classmethod
    def _rate_pass(cls, key: str, interval_s: float) -> bool:
        now = time.monotonic()
        identity = (cls.__name__, key)
        if now - cls._rate_last.get(identity, 0.0) < interval_s:
            return False
        cls._rate_last[identity] = now
        return True
