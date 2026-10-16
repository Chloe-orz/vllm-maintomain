# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the wire-test branch's Worker check after normal HTTP startup."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    import uvicorn

    from vllm.engine.protocol import EngineClient

logger = init_logger(__name__)
_STARTUP_POLL_INTERVAL_S = 0.1
_PROBE_TIMEOUT_S = 1800.0


async def run_lwd_probe(server: uvicorn.Server, engine: EngineClient) -> None:
    config = engine.vllm_config
    # Multiple API processes share Workers; only one may submit the test round.
    if config.lwd_config is None or config.parallel_config._api_process_rank != 0:
        return

    while not server.started and not server.should_exit:
        await asyncio.sleep(_STARTUP_POLL_INTERVAL_S)
    if server.should_exit:
        return

    try:
        logger.info(
            "[LWD][wire-test] HTTP server ready; starting Worker transfers role=%s",
            config.lwd_config.role,
        )
        results = await engine.collective_rpc("lwd_run_probe", timeout=_PROBE_TIMEOUT_S)
        if len(results) != config.parallel_config.tensor_parallel_size or any(
            not isinstance(result, dict) or result.get("status") != "PASS"
            for result in results
        ):
            raise RuntimeError("Not all local Workers passed the data-plane check")
        logger.info(
            "[LWD][wire-test] all %d local Workers PASS role=%s",
            len(results),
            config.lwd_config.role,
        )
    except Exception:
        logger.exception("[LWD][wire-test] FAIL; stopping HTTP server")
        server.should_exit = True
