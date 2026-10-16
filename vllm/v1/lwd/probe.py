# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the wire-test branch's Worker check after normal HTTP startup."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    import uvicorn

    from vllm.engine.protocol import EngineClient

logger = init_logger(__name__)
_STARTUP_POLL_INTERVAL_S = 0.1
_PROBE_TIMEOUT_S = 1800.0


def probe_cases(hidden_size: int) -> list[dict[str, str | int | bool | list[int]]]:
    cases: list[dict[str, str | int | bool | list[int]]] = [
        dict(label="batch_up", channel="up", batches=200, window=4),
        dict(label="batch_down", channel="down", batches=200, window=4),
        dict(label="interleaved", channel="mixed", batches=200, window=8),
        dict(label="large_up", channel="up", batches=32, window=2),
        dict(label="large_down", channel="down", batches=32, window=2),
        dict(label="early_recv", channel="up", batches=200, window=3),
    ]
    return [
        {
            **case,
            "hidden_size": hidden_size,
            "tokens": [128, 512, 1024, 2048]
            if case["label"] in ("large_up", "large_down")
            else [1, 4, 2, 8],
            "early_recv": case["label"] == "early_recv",
        }
        for case in cases
    ]


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
        if not config.lwd_config.topology.feature_ctrl.enable_early_recv:
            raise ValueError(
                "The full wire-test suite requires YAML generated with "
                "--enable-early-recv"
            )
        logger.info(
            "[LWD][wire-test] HTTP server ready; starting Worker transfers role=%s",
            config.lwd_config.role,
        )
        cases = probe_cases(config.model_config.get_hidden_size())
        for case in cases:
            logger.info(
                "[LWD][wire-test] START case=%s role=%s",
                case["label"],
                config.lwd_config.role,
            )
            results = await engine.collective_rpc(
                "lwd_run_probe",
                timeout=_PROBE_TIMEOUT_S,
                args=(json.dumps(case),),
            )
            if len(results) != config.parallel_config.tensor_parallel_size or any(
                not isinstance(result, dict)
                or result.get("status") != "PASS"
                or result.get("case") != case["label"]
                or result.get("batches") != case["batches"]
                for result in results
            ):
                raise RuntimeError(f"Not all local Workers passed case={case['label']}")
            logger.info(
                "[LWD][wire-test] CASE PASS case=%s role=%s workers=%d",
                case["label"],
                config.lwd_config.role,
                len(results),
            )
        logger.info(
            "[LWD][wire-test] SUITE PASS role=%s cases=%d workers=%d",
            config.lwd_config.role,
            len(cases),
            len(results),
        )
    except Exception:
        logger.exception("[LWD][wire-test] FAIL; stopping HTTP server")
        server.should_exit = True
