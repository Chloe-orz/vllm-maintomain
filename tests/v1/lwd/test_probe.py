# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm.v1.lwd.probe import run_lwd_probe


@pytest.fixture
def runtime() -> tuple[SimpleNamespace, SimpleNamespace]:
    server = SimpleNamespace(started=False, should_exit=False)
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            lwd_config=SimpleNamespace(role="edge"),
            parallel_config=SimpleNamespace(
                _api_process_rank=0, tensor_parallel_size=2
            ),
        ),
        collective_rpc=AsyncMock(return_value=[{"status": "PASS"}] * 2),
    )
    return server, engine


def test_transfers_wait_for_http_startup(
    runtime: tuple[SimpleNamespace, SimpleNamespace],
) -> None:
    server, engine = runtime

    async def scenario() -> None:
        task = asyncio.create_task(run_lwd_probe(server, engine))
        await asyncio.sleep(0)
        engine.collective_rpc.assert_not_called()
        server.started = True
        await asyncio.wait_for(task, 1)

    asyncio.run(scenario())
    engine.collective_rpc.assert_awaited_once_with("lwd_run_probe", timeout=1800.0)
    assert not server.should_exit


@pytest.mark.parametrize("skip", ["centralized", "other_api_process", "stopping"])
def test_non_lwd_duplicate_or_stopping_server_never_starts_transfers(
    runtime: tuple[SimpleNamespace, SimpleNamespace], skip: str
) -> None:
    server, engine = runtime
    if skip == "centralized":
        engine.vllm_config.lwd_config = None
    elif skip == "other_api_process":
        engine.vllm_config.parallel_config._api_process_rank = 1
    else:
        server.should_exit = True
    asyncio.run(run_lwd_probe(server, engine))
    engine.collective_rpc.assert_not_called()


@pytest.mark.parametrize("result", [[], [{"status": "PASS"}], [{"status": "FAIL"}] * 2])
def test_missing_or_failed_worker_result_stops_server(
    runtime: tuple[SimpleNamespace, SimpleNamespace], result: list[dict[str, str]]
) -> None:
    server, engine = runtime
    server.started = True
    engine.collective_rpc.return_value = result
    asyncio.run(run_lwd_probe(server, engine))
    assert server.should_exit


def test_worker_timeout_stops_server(
    runtime: tuple[SimpleNamespace, SimpleNamespace],
) -> None:
    server, engine = runtime
    server.started = True
    engine.collective_rpc.side_effect = TimeoutError("peer absent")
    asyncio.run(run_lwd_probe(server, engine))
    assert server.should_exit


@pytest.mark.parametrize("started", [False, True])
def test_shutdown_cancels_wait_or_inflight_rpc_without_reporting_failure(
    runtime: tuple[SimpleNamespace, SimpleNamespace], started: bool
) -> None:
    server, engine = runtime
    server.started = started

    async def wait_for_worker(method: str, timeout: float) -> None:
        await asyncio.Event().wait()

    async def scenario() -> None:
        engine.collective_rpc.side_effect = wait_for_worker
        task = asyncio.create_task(run_lwd_probe(server, engine))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert not server.should_exit
    assert engine.collective_rpc.call_count == int(started)
