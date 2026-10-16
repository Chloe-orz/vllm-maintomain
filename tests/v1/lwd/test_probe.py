# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm.v1.lwd.probe import probe_cases, run_lwd_probe


@pytest.fixture
def runtime() -> tuple[SimpleNamespace, SimpleNamespace]:
    server = SimpleNamespace(started=False, should_exit=False)
    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(
            lwd_config=SimpleNamespace(
                role="edge",
                topology=SimpleNamespace(
                    feature_ctrl=SimpleNamespace(enable_early_recv=True)
                ),
            ),
            model_config=SimpleNamespace(get_hidden_size=lambda: 5120),
            parallel_config=SimpleNamespace(
                _api_process_rank=0, tensor_parallel_size=2
            ),
        ),
        collective_rpc=AsyncMock(return_value=[{"status": "PASS"}] * 2),
    )

    async def run_case(method: str, timeout: float, args: tuple[str]) -> list[dict]:
        plan = json.loads(args[0])
        return [
            {"status": "PASS", "case": plan["label"], "batches": plan["batches"]}
        ] * 2

    engine.collective_rpc.side_effect = run_case
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
    calls = engine.collective_rpc.await_args_list
    assert len(calls) == 6
    assert [json.loads(call.kwargs["args"][0]) for call in calls] == probe_cases(5120)
    assert all(
        call.args == ("lwd_run_probe",) and call.kwargs["timeout"] == 1800.0
        for call in calls
    )
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


@pytest.mark.parametrize(
    "result",
    [
        [],
        [{"status": "PASS"}],
        [{"status": "FAIL"}] * 2,
        [{"status": "PASS", "case": "wrong", "batches": 200}] * 2,
        [{"status": "PASS", "case": "batch_up", "batches": 1}] * 2,
    ],
)
def test_missing_or_failed_worker_result_stops_server(
    runtime: tuple[SimpleNamespace, SimpleNamespace], result: list[dict[str, str]]
) -> None:
    server, engine = runtime
    server.started = True
    engine.collective_rpc.side_effect = None
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

    async def wait_for_worker(method: str, timeout: float, args: tuple[str]) -> None:
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


def test_disabled_early_receive_does_not_report_a_complete_suite(
    runtime: tuple[SimpleNamespace, SimpleNamespace],
) -> None:
    server, engine = runtime
    server.started = True
    engine.vllm_config.lwd_config.topology.feature_ctrl.enable_early_recv = False
    asyncio.run(run_lwd_probe(server, engine))
    engine.collective_rpc.assert_not_called()
    assert server.should_exit


def test_suite_stops_at_first_failed_case(
    runtime: tuple[SimpleNamespace, SimpleNamespace],
) -> None:
    server, engine = runtime
    server.started = True
    engine.collective_rpc.side_effect = [
        [{"status": "PASS", "case": "batch_up", "batches": 200}] * 2,
        RuntimeError("DOWN failed"),
    ]
    asyncio.run(run_lwd_probe(server, engine))
    assert engine.collective_rpc.call_count == 2
    assert server.should_exit
