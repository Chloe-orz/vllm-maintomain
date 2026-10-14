# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.lwd_debug import LwdControlLog, LwdDebug, LwdLogBase


def test_disabled_diagnostics_do_not_inspect_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default path must not decode tokens or inspect request state."""
    monkeypatch.setattr(LwdLogBase, "DEBUG", False)
    LwdDebug.cloud_step(None, None, None)


def test_cloud_diagnostics_use_refactored_scheduler_finish_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control PR moved this helper out of the cloud engine."""
    monkeypatch.setattr(LwdLogBase, "DEBUG", True)
    extract = Mock(return_value=[0])
    scheduler = SimpleNamespace(_lwd_c2e_finish_reasons=extract)
    LwdDebug.cloud_step(scheduler, SimpleNamespace(req_ids=["r1"]), {})
    extract.assert_called_once_with(["r1"], {})


def test_flight_log_is_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.v1.lwd_debug import control_log, log_base

    output = Mock()
    monkeypatch.setattr(LwdLogBase, "_rate_last", {})
    monkeypatch.setattr(control_log.logger, "info", output)
    monkeypatch.setattr(
        log_base.time, "monotonic", Mock(side_effect=[10.0, 11.0, 16.0])
    )
    for _ in range(3):
        LwdControlLog.flight(1, 2)
    assert output.call_count == 2
