# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import pickle
from pathlib import Path

import pytest
from lwd_config_under_test import resolve_lwd_config

FIXTURE = Path(__file__).parent / "fixtures/s3_cloud_share_2dp.yaml"


def test_snapshot_survives_handoff_without_yaml_or_key_access(tmp_path: Path) -> None:
    path = tmp_path / "topology.yaml"
    path.write_bytes(FIXTURE.read_bytes())
    raw = {
        "lwd_config": {"path": str(path), "role": "edge", "instance_id": 1},
        "lwd_coordination": {
            "enabled": True,
            "control_url": "http://127.0.0.1:9133/v1/chat/completions",
            "tenant_key_file": str(tmp_path / "does-not-exist"),
            "consumer_id": "enterprise-a-edge1",
            "connect_timeout": 5.0,
        },
    }
    original = copy.deepcopy(raw)
    snapshot = resolve_lwd_config(raw)
    path.unlink()
    received, coordination = pickle.loads(pickle.dumps(snapshot))
    assert received.instance.id == 1
    assert [link.dp_idx for link in received.connections] == [0, 1]
    assert received.topology.digest
    assert coordination.consumer_id == "enterprise-a-edge1"
    assert coordination.connect_timeout == 5.0
    assert raw == original


@pytest.mark.parametrize("raw", [{}, {"enable_cpu_binding": True}, None])
def test_absent_lwd_leaves_centralized_configuration_empty(raw: object) -> None:
    assert resolve_lwd_config(raw) == (None, None)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"instance_id": 2}, "Instance not found"),
        ({"instance_id": True}, "integer"),
        ({"role": "unknown"}, "role"),
        ({"scheduler": "mixed"}, "unknown fields"),
    ],
)
def test_invalid_entry_fails_during_configuration(change: dict, match: str) -> None:
    entry = {"path": str(FIXTURE), "role": "edge", "instance_id": 0, **change}
    with pytest.raises(ValueError, match=match):
        resolve_lwd_config({"lwd_config": entry})


@pytest.mark.parametrize(
    "coordination",
    [
        {"enabled": "false"},
        {"listen_port": 70000},
        {"connect_timeout": 0},
        {"connect_timeout": float("nan")},
        {"enabled": True},
        {"control_url": "file:///tmp/control"},
        {"unknown": 1},
    ],
)
def test_invalid_coordination_is_not_silently_forwarded(coordination: dict) -> None:
    with pytest.raises(ValueError):
        resolve_lwd_config({"lwd_coordination": coordination})
