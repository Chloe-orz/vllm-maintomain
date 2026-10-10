# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import pickle
from pathlib import Path

import pytest
from lwd_config_under_test import LwdAPIConfig, LwdConfig, resolve_lwd_config

FIXTURE = Path(__file__).parent / "fixtures/s3_cloud_share_2dp.yaml"


def test_snapshot_survives_handoff_without_yaml_or_key_access(tmp_path: Path) -> None:
    path = tmp_path / "topology.yaml"
    path.write_bytes(FIXTURE.read_bytes())
    raw = {
        "lwd_config": {
            "path": str(path),
            "role": "edge",
            "instance_id": 1,
            "coordination": {
                "enabled": True,
                "control_url": "http://127.0.0.1:9133/v1/chat/completions",
                "tenant_key_file": str(tmp_path / "does-not-exist"),
                "consumer_id": "enterprise-a-edge1",
                "connect_timeout": 5.0,
            },
        },
    }
    original = copy.deepcopy(raw)
    snapshot = resolve_lwd_config(raw)
    path.unlink()
    received, coordination = pickle.loads(pickle.dumps(snapshot))
    assert received.instance.id == 1
    assert [link.dp_idx for link in received.connections] == [0, 1]
    assert received.topology.digest
    assert received.coordination is coordination
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
        None,
        [],
        {"enabled": "false"},
        {"listen_port": 70000},
        {"connect_timeout": 0},
        {"connect_timeout": float("nan")},
        {"enabled": True},
        {"control_url": "file:///tmp/control"},
        {"unknown": 1},
    ],
)
def test_invalid_coordination_is_not_silently_forwarded(coordination: object) -> None:
    with pytest.raises(ValueError):
        resolve_lwd_config(
            {
                "lwd_config": {
                    "path": str(FIXTURE),
                    "role": "edge",
                    "instance_id": 0,
                    "coordination": coordination,
                }
            }
        )


@pytest.mark.parametrize("with_entry", [False, True])
def test_legacy_coordination_cannot_shadow_nested_configuration(
    with_entry: bool,
) -> None:
    raw: dict[str, object] = {"lwd_coordination": {"enabled": False}}
    if with_entry:
        raw["lwd_config"] = {
            "path": str(FIXTURE),
            "role": "edge",
            "instance_id": 0,
            "coordination": {"enabled": False},
        }
    with pytest.raises(ValueError, match="move it into lwd_config.coordination"):
        resolve_lwd_config(raw)


@pytest.mark.parametrize("cloud_id", [0, 1])
def test_cloud_coordination_name_defaults_to_selected_cloud(cloud_id: int) -> None:
    supplied = {"enabled": True, "listen_port": 9133}
    entry = {
        "path": str(FIXTURE.parent / "s2_edge_share_1dp.yaml"),
        "role": "cloud",
        "instance_id": cloud_id,
        "coordination": supplied,
    }
    selected, coordination = resolve_lwd_config({"lwd_config": entry})
    assert selected.coordination is coordination
    assert coordination.instance_id == f"cloud-{cloud_id}"
    assert "instance_id" not in supplied


def test_absent_coordination_stays_disabled() -> None:
    selected, coordination = resolve_lwd_config(
        {
            "lwd_config": {
                "path": str(FIXTURE),
                "role": "cloud",
                "instance_id": 0,
            }
        }
    )
    assert selected.coordination is coordination is None


@pytest.mark.parametrize(
    "identity,flag", [(0, "--api-server-rpc-port"), (1, "--api-server-attach")]
)
def test_api_cli_metadata_survives_snapshot_handoff(
    tmp_path: Path, identity: int, flag: str
) -> None:
    parser = argparse.ArgumentParser()
    LwdAPIConfig.add_cli_args(parser)
    args = parser.parse_args([flag, "10.0.0.1:25100"])
    path = tmp_path / "topology.yaml"
    args.additional_config = {
        "lwd_config": {
            "path": str(path),
            "role": "edge",
            "instance_id": identity,
            "coordination": {"enabled": False},
        }
    }
    # CLI identity checks do not read a YAML file; the snapshot owns that read.
    api = LwdAPIConfig.from_namespace(args)
    path.write_bytes((FIXTURE.parent / "s2_edge_share_2dp.yaml").read_bytes())
    entry = LwdConfig.from_dict(args.additional_config["lwd_config"])
    api.validate_topology(entry)
    path.unlink()
    # Only round-trip our own objects, never externally supplied pickle data.
    received_api, received_entry = pickle.loads(pickle.dumps((api, entry)))
    received_api.validate_topology(received_entry)
    assert received_api == api
    assert [c.dp_idx for c in received_entry.connections] == [0, 1]
    assert received_entry.instance_id == identity
