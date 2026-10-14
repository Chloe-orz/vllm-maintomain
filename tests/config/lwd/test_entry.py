# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
from lwd_config_under_test import LwdAPIConfig, lwd_spec_factor, resolve_lwd_config

FIXTURE_NAME = "s3_cloud_share_2dp.yaml"


@pytest.mark.parametrize("drafts,expected", [(None, 1), (1, 2), (3, 4)])
def test_spec_factor_reserves_target_and_draft_tokens(
    drafts: int | None, expected: int
) -> None:
    speculative = (
        None if drafts is None else SimpleNamespace(num_speculative_tokens=drafts)
    )
    assert lwd_spec_factor(SimpleNamespace(speculative_config=speculative)) == expected


@pytest.mark.parametrize("role", ["edge", "cloud"])
def test_runtime_selects_the_same_link_on_both_sides(
    role: str, lwd_yaml_dir: Path
) -> None:
    config = resolve_lwd_config(
        {
            "lwd_config": {
                "path": str(lwd_yaml_dir / "s1_single_instance_1dp.yaml"),
                "role": role,
                "instance_id": 0,
            }
        }
    )
    # Runtime connection selection must survive a configuration handoff.
    received = pickle.loads(pickle.dumps(config))
    link = received.runtime_connection
    assert (link.edge.addr, link.cloud.addr) == ("10.0.0.1", "10.1.0.1")
    assert len(link.cloud.ranks) == 8


@pytest.mark.parametrize(
    "name",
    [
        "s1_single_instance_2dp.yaml",
        "s2_edge_share_1dp.yaml",
        "s3_cloud_share_1dp.yaml",
    ],
)
def test_runtime_rejects_unsupported_layout_without_discarding_links(
    name: str, lwd_yaml_dir: Path
) -> None:
    config = resolve_lwd_config(
        {
            "lwd_config": {
                "path": str(lwd_yaml_dir / name),
                "role": "edge",
                "instance_id": 0,
            }
        }
    )
    links_before = config.connections
    with pytest.raises(ValueError, match="single_instance, DP=1") as error:
        _ = config.runtime_connection
    message = str(error.value)
    assert f"path={config.path!r}" in message
    assert f"scene={config.topology.deployment.scene!r}" in message
    assert "role='edge', instance_id=0" in message
    selected = [(link.edge_id, link.cloud_id, link.dp_idx) for link in links_before]
    assert f"connections(edge_id, cloud_id, dp_idx)={selected!r}" in message
    assert config.connections == links_before


def test_snapshot_survives_handoff_without_yaml_or_key_access(
    tmp_path: Path, lwd_yaml_dir: Path
) -> None:
    path = tmp_path / "topology.yaml"
    path.write_bytes((lwd_yaml_dir / FIXTURE_NAME).read_bytes())
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
    received = pickle.loads(pickle.dumps(snapshot))
    assert received.instance.id == 1
    assert [link.dp_idx for link in received.connections] == [0, 1]
    assert received.topology.digest
    assert received.coordination.consumer_id == "enterprise-a-edge1"
    assert received.coordination.connect_timeout == 5.0
    assert not received.api.requested
    assert raw == original


@pytest.mark.parametrize("raw", [{}, {"enable_cpu_binding": True}, None])
def test_absent_lwd_leaves_centralized_configuration_empty(raw: object) -> None:
    assert resolve_lwd_config(raw) is None
    assert resolve_lwd_config(raw, api=LwdAPIConfig()) is None
    with pytest.raises(ValueError, match="require lwd_config.path"):
        resolve_lwd_config(raw, api=LwdAPIConfig(api_server_rpc_port="10.0.0.1:25100"))


@pytest.mark.parametrize(
    "change,match",
    [
        ({"instance_id": 2}, "Instance not found"),
        ({"instance_id": True}, "integer"),
        ({"role": "unknown"}, "role"),
        ({"scheduler": "mixed"}, "scheduler"),
    ],
)
def test_invalid_entry_fails_during_configuration(
    change: dict, match: str, lwd_yaml_dir: Path
) -> None:
    entry = {
        "path": str(lwd_yaml_dir / FIXTURE_NAME),
        "role": "edge",
        "instance_id": 0,
        **change,
    }
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
def test_invalid_coordination_is_not_silently_forwarded(
    coordination: object, lwd_yaml_dir: Path
) -> None:
    with pytest.raises(ValueError):
        resolve_lwd_config(
            {
                "lwd_config": {
                    "path": str(lwd_yaml_dir / FIXTURE_NAME),
                    "role": "edge",
                    "instance_id": 0,
                    "coordination": coordination,
                }
            }
        )


@pytest.mark.parametrize("with_entry", [False, True])
def test_legacy_coordination_cannot_shadow_nested_configuration(
    with_entry: bool,
    lwd_yaml_dir: Path,
) -> None:
    raw: dict[str, object] = {"lwd_coordination": {"enabled": False}}
    if with_entry:
        raw["lwd_config"] = {
            "path": str(lwd_yaml_dir / FIXTURE_NAME),
            "role": "edge",
            "instance_id": 0,
            "coordination": {"enabled": False},
        }
    with pytest.raises(ValueError, match="move it into lwd_config.coordination"):
        resolve_lwd_config(raw)


@pytest.mark.parametrize("cloud_id,name", [(0, None), (1, None), (1, "custom-cloud")])
def test_cloud_coordination_name_preserves_override_or_defaults_to_selected_cloud(
    cloud_id: int, name: str | None, lwd_yaml_dir: Path
) -> None:
    supplied = {
        "enabled": True,
        "listen_port": 9133,
        **({"instance_id": name} if name else {}),
    }
    original = supplied.copy()
    entry = {
        "path": str(lwd_yaml_dir / "s2_edge_share_1dp.yaml"),
        "role": "cloud",
        "instance_id": cloud_id,
        "coordination": supplied,
    }
    selected = resolve_lwd_config({"lwd_config": entry})
    assert selected.coordination.instance_id == (name or f"cloud-{cloud_id}")
    assert supplied == original


def test_absent_coordination_stays_disabled(lwd_yaml_dir: Path) -> None:
    selected = resolve_lwd_config(
        {
            "lwd_config": {
                "path": str(lwd_yaml_dir / FIXTURE_NAME),
                "role": "cloud",
                "instance_id": 0,
            }
        }
    )
    assert selected.coordination is None


@pytest.mark.parametrize(
    "identity,flag", [(0, "--api-server-rpc-port"), (1, "--api-server-attach")]
)
def test_api_cli_metadata_survives_snapshot_handoff(
    tmp_path: Path, identity: int, flag: str, lwd_yaml_dir: Path
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
    # Capturing CLI options does not read YAML; snapshot construction owns that read.
    api = LwdAPIConfig.from_namespace(args)
    path.write_bytes((lwd_yaml_dir / "s2_edge_share_2dp.yaml").read_bytes())
    original = copy.deepcopy(args.additional_config)
    entry = resolve_lwd_config(args.additional_config, api=api)
    path.unlink()
    # Only round-trip our own objects, never externally supplied pickle data.
    received_entry = pickle.loads(pickle.dumps(entry))
    assert received_entry.api == api
    assert received_entry.coordination.enabled is False
    assert [c.dp_idx for c in received_entry.connections] == [0, 1]
    assert received_entry.instance_id == identity
    assert args.additional_config == original
