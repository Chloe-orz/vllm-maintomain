# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import asdict
from pathlib import Path

import pytest
from lwd_config_under_test import LwdConfig, LwdTopology, LwdTransportConfig
from sample_topologies import topology_payload


def test_transport_uses_yaml_addresses_without_reusing_ctrl_port() -> None:
    topology = LwdTopology.from_dict(topology_payload("s1_single_instance_1dp.yaml"))
    link = topology.connections("edge", 0)[0]
    assert LwdTransportConfig().endpoints(link) == {
        "pre_out": "tcp://10.1.0.1:5558",
        "post_out": "tcp://10.0.0.1:5559",
        "wire_store": "tcp://10.0.0.1:29600",
    }
    assert link.cloud.ctrl_port == 5550


def test_legacy_overrides_are_snapshotted_with_their_source(
    monkeypatch: pytest.MonkeyPatch, lwd_yaml_dir: Path
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_LWD_PRE_OUT_PORT", "6100")
    monkeypatch.setenv("VLLM_ASCEND_LWD_PRE_OUT_HOST", "10.2.2.2")
    cfg = LwdConfig.from_dict(
        {
            "path": str(lwd_yaml_dir / "s1_single_instance_1dp.yaml"),
            "role": "cloud",
            "instance_id": 0,
            "transport": {"pre_out_port": 6000, "wire_store_port": 29601},
        }
    )
    monkeypatch.setenv("VLLM_ASCEND_LWD_PRE_OUT_PORT", "6200")
    assert cfg.transport.pre_out_port == 6100
    assert cfg.transport.pre_out_host == "10.2.2.2"
    assert cfg.transport.wire_store_port == 29601
    assert asdict(cfg)["transport"]["sources"] == {
        "pre_out_port": "VLLM_ASCEND_LWD_PRE_OUT_PORT",
        "pre_out_host": "VLLM_ASCEND_LWD_PRE_OUT_HOST",
        "wire_store_port": "lwd_config.transport",
    }


@pytest.mark.parametrize(
    "raw",
    [
        {"pre_out_port": 0},
        {"post_out_port": 65536},
        {"wire_store_port": True},
        {"pre_out_host": "0.0.0.0"},
        {"post_out_host": "tcp://10.0.0.1:5559"},
        {"hello_timeout_s": float("nan")},
        {"wire_timeout_s": 0},
        {"publish_queue_max": -1},
        {"unknown": 1},
    ],
)
def test_invalid_transport_fails_before_sockets_are_opened(raw: dict) -> None:
    with pytest.raises(ValueError):
        LwdTransportConfig.from_dict(raw, environ={})


def test_bad_legacy_port_does_not_silently_use_a_different_port() -> None:
    with pytest.raises(ValueError, match="VLLM_ASCEND_LWD_PRE_OUT_PORT"):
        LwdTransportConfig.from_dict(
            {}, environ={"VLLM_ASCEND_LWD_PRE_OUT_PORT": "bad"}
        )


def test_colocated_listener_collision_is_rejected() -> None:
    topology = LwdTopology.from_dict(topology_payload("s1_single_instance_1dp.yaml"))
    config = LwdTransportConfig(pre_out_host="::1", post_out_host="::1")
    link = topology.connections("edge", 0)[0]
    assert config.endpoints(link)["pre_out"] == "tcp://[::1]:5558"
    with pytest.raises(ValueError, match="distinct endpoints"):
        LwdTransportConfig(
            pre_out_host="::1",
            post_out_host="::1",
            post_out_port=5558,
        ).endpoints(link)


def test_multicard_single_instance_has_the_same_key_at_both_ends() -> None:
    raw = topology_payload("s1_single_instance_1dp.yaml")
    raw["edges"][0]["dp"][0]["ranks"] = [8, 0]
    raw["clouds"][0]["dp"][0]["ranks"] = [7, 6, 5, 4, 3, 2, 1]
    topology = LwdTopology.from_dict(raw)
    edge = topology.connections("edge", 0)[0]
    cloud = topology.connections("cloud", 0)[0]
    assert edge.key == cloud.key
    assert asdict(edge.key) == {"edge_id": 0, "cloud_id": 0, "dp_idx": 0}
    assert edge.edge.ranks == (8, 0)
    assert edge.cloud_leader_rank == 7
