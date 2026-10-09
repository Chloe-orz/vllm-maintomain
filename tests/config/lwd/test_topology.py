# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import yaml
from lwd_config_under_test import LwdTopology

FIXTURES = Path(__file__).parent / "fixtures"


def cloud_reuse(dp: int = 2) -> dict:
    return yaml.safe_load((FIXTURES / f"s3_cloud_share_{dp}dp.yaml").read_text())


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.yaml")), ids=lambda p: p.stem)
def test_original_eight_layouts_keep_rank_and_link_contract(path: Path) -> None:
    topology = LwdTopology.from_file(str(path))
    expected = {
        "single_instance": (9, 1),
        "edge_share": (17, 2),
        "cloud_share": (10, 2),
        "lwd_cluster": (18, 4),
    }
    world, links = expected[topology.deployment.scene]
    assert topology.deployment.hccl_world_size == world
    assert len(topology.instance_links) == links
    dp_count = len(topology.clouds[0].dp)
    assert (
        sum(len(topology.connections("edge", e.id)) for e in topology.edges)
        == links * dp_count
    )


@pytest.mark.parametrize("dp_count", [1, 2])
def test_colocated_edges_keep_distinct_ranks_and_share_cloud(dp_count: int) -> None:
    raw = cloud_reuse(dp_count)
    for dp in raw["edges"][1]["dp"]:
        dp["addr"] = raw["edges"][0]["dp"][0]["addr"]
    topology = LwdTopology.from_dict(raw)
    assert [e.dp[0].ranks for e in topology.edges] == [(0,), (1,)]
    assert topology.deployment.hccl_world_size == 10
    for edge in topology.edges:
        assert {d.ranks for d in edge.dp} == {(edge.id,)}
        assert [c.cloud_id for c in topology.connections("edge", edge.id)] == [
            0
        ] * dp_count


@pytest.mark.parametrize(
    "fault,match",
    [
        ("duplicate_id", "unique and contiguous"),
        ("rank_alias", "distinct rank"),
        ("missing_link", "isolated instance"),
        ("port_collision", "different ctrl_port"),
        ("world_gap", "without gaps"),
        ("dp_mismatch", "matching DP"),
    ],
)
def test_invalid_topology_fails_before_runtime(fault: str, match: str) -> None:
    raw = cloud_reuse()
    if fault == "duplicate_id":
        raw["edges"][1]["id"] = 0
    elif fault == "rank_alias":
        for dp in raw["edges"][1]["dp"]:
            dp.update(addr=raw["edges"][0]["dp"][0]["addr"], ranks=[0])
        raw["deployment"]["hccl_world_size"] = 9
        for dp in raw["clouds"][0]["dp"]:
            dp["ranks"] = [r - 1 for r in dp["ranks"]]
    elif fault == "missing_link":
        raw["instance_links"].pop()
    elif fault == "port_collision":
        raw["clouds"][0]["dp"][1]["ctrl_port"] = 5550
    elif fault == "world_gap":
        raw["deployment"]["hccl_world_size"] += 1
    elif fault == "dp_mismatch":
        raw["edges"][1]["dp"].pop()
    with pytest.raises(ValueError, match=match):
        LwdTopology.from_dict(raw)


def test_duplicate_yaml_keys_are_not_silently_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("deployment: {}\ndeployment: {}\n")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        LwdTopology.from_file(str(path))


def test_role_and_instance_lookup_does_not_fall_back() -> None:
    topology = LwdTopology.from_dict(cloud_reuse())
    with pytest.raises(ValueError, match="Instance not found"):
        topology.instance("edge", 2)
    with pytest.raises(ValueError, match="DP not found"):
        topology.dp("cloud", 0, 2)
