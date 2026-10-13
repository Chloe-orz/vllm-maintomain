# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from lwd_config_under_test import LwdTopology
from sample_topologies import topology_payload

ROOT = Path(__file__).resolve().parents[3]
GENERATOR = ROOT / "tools/gen_lwd_topology.py"


def generate(output: Path, **options: str | int) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(GENERATOR), "-o", str(output)]
    for name, value in options.items():
        command.extend([f"--{name.replace('_', '-')}", str(value)])
    return subprocess.run(command, capture_output=True, text=True)


@pytest.mark.parametrize(
    "scene,edges,clouds,prefix",
    [
        ("single_instance", "10.0.0.1", "10.1.0.1", "s1"),
        ("edge_share", "10.0.0.1", "0@10.1.0.1,1@10.1.0.2", "s2"),
        ("cloud_share", "0@10.0.0.1,1@10.0.0.2", "10.1.0.1", "s3"),
        ("lwd_cluster", "10.0.0.1,10.0.0.2", "10.1.0.1,10.1.0.2", "s4"),
    ],
)
@pytest.mark.parametrize("dp", [1, 2])
def test_explicit_card_counts_reproduce_original_payloads(
    tmp_path: Path, scene: str, edges: str, clouds: str, prefix: str, dp: int
) -> None:
    output = tmp_path / "topology.yaml"
    generate(
        output,
        scene=scene,
        edge_machines=edges,
        cloud_machines=clouds,
        cloud_cards="0:8" if prefix in ("s1", "s3") else "0:8,1:8",
        dp=str(dp),
    ).check_returncode()
    expected = topology_payload(f"{prefix}_{scene}_{dp}dp.yaml")
    assert yaml.safe_load(output.read_text()) == expected
    assert list(tmp_path.iterdir()) == [output]


def test_explicit_colocated_ids_generate_independent_ranks(tmp_path: Path) -> None:
    output = tmp_path / "topology.yaml"
    generate(
        output,
        scene="cloud_share",
        edge_machines="1@76.76.26.17,0@76.76.26.17",
        cloud_machines="76.76.26.232",
        cloud_cards="0:8",
        dp="2",
        port_base="15783",
    ).check_returncode()
    raw = yaml.safe_load(output.read_text())
    assert [[d["ranks"] for d in e["dp"]] for e in raw["edges"]] == [
        [[0], [0]],
        [[1], [1]],
    ]
    assert [d["ctrl_port"] for d in raw["clouds"][0]["dp"]] == [15783, 15784]
    assert raw["deployment"]["hccl_world_size"] == 10


@pytest.mark.parametrize(
    "edges,dp",
    [
        ("0@10.0.0.1,0@10.0.0.2", "1"),
        ("0@10.0.0.1,2@10.0.0.2", "1"),
        ("0@10.0.0.1,10.0.0.2", "1"),
        ("0@10.0.0.1,1@10.0.0.2", "1,2"),
    ],
)
def test_invalid_inputs_do_not_overwrite_existing_yaml(
    tmp_path: Path, edges: str, dp: str
) -> None:
    output = tmp_path / "topology.yaml"
    output.write_text("preserve this file\n")
    result = generate(
        output,
        scene="cloud_share",
        edge_machines=edges,
        cloud_machines="10.1.0.1",
        cloud_cards="0:8",
        dp=dp,
    )
    assert result.returncode != 0
    assert output.read_text() == "preserve this file\n"


@pytest.mark.parametrize(
    "clouds,cards,dp,expected_ranks,world",
    [
        ("0@10.1.0.1,1@10.1.0.1", "0:2,1:4", 2, [[[1], [2]], [[3, 4], [5, 6]]], 7),
        ("0@10.1.0.1,1@10.1.0.2", "0:4,1:2", 1, [[[1, 2, 3, 4]], [[5, 6]]], 7),
        ("1@10.1.0.1,0@10.1.0.1", "1:4,0:4", 1, [[[1, 2, 3, 4]], [[5, 6, 7, 8]]], 9),
        (
            "0@10.1.0.1,1@10.1.0.1",
            "0:4,1:4",
            2,
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            9,
        ),
        (
            "0@10.1.0.1,1@10.1.0.2",
            "0:8,1:8",
            1,
            [[list(range(1, 9))], [list(range(9, 17))]],
            17,
        ),
        (
            "0@10.1.0.1,1@10.1.0.2",
            "0:8,1:8",
            2,
            [[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]],
            17,
        ),
    ],
)
def test_edge_share_allocates_explicit_cards_per_instance(
    tmp_path: Path, clouds: str, cards: str, dp: int, expected_ranks: list, world: int
) -> None:
    output = tmp_path / "topology.yaml"
    generate(
        output,
        scene="edge_share",
        edge_machines="10.0.0.1",
        cloud_machines=clouds,
        cloud_cards=cards,
        dp=str(dp),
        port_base="6020",
    ).check_returncode()
    raw = yaml.safe_load(output.read_text())
    assert raw["deployment"]["hccl_world_size"] == world
    assert [[d["ranks"] for d in c["dp"]] for c in raw["clouds"]] == expected_ranks
    assert [[d["ctrl_port"] for d in c["dp"]] for c in raw["clouds"]] == [
        list(range(6020, 6020 + dp))
    ] * 2
    assert {tuple(d["ranks"]) for e in raw["edges"] for d in e["dp"]} == {(0,)}
    assert raw["instance_links"] == [{"edge": 0, "cloud": 0}, {"edge": 1, "cloud": 1}]


@pytest.mark.parametrize(
    "clouds,dp,port",
    [
        ("0@10.1.0.1,0@10.1.0.2", "1", "6020"),
        ("0@10.1.0.1,2@10.1.0.2", "1", "6020"),
        ("0@10.1.0.1,10.1.0.2", "1", "6020"),
        ("10.1.0.1,10.1.0.1", "1", "6020"),
        ("0@10.1.0.1,1@10.1.0.1,2@10.1.0.1", "1", "6020"),
        ("0@10.1.0.1,1@10.1.0.1", "8", "6020"),
        ("0@10.1.0.1,1@10.1.0.2", "2", "65535"),
    ],
)
def test_invalid_cloud_allocation_preserves_output(
    tmp_path: Path, clouds: str, dp: str, port: str
) -> None:
    output = tmp_path / "topology.yaml"
    output.write_text("preserve this file\n")
    result = generate(
        output,
        scene="edge_share",
        edge_machines="10.0.0.1",
        cloud_machines=clouds,
        cloud_cards="0:4,1:4",
        dp=dp,
        port_base=port,
    )
    assert result.returncode != 0
    assert output.read_text() == "preserve this file\n"


@pytest.mark.parametrize(
    "scene,edges",
    [
        ("single_instance", "10.0.0.1"),
        ("cloud_share", "0@10.0.0.1,1@10.0.0.1"),
    ],
)
@pytest.mark.parametrize("cards,dp", [(4, 1), (4, 2), (16, 2)])
def test_cloud_size_is_not_inferred_from_host_address(
    tmp_path: Path, scene: str, edges: str, cards: int, dp: int
) -> None:
    output = tmp_path / "topology.yaml"
    generate(
        output,
        scene=scene,
        edge_machines=edges,
        cloud_machines="10.1.0.1",
        cloud_cards=f"0:{cards}",
        dp=str(dp),
    ).check_returncode()
    topology = LwdTopology.from_file(str(output))
    edge_count = 1 if scene == "single_instance" else 2
    assert topology.deployment.hccl_world_size == edge_count + cards
    assert [len(d.ranks) for d in topology.clouds[0].dp] == [cards // dp] * dp


@pytest.mark.parametrize(
    "cards,dp,match",
    [
        (None, "1", "required"),
        ("0:0", "1", "positive"),
        ("0:-1", "1", "ID:count"),
        ("0:3", "2", "divisible"),
        ("0:4,0:4", "1", "duplicate"),
        ("1:4", "1", "every cloud instance"),
        ("0:4,1:4", "1", "every cloud instance"),
        ("4", "1", "ID:count"),
        ("0:4", "0", "positive integer"),
    ],
)
def test_invalid_card_counts_preserve_existing_yaml(
    tmp_path: Path, cards: str | None, dp: str, match: str
) -> None:
    output = tmp_path / "topology.yaml"
    output.write_text("keep existing yaml\n")
    command = [
        sys.executable,
        str(GENERATOR),
        "--scene",
        "single_instance",
        "--edge-machines",
        "10.0.0.1",
        "--cloud-machines",
        "10.1.0.1",
        "--dp",
        dp,
        "-o",
        str(output),
    ]
    if cards is not None:
        command.extend(["--cloud-cards", cards])
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert match in result.stderr
    assert output.read_text() == "keep existing yaml\n"


@pytest.mark.parametrize(
    "scene,edges,clouds,cards",
    [
        ("single_instance", "10.1.0.1", "10.1.0.1", "0:6"),
        ("cloud_share", "0@10.0.0.1,1@10.0.0.1,2@10.0.0.2", "10.1.0.1", "0:12"),
        ("edge_share", "10.0.0.1", "0@10.1.0.1,1@10.1.0.2,2@10.1.0.1", "0:3,1:6,2:9"),
        (
            "lwd_cluster",
            "0@10.0.0.1,1@10.0.0.1,2@10.0.0.2",
            "0@10.1.0.1,1@10.1.0.2,2@10.1.0.1",
            "0:3,1:6,2:9",
        ),
    ],
)
def test_allocation_rules_scale_beyond_two_instances_and_two_dps(
    tmp_path: Path, scene: str, edges: str, clouds: str, cards: str
) -> None:
    output = tmp_path / "topology.yaml"
    result = generate(
        output,
        scene=scene,
        edge_machines=edges,
        cloud_machines=clouds,
        cloud_cards=cards,
        dp=3,
        port_base=6201,
    )
    assert result.returncode == 0, result.stderr
    topology = LwdTopology.from_file(str(output))
    counts = {
        int(identity): int(count)
        for identity, count in (entry.split(":") for entry in cards.split(","))
    }
    for cloud in topology.clouds:
        assert [len(dp.ranks) for dp in cloud.dp] == [counts[cloud.id] // 3] * 3
    edge_ranks = {r for edge in topology.edges for dp in edge.dp for r in dp.ranks}
    cloud_ranks = [r for cloud in topology.clouds for dp in cloud.dp for r in dp.ranks]
    assert len(cloud_ranks) == len(set(cloud_ranks)) == sum(counts.values())
    assert edge_ranks.isdisjoint(cloud_ranks)
    assert edge_ranks | set(cloud_ranks) == set(
        range(topology.deployment.hccl_world_size)
    )
    if scene == "lwd_cluster":
        endpoints = [
            (dp.addr, dp.ctrl_port) for cloud in topology.clouds for dp in cloud.dp
        ]
        assert len(set(endpoints)) == len(endpoints)


def test_colocated_cloud_port_overflow_preserves_output(tmp_path: Path) -> None:
    output = tmp_path / "topology.yaml"
    output.write_text("keep existing yaml\n")
    result = generate(
        output,
        scene="lwd_cluster",
        edge_machines="10.0.0.1,10.0.0.2",
        cloud_machines="0@10.1.0.1,1@10.1.0.1",
        cloud_cards="0:4,1:4",
        port_base=65535,
    )
    assert result.returncode != 0
    assert "ctrl_port" in result.stderr
    assert output.read_text() == "keep existing yaml\n"


@pytest.mark.parametrize(
    "scene,edges,clouds,cards,match",
    [
        (
            "edge_share",
            "10.0.0.1",
            "10.1.0.1,10.1.0.2",
            "0:4,1:4",
            "--cloud-machines ID@IP",
        ),
        (
            "edge_share",
            "0@10.0.0.1,1@10.0.0.1",
            "0@10.1.0.1,1@10.1.0.1",
            "0:4,1:4",
            "one plain edge IP",
        ),
        (
            "cloud_share",
            "10.0.0.1,10.0.0.2",
            "10.1.0.1",
            "0:8",
            "--edge-machines ID@IP",
        ),
    ],
)
def test_scene_requires_ids_only_where_instances_are_listed(
    tmp_path: Path, scene: str, edges: str, clouds: str, cards: str, match: str
) -> None:
    output = tmp_path / "topology.yaml"
    output.write_text("keep existing yaml\n")
    result = generate(
        output,
        scene=scene,
        edge_machines=edges,
        cloud_machines=clouds,
        cloud_cards=cards,
    )
    assert result.returncode != 0
    assert match in result.stderr
    assert output.read_text() == "keep existing yaml\n"
