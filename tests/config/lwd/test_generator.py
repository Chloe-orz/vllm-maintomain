# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
GENERATOR = ROOT / "tools/gen_lwd_topology.py"
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "scene,edges,clouds,prefix",
    [
        ("single_instance", "10.0.0.1", "10.1.0.1", "s1"),
        ("edge_share", "10.0.0.1", "10.1.0.1,10.1.0.2", "s2"),
        ("cloud_share", "10.0.0.1,10.0.0.2", "10.1.0.1", "s3"),
        ("lwd_cluster", "10.0.0.1,10.0.0.2", "10.1.0.1,10.1.0.2", "s4"),
    ],
)
@pytest.mark.parametrize("dp", [1, 2])
def test_legacy_commands_reproduce_original_payloads(
    tmp_path: Path, scene: str, edges: str, clouds: str, prefix: str, dp: int
) -> None:
    output = tmp_path / "topology.yaml"
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--scene",
            scene,
            "--edge-machines",
            edges,
            "--cloud-machines",
            clouds,
            "--dp",
            str(dp),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    expected = yaml.safe_load((FIXTURES / f"{prefix}_{scene}_{dp}dp.yaml").read_text())
    assert yaml.safe_load(output.read_text()) == expected
    assert list(tmp_path.iterdir()) == [output]


def test_explicit_colocated_ids_generate_independent_ranks(tmp_path: Path) -> None:
    output = tmp_path / "topology.yaml"
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--scene",
            "cloud_share",
            "--edge-machines",
            "1@76.76.26.17,0@76.76.26.17",
            "--cloud-machines",
            "76.76.26.232",
            "--dp",
            "2",
            "--port-base",
            "15783",
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
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
        ("10.0.0.1,10.0.0.2", "1,2"),
    ],
)
def test_invalid_inputs_do_not_overwrite_existing_yaml(
    tmp_path: Path, edges: str, dp: str
) -> None:
    output = tmp_path / "topology.yaml"
    output.write_text("preserve this file\n")
    result = subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            "--scene",
            "cloud_share",
            "--edge-machines",
            edges,
            "--cloud-machines",
            "10.1.0.1",
            "--dp",
            dp,
            "-o",
            str(output),
        ],
        capture_output=True,
    )
    assert result.returncode != 0
    assert output.read_text() == "preserve this file\n"
