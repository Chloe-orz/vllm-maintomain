# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate one LWD topology without importing the inference runtime."""

import argparse
import importlib.util
import sys
from ipaddress import ip_address
from pathlib import Path
from typing import Any

import yaml

CLOUD_CARDS = 8
SCENES = ("single_instance", "edge_share", "cloud_share", "lwd_cluster")


def _addresses(value: str) -> list[str]:
    addresses = [str(ip_address(item.strip())) for item in value.split(",")]
    if len(set(addresses)) != len(addresses):
        raise ValueError(
            "Machine addresses must be unique; use ID@address for instances"
        )
    return addresses


def _edge_instances(scene: str, value: str, cloud_count: int) -> list[tuple[int, str]]:
    items = [item.strip() for item in value.split(",")]
    if any("@" in item for item in items):
        if not all(item.count("@") == 1 for item in items):
            raise ValueError("Do not mix plain addresses and ID@address entries")
        instances = []
        for item in items:
            identity, address = item.split("@")
            if not identity.isdecimal():
                raise ValueError("Edge instance IDs must be non-negative integers")
            instances.append((int(identity), str(ip_address(address))))
        instances.sort()
        if [identity for identity, _ in instances] != list(range(len(instances))):
            raise ValueError("Edge instance IDs must be unique and contiguous from 0")
        return instances
    addresses = _addresses(value)
    if scene == "edge_share":
        if len(addresses) != 1:
            raise ValueError("edge_share requires one edge machine")
        addresses *= cloud_count
    return list(enumerate(addresses))


def build_topology(
    scene: str, edge_machines: str, cloud_machines: str, dp: int, port_base: int
) -> dict[str, Any]:
    """Allocate global ranks; host-local physical devices remain launch settings."""
    if scene not in SCENES:
        raise ValueError(f"Unknown scene: {scene}")
    if dp not in (1, 2, 4, 8):
        raise ValueError("DP must be one integer in 1, 2, 4, 8")
    if not 1 <= port_base <= 65535 - dp + 1:
        raise ValueError("All generated control ports must be within 1..65535")
    cloud_addresses = _addresses(cloud_machines)
    instances = _edge_instances(scene, edge_machines, len(cloud_addresses))
    edges = []
    for identity, address in instances:
        rank = 0 if scene == "edge_share" else identity
        edges.append(
            {
                "id": identity,
                "dp": [
                    {"dp_idx": index, "addr": address, "ranks": [rank]}
                    for index in range(dp)
                ],
            }
        )
    # Distinct cloud-reuse instances retain distinct ranks on the same host.
    edge_rank_count = 1 if scene == "edge_share" else len(edges)
    clouds = []
    for identity, address in enumerate(cloud_addresses):
        base = edge_rank_count + identity * CLOUD_CARDS
        per_dp = CLOUD_CARDS // dp
        clouds.append(
            {
                "id": identity,
                "dp": [
                    {
                        "dp_idx": index,
                        "addr": address,
                        "ranks": list(
                            range(base + index * per_dp, base + (index + 1) * per_dp)
                        ),
                        "ctrl_port": port_base + index,
                    }
                    for index in range(dp)
                ],
            }
        )
    links = (
        [{"edge": edge["id"], "cloud": edge["id"]} for edge in edges]
        if scene == "edge_share"
        else [
            {"edge": edge["id"], "cloud": cloud["id"]}
            for edge in edges
            for cloud in clouds
        ]
    )
    return {
        "deployment": {
            "mode": 0,
            "scene": scene,
            "hccl_world_size": edge_rank_count + CLOUD_CARDS * len(clouds),
            "edges_num": len(edges),
            "clouds_num": len(clouds),
        },
        "feature_ctrl": {"enable_early_recv": False, "enable_scramble": False},
        "edges": edges,
        "clouds": clouds,
        "instance_links": links,
    }


def _validate(raw: dict[str, Any]) -> None:
    # Use the runtime schema without importing torch, plugins or a device backend.
    path = Path(__file__).resolve().parents[1] / "vllm/config/lwd/topology.py"
    spec = importlib.util.spec_from_file_location("lwd_topology_standalone", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.LwdTopology.from_dict(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, choices=SCENES)
    parser.add_argument(
        "--edge-machines",
        required=True,
        help="Comma-separated IPs or ID@IP entries; never physical NPU IDs",
    )
    parser.add_argument(
        "--cloud-machines", required=True, help="Comma-separated cloud IPs"
    )
    parser.add_argument("--dp", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--port-base", type=int, default=5550)
    parser.add_argument("--enable-early-recv", action="store_true")
    parser.add_argument("--enable-scramble", action="store_true")
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="One output YAML file"
    )
    args = parser.parse_args()
    try:
        raw = build_topology(
            args.scene, args.edge_machines, args.cloud_machines, args.dp, args.port_base
        )
        raw["feature_ctrl"] = {
            "enable_early_recv": args.enable_early_recv,
            "enable_scramble": args.enable_scramble,
        }
        _validate(raw)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
