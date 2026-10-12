# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed reference layouts, independent of the production generator."""

from copy import deepcopy
from typing import Any

# Preserve the original eight reference YAML payloads as test data.
PAYLOADS: dict[str, dict[str, Any]] = {
    "s1_single_instance_1dp.yaml": {
        "deployment": {
            "mode": 0,
            "scene": "single_instance",
            "hccl_world_size": 9,
            "edges_num": 1,
            "clouds_num": 1,
        },
        "feature_ctrl": {"enable_early_recv": False, "enable_scramble": False},
        "edges": [{"id": 0, "dp": [{"dp_idx": 0, "addr": "10.0.0.1", "ranks": [0]}]}],
        "clouds": [
            {
                "id": 0,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "10.1.0.1",
                        "ranks": [1, 2, 3, 4, 5, 6, 7, 8],
                        "ctrl_port": 5550,
                    }
                ],
            }
        ],
        "instance_links": [{"edge": 0, "cloud": 0}],
    },
    "s1_single_instance_2dp.yaml": {
        "deployment": {
            "mode": 0,
            "scene": "single_instance",
            "hccl_world_size": 9,
            "edges_num": 1,
            "clouds_num": 1,
        },
        "feature_ctrl": {"enable_early_recv": False, "enable_scramble": False},
        "edges": [
            {
                "id": 0,
                "dp": [
                    {"dp_idx": 0, "addr": "10.0.0.1", "ranks": [0]},
                    {"dp_idx": 1, "addr": "10.0.0.1", "ranks": [0]},
                ],
            }
        ],
        "clouds": [
            {
                "id": 0,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "10.1.0.1",
                        "ranks": [1, 2, 3, 4],
                        "ctrl_port": 5550,
                    },
                    {
                        "dp_idx": 1,
                        "addr": "10.1.0.1",
                        "ranks": [5, 6, 7, 8],
                        "ctrl_port": 5551,
                    },
                ],
            }
        ],
        "instance_links": [{"edge": 0, "cloud": 0}],
    },
    "s2_edge_share_1dp.yaml": {
        "deployment": {
            "mode": 0,
            "scene": "edge_share",
            "hccl_world_size": 17,
            "edges_num": 2,
            "clouds_num": 2,
        },
        "feature_ctrl": {"enable_early_recv": False, "enable_scramble": False},
        "edges": [
            {"id": 0, "dp": [{"dp_idx": 0, "addr": "10.0.0.1", "ranks": [0]}]},
            {"id": 1, "dp": [{"dp_idx": 0, "addr": "10.0.0.1", "ranks": [0]}]},
        ],
        "clouds": [
            {
                "id": 0,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "10.1.0.1",
                        "ranks": [1, 2, 3, 4, 5, 6, 7, 8],
                        "ctrl_port": 5550,
                    }
                ],
            },
            {
                "id": 1,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "10.1.0.2",
                        "ranks": [9, 10, 11, 12, 13, 14, 15, 16],
                        "ctrl_port": 5550,
                    }
                ],
            },
        ],
        "instance_links": [{"edge": 0, "cloud": 0}, {"edge": 1, "cloud": 1}],
    },
    "s2_edge_share_2dp.yaml": {
        "deployment": {
            "mode": 0,
            "scene": "edge_share",
            "hccl_world_size": 17,
            "edges_num": 2,
            "clouds_num": 2,
        },
        "feature_ctrl": {"enable_early_recv": False, "enable_scramble": False},
        "edges": [
            {
                "id": 0,
                "dp": [
                    {"dp_idx": 0, "addr": "10.0.0.1", "ranks": [0]},
                    {"dp_idx": 1, "addr": "10.0.0.1", "ranks": [0]},
                ],
            },
            {
                "id": 1,
                "dp": [
                    {"dp_idx": 0, "addr": "10.0.0.1", "ranks": [0]},
                    {"dp_idx": 1, "addr": "10.0.0.1", "ranks": [0]},
                ],
            },
        ],
        "clouds": [
            {
                "id": 0,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "10.1.0.1",
                        "ranks": [1, 2, 3, 4],
                        "ctrl_port": 5550,
                    },
                    {
                        "dp_idx": 1,
                        "addr": "10.1.0.1",
                        "ranks": [5, 6, 7, 8],
                        "ctrl_port": 5551,
                    },
                ],
            },
            {
                "id": 1,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "10.1.0.2",
                        "ranks": [9, 10, 11, 12],
                        "ctrl_port": 5550,
                    },
                    {
                        "dp_idx": 1,
                        "addr": "10.1.0.2",
                        "ranks": [13, 14, 15, 16],
                        "ctrl_port": 5551,
                    },
                ],
            },
        ],
        "instance_links": [{"edge": 0, "cloud": 0}, {"edge": 1, "cloud": 1}],
    },
    "s3_cloud_share_1dp.yaml": {
        "deployment": {
            "mode": 0,
            "scene": "cloud_share",
            "hccl_world_size": 10,
            "edges_num": 2,
            "clouds_num": 1,
        },
        "feature_ctrl": {"enable_early_recv": False, "enable_scramble": False},
        "edges": [
            {"id": 0, "dp": [{"dp_idx": 0, "addr": "10.0.0.1", "ranks": [0]}]},
            {"id": 1, "dp": [{"dp_idx": 0, "addr": "10.0.0.2", "ranks": [1]}]},
        ],
        "clouds": [
            {
                "id": 0,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "10.1.0.1",
                        "ranks": [2, 3, 4, 5, 6, 7, 8, 9],
                        "ctrl_port": 5550,
                    }
                ],
            }
        ],
        "instance_links": [{"edge": 0, "cloud": 0}, {"edge": 1, "cloud": 0}],
    },
    "s3_cloud_share_2dp.yaml": {
        "deployment": {
            "mode": 0,
            "scene": "cloud_share",
            "hccl_world_size": 10,
            "edges_num": 2,
            "clouds_num": 1,
        },
        "feature_ctrl": {"enable_early_recv": False, "enable_scramble": False},
        "edges": [
            {
                "id": 0,
                "dp": [
                    {"dp_idx": 0, "addr": "10.0.0.1", "ranks": [0]},
                    {"dp_idx": 1, "addr": "10.0.0.1", "ranks": [0]},
                ],
            },
            {
                "id": 1,
                "dp": [
                    {"dp_idx": 0, "addr": "10.0.0.2", "ranks": [1]},
                    {"dp_idx": 1, "addr": "10.0.0.2", "ranks": [1]},
                ],
            },
        ],
        "clouds": [
            {
                "id": 0,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "10.1.0.1",
                        "ranks": [2, 3, 4, 5],
                        "ctrl_port": 5550,
                    },
                    {
                        "dp_idx": 1,
                        "addr": "10.1.0.1",
                        "ranks": [6, 7, 8, 9],
                        "ctrl_port": 5551,
                    },
                ],
            }
        ],
        "instance_links": [{"edge": 0, "cloud": 0}, {"edge": 1, "cloud": 0}],
    },
    "s4_lwd_cluster_1dp.yaml": {
        "deployment": {
            "mode": 0,
            "scene": "lwd_cluster",
            "hccl_world_size": 18,
            "edges_num": 2,
            "clouds_num": 2,
        },
        "feature_ctrl": {"enable_early_recv": False, "enable_scramble": False},
        "edges": [
            {"id": 0, "dp": [{"dp_idx": 0, "addr": "10.0.0.1", "ranks": [0]}]},
            {"id": 1, "dp": [{"dp_idx": 0, "addr": "10.0.0.2", "ranks": [1]}]},
        ],
        "clouds": [
            {
                "id": 0,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "10.1.0.1",
                        "ranks": [2, 3, 4, 5, 6, 7, 8, 9],
                        "ctrl_port": 5550,
                    }
                ],
            },
            {
                "id": 1,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "10.1.0.2",
                        "ranks": [10, 11, 12, 13, 14, 15, 16, 17],
                        "ctrl_port": 5550,
                    }
                ],
            },
        ],
        "instance_links": [
            {"edge": 0, "cloud": 0},
            {"edge": 0, "cloud": 1},
            {"edge": 1, "cloud": 0},
            {"edge": 1, "cloud": 1},
        ],
    },
    "s4_lwd_cluster_2dp.yaml": {
        "deployment": {
            "mode": 0,
            "scene": "lwd_cluster",
            "hccl_world_size": 18,
            "edges_num": 2,
            "clouds_num": 2,
        },
        "feature_ctrl": {"enable_early_recv": False, "enable_scramble": False},
        "edges": [
            {
                "id": 0,
                "dp": [
                    {"dp_idx": 0, "addr": "10.0.0.1", "ranks": [0]},
                    {"dp_idx": 1, "addr": "10.0.0.1", "ranks": [0]},
                ],
            },
            {
                "id": 1,
                "dp": [
                    {"dp_idx": 0, "addr": "10.0.0.2", "ranks": [1]},
                    {"dp_idx": 1, "addr": "10.0.0.2", "ranks": [1]},
                ],
            },
        ],
        "clouds": [
            {
                "id": 0,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "10.1.0.1",
                        "ranks": [2, 3, 4, 5],
                        "ctrl_port": 5550,
                    },
                    {
                        "dp_idx": 1,
                        "addr": "10.1.0.1",
                        "ranks": [6, 7, 8, 9],
                        "ctrl_port": 5551,
                    },
                ],
            },
            {
                "id": 1,
                "dp": [
                    {
                        "dp_idx": 0,
                        "addr": "10.1.0.2",
                        "ranks": [10, 11, 12, 13],
                        "ctrl_port": 5550,
                    },
                    {
                        "dp_idx": 1,
                        "addr": "10.1.0.2",
                        "ranks": [14, 15, 16, 17],
                        "ctrl_port": 5551,
                    },
                ],
            },
        ],
        "instance_links": [
            {"edge": 0, "cloud": 0},
            {"edge": 0, "cloud": 1},
            {"edge": 1, "cloud": 0},
            {"edge": 1, "cloud": 1},
        ],
    },
}


def topology_payload(name: str) -> dict[str, Any]:
    return deepcopy(PAYLOADS[name])
