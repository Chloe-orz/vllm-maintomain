# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read distributed initialization parameters from the LWD topology."""

from typing import TYPE_CHECKING

from vllm import envs
from vllm.utils.network_utils import get_distributed_init_method

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.lwd import LwdConfig


def get_lwd_rank_groups(config: "LwdConfig") -> list[list[int]]:
    if config.topology.deployment.scene != "single_instance":
        raise ValueError("[LWD] no-PP bootstrap requires single_instance")
    # Edge DPs share an allocation; each physical Worker joins the world once.
    edge_groups = [connection.edge.ranks for connection in config.connections]
    cloud_groups = [connection.cloud.ranks for connection in config.connections]
    allocations = dict.fromkeys(edge_groups + cloud_groups)
    return [list(ranks) for ranks in allocations]


def get_lwd_init_method(config: "LwdConfig") -> str:
    try:
        port = envs.VLLM_LWD_WIRE_STORE_PORT
    except ValueError as exc:
        raise ValueError("[LWD] VLLM_LWD_WIRE_STORE_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("[LWD] VLLM_LWD_WIRE_STORE_PORT must be in [1, 65535]")
    root = next(
        dp
        for connection in config.connections
        for dp in (connection.edge, connection.cloud)
        if 0 in dp.ranks
    )
    return get_distributed_init_method(root.addr, port)


def get_lwd_distributed_args(
    config: "VllmConfig", tp_rank: int
) -> tuple[int, int, str]:
    """Map the executor's TP rank to the global rank assigned by YAML."""
    lwd_config = config.lwd_config
    parallel_config = config.parallel_config
    assert lwd_config is not None
    if lwd_config.topology.deployment.scene != "single_instance":
        raise ValueError("[LWD] no-PP bootstrap requires single_instance")

    dp_idx = parallel_config.data_parallel_index
    ranks = lwd_config.instance.dp[dp_idx].ranks
    if parallel_config.tensor_parallel_size != len(ranks):
        raise ValueError("[LWD] local TP size must match YAML ranks")

    global_rank = ranks[tp_rank]
    return (
        lwd_config.topology.deployment.hccl_world_size,
        global_rank,
        get_lwd_init_method(lwd_config),
    )
