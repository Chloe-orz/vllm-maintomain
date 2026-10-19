# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check LWD rank layout and local runtime world without NPU hardware."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.distributed import parallel_state as ps
from vllm.distributed.lwd import get_lwd_distributed_args, get_lwd_rank_groups


def config_for(
    role: str, edge: int = 1, cloud: int = 8, dp: int = 1, dp_rank: int = 0
) -> SimpleNamespace:
    links = tuple(
        SimpleNamespace(
            dp_idx=i,
            edge=SimpleNamespace(addr="76.76.26.18", ranks=tuple(range(edge))),
            cloud=SimpleNamespace(
                addr="76.76.26.234",
                ranks=tuple(range(edge + i * cloud, edge + (i + 1) * cloud)),
            ),
        )
        for i in range(dp)
    )
    return SimpleNamespace(
        lwd_config=SimpleNamespace(
            role=role,
            connections=links,
            instance=SimpleNamespace(dp=tuple(getattr(link, role) for link in links)),
            topology=SimpleNamespace(
                deployment=SimpleNamespace(
                    scene="single_instance",
                    hccl_world_size=edge + dp * cloud,
                )
            ),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=edge if role == "edge" else cloud,
            pipeline_parallel_size=1,
            data_parallel_size=1 if role == "edge" else dp,
            data_parallel_rank=dp_rank,
            data_parallel_index=dp_rank,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            nnodes=1,
            enable_elastic_ep=False,
            use_all2all=False,
            enable_eplb=False,
        ),
        engram_config=None,
        model_config=SimpleNamespace(is_moe=False),
    )


@pytest.mark.parametrize(
    "role,rank,global_rank", [("edge", 0, 0), ("cloud", 0, 1), ("cloud", 7, 8)]
)
@pytest.mark.parametrize("port", [None, 29700])
def test_both_sides_share_bootstrap_address_but_keep_local_device_rank(
    role: str,
    rank: int,
    global_rank: int,
    port: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_LWD_WIRE_STORE_PORT", raising=False)
    if port is not None:
        monkeypatch.setenv("VLLM_LWD_WIRE_STORE_PORT", str(port))
    config = config_for(role)
    assert get_lwd_distributed_args(config, rank) == (
        9,
        global_rank,
        f"tcp://76.76.26.18:{port if port is not None else 29600}",
    )


@pytest.mark.parametrize("port", ["0", "-1", "65536", "invalid", ""])
def test_invalid_wire_store_env_fails_before_rendezvous(
    port: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_LWD_WIRE_STORE_PORT", port)
    with pytest.raises(ValueError, match="VLLM_LWD_WIRE_STORE_PORT"):
        get_lwd_distributed_args(config_for("edge"), 0)


def test_mismatched_tp_size_fails_before_rendezvous() -> None:
    config = config_for("edge")
    config.parallel_config.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="local TP size"):
        get_lwd_distributed_args(config, 0)


@pytest.mark.parametrize("dp", [1, 2, 3])
@pytest.mark.parametrize("rank", [0, 1, 2, 5])
def test_asymmetric_tp_and_singleton_pp_preserve_bootstrap_for_channels(
    monkeypatch: pytest.MonkeyPatch, rank: int, dp: int
) -> None:
    config = config_for("edge" if rank < 2 else "cloud", 2, 4, dp)
    world_size = 2 + 4 * dp
    bootstrap = SimpleNamespace(
        world_size=world_size,
        local_rank=rank if rank < 2 else rank - 2,
        ranks=list(range(world_size)),
    )
    monkeypatch.setattr(ps, "_WORLD", bootstrap)
    for name in (
        "_TP",
        "_ETP",
        "_PP",
        "_DP",
        "_PCP",
        "_DCP",
        "_EP",
        "_EPLB",
        "_LWD_BOOTSTRAP_WORLD",
        "_NODE_COUNT",
    ):
        monkeypatch.setattr(ps, name, None)
    calls = {}

    def create(
        groups: list[list[int]], local_rank: int, backend: str, **kwargs: object
    ) -> SimpleNamespace:
        calls[kwargs["group_name"]] = groups
        ranks = next(group for group in groups if rank in group)
        return SimpleNamespace(ranks=ranks, cpu_group=object(), destroy=Mock())

    monkeypatch.setattr(ps, "init_model_parallel_group", create)
    monkeypatch.setattr(ps, "_node_count", lambda group: 1)
    ps._initialize_lwd_model_parallel(config, "gloo")
    assert calls["tp"] == [[0, 1]] + [
        list(range(2 + 4 * i, 6 + 4 * i)) for i in range(dp)
    ]
    assert calls["dp"] == [[0], [1]] + [
        [2 + pos + 4 * i for i in range(dp)] for pos in range(4)
    ]
    assert all(
        calls[name] == [[i] for i in range(world_size)] for name in ("pp", "pcp", "dcp")
    )
    assert ps.get_lwd_bootstrap_world() is bootstrap
    assert ps.get_world_group() is ps.get_tp_group()
    tp = ps.get_tp_group()
    ps.destroy_model_parallel()
    tp.destroy.assert_called_once()
    assert ps._WORLD is None


@pytest.mark.parametrize("dp", [2, 3])
@pytest.mark.parametrize("edge_size", [1, 2])
def test_shared_edge_and_all_cloud_dps_cover_yaml_world_once(
    dp: int, edge_size: int
) -> None:
    edge = config_for("edge", edge_size, 4, dp)
    ranks = [get_lwd_distributed_args(edge, i)[1] for i in range(edge_size)]
    for dp_rank in range(dp):
        cloud = config_for("cloud", edge_size, 4, dp, dp_rank)
        assert get_lwd_rank_groups(edge.lwd_config) == get_lwd_rank_groups(
            cloud.lwd_config
        )
        for worker_rank in range(4):
            size, rank, address = get_lwd_distributed_args(cloud, worker_rank)
            assert size == edge_size + 4 * dp
            assert address == get_lwd_distributed_args(edge, 0)[2]
            ranks.append(rank)
    assert ranks == list(range(edge_size + 4 * dp))


def test_mapping_uses_yaml_order_instead_of_edge_first_arithmetic() -> None:
    config = config_for("cloud", 1, 2, 2, 1)
    for link, ranks in zip(config.lwd_config.connections, [(0, 3), (1, 2)]):
        link.edge.ranks = (4,)
        link.cloud.ranks = ranks
    assert get_lwd_rank_groups(config.lwd_config) == [[4], [0, 3], [1, 2]]
    assert get_lwd_distributed_args(config, 0) == (5, 1, "tcp://76.76.26.234:29600")


@pytest.mark.parametrize(
    "role,dp_idx,tp_rank,global_rank",
    [("edge", 0, 0, 0), ("cloud", 0, 2, 3), ("cloud", 1, 2, 7)],
)
def test_independent_engine_dp_reset_preserves_yaml_identity(
    role: str, dp_idx: int, tp_rank: int, global_rank: int
) -> None:
    config = config_for(role, 1, 4, 2, dp_idx)
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.data_parallel_rank = 0
    assert get_lwd_distributed_args(config, tp_rank) == (
        9,
        global_rank,
        "tcp://76.76.26.18:29600",
    )


def test_multiple_instances_remain_outside_this_bootstrap() -> None:
    config = config_for("edge")
    config.lwd_config.topology.deployment.scene = "cloud_share"
    with pytest.raises(ValueError, match="single_instance"):
        get_lwd_distributed_args(config, 0)


def test_native_dp_offsets_do_not_overwrite_yaml_rendezvous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.config

    config = config_for("cloud", 1, 4, 2, 1)
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.distributed_executor_backend = "mp"
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(ps.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        ps.torch.distributed, "is_backend_available", lambda backend: True
    )
    monkeypatch.setattr(ps.envs, "VLLM_DISTRIBUTED_USE_SPLIT_GROUP", False)
    init = Mock(side_effect=RuntimeError("captured rendezvous"))
    monkeypatch.setattr(ps.torch.distributed, "init_process_group", init)
    with pytest.raises(RuntimeError, match="captured rendezvous"):
        ps.init_distributed_environment(4, 3, "tcp://ignored:1234", 3, "gloo")
    assert init.call_args.kwargs["world_size"] == 9
    assert init.call_args.kwargs["rank"] == 8
    assert init.call_args.kwargs["init_method"] == "tcp://76.76.26.18:29600"
