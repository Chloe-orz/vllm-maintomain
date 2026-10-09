# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""File schema for LWD, independent of devices and distributed initialization."""

from __future__ import annotations

import logging
from collections.abc import Hashable
from dataclasses import dataclass
from hashlib import sha256
from ipaddress import ip_address
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

CLOUD_CARDS = 8


def _mapping(
    value: Any, field: str, required: set[str], optional: tuple[str, ...] = ()
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"[LWD] {field} must be a mapping")
    missing = required - value.keys()
    unknown = value.keys() - required - set(optional)
    if missing or unknown:
        raise ValueError(
            f"[LWD] {field}: missing fields={sorted(missing)}, "
            f"unknown fields={sorted(map(str, unknown))}"
        )
    return value


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"[LWD] {field} must be an integer >= {minimum}")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"[LWD] {field} must be a non-empty list")
    return value


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys instead of silently replacing values."""

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Hashable, Any]:
        result: dict[Hashable, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("[LWD] YAML mapping keys must be strings")
            if key in result:
                raise ValueError(f"[LWD] duplicate YAML key: {key!r}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


@dataclass(frozen=True)
class LwdDeployment:
    mode: int
    scene: str
    hccl_world_size: int
    edges_num: int
    clouds_num: int


@dataclass(frozen=True)
class LwdFeatures:
    enable_early_recv: bool = False
    enable_scramble: bool = False


@dataclass(frozen=True)
class LwdDP:
    dp_idx: int
    addr: str
    ranks: tuple[int, ...]
    ctrl_port: int | None = None


@dataclass(frozen=True)
class LwdInstance:
    id: int
    dp: tuple[LwdDP, ...]


@dataclass(frozen=True)
class LwdLink:
    edge: int
    cloud: int


@dataclass(frozen=True)
class LwdDPConnection:
    """One explicit instance link expanded at matching DP indices."""

    edge_id: int
    cloud_id: int
    dp_idx: int
    edge: LwdDP
    cloud: LwdDP

    @property
    def cloud_leader_rank(self) -> int:
        return self.cloud.ranks[0]


@dataclass(frozen=True)
class LwdTopology:
    deployment: LwdDeployment
    feature_ctrl: LwdFeatures
    edges: tuple[LwdInstance, ...]
    clouds: tuple[LwdInstance, ...]
    instance_links: tuple[LwdLink, ...]
    digest: str = ""
    """First 16 hex characters of the source file digest; empty for from_dict."""

    @classmethod
    def from_file(cls, path: str) -> LwdTopology:
        try:
            file_bytes = Path(path).read_bytes()
            raw = yaml.load(file_bytes.decode("utf-8"), Loader=_UniqueKeyLoader)
            topology = cls.from_dict(raw)
            return LwdTopology(
                deployment=topology.deployment,
                feature_ctrl=topology.feature_ctrl,
                edges=topology.edges,
                clouds=topology.clouds,
                instance_links=topology.instance_links,
                digest=sha256(file_bytes).hexdigest()[:16],
            )
        except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
            raise ValueError(f"[LWD] Invalid topology file {path!r}: {exc}") from exc

    @classmethod
    def from_dict(cls, raw: Any) -> LwdTopology:
        raw = _mapping(
            raw,
            "topology",
            {"deployment", "feature_ctrl", "edges", "clouds", "instance_links"},
        )
        deploy = _mapping(
            raw["deployment"],
            "deployment",
            {"mode", "scene", "hccl_world_size", "edges_num", "clouds_num"},
        )
        mode = _integer(deploy["mode"], "deployment.mode")
        if mode != 0:
            raise ValueError(
                "[LWD] deployment.mode: only 0 (prefill_only) is supported"
            )
        scene = deploy["scene"]
        if scene not in ("single_instance", "edge_share", "cloud_share", "lwd_cluster"):
            raise ValueError("[LWD] deployment.scene is not a recognized scene")
        deployment = LwdDeployment(
            mode=mode,
            scene=scene,
            hccl_world_size=_integer(
                deploy["hccl_world_size"], "deployment.hccl_world_size", 1
            ),
            edges_num=_integer(deploy["edges_num"], "deployment.edges_num", 1),
            clouds_num=_integer(deploy["clouds_num"], "deployment.clouds_num", 1),
        )
        features = _mapping(
            raw["feature_ctrl"],
            "feature_ctrl",
            set(),
            ("enable_early_recv", "enable_scramble"),
        )
        for name, value in features.items():
            if type(value) is not bool:
                raise ValueError(f"[LWD] feature_ctrl.{name} must be a boolean")
        # Preserve source order for cloud-machine rank allocation checks.
        preserve_order = scene in ("cloud_share", "lwd_cluster")
        edges = _instances(raw["edges"], "edges", preserve_order=preserve_order)
        clouds = _instances(raw["clouds"], "clouds", preserve_order=preserve_order)
        if deployment.edges_num != len(edges) or deployment.clouds_num != len(clouds):
            raise ValueError(
                "[LWD] deployment instance counts do not match edges/clouds"
            )
        links = []
        for index, value in enumerate(_list(raw["instance_links"], "instance_links")):
            field = f"instance_links[{index}]"
            value = _mapping(value, field, {"edge", "cloud"})
            link = LwdLink(
                _integer(value["edge"], f"{field}.edge"),
                _integer(value["cloud"], f"{field}.cloud"),
            )
            if link in links:
                raise ValueError(f"[LWD] {field}: duplicate link")
            if link.edge >= len(edges) or link.cloud >= len(clouds):
                raise ValueError(f"[LWD] {field}: unknown edge/cloud instance")
            links.append(link)
        topology = cls(deployment, LwdFeatures(**features), edges, clouds, tuple(links))
        topology.validate_layout()
        return topology

    def validate_layout(self) -> None:
        """Validate config structure, including the future multi-DP layout.

        Shared edge ranks count once in the world. Cloud DPs are disjoint.
        This does not create workers, connections or communication groups.
        """
        self._validate_scene()
        rank_addresses: dict[int, str] = {}
        edge_ranks: set[int] = set()
        cloud_ranks: set[int] = set()
        endpoints: set[tuple[str, int | None]] = set()
        role_addresses: dict[str, set[str]] = {"edges": set(), "clouds": set()}
        for role, instances in (("edges", self.edges), ("clouds", self.clouds)):
            for instance in instances:
                for dp in instance.dp:
                    role_addresses[role].add(dp.addr)
                    if len(set(dp.ranks)) != len(dp.ranks):
                        raise ValueError(
                            f"[LWD] {role}[{instance.id}].dp[{dp.dp_idx}]: "
                            "duplicate ranks within one DP"
                        )
                    for rank in dp.ranks:
                        if rank in rank_addresses and rank_addresses[rank] != dp.addr:
                            raise ValueError(
                                f"[LWD] rank {rank} is assigned to different addresses"
                            )
                        rank_addresses[rank] = dp.addr
                        if role == "clouds" and rank in cloud_ranks:
                            raise ValueError("[LWD] Cloud DP ranks must not overlap")
                    if role == "edges":
                        edge_ranks.update(dp.ranks)
                    else:
                        cloud_ranks.update(dp.ranks)
                        endpoint = (dp.addr, dp.ctrl_port)
                        if endpoint in endpoints:
                            raise ValueError(
                                "[LWD] Cloud DPs on the same address must use "
                                "different ctrl_port values"
                            )
                        endpoints.add(endpoint)
        if edge_ranks & cloud_ranks:
            raise ValueError("[LWD] Edge and cloud ranks must not overlap")
        if mixed := role_addresses["edges"] & role_addresses["clouds"]:
            raise ValueError(
                f"[LWD] Addresses must not host both roles: {sorted(mixed)}"
            )
        # IDs belong to separate role namespaces. Check each role independently.
        linked_edges = {link.edge for link in self.instance_links}
        linked_clouds = {link.cloud for link in self.instance_links}
        for role, instances, linked in (
            ("edges", self.edges, linked_edges),
            ("clouds", self.clouds, linked_clouds),
        ):
            for instance in instances:
                if instance.id not in linked:
                    raise ValueError(
                        f"[LWD] {role}[{instance.id}] is not referenced by any "
                        "instance_links entry (isolated instance)"
                    )
        world = self.deployment.hccl_world_size
        if len(rank_addresses) != world or sorted(rank_addresses) != list(range(world)):
            raise ValueError(
                "[LWD] Unique ranks must cover [0, hccl_world_size) without gaps"
            )
        if edge_ranks != set(range(len(edge_ranks))):
            raise ValueError("[LWD] Global ranks must place all edges before clouds")
        for link in self.instance_links:
            edge = self.instance("edge", link.edge)
            cloud = self.instance("cloud", link.cloud)
            if len(edge.dp) != len(cloud.dp):
                raise ValueError(
                    f"[LWD] Linked edge={link.edge}, cloud={link.cloud} "
                    "must have matching DP indices"
                )
        if self.deployment.scene in ("cloud_share", "lwd_cluster"):
            self._validate_cloud_reuse_layout()
        self._warn_layout_recommendations()

    def _validate_scene(self) -> None:
        """Validate logical instance shape, independently of physical host count."""
        scene = self.deployment.scene
        edge_count, cloud_count = len(self.edges), len(self.clouds)
        if scene == "single_instance" and (edge_count, cloud_count) != (1, 1):
            raise ValueError("[LWD] single_instance requires one edge and one cloud")
        if scene == "cloud_share" and (edge_count < 2 or cloud_count != 1):
            raise ValueError("[LWD] cloud_share requires multiple edges and one cloud")
        if scene == "lwd_cluster" and (edge_count < 2 or cloud_count < 2):
            raise ValueError("[LWD] lwd_cluster requires multiple edges and clouds")
        if scene == "edge_share":
            if edge_count != cloud_count or cloud_count < 2:
                raise ValueError(
                    "[LWD] edge_share requires equal multiple edge/cloud IDs"
                )
            placements = {(dp.addr, dp.ranks) for e in self.edges for dp in e.dp}
            if len(placements) != 1 or any(
                dp.ranks != (0,) for e in self.edges for dp in e.dp
            ):
                raise ValueError("[LWD] edge_share instances must share one edge rank")
            if set(self.instance_links) != {LwdLink(e.id, e.id) for e in self.edges}:
                raise ValueError("[LWD] edge_share requires one-to-one instance links")

    def _warn_layout_recommendations(self) -> None:
        """Warn about port numbering without inferring scenes from host count."""
        machines: dict[str, list[LwdDP]] = {}
        for cloud in self.clouds:
            for dp in cloud.dp:
                machines.setdefault(dp.addr, []).append(dp)
        for addr, dps in machines.items():
            ordered = sorted(dps, key=lambda dp: dp.dp_idx)
            first = ordered[0]
            assert first.ctrl_port is not None
            # port_base is a generator input, not a YAML field. Infer a local
            # base from this machine's first DP; never assume port 5550 or
            # compare different machines (a remote DP may restart at 5550).
            base = first.ctrl_port - first.dp_idx
            if any(dp.ctrl_port != base + dp.dp_idx for dp in ordered):
                logger.warning(
                    "[LWD][config] Cloud machine %s ctrl_port values %s deviate "
                    "from port_base + dp_idx (inferred port_base=%d). "
                    "Keeping the configured ports.",
                    addr,
                    [(dp.dp_idx, dp.ctrl_port) for dp in ordered],
                    base,
                )

    def _validate_cloud_reuse_layout(self) -> None:
        """Validate HTML sections 4.2/4.4 without starting shared execution.

        Connections remain explicit: reject missing pairs, never add them.
        Each edge instance owns a distinct rank, including colocated instances.
        DPs of one edge instance share that rank and address.
        """
        expected_links = {
            LwdLink(edge.id, cloud.id) for edge in self.edges for cloud in self.clouds
        }
        if set(self.instance_links) != expected_links:
            raise ValueError(
                "[LWD] Cloud reuse requires explicit full-mesh instance_links "
                "(every edge instance paired with every cloud instance)"
            )

        for edge in self.edges:
            if any(dp.ranks != (edge.id,) for dp in edge.dp):
                raise ValueError(
                    "[LWD] Cloud reuse requires one distinct rank per edge instance; "
                    "all DPs of that instance must share its rank"
                )
            if len({dp.addr for dp in edge.dp}) != 1:
                raise ValueError("[LWD] Edge instance DPs must share one address")

        cloud_machines: dict[str, list[LwdDP]] = {}
        for cloud in self.clouds:
            for dp in cloud.dp:
                cloud_machines.setdefault(dp.addr, []).append(dp)
        cloud_cards = CLOUD_CARDS
        for machine_index, (addr, dps) in enumerate(cloud_machines.items()):
            ranks = sorted(rank for dp in dps for rank in dp.ranks)
            if len(ranks) != cloud_cards or ranks != list(
                range(ranks[0], ranks[0] + cloud_cards)
            ):
                raise ValueError(
                    f"[LWD] Cloud machine {addr} must own eight contiguous ranks"
                )
            expected_start = len(self.edges) + machine_index * cloud_cards
            if ranks[0] != expected_start:
                raise ValueError(
                    f"[LWD] Cloud machine {addr} ranks must follow machine "
                    f"declaration order: expected start {expected_start}, "
                    f"got {ranks[0]}"
                )
            if cloud_cards % len(dps) or any(
                len(dp.ranks) != cloud_cards // len(dps) for dp in dps
            ):
                raise ValueError(
                    f"[LWD] Cloud machine {addr} must split eight ranks "
                    "equally among its DPs"
                )
            # Stable sorting preserves instance declaration order for ties.
            for index, dp in enumerate(sorted(dps, key=lambda item: item.dp_idx)):
                start = ranks[0] + index * (cloud_cards // len(dps))
                if dp.ranks != tuple(range(start, start + len(dp.ranks))):
                    raise ValueError(
                        f"[LWD] Cloud machine {addr} DP ranks must be ordered "
                        "contiguous blocks in DP-index order"
                    )
        expected_world = len(self.edges) + cloud_cards * len(cloud_machines)
        if self.deployment.hccl_world_size != expected_world:
            raise ValueError(
                "[LWD] Cloud reuse hccl_world_size must equal "
                "edge instance count + 8 * cloud machine count"
            )

    def downstream_clouds(self, edge_id: int) -> tuple[LwdInstance, ...]:
        """Return only clouds explicitly linked to this edge instance."""
        self.instance("edge", edge_id)
        return tuple(
            self.instance("cloud", link.cloud)
            for link in self.instance_links
            if link.edge == edge_id
        )

    def upstream_edges(self, cloud_id: int) -> tuple[LwdInstance, ...]:
        """Return only edges explicitly linked to this cloud instance."""
        self.instance("cloud", cloud_id)
        return tuple(
            self.instance("edge", link.edge)
            for link in self.instance_links
            if link.cloud == cloud_id
        )

    def connections(self, role: str, instance_id: int) -> tuple[LwdDPConnection, ...]:
        """Describe every DP connection for an instance without opening sockets."""
        self.instance(role, instance_id)
        result = []
        for link in self.instance_links:
            local_id = link.edge if role == "edge" else link.cloud
            if local_id != instance_id:
                continue
            for edge_dp in self.instance("edge", link.edge).dp:
                result.append(
                    LwdDPConnection(
                        link.edge,
                        link.cloud,
                        edge_dp.dp_idx,
                        edge_dp,
                        self.dp("cloud", link.cloud, edge_dp.dp_idx),
                    )
                )
        return tuple(result)

    def instance(self, role: str, instance_id: int) -> LwdInstance:
        if role not in ("edge", "cloud"):
            raise ValueError("[LWD] role must be 'edge' or 'cloud'")
        instances = self.edges if role == "edge" else self.clouds
        for instance in instances:
            if instance.id == instance_id:
                return instance
        raise ValueError(
            f"[LWD] Instance not found: role={role}, instance_id={instance_id}"
        )

    def dp(self, role: str, instance_id: int, dp_idx: int) -> LwdDP:
        """Resolve an explicit DP; never silently select DP 0 for multi-DP."""
        for dp in self.instance(role, instance_id).dp:
            if dp.dp_idx == dp_idx:
                return dp
        raise ValueError(
            f"[LWD] DP not found: role={role}, instance_id={instance_id}, "
            f"dp_idx={dp_idx}"
        )


def _instances(
    raw: Any, role: str, *, preserve_order: bool = False
) -> tuple[LwdInstance, ...]:
    result = []
    for index, value in enumerate(_list(raw, role)):
        field = f"{role}[{index}]"
        value = _mapping(value, field, {"id", "dp"})
        instance_id = _integer(value["id"], f"{field}.id")
        dps = []
        for dp_index, dp in enumerate(_list(value["dp"], f"{field}.dp")):
            dp_field = f"{field}.dp[{dp_index}]"
            keys = {"dp_idx", "addr", "ranks"}
            if role == "clouds":
                keys.add("ctrl_port")
            dp = _mapping(dp, dp_field, keys)
            dp_idx = _integer(dp["dp_idx"], f"{dp_field}.dp_idx")
            addr = dp["addr"]
            if not isinstance(addr, str):
                raise ValueError(f"[LWD] {dp_field}.addr must be an IP address string")
            try:
                ip = ip_address(addr)
            except ValueError as exc:
                raise ValueError(f"[LWD] {dp_field}.addr: invalid IP {addr!r}") from exc
            if ip.is_unspecified or ip.is_multicast:
                raise ValueError(f"[LWD] {dp_field}.addr must be a unicast endpoint IP")
            ranks = tuple(
                _integer(rank, f"{dp_field}.ranks[{i}]")
                for i, rank in enumerate(_list(dp["ranks"], f"{dp_field}.ranks"))
            )
            port = None
            if role == "clouds":
                port = _integer(dp["ctrl_port"], f"{dp_field}.ctrl_port", 1)
                if port > 65535:
                    raise ValueError(f"[LWD] {dp_field}.ctrl_port must be <= 65535")
            dps.append(LwdDP(dp_idx, str(ip), ranks, port))
        dps.sort(key=lambda dp: dp.dp_idx)
        if [dp.dp_idx for dp in dps] != list(range(len(dps))):
            raise ValueError(
                f"[LWD] {field}.dp_idx must be unique and contiguous from 0"
            )
        result.append(LwdInstance(instance_id, tuple(dps)))
    if sorted(instance.id for instance in result) != list(range(len(result))):
        raise ValueError(f"[LWD] {role}.id must be unique and contiguous from 0")
    if not preserve_order:
        result.sort(key=lambda instance: instance.id)
    return tuple(result)
