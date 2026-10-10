# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Original path/role/instance entry, without parallel or model projection."""

from dataclasses import dataclass
from typing import Any

from .coordination import LwdCoordinationConfig
from .topology import LwdDPConnection, LwdInstance, LwdTopology, _integer, _mapping


def lwd_entry_from_dict(raw: Any) -> tuple[str, str, int]:
    """Validate the entry without reading YAML during early CLI checks."""
    raw = _mapping(
        raw, "lwd_config", {"path", "role", "instance_id"}, ("coordination",)
    )
    path = raw["path"]
    if not isinstance(path, str) or not path.strip():
        raise ValueError("[LWD] lwd_config.path must be a non-empty string")
    if raw["role"] not in ("edge", "cloud"):
        raise ValueError("[LWD] lwd_config.role must be 'edge' or 'cloud'")
    identity = _integer(raw["instance_id"], "lwd_config.instance_id")
    return path, raw["role"], identity


@dataclass(frozen=True)
class LwdConfig:
    path: str
    role: str
    instance_id: int
    topology: LwdTopology
    coordination: LwdCoordinationConfig | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> "LwdConfig":
        path, role, identity = lwd_entry_from_dict(raw)
        coordination = None
        if "coordination" in raw:
            coordination = LwdCoordinationConfig.from_dict(
                raw["coordination"], cloud_id=identity if role == "cloud" else None
            )
            coordination.validate_role(role)
        topology = LwdTopology.from_file(path)
        topology.instance(role, identity)
        return cls(path, role, identity, topology, coordination)

    @property
    def instance(self) -> LwdInstance:
        return self.topology.instance(self.role, self.instance_id)

    @property
    def connections(self) -> tuple[LwdDPConnection, ...]:
        return self.topology.connections(self.role, self.instance_id)
