# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Original path/role/instance entry, without parallel or model projection."""

from dataclasses import dataclass
from typing import Any

from .topology import LwdDPConnection, LwdInstance, LwdTopology, _integer, _mapping


@dataclass(frozen=True)
class LwdConfig:
    path: str
    role: str
    instance_id: int
    topology: LwdTopology

    @classmethod
    def from_dict(cls, raw: Any) -> "LwdConfig":
        raw = _mapping(raw, "lwd_config", {"path", "role", "instance_id"})
        path = raw["path"]
        if not isinstance(path, str) or not path.strip():
            raise ValueError("[LWD] lwd_config.path must be a non-empty string")
        if raw["role"] not in ("edge", "cloud"):
            raise ValueError("[LWD] lwd_config.role must be 'edge' or 'cloud'")
        identity = _integer(raw["instance_id"], "lwd_config.instance_id")
        topology = LwdTopology.from_file(path)
        topology.instance(raw["role"], identity)
        return cls(path, raw["role"], identity, topology)

    @property
    def instance(self) -> LwdInstance:
        return self.topology.instance(self.role, self.instance_id)

    @property
    def connections(self) -> tuple[LwdDPConnection, ...]:
        return self.topology.connections(self.role, self.instance_id)
