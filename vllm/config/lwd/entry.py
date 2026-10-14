# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Original path/role/instance entry, without parallel or model projection."""

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ConfigDict, TypeAdapter, with_config
from typing_extensions import NotRequired, TypedDict

from .api import LwdAPIConfig
from .coordination import LwdCoordinationConfig
from .topology import LwdDPConnection, LwdId, LwdInstance, LwdString, LwdTopology
from .transport import LwdTransportConfig


@with_config(ConfigDict(extra="forbid"))
class _LwdEntry(TypedDict):
    path: LwdString
    role: Literal["edge", "cloud"]
    instance_id: LwdId
    coordination: NotRequired[dict[str, Any]]
    transport: NotRequired[dict[str, Any]]


@dataclass(frozen=True)
class LwdConfig:
    path: str
    role: str
    instance_id: int
    topology: LwdTopology
    coordination: LwdCoordinationConfig | None = None
    api: LwdAPIConfig = field(default_factory=LwdAPIConfig)
    transport: LwdTransportConfig = field(default_factory=LwdTransportConfig)

    def __post_init__(self) -> None:
        self.api.validate_topology(self)

    @classmethod
    def from_dict(cls, raw: Any, *, api: LwdAPIConfig | None = None) -> "LwdConfig":
        entry = TypeAdapter(_LwdEntry).validate_python(raw)
        path, role, identity = entry["path"], entry["role"], entry["instance_id"]
        coordination = None
        if "coordination" in entry:
            coordination = LwdCoordinationConfig.from_dict(
                entry["coordination"], cloud_id=identity if role == "cloud" else None
            )
            coordination.validate_role(role)
        topology = LwdTopology.from_file(path)
        topology.instance(role, identity)
        return cls(
            path,
            role,
            identity,
            topology,
            coordination,
            api=api if api is not None else LwdAPIConfig(),
            transport=LwdTransportConfig.from_dict(entry.get("transport", {})),
        )

    @property
    def instance(self) -> LwdInstance:
        return self.topology.instance(self.role, self.instance_id)

    @property
    def connections(self) -> tuple[LwdDPConnection, ...]:
        return self.topology.connections(self.role, self.instance_id)

    @property
    def runtime_connection(self) -> LwdDPConnection:
        """Select the supported runtime link without changing the YAML schema."""
        links = self.connections
        if (
            self.topology.deployment.scene != "single_instance"
            or len(links) != 1
            or links[0].dp_idx != 0
        ):
            raise ValueError("[LWD] runtime currently requires single_instance, DP=1")
        return links[0]
