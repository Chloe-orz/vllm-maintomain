# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup snapshot of the existing PUSH/PULL and wire rendezvous settings."""

import os
from collections.abc import Mapping
from dataclasses import field
from typing import Annotated, Any

from pydantic import ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass

from .topology import LwdCount, LwdDPConnection, LwdPort, LwdString

LwdTimeout = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]

# Keep existing launch scripts usable; read overrides only during parsing.
_ENV_FIELDS = {
    "pre_out_host": ("VLLM_ASCEND_LWD_PRE_OUT_HOST", str),
    "pre_out_port": ("VLLM_ASCEND_LWD_PRE_OUT_PORT", int),
    "post_out_port": ("VLLM_ASCEND_LWD_POST_OUT_PORT", int),
    "hello_timeout_s": ("VLLM_ASCEND_LWD_HELLO_TIMEOUT_S", float),
}


def _endpoint(host: str, port: int) -> str:
    return f"tcp://{'[' + host + ']' if ':' in host else host}:{port}"


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LwdTransportConfig:
    pre_out_host: LwdString | None = None
    post_out_host: LwdString | None = None
    pre_out_port: LwdPort = 5558
    post_out_port: LwdPort = 5559
    wire_store_port: LwdPort = 29600
    hello_timeout_s: LwdTimeout = 600.0
    wire_timeout_s: LwdTimeout = 1200.0
    publish_queue_max: LwdCount = 1000
    sources: dict[str, str] = field(default_factory=dict, init=False)

    @field_validator("pre_out_host", "post_out_host")
    @classmethod
    def validate_host(cls, value: str | None) -> str | None:
        if value is not None and (
            value in ("*", "0.0.0.0", "::")
            or any(char.isspace() for char in value)
            or any(char in value for char in "/[]")
        ):
            raise ValueError("[LWD] transport host must be a peer address, not a URL")
        return value

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, environ: Mapping[str, str] | None = None
    ) -> "LwdTransportConfig":
        values = dict(raw)
        sources = {name: "lwd_config.transport" for name in raw}
        for name, (variable, convert) in _ENV_FIELDS.items():
            value = (os.environ if environ is None else environ).get(variable)
            if value is not None:
                try:
                    values[name] = convert(value)
                except ValueError as exc:
                    raise ValueError(f"[LWD] invalid {variable}={value!r}") from exc
                sources[name] = variable
        config = cls(**values)
        config.sources.update(sources)
        return config

    def endpoints(self, connection: LwdDPConnection) -> dict[str, str]:
        """Resolve against one explicit link; YAML ctrl_port is kept separate."""
        cloud_host = self.pre_out_host or connection.cloud.addr
        edge_host = self.post_out_host or connection.edge.addr
        endpoints = {
            "pre_out": _endpoint(cloud_host, self.pre_out_port),
            "post_out": _endpoint(edge_host, self.post_out_port),
            "wire_store": _endpoint(edge_host, self.wire_store_port),
        }
        if len(set(endpoints.values())) != len(endpoints):
            raise ValueError("[LWD] transport listeners must use distinct endpoints")
        return endpoints
