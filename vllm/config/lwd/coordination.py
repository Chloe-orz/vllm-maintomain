# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache coordination metadata only; no key reads, clients or server startup."""

import math
from dataclasses import dataclass, fields
from typing import Any
from urllib.parse import urlsplit

from .topology import _mapping


@dataclass(frozen=True)
class LwdCoordinationConfig:
    enabled: bool = False
    control_url: str | None = None
    tenant_key_file: str | None = None
    consumer_id: str | None = None
    listen_host: str = "0.0.0.0"
    listen_port: int = 8100
    instance_id: str = "cloud-0"
    connect_timeout: float = 10.0

    @classmethod
    def from_dict(cls, raw: Any) -> "LwdCoordinationConfig":
        raw = _mapping(
            raw, "lwd_coordination", set(), tuple(f.name for f in fields(cls))
        )
        config = cls(**raw)
        if type(config.enabled) is not bool:
            raise ValueError("[LWD] lwd_coordination.enabled must be a boolean")
        for name in ("control_url", "tenant_key_file", "consumer_id"):
            value = getattr(config, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    f"[LWD] lwd_coordination.{name} must be a non-empty string"
                )
        for name in ("listen_host", "instance_id"):
            value = getattr(config, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"[LWD] lwd_coordination.{name} must be a non-empty string"
                )
        if type(config.listen_port) is not int or not 1 <= config.listen_port <= 65535:
            raise ValueError(
                "[LWD] lwd_coordination.listen_port must be within 1..65535"
            )
        if (
            type(config.connect_timeout) not in (int, float)
            or not math.isfinite(config.connect_timeout)
            or config.connect_timeout <= 0
        ):
            raise ValueError("[LWD] connect_timeout must be finite and positive")
        if config.control_url is not None:
            url = urlsplit(config.control_url)
            if url.scheme not in ("http", "https") or not url.hostname:
                raise ValueError("[LWD] control_url must be an HTTP(S) endpoint")
        return config

    def validate_role(self, role: str | None) -> None:
        if not self.enabled:
            return
        if role is None:
            raise ValueError("[LWD] enabled coordination requires lwd_config")
        if role == "edge" and (not self.control_url or not self.tenant_key_file):
            raise ValueError(
                "[LWD] edge coordination requires control_url and tenant_key_file"
            )
