# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-independent LWD configuration, without execution hooks."""

from typing import Any

from .coordination import LwdCoordinationConfig
from .entry import LwdConfig
from .topology import LwdTopology

__all__ = ["LwdConfig", "LwdCoordinationConfig", "LwdTopology", "resolve_lwd_config"]


def resolve_lwd_config(
    additional_config: Any,
) -> tuple[LwdConfig | None, LwdCoordinationConfig | None]:
    """Parse once into the VllmConfig snapshot forwarded to engines/workers."""
    if not isinstance(additional_config, dict):
        return None, None
    if "edge_cloud_config" in additional_config:
        raise ValueError("[LWD] edge_cloud_config is retired; use lwd_config.path")
    lwd = (
        LwdConfig.from_dict(additional_config["lwd_config"])
        if "lwd_config" in additional_config
        else None
    )
    coordination = (
        LwdCoordinationConfig.from_dict(additional_config["lwd_coordination"])
        if "lwd_coordination" in additional_config
        else None
    )
    if coordination is not None:
        coordination.validate_role(lwd.role if lwd is not None else None)
    return lwd, coordination
