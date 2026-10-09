# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect the same LWD snapshot as VllmConfig, without starting inference."""

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--additional-config", required=True, help="JSON launch settings"
    )
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[2] / "vllm/config/lwd/__init__.py"
    spec = importlib.util.spec_from_file_location("lwd_config_standalone", package)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        additional = json.loads(args.additional_config)
        if not isinstance(additional, dict):
            raise ValueError("--additional-config must be a JSON object")
        lwd, coordination = module.resolve_lwd_config(additional)
    except ValueError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "metadata_only": True,
                "lwd_config": asdict(lwd) if lwd is not None else None,
                "lwd_coordination": asdict(coordination)
                if coordination is not None
                else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
