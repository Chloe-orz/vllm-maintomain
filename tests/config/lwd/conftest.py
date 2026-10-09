# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the pure configuration package without importing the inference runtime.

Run with --confcutdir=tests/config/lwd to avoid GPU fixtures in tests/conftest.py.
"""

import importlib.util
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[3] / "vllm/config/lwd"
spec = importlib.util.spec_from_file_location(
    "lwd_config_under_test", PACKAGE / "__init__.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
