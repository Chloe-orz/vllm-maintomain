# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Seam tests for the prompt-embeds sinking batch.

Anchors the sinking of the three data-condition weave blocks (shape-only
placeholder embeds on the new/resumed request paths, and the MM-encoder
gate) from ``GPUModelRunner`` into ``LwdCloudModelRunner`` overrides:

- the base must no longer know the shape-only transport protocol
  (``prompt_embeds_shape`` disappears from it entirely),
- the base MM-encoder decision is the native rule behind
  ``_should_run_mm_encoder`` (no ``has_prompt_embeds`` local),
- call sites go through the seams only.

Source-level assertions run everywhere (no torch); runtime half is
skipped without torch and re-runs on the NPU env.

Companion: vllm-ascend ``tests/ut/worker/test_lwd_prompt_embeds_seam.py``
asserts the cloud-side overrides.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
BASE = _REPO_ROOT / "vllm" / "v1" / "worker" / "gpu_model_runner.py"


def _src() -> str:
    return BASE.read_text(encoding="utf-8")


def _method(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name} not found in {BASE}")


def _only_return(src: str, fn: ast.FunctionDef) -> str:
    rets = [s for s in fn.body if isinstance(s, ast.Return)]
    assert len(rets) == 1, "seam must be a single-return function"
    seg = ast.get_source_segment(src, rets[0].value)
    assert seg is not None
    return seg


@pytest.mark.parametrize(
    "fragment",
    [
        "def _resolve_prompt_embeds(",
        "def _should_run_mm_encoder(",
        "prompt_embeds=self._resolve_prompt_embeds(new_req_data)",
        "req_state.prompt_embeds = self._resolve_prompt_embeds(new_req_data)",
        "if self._should_run_mm_encoder(is_first_rank, is_encoder_decoder):",
    ],
)
def test_seam_call_sites_present(fragment: str) -> None:
    assert fragment in _src()


def test_base_dropped_shape_protocol() -> None:
    # Placeholder-by-shape reconstruction is LWD cloud behavior: the
    # shared base must not reference the transport protocol anymore.
    assert "prompt_embeds_shape" not in _src()


def test_base_mm_gate_is_native_rule() -> None:
    src = _src()
    assert "has_prompt_embeds" not in src
    seg = _only_return(src, _method(ast.parse(src), "_should_run_mm_encoder"))
    assert "supports_mm_inputs" in seg
    assert "is_first_rank" in seg
    assert "is_encoder_decoder" in seg


def test_base_resolve_prompt_embeds_is_passthrough() -> None:
    src = _src()
    seg = _only_return(src, _method(ast.parse(src), "_resolve_prompt_embeds"))
    assert seg == "new_req_data.prompt_embeds"


def _runner_cls():
    pytest.importorskip(
        "torch", reason="re-run on NPU env: python>=3.10 + torch + vllm"
    )
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    return GPUModelRunner


def test_runtime_native_passthrough_ignores_shape() -> None:
    cls = _runner_cls()
    runner = cls.__new__(cls)

    class _ND:
        prompt_embeds = "EMBEDS"
        prompt_embeds_shape = (4, 8)
        prompt_embeds_dtype = None

    # Native rule: embeds pass through even when a shape is present.
    assert runner._resolve_prompt_embeds(_ND) == "EMBEDS"


@pytest.mark.parametrize(
    "supports_mm,first,encdec,want",
    [
        (True, True, False, True),
        (True, False, False, False),
        (False, True, False, False),
        (True, True, True, False),
    ],
)
def test_runtime_native_mm_gate(
    supports_mm: bool, first: bool, encdec: bool, want: bool
) -> None:
    cls = _runner_cls()
    runner = cls.__new__(cls)
    runner.supports_mm_inputs = supports_mm
    assert runner._should_run_mm_encoder(first, encdec) is want
