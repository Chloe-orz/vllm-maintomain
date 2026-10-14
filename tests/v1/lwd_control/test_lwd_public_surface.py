# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""外部接入面守卫测试 + 包内依赖方向守卫。

core.py 内联读 vllm_config.lwd_config 选引擎类;serve.py 不再接触
lwd_control;重量级模块(需 torch 的引擎/调度器)以 find_spec 断言
存在性,不触发执行。
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

try:
    from vllm.v1.lwd_control.lwd_control_diag import LwdControlDiag
except Exception:  # pragma: no cover
    pytest.skip("vllm 包入口不可导入", allow_module_level=True)

assert callable(LwdControlDiag.token_passthrough_enabled)
assert callable(LwdControlDiag.disable_down_enabled)
assert callable(LwdControlDiag.bypass_unembed_rows)

REPO_ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = REPO_ROOT / "vllm"


def _spec(name: str):
    return importlib.util.find_spec(name)


def test_module_inventory():
    """核心模块存在性(不触发 torch 依赖的执行)。"""
    for name in [
        "vllm.v1.lwd_control.control_scheduler.lwd_base_scheduler",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_engine",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_scheduler",
        "vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_engine",
        "vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler",
        "vllm.v1.lwd_control.control_cloud_scheduler.lwd_prefill_notify_queue",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_picker",
    ]:
        assert _spec(name) is not None, name


def test_deleted_modules_gone():
    """PD 分离相位机、旧配置/台账模块已删,不可再 import。"""
    for name in [
        "vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_phase_scheduler",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_assemble",
        "vllm.v1.lwd_control.control_edge_scheduler.lwd_edge_awaiting_tracker",
        "vllm.v1.lwd_control.lwd_config",
    ]:
        assert _spec(name) is None, name


def test_external_entry_sites():
    """core.py 内联选择块在;serve.py 零 lwd_control 引用。"""
    core = (VLLM_DIR / "v1" / "engine" / "core.py").read_text()
    assert "vllm_config.lwd_config" in core
    assert "LwdEdgeEngineCore" in core and "LwdCloudEngineCore" in core
    serve = (VLLM_DIR / "entrypoints" / "cli" / "serve.py").read_text()
    assert "lwd_control" not in serve


@pytest.mark.parametrize("role", [None, "edge", "cloud"])
def test_engine_selection_accepts_absent_config_and_current_roles(
    role: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute the real selection block without loading a device runtime."""
    tree = ast.parse((VLLM_DIR / "v1/engine/core.py").read_text())
    selection = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.ImportFrom)
            and any(alias.name == "lwd_edge_engine" for alias in child.names)
            for child in ast.walk(node)
        )
        and "lwd" in ast.unparse(node.test)
    )
    central, edge, cloud = object(), object(), object()
    for name, expected in [("edge", edge), ("cloud", cloud)]:
        path = f"vllm.v1.lwd_control.control_{name}_scheduler.lwd_{name}_engine"
        module = ModuleType(path)
        setattr(module, f"Lwd{name.title()}EngineCore", expected)
        monkeypatch.setitem(sys.modules, path, module)
    scope = {
        "lwd": None if role is None else SimpleNamespace(role=role),
        "engine_cls": central,
    }
    exec(
        compile(ast.Module(body=[selection], type_ignores=[]), "core.py", "exec"), scope
    )
    assert scope["engine_cls"] is {None: central, "edge": edge, "cloud": cloud}[role]


def test_cloud_scheduler_keeps_debug_static_method():
    """lwd_debug 引用的静态方法名仍在(文本守卫,免 torch);core.py 无
    LWD 模型输出钩子(c2e 收编调度器 update_from_output + 引擎 post_step)。"""
    source = (
        VLLM_DIR
        / "v1"
        / "lwd_control"
        / "control_cloud_scheduler"
        / "lwd_cloud_scheduler.py"
    ).read_text()
    assert "def _lwd_c2e_finish_reasons(" in source
    core = (VLLM_DIR / "v1" / "engine" / "core.py").read_text()
    assert "lwd_process_model_output" not in core
    assert "lwd_handle_model_output" not in core


def test_cloud_package_never_imports_edge_package():
    """依赖方向守卫:云侧包禁止 import 边侧包。"""
    cloud_dir = VLLM_DIR / "v1" / "lwd_control" / "control_cloud_scheduler"
    offenders = [
        py.name
        for py in cloud_dir.glob("*.py")
        if "control_edge_scheduler" in py.read_text()
    ]
    assert offenders == []


def test_diag_package_purity():
    """诊断包纯度:零功能依赖(不 import 调度/引擎/通信类),不持
    执行适配器(future/伪输出归引擎);违者即诊断与功能再次糅合。"""
    diag_dir = VLLM_DIR / "v1" / "lwd_control" / "lwd_control_diag"
    files = list(diag_dir.glob("*.py"))
    assert files, "lwd_control_diag 包缺失"
    banned = (
        "control_edge_scheduler",
        "control_cloud_scheduler",
        "control_communication",
        "control_scheduler",
        "LwdBypassFuture",
        "LwdStubModelOutput",
        "LWD_EMPTY_MODEL_OUTPUT",
    )
    for py in files:
        text = py.read_text()
        for marker in banned:
            assert marker not in text, (py.name, marker)


def test_no_scheduler_mode_strings_left():
    """PD 分离判据字符串在包内清零(唯一形态 PD_mix)。"""
    for py in (VLLM_DIR / "v1" / "lwd_control").rglob("*.py"):
        text = py.read_text()
        assert "prefill_first" not in text, py
        assert "decode_first" not in text, py
