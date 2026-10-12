"""vllm/config/lwd.py 的 LwdConfig 解析与 env 覆盖测试。

依赖 vllm.logger -> vllm.envs 完整链(Python >=3.10);轻量环境
(无 torch/py<3.10)模块级跳过。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

try:
    from vllm.config.lwd import LwdConfig, lwd_spec_factor
except Exception:  # pragma: no cover - 轻量环境(见 conftest)
    pytest.skip(
        "vllm.logger/envs 完整链不可用(需 Python>=3.10)",
        allow_module_level=True,
    )


def test_transport_defaults():
    cfg = LwdConfig.from_dict({})
    assert cfg.pre_out_host == "127.0.0.1"
    assert cfg.pre_out_port == 5558
    assert cfg.post_out_port == 5559
    assert cfg.hello_timeout_s == 600.0
    assert cfg.debug is False
    assert "scheduler_name" not in LwdConfig.__dataclass_fields__


def test_section_transport_fields():
    cfg = LwdConfig.from_dict(
        {"role": "cloud", "pre_out_host": "10.0.0.1", "pre_out_port": 6000}
    )
    assert cfg.is_edge is False
    assert cfg.pre_out_host == "10.0.0.1"
    assert cfg.pre_out_port == 6000


def test_endpoints():
    cfg = LwdConfig.from_dict({"pre_out_host": "10.1.1.1", "pre_out_port": 7})
    assert cfg.lwd_pre_out_endpoint() == "tcp://10.1.1.1:7"
    assert cfg.lwd_post_out_bind_endpoint() == "tcp://*:5559"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LWD_PRE_OUT_PORT", "6100")
    monkeypatch.setenv("VLLM_ASCEND_LWD_PRE_OUT_HOST", "10.2.2.2")
    monkeypatch.setenv("VLLM_ASCEND_LWD_HELLO_TIMEOUT_S", "12.5")
    cfg = LwdConfig.from_dict({})
    assert cfg.pre_out_port == 6100
    assert cfg.pre_out_host == "10.2.2.2"
    assert cfg.hello_timeout_s == 12.5


def test_env_bad_value_keeps_default(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LWD_PRE_OUT_PORT", "not-a-port")
    cfg = LwdConfig.from_dict({})
    assert cfg.pre_out_port == 5558


def test_env_debug_strict_one(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_LWD_DEBUG", "1")
    assert LwdConfig.from_dict({}).debug is True
    monkeypatch.setenv("VLLM_ASCEND_LWD_DEBUG", "true")
    assert LwdConfig.from_dict({}).debug is False


def test_stale_scheduler_key_ignored():
    """PD 分离已删:残留 scheduler 键不拦截、无对应字段。"""
    cfg = LwdConfig.from_dict({"scheduler": "prefill_first"})
    assert "scheduler_name" not in LwdConfig.__dataclass_fields__
    assert cfg.pre_out_port == 5558  # 其余解析不受影响


def test_lwd_diag_helpers():
    from vllm.v1.lwd_control import (
        lwd_diag_disable_down_enabled,
        lwd_diag_token_passthrough_enabled,
    )

    assert lwd_diag_token_passthrough_enabled() is False
    assert lwd_diag_disable_down_enabled() is False


def test_spec_factor():
    assert lwd_spec_factor(SimpleNamespace(speculative_config=None)) == 1
    assert (
        lwd_spec_factor(
            SimpleNamespace(
                speculative_config=SimpleNamespace(num_speculative_tokens=4)
            )
        )
        == 5
    )
