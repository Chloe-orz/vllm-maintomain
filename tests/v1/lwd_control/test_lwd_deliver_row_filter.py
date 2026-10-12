"""零接受活行剔除测试(修复:云组行 364/364 定案后的对症修复)。

语义对当:修复前 c2e 行集原样带 accepted=0 且 finish=-1 的行,边侧按
"行缺且未完结"判 ERROR;修复后此类行在云侧组 c2e 时即被剔除(四数组
按位同步),带完结码的收尾行保留,正接受行不动。
"""

from __future__ import annotations

import pytest

try:
    from vllm.v1.lwd_control.control_cloud_scheduler.lwd_cloud_scheduler import (
        LwdCloudScheduler,
    )
except Exception:  # pragma: no cover - 轻量环境(见 conftest)
    pytest.skip(
        "lwd_cloud_scheduler 完整链不可用(需 torch/Python>=3.10)",
        allow_module_level=True,
    )

_NOT_FINISHED = -1


def test_zero_accept_live_rows_dropped():
    """(0, -1) 剔除;其余全保留,四数组按位对齐。"""
    req_ids = ["live0", "ok", "fin0", "live1"]
    accepted = [0, 2, 0, 1]
    finishes = [_NOT_FINISHED, _NOT_FINISHED, 0, _NOT_FINISHED]
    top_ths = [[], [7, 8], [], [9]]
    r, a, f, t = LwdCloudScheduler._filter_deliver_rows(
        req_ids, accepted, finishes, top_ths)
    assert r == ["ok", "fin0", "live1"]
    assert a == [2, 0, 1]
    assert f == [_NOT_FINISHED, 0, _NOT_FINISHED]
    assert t == [[7, 8], [], [9]]


def test_all_zero_accept_live_rows_empty():
    """全零接受活行 → 四数组全空(调用方据此不发空通告)。"""
    r, a, f, t = LwdCloudScheduler._filter_deliver_rows(
        ["x", "y"], [0, 0], [_NOT_FINISHED, _NOT_FINISHED], [[], []])
    assert r == [] and a == [] and f == [] and t == []


def test_zero_accept_finished_rows_kept():
    """零接受但带完结码的行(正常收尾/pure-finish)必须保留——
    剔除它们会吞掉终结信号,请求永远等不到完成码。"""
    r, _, f, _ = LwdCloudScheduler._filter_deliver_rows(
        ["fin"], [0], [1], [[]])
    assert r == ["fin"] and f == [1]
