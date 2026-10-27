# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for edge-side LWD prefix negotiation.

- ``LwdEdgePrefixControl`` is loaded with vllm runtime deps stubbed in
  ``sys.modules`` (real ``lwd_notify`` is importable: msgspec-only);
- 调度器侧方法(READY 入队/refill 组批/重 embed 源保留/在途闸门)按
  仓内既有 LWD 测试惯例 AST 抽取,包进带 ``_Base`` 的 shim 类执行
  (保留 super() 语义),不重导 torch/vllm 全栈。
"""

import ast
import importlib.util
import logging
import queue
import sys
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3] / "vllm/v1/lwd_control"

logger = logging.getLogger("test_lwd_edge_prefix_control")

# 真实 lwd_notify(仅依赖 msgspec):READY 结构与 nonce 对账按真类型测
_notify_spec = importlib.util.spec_from_file_location(
    "vllm.v1.lwd_control.control_communication.lwd_notify",
    ROOT / "control_communication/lwd_notify.py",
)
NOTIFY = importlib.util.module_from_spec(_notify_spec)
sys.modules[_notify_spec.name] = NOTIFY
_notify_spec.loader.exec_module(NOTIFY)
LwdPrefixReadyNotify = NOTIFY.LwdPrefixReadyNotify


def _load_edge_control():
    """stub vllm 依赖后加载 LwdEdgePrefixControl 所在模块。"""
    stub_logger = types.ModuleType("vllm.logger")
    stub_logger.init_logger = lambda name: logging.getLogger(name)
    stub_hashing = types.ModuleType("vllm.utils.hashing")
    stub_hashing.get_hash_fn_by_name = Mock(return_value="hash_fn")
    stub_kvcu = types.ModuleType("vllm.v1.core.kv_cache_utils")
    stub_kvcu.get_request_block_hasher = Mock(return_value="cloud_hasher")
    stub_kvcu.init_none_hash = Mock()
    stub_engine = types.ModuleType("vllm.v1.engine")
    stub_engine.EngineCoreRequestType = NS(WAKEUP="wakeup")
    stubs = {
        "vllm.logger": stub_logger,
        "vllm.utils.hashing": stub_hashing,
        "vllm.v1.core.kv_cache_utils": stub_kvcu,
        "vllm.v1.engine": stub_engine,
    }
    spec = importlib.util.spec_from_file_location(
        "lwd_edge_prefix_control_under_test",
        ROOT / "control_edge_scheduler/lwd_edge_prefix_control.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: module, **stubs}):
        spec.loader.exec_module(module)
    return module, stub_kvcu


CONTROL_MOD, STUB_KVCU = _load_edge_control()
LwdEdgePrefixControl = CONTROL_MOD.LwdEdgePrefixControl


class FakeRequest:
    """边侧请求桩:nonce/哈希链重建/入队所需的最小字段。"""

    def __init__(self, rid="r", prompt=11):
        self.request_id = rid
        self.num_prompt_tokens = prompt
        self.block_hashes = [b"old"]
        self._block_hasher = None
        self.num_computed_tokens = 0

    def update_block_hashes(self):
        # 模拟原生:用当前 _block_hasher 重算(prompt 11/块 4 -> 2 块)
        if self._block_hasher is not None:
            self.block_hashes = [b"h0", b"h1"]


def make_engine():
    return NS(
        scheduler=NS(
            lwd_edge_add_request=Mock(),
            lwd_edge_prefix_ready=Mock(),
        ),
        input_queue=queue.Queue(),
        request_block_hasher="engine_hasher",
        vllm_config=NS(cache_config=NS(prefix_caching_hash_algo="builtin")),
        abort_requests=Mock(),
    )


class TestLwdEdgePrefixControl(unittest.TestCase):
    def setUp(self):
        STUB_KVCU.get_request_block_hasher.reset_mock()
        STUB_KVCU.init_none_hash.reset_mock()
        self.engine = make_engine()
        self.control = LwdEdgePrefixControl(self.engine, timeout_s=60.0)
        self.addCleanup(self.control.shutdown)

    def test_configure_rebuilds_hasher_with_cloud_granularity(self):
        hello = NS(
            hash_block_size=16, prefix_alignment_tokens=16,
            prefix_alignment_required=False, prefix_use_eagle=False,
        )
        self.control.configure(hello)
        STUB_KVCU.get_request_block_hasher.assert_called_once_with(
            16, "hash_fn"
        )
        self.assertEqual(self.engine.request_block_hasher, "cloud_hasher")
        self.assertEqual(self.control._hash_block_size, 16)
        # 引擎已有 hasher:不需 init_none_hash
        STUB_KVCU.init_none_hash.assert_not_called()

    def test_configure_init_none_hash_when_engine_hasher_missing(self):
        self.engine.request_block_hasher = None
        self.control.configure(
            NS(hash_block_size=8, prefix_alignment_tokens=0,
               prefix_alignment_required=False, prefix_use_eagle=False)
        )
        STUB_KVCU.init_none_hash.assert_called_once_with("hash_fn")
        STUB_KVCU.get_request_block_hasher.assert_called_once_with(
            8, "hash_fn"
        )

    def test_configure_rejects_invalid_block_size(self):
        with self.assertRaises(ValueError):
            self.control.configure(
                NS(hash_block_size=0, prefix_alignment_tokens=0,
                   prefix_alignment_required=False, prefix_use_eagle=False)
            )

    def test_add_request_assigns_nonce_rebuilds_chain_and_waits(self):
        req = FakeRequest()
        self.control.add_request(req)
        self.assertEqual(req.lwd_prefix_nonce, 1)
        # 协商 hasher 生效:旧链被清,按引擎 hasher 重建
        self.assertIs(req._block_hasher, "engine_hasher")
        self.assertEqual(req.block_hashes, [b"h0", b"h1"])
        # 未入本地队列:只发了预告(lwd_edge_add_request),等 READY
        self.engine.scheduler.lwd_edge_add_request.assert_called_once_with(req)
        self.engine.scheduler.lwd_edge_prefix_ready.assert_not_called()
        self.assertIn("r", self.control.pending)
        _req, timer = self.control.pending["r"]
        self.assertTrue(timer.is_alive() or not timer.finished.is_set())
        # 第二个请求 nonce 单调递增
        req2 = FakeRequest("r2")
        self.control.add_request(req2)
        self.assertEqual(req2.lwd_prefix_nonce, 2)

    def test_add_request_failure_rolls_back_pending(self):
        self.engine.scheduler.lwd_edge_add_request.side_effect = RuntimeError(
            "publish queue full"
        )
        with self.assertRaises(RuntimeError):
            self.control.add_request(FakeRequest())
        self.assertNotIn("r", self.control.pending)

    def test_ready_with_matching_nonce_enqueues_with_hit(self):
        req = FakeRequest()
        self.control.add_request(req)
        handled = self.control.handle_client_request(
            "wakeup", LwdPrefixReadyNotify("r", 8, 1)
        )
        self.assertTrue(handled)
        self.engine.scheduler.lwd_edge_prefix_ready.assert_called_once_with(
            req, 8
        )
        self.assertNotIn("r", self.control.pending)
        self.engine.abort_requests.assert_not_called()

    def test_stale_or_unknown_ready_is_swallowed(self):
        req = FakeRequest()
        self.control.add_request(req)
        # 未知 rid
        self.assertTrue(
            self.control.handle_client_request(
                "wakeup", LwdPrefixReadyNotify("ghost", 8, 9)
            )
        )
        # nonce 错号(迟到的旧回执)
        self.assertTrue(
            self.control.handle_client_request(
                "wakeup", LwdPrefixReadyNotify("r", 8, 99)
            )
        )
        self.engine.scheduler.lwd_edge_prefix_ready.assert_not_called()
        self.assertIn("r", self.control.pending)
        # 非 READY 的 WAKEUP 不接管
        self.assertFalse(self.control.handle_client_request("wakeup", None))

    def test_error_ready_enqueues_zero_hit_then_aborts(self):
        req = FakeRequest()
        self.control.add_request(req)
        self.control.handle_client_request(
            "wakeup", LwdPrefixReadyNotify("r", 0, 1, "reserve failed")
        )
        # 先零命中入队(让原生取消找得到请求),再走 abort 出口
        self.engine.scheduler.lwd_edge_prefix_ready.assert_called_once_with(
            req, 0
        )
        self.engine.abort_requests.assert_called_once_with(["r"])
        self.assertNotIn("r", self.control.pending)

    def test_timeout_posts_error_ready_wakeup(self):
        req = FakeRequest()
        self.control.add_request(req)
        _req, timer = self.control.pending["r"]
        timer.cancel()
        # 直接触发定时器载荷:只投递,不改等待表(主线程消费时清理)
        timer.function(*timer.args)
        msg_type, payload = self.engine.input_queue.get_nowait()
        self.assertEqual(msg_type, "wakeup")
        self.assertIsInstance(payload, LwdPrefixReadyNotify)
        self.assertEqual((payload.request_id, payload.nonce), ("r", 1))
        self.assertTrue(payload.error)

    def test_prepare_abort_registers_zero_hit(self):
        req = FakeRequest()
        self.control.add_request(req)
        self.control.prepare_abort(["r"])
        self.assertNotIn("r", self.control.pending)
        self.engine.scheduler.lwd_edge_prefix_ready.assert_called_once_with(
            req, 0
        )

    def test_shutdown_cancels_all_timers(self):
        self.control.add_request(FakeRequest("a"))
        self.control.add_request(FakeRequest("b"))
        timers = [t for _r, t in self.control.pending.values()]
        self.control.shutdown()
        for timer in timers:
            self.assertTrue(timer.finished.is_set() or not timer.is_alive())


# --------------------------------------------------------------------- #
# 调度器侧方法:AST 抽取 + shim 类执行                                     #
# --------------------------------------------------------------------- #


class _Base:
    """shim 基类:接住 super().add_request / finish_requests。"""

    def add_request(self, request):
        self.added.append(request)

    def finish_requests(self, request_ids, status):
        self.finished = (list(request_ids), status)
        for rid in request_ids:
            self.requests.pop(rid, None)


def _load_scheduler_class():
    path = ROOT / "control_edge_scheduler/lwd_edge_scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LwdEdgeScheduler"
    )
    wanted = {
        "lwd_edge_max_num_seqs_check",
        "_lwd_engine_load",
        "_lwd_route_request",
        "lwd_edge_prefix_ready",
        "lwd_edge_build_refill",
        "lwd_edge_update_progress",
        "lwd_edge_deliver_tokens",
        "lwd_edge_abort",
    }
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert len(methods) == len(wanted), wanted - {m.name for m in methods}
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias("annotations")], level=0
    )
    shim = ast.ClassDef(
        name="EdgeShim",
        bases=[ast.Name(id="_Base", ctx=ast.Load())],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(body=[future, shim], type_ignores=[])
    ast.fix_missing_locations(module)

    class _Struct:
        """msgspec.Struct 桩:kwargs 构造 + 属性读取。"""

        _fields: tuple = ()

        def __init__(self, **kwargs):
            for field in self._fields:
                setattr(self, field, kwargs.get(field))

    class LwdRangeItem(_Struct):
        _fields = (
            "request_id", "offset", "num_tokens", "has_mrope",
            "mrope_positions", "is_refill",
        )

    class LwdRangeNotify(_Struct):
        _fields = ("request_id", "offset", "num_tokens", "seqno", "items")

    class LwdAbortNotify(_Struct):
        _fields = ("request_id",)

    class LwdEmbedBatch(_Struct):
        _fields = (
            "req_ids", "token_ids", "prompt_offsets", "has_mrope",
            "mrope_positions", "prompt_lens", "aborted_req_ids",
            "engine_index", "is_refill",
        )

    class NewRequestData(_Struct):
        _fields = (
            "req_id", "prompt_token_ids", "mm_features", "sampling_params",
            "pooling_params", "block_ids", "num_computed_tokens",
            "lora_request",
        )

    class LwdBatch(_Struct):
        _fields = ("batch_type", "seqno", "batch_meta")

    class SchedulerOutput:
        @staticmethod
        def make_empty():
            return NS(
                num_scheduled_tokens={}, scheduled_new_reqs=[], lwd_batch=None
            )

    namespace = {
        "_Base": _Base,
        "logger": logger,
        "time": __import__("time"),
        "RequestStatus": NS(FINISHED_STOPPED="stopped", FINISHED_ABORTED="aborted"),
        "LwdAbortNotify": LwdAbortNotify,
        "LwdRangeItem": LwdRangeItem,
        "LwdRangeNotify": LwdRangeNotify,
        "LwdBatch": LwdBatch,
        "LwdBatchType": NS(LWD_EMBED="lwd_embed"),
        "LwdEmbedBatch": LwdEmbedBatch,
        "NewRequestData": NewRequestData,
        "SchedulerOutput": SchedulerOutput,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["EdgeShim"]


EdgeShim = _load_scheduler_class()


def make_scheduler(**overrides):
    sched = EdgeShim()
    sched.added = []
    sched.finished = None
    sched.requests = {}
    sched.running = []
    sched.waiting = []
    sched.skipped_waiting = []
    sched._lwd_awaiting = {}
    sched._lwd_prefix_enabled = True
    sched._lwd_prefix_pending = None
    sched._lwd_prefix_hash_block_size = 4
    sched._lwd_embed_sources = {}
    sched._lwd_mrope_positions_dict = {}
    sched._lwd_cloud_dp = 1
    sched._lwd_route = {
        rid: 0 for rid in ("r", "p", "a", "running", "live", "m", "gone")
    }
    sched._lwd_last_engine = -1
    sched._lwd_seqno = [0]
    sched._lwd_batch_requests = {}
    sched.max_num_running_reqs = 4
    sched.lwd_edge_publisher = NS(publish=Mock(return_value=True))
    sched.lwd_edge_publishers = {0: sched.lwd_edge_publisher}
    for key, value in overrides.items():
        setattr(sched, key, value)
    return sched


class TestEdgeSeqsGate(unittest.TestCase):
    def test_ready_waiters_count_toward_inflight_gate(self):
        sched = make_scheduler(
            running=[NS(request_id="running")], _lwd_awaiting={"a": 0.0},
            _lwd_prefix_pending={"p": None},
        )
        # 1 running + 1 awaiting + 1 等待 READY = 3 < 4:可开新
        self.assertTrue(sched.lwd_edge_max_num_seqs_check())
        sched.max_num_running_reqs = 3
        self.assertFalse(sched.lwd_edge_max_num_seqs_check())

    def test_no_pending_table_is_backward_compatible(self):
        sched = make_scheduler(running=[NS(request_id="running")])
        self.assertTrue(sched.lwd_edge_max_num_seqs_check())

    def test_pending_is_charged_only_to_its_routed_engine(self):
        sched = make_scheduler(
            _lwd_cloud_dp=2,
            _lwd_route={"p": 1},
            _lwd_prefix_pending={"p": None},
        )
        self.assertEqual(sched._lwd_engine_load(0), (0, 0))
        self.assertEqual(sched._lwd_engine_load(1), (1, 0))

    def test_route_score_includes_ready_waiters(self):
        sched = make_scheduler(
            _lwd_cloud_dp=2,
            _lwd_route={"p": 0},
            _lwd_prefix_pending={"p": None},
        )
        self.assertEqual(sched._lwd_route_request("new"), 1)
        self.assertEqual(sched._lwd_route["new"], 1)


class TestEdgePrefixReady(unittest.TestCase):
    def test_hit_sets_computed_then_enqueues(self):
        sched = make_scheduler()
        req = NS(request_id="r", num_prompt_tokens=11, num_computed_tokens=0)
        sched.lwd_edge_prefix_ready(req, 8)
        self.assertEqual(req.num_computed_tokens, 8)
        self.assertEqual(sched.added, [req])

    def _assert_aborted(self, sched, req):
        """非法 hit 的请求级处理:入队即 abort(不抛出),abort 信号发云。"""
        self.assertEqual(req.num_computed_tokens, 0)
        self.assertEqual(sched.added, [req])
        self.assertEqual(sched.finished, (["r"], "aborted"))
        sched.lwd_edge_publisher.publish.assert_called_once()

    def test_out_of_range_hit_aborts_request_without_raising(self):
        # 越界 hit 不再抛出(READY 走主循环,抛出会打死边引擎)
        for bad_hit in (11, -1):
            with self.subTest(hit=bad_hit):
                sched = make_scheduler()
                req = NS(
                    request_id="r", num_prompt_tokens=11,
                    num_computed_tokens=0,
                )
                sched.lwd_edge_prefix_ready(req, bad_hit)
                self._assert_aborted(sched, req)

    def test_misaligned_hit_aborts_request(self):
        # hit 必须是 hash 块整数倍(5 % 4 != 0 → abort)
        sched = make_scheduler()
        req = NS(request_id="r", num_prompt_tokens=11, num_computed_tokens=0)
        sched.lwd_edge_prefix_ready(req, 5)
        self._assert_aborted(sched, req)

    def test_zero_hit_always_valid(self):
        sched = make_scheduler()
        req = NS(request_id="r", num_prompt_tokens=11, num_computed_tokens=0)
        sched.lwd_edge_prefix_ready(req, 0)
        self.assertEqual(sched.added, [req])
        self.assertIsNone(sched.finished)


class TestEdgeRefill(unittest.TestCase):
    def setUp(self):
        self.sched = make_scheduler()
        self.sched._lwd_embed_sources["r"] = (list(range(20)), [])

    def refill(self, rid="r", start=4, end=8):
        return self.sched.lwd_edge_build_refill(
            NS(request_id=rid, start_row=start, end_row=end)
        )

    def test_refill_builds_out_of_band_embed_batch(self):
        out = self.refill()
        self.assertIsNotNone(out)
        # RangeNotify 已发布:is_refill 标记 + prompt 绝对坐标
        notify = self.sched.lwd_edge_publisher.publish.call_args.args[0]
        self.assertEqual((notify.offset, notify.num_tokens, notify.seqno),
                         (4, 4, 0))
        self.assertTrue(notify.items[0].is_refill)
        # seqno 单调进位 + pending 闸门登记
        self.assertEqual(self.sched._lwd_seqno, [1])
        self.assertEqual(self.sched._lwd_batch_requests, {(0, 0): ["r"]})
        # EMBED 批:保留源切片,不参与本地水位记账
        batch = out.lwd_batch
        self.assertEqual(batch.batch_type, "lwd_embed")
        meta = batch.batch_meta
        self.assertEqual(meta.req_ids, ["r"])
        self.assertEqual(meta.token_ids, [[4, 5, 6, 7]])
        self.assertEqual(meta.prompt_offsets, [4])
        self.assertEqual(meta.prompt_lens, [20])
        self.assertEqual(meta.has_mrope, [False])
        self.assertTrue(meta.is_refill)  # worker 据此摘除重登记的 mm 特征
        self.assertEqual(out.num_scheduled_tokens, {})
        self.assertEqual(out.scheduled_new_reqs, [])  # 纯文本无 mm 重登记

    def test_refill_falls_back_to_live_request(self):
        # 部分到达即被抢占的命中请求:嵌入未完结、无保留源,但本体还在
        # 调度器里——用活体的 prompt_token_ids/mm 组批,不误杀健康请求
        self.sched._lwd_embed_sources.clear()
        self.sched.requests["live"] = NS(
            request_id="live",
            prompt_token_ids=list(range(20)),
            mm_features=[],
        )
        out = self.sched.lwd_edge_build_refill(
            NS(request_id="live", start_row=4, end_row=8)
        )
        self.assertIsNotNone(out)
        self.assertEqual(out.lwd_batch.batch_meta.token_ids, [[4, 5, 6, 7]])
        self.assertTrue(out.lwd_batch.batch_meta.is_refill)

    def test_refill_rejected_only_when_request_truly_gone(self):
        # 保留源与活体两处都没有(请求真正终结)才拒绝
        self.sched._lwd_embed_sources.clear()
        self.assertIsNone(self.refill(rid="gone"))
        self.sched.lwd_edge_publisher.publish.assert_not_called()
        self.assertEqual(self.sched._lwd_seqno, [0])

    def test_refill_without_publisher_leaves_no_trace(self):
        # publisher=None 仅测试桩可达;缺守卫会静默跳过发布还进位 seqno
        self.sched.lwd_edge_publisher = None
        self.sched.lwd_edge_publishers = {}
        self.assertIsNone(self.refill())
        self.assertEqual(self.sched._lwd_seqno, [0])
        self.assertEqual(self.sched._lwd_batch_requests, {})

    def test_refill_uses_original_engine_and_independent_seqno(self):
        publishers = {0: NS(publish=Mock(return_value=True)),
                      1: NS(publish=Mock(return_value=True))}
        self.sched._lwd_cloud_dp = 2
        self.sched._lwd_route["r"] = 1
        self.sched.lwd_edge_publishers = publishers
        self.sched._lwd_seqno = [3, 7]
        out = self.refill()
        publishers[1].publish.assert_called_once()
        publishers[0].publish.assert_not_called()
        self.assertEqual(self.sched._lwd_seqno, [3, 8])
        self.assertEqual(self.sched._lwd_batch_requests, {(1, 7): ["r"]})
        self.assertEqual(out.lwd_batch.batch_meta.engine_index, 1)

    def test_refill_reregisters_mm_features(self):
        self.sched._lwd_embed_sources["m"] = (list(range(10)), ["img0"])
        out = self.sched.lwd_edge_build_refill(
            NS(request_id="m", start_row=2, end_row=6)
        )
        self.assertEqual(len(out.scheduled_new_reqs), 1)
        req_data = out.scheduled_new_reqs[0]
        self.assertEqual(req_data.req_id, "m")
        self.assertEqual(req_data.mm_features, ["img0"])
        self.assertIsNone(req_data.prompt_token_ids)  # 不重复上运输层

    def test_refill_missing_source_returns_none(self):
        # 请求已终结(保留源随 awaiting 释放):幂等拒绝,由引擎回 abort
        self.assertIsNone(self.refill(rid="gone"))
        self.sched.lwd_edge_publisher.publish.assert_not_called()
        self.assertEqual(self.sched._lwd_seqno, [0])

    def test_refill_out_of_bounds_returns_none(self):
        self.assertIsNone(self.refill(start=8, end=4))
        self.assertIsNone(self.refill(start=0, end=21))
        self.assertEqual(self.sched._lwd_seqno, [0])

    def test_refill_publish_failure_leaves_no_trace(self):
        self.sched.lwd_edge_publisher.publish.return_value = False
        self.assertIsNone(self.refill())
        self.assertEqual(self.sched._lwd_seqno, [0])  # 无空洞契约:不进位
        self.assertEqual(self.sched._lwd_batch_requests, {})


class TestEdgeEmbedSourceRetention(unittest.TestCase):
    """重 embed 源保留:随嵌入完结登记,随 awaiting 终结释放。"""

    def _request(self, rid, prompt=8, computed=8):
        return NS(
            request_id=rid,
            num_prompt_tokens=prompt,
            num_computed_tokens=computed,
            prompt_token_ids=list(range(prompt)),
            mm_features=[],
        )

    def test_retained_at_embed_done_released_on_final_delivery(self):
        sched = make_scheduler()
        sched.requests["r"] = self._request("r")
        sched.lwd_edge_update_progress({"r": 8})
        # 嵌入完结:本体清出(finish_requests 桩已摘除),源保留
        self.assertNotIn("r", sched.requests)
        self.assertIn("r", sched._lwd_awaiting)
        self.assertEqual(sched._lwd_embed_sources["r"][0], list(range(8)))
        # 云结果终结:源随 awaiting 一并释放
        self.assertTrue(sched.lwd_edge_deliver_tokens("r", [101], True))
        self.assertNotIn("r", sched._lwd_embed_sources)
        self.assertNotIn("r", sched._lwd_awaiting)

    def test_abort_releases_source(self):
        sched = make_scheduler()
        sched.requests["r"] = self._request("r")
        sched.lwd_edge_update_progress({"r": 8})
        sched.lwd_edge_abort(["r"])
        self.assertNotIn("r", sched._lwd_embed_sources)
        # abort 预告照常发云
        sched.lwd_edge_publisher.publish.assert_called_once()

    def test_retention_disabled_without_prefix_caching(self):
        sched = make_scheduler(_lwd_prefix_enabled=False)
        sched.requests["r"] = self._request("r")
        sched.lwd_edge_update_progress({"r": 8})
        self.assertNotIn("r", sched._lwd_embed_sources)

    def test_unfinished_request_keeps_source_in_request(self):
        sched = make_scheduler()
        sched.requests["r"] = self._request("r", computed=4)
        sched.lwd_edge_update_progress({"r": 4})
        self.assertIn("r", sched.requests)  # 未完结:不入 awaiting/保留表
        self.assertNotIn("r", sched._lwd_embed_sources)


class TestEdgeSchedulerConfigCopy(unittest.TestCase):
    """lwd_edge_scheduler_config:调度器视角 PC-off 浅拷贝。

    边侧 0 KV 组,KVCacheManager 若以 enable_caching=True 构造会掉进
    hybrid coordinator 断言(>=2 attention 组)构造即炸;开关必须留在
    引擎的原始配置上(hasher/HELLO 互校同源),调度器拿浅拷贝。
    """

    @staticmethod
    def _load_fn():
        import copy as _copy

        path = ROOT / "control_edge_scheduler/lwd_edge_scheduler.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fn = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "lwd_edge_scheduler_config"
        )
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias("annotations")], level=0
        )
        module = ast.Module(body=[future, fn], type_ignores=[])
        ast.fix_missing_locations(module)
        ns = {"copy": _copy}
        exec(compile(module, str(path), "exec"), ns)
        return ns["lwd_edge_scheduler_config"]

    def test_copy_disables_prefix_caching(self):
        fn = self._load_fn()
        config = NS(
            cache_config=NS(enable_prefix_caching=True),
            scheduler_config=NS(x=1),
        )
        copied = fn(config)
        self.assertFalse(copied.cache_config.enable_prefix_caching)

    def test_original_config_untouched(self):
        fn = self._load_fn()
        config = NS(cache_config=NS(enable_prefix_caching=True))
        fn(config)
        self.assertTrue(config.cache_config.enable_prefix_caching)

    def test_copy_is_shallow_except_cache_config(self):
        fn = self._load_fn()
        config = NS(
            cache_config=NS(enable_prefix_caching=False),
            scheduler_config=NS(x=1),
        )
        copied = fn(config)
        self.assertIsNot(copied, config)
        self.assertIsNot(copied.cache_config, config.cache_config)
        # 其余字段共享(浅拷贝语义)
        self.assertIs(copied.scheduler_config, config.scheduler_config)


if __name__ == "__main__":
    unittest.main()
