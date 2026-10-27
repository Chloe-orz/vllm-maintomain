# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the cloud-side LWD prefix reservations.

Loads ``lwd_prefix_cache.py`` standalone (module level is stdlib-only by
design); vllm runtime deps used by ``init_lwd_prefix_cache`` are stubbed
in ``sys.modules``.
"""

import ast
import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

ROOT = (
    Path(__file__).resolve().parents[3]
    / "vllm/v1/lwd_control/control_cloud_scheduler/lwd_prefix_cache.py"
)
CLOUD_ENGINE = (
    Path(__file__).resolve().parents[3]
    / "vllm/v1/lwd_control/control_cloud_scheduler/lwd_cloud_engine.py"
)

_spec = importlib.util.spec_from_file_location("lwd_prefix_cache_under_test", ROOT)
CACHE = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = CACHE
_spec.loader.exec_module(CACHE)

LwdPrefixReservations = CACHE.LwdPrefixReservations
prefix_start = CACHE.prefix_start


def make_request(rid="r", prompt=11, hit_cap=None, skip=False, computed=0):
    req = NS(
        request_id=rid,
        num_prompt_tokens=prompt,
        num_tokens=prompt,
        block_hashes=[b"a", b"b"],
        skip_reading_prefix_cache=skip,
        num_computed_tokens=computed,
    )
    if hit_cap is not None:
        req.lwd_resume_cap_tokens = hit_cap
    return req


def make_manager(
    hit_groups=(), hit=0, hash_block_size=16, group_block_size=16,
    scheduler_block_size=16,
):
    """manager 桩:find_longest_cache_hit + block_pool + 组块/调度块大小。"""
    manager = NS(
        coordinator=Mock(),
        block_pool=Mock(),
        enable_caching=True,
        lwd_prefix_caching=False,
    )
    manager.coordinator.find_longest_cache_hit.return_value = (hit_groups, hit)
    manager.coordinator.single_type_managers = (
        NS(block_size=group_block_size),
    )
    manager.coordinator.scheduler_block_size = scheduler_block_size
    manager.block_pool.hash_block_size = hash_block_size
    return manager


class TestLwdPrefixReservations(unittest.TestCase):
    def setUp(self):
        self.blk = NS(is_null=False)
        self.null_blk = NS(is_null=True)
        self.manager = NS(
            coordinator=Mock(),
            block_pool=Mock(),
            enable_caching=True,
            lwd_prefix_caching=False,
        )
        self.manager.coordinator.find_longest_cache_hit.return_value = (
            ([self.blk, self.null_blk],),
            8,
        )
        self.reservations = LwdPrefixReservations(self.manager)

    def test_attach_flips_manager_hook(self):
        self.assertIs(self.manager.lwd_prefix_reservations, self.reservations)
        self.assertTrue(self.manager.lwd_prefix_caching)

    def test_reserve_pins_real_blocks_and_records_snapshot(self):
        req = make_request()
        hit = self.reservations.reserve(req)
        self.assertEqual(hit, 8)
        # null 槽不钉,实体块 touch
        self.manager.block_pool.touch.assert_called_once_with([self.blk])
        self.assertEqual(req.lwd_resume_cap_tokens, 8)
        self.assertTrue(self.reservations.has("r"))
        self.assertEqual(self.reservations.hit_of("r"), 8)
        # 快照保留原生组布局(含 null 槽原位)
        groups, snap_hit = self.reservations.get_computed_blocks(req)
        self.assertEqual(snap_hit, 8)
        self.assertEqual(groups, ((self.blk, self.null_blk),))

    def test_duplicate_reserve_raises(self):
        req = make_request()
        self.reservations.reserve(req)
        with self.assertRaises(ValueError):
            self.reservations.reserve(req)

    def test_skip_reading_prefix_cache_reserves_zero_without_touch(self):
        req = make_request(skip=True)
        hit = self.reservations.reserve(req)
        self.assertEqual(hit, 0)
        self.manager.block_pool.touch.assert_not_called()
        self.manager.coordinator.find_longest_cache_hit.assert_not_called()
        self.assertEqual(req.lwd_resume_cap_tokens, 0)
        self.assertEqual(self.reservations.get_computed_blocks(req), ((), 0))

    def test_snapshot_is_fail_closed_after_cache_growth(self):
        """协商快照不因后续缓存增长改变边界(零命中承诺恒零)。"""
        self.manager.coordinator.find_longest_cache_hit.return_value = ((), 0)
        req = make_request()
        self.assertEqual(self.reservations.reserve(req), 0)
        # 缓存"后来增长":若重查会得到 8,快照仍须为零命中
        self.manager.coordinator.find_longest_cache_hit.return_value = (
            ([self.blk],),
            8,
        )
        self.assertEqual(self.reservations.get_computed_blocks(req), ((), 0))

    def test_missing_snapshot_with_cap_raises(self):
        """已协商但快照缺失 -> fail-closed 报错,不退回重查。"""
        req = make_request(hit_cap=8)
        with self.assertRaises(RuntimeError):
            self.reservations.get_computed_blocks(req)

    def test_no_negotiation_falls_back_to_native_query(self):
        req = make_request()
        _groups, hit = self.reservations.get_computed_blocks(req)
        self.assertEqual(hit, 8)
        self.manager.coordinator.find_longest_cache_hit.assert_called_once_with(
            req.block_hashes, req.num_tokens - 1
        )

    def test_reserve_hit_does_not_move_floor(self):
        """抢占重准入:reserve_hit 只钉残块记快照,不动暂存地板 S。"""
        req = make_request(hit_cap=8)
        hit = self.reservations.reserve_hit(req, ((self.blk,),), 4)
        self.assertEqual(hit, 4)
        self.manager.block_pool.touch.assert_called_once_with([self.blk])
        self.assertEqual(req.lwd_resume_cap_tokens, 8)  # S 不变
        self.assertEqual(
            self.reservations.get_computed_blocks(req), (((self.blk,),), 4)
        )

    def test_release_frees_pinned_blocks_once(self):
        req = make_request()
        self.reservations.reserve(req)
        self.reservations.release("r")
        self.manager.block_pool.free_blocks.assert_called_once_with([self.blk])
        # 幂等:重复 release / 未知 rid 均为 no-op
        self.reservations.release("r")
        self.reservations.release("unknown")
        self.manager.block_pool.free_blocks.assert_called_once()
        self.assertFalse(self.reservations.has("r"))

    def test_release_zero_hit_does_not_free(self):
        self.manager.coordinator.find_longest_cache_hit.return_value = ((), 0)
        self.reservations.reserve(make_request())
        self.reservations.release("r")
        self.manager.block_pool.free_blocks.assert_not_called()


class TestPrefixStart(unittest.TestCase):
    def test_running_request_uses_native_watermark(self):
        req = make_request(computed=6, hit_cap=8)
        self.assertEqual(prefix_start(req), 6)

    def test_fresh_request_uses_active_reservation_hit(self):
        manager = NS(coordinator=Mock(), block_pool=Mock())
        manager.coordinator.find_longest_cache_hit.return_value = ((["b"],), 4)
        reservations = LwdPrefixReservations(manager)
        req = make_request()
        reservations.reserve(req)
        # 活跃预留优先于协商边界(重准入 H' 可能 != S)
        self.assertEqual(prefix_start(req, reservations), 4)
        reservations.release("r")
        self.assertEqual(prefix_start(req, reservations), 4)  # 回退协商边界
        self.assertEqual(prefix_start(req), 4)

    def test_no_cap_defaults_to_zero(self):
        self.assertEqual(prefix_start(make_request()), 0)


def _run_init(
    *,
    enabled,
    specs,
    pcp=1,
    manager_caching=True,
    scheduler_block_size=16,
    lcm_block_size=16,
):
    """以 stub 模块注入 vllm 依赖后调用 init_lwd_prefix_cache。

    specs: "full" / "sliding" / ("mamba", mode) / "unknown" 列表,
    用 stub 模块里的同名类实例化(isinstance 校验对类型身份敏感)。
    """

    class FullAttentionSpec:
        pass

    class SlidingWindowSpec:
        pass

    class MambaSpec:
        def __init__(self, mamba_cache_mode="all"):
            self.mamba_cache_mode = mamba_cache_mode

    class UniformTypeKVCacheSpecs:
        def __init__(self, kv_cache_specs):
            self.kv_cache_specs = kv_cache_specs

    instances = []
    for spec in specs:
        if spec == "full":
            instances.append(FullAttentionSpec())
        elif spec == "sliding":
            instances.append(SlidingWindowSpec())
        elif isinstance(spec, tuple) and spec[0] == "mamba":
            instances.append(MambaSpec(spec[1]))
        else:
            instances.append(type("UnknownSpec", (), {})())

    kv_utils = types.ModuleType("vllm.v1.core.kv_cache_utils")
    kv_utils.resolve_kv_cache_block_sizes = Mock(return_value=(16, 16))
    iface = types.ModuleType("vllm.v1.kv_cache_interface")
    iface.FullAttentionSpec = FullAttentionSpec
    iface.SlidingWindowSpec = SlidingWindowSpec
    iface.MambaSpec = MambaSpec
    iface.UniformTypeKVCacheSpecs = UniformTypeKVCacheSpecs

    coordinator = NS(
        scheduler_block_size=scheduler_block_size,
        lcm_block_size=lcm_block_size,
    )
    manager = NS(
        coordinator=coordinator,
        enable_caching=manager_caching,
        lwd_prefix_caching=False,
    )
    vllm_config = NS(
        cache_config=NS(enable_prefix_caching=enabled),
        parallel_config=NS(prefill_context_parallel_size=pcp),
    )
    kv_cache_config = NS(kv_cache_groups=[NS(kv_cache_spec=s) for s in instances])
    with patch.dict(
        sys.modules,
        {
            "vllm.v1.core.kv_cache_utils": kv_utils,
            "vllm.v1.kv_cache_interface": iface,
        },
    ):
        state = CACHE.init_lwd_prefix_cache(
            vllm_config, kv_cache_config, manager, use_eagle=True
        )
    return state, manager


class TestInitLwdPrefixCache(unittest.TestCase):
    def test_disabled_skips_validation(self):
        state, manager = _run_init(enabled=False, specs=[])
        self.assertFalse(state.enabled)
        self.assertIsNone(state.reservations)
        self.assertFalse(manager.lwd_prefix_caching)

    def test_full_attention_ok(self):
        state, manager = _run_init(enabled=True, specs=["full"])
        self.assertTrue(state.enabled)
        self.assertIsNotNone(state.reservations)
        self.assertTrue(manager.lwd_prefix_caching)
        self.assertEqual(state.hash_block_size, 16)
        self.assertEqual(state.alignment_tokens, 16)
        self.assertFalse(state.alignment_required)
        self.assertTrue(state.use_eagle)

    def test_mixed_groups_and_alignment_lcm(self):
        state, _manager = _run_init(
            enabled=True,
            specs=["full", "sliding", ("mamba", "align")],
            scheduler_block_size=16,
            lcm_block_size=48,
        )
        self.assertTrue(state.alignment_required)
        self.assertEqual(state.alignment_tokens, 48)  # lcm(16, 48)

    def test_mamba_all_mode_ok_without_alignment(self):
        state, _manager = _run_init(enabled=True, specs=[("mamba", "all")])
        self.assertTrue(state.enabled)
        self.assertFalse(state.alignment_required)

    def test_manager_caching_disabled_fails(self):
        with self.assertRaises(ValueError):
            _run_init(enabled=True, specs=["full"], manager_caching=False)

    def test_pcp_fails(self):
        with self.assertRaises(ValueError):
            _run_init(enabled=True, specs=["full"], pcp=2)

    def test_unsupported_spec_fails(self):
        with self.assertRaises(ValueError):
            _run_init(enabled=True, specs=["unknown"])

    def test_mamba_non_align_mode_fails(self):
        with self.assertRaises(ValueError):
            _run_init(enabled=True, specs=[("mamba", "write_through")])


class TestCloudAddRequestGuard(unittest.TestCase):
    """云引擎 add_request 的前缀协商护栏(M1 纵深防御)。

    AST 抽取 lwd_cloud_engine.py 的 add_request,包进带 _CloudBase 的
    shim 类执行(保留 super() 语义),不重导 vllm 全栈。"""

    @classmethod
    def setUpClass(cls):
        tree = ast.parse(CLOUD_ENGINE.read_text(encoding="utf-8"))
        cls_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "LwdCloudEngineCoreMixin"
        )
        method = next(
            node
            for node in cls_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "add_request"
        )

        class _CloudBase:
            def add_request(self, request, request_wave=0):
                self.admitted.append(request)

        shim = ast.ClassDef(
            name="CloudShim",
            bases=[ast.Name(id="_CloudBase", ctx=ast.Load())],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        module = ast.Module(body=[shim], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "_CloudBase": _CloudBase,
            "logger": logging.getLogger("test_lwd_prefix_cache"),
            "Request": object,
        }
        exec(compile(module, str(CLOUD_ENGINE), "exec"), namespace)
        cls.shim_cls = namespace["CloudShim"]

    def engine(self, prefix_enabled, reservations=None):
        eng = self.shim_cls()
        eng.admitted = []
        eng._lwd_prefix_enabled = prefix_enabled
        eng._lwd_send_prefix_ready = Mock()
        eng.scheduler = NS(lwd_prefix_reservations=reservations)
        return eng

    @staticmethod
    def request(nonce=0, adopted=True):
        req = NS(request_id="r", num_prompt_tokens=64)
        req.lwd_prefix_nonce = nonce
        req.lwd_wire_hashes_adopted = adopted
        return req

    def test_nonce_zero_rejected_when_prefix_enabled(self):
        eng = self.engine(prefix_enabled=True)
        eng.add_request(self.request(nonce=0))
        self.assertEqual(eng.admitted, [])
        # nonce==0 无协商号可回执:不发 READY,只记日志
        eng._lwd_send_prefix_ready.assert_not_called()

    def test_nonce_zero_admitted_when_prefix_disabled(self):
        eng = self.engine(prefix_enabled=False)
        eng.add_request(self.request(nonce=0))
        self.assertEqual(len(eng.admitted), 1)

    def test_nonce_without_reservations_gets_error_ready(self):
        eng = self.engine(prefix_enabled=False, reservations=None)
        eng.add_request(self.request(nonce=7))
        self.assertEqual(eng.admitted, [])
        args = eng._lwd_send_prefix_ready.call_args.args
        self.assertEqual(args[:3], ("r", 0, 7))
        # error 非空(keyword error= 传参)
        self.assertTrue(eng._lwd_send_prefix_ready.call_args.kwargs["error"])

    def test_unadopted_wire_chain_gets_error_ready(self):
        reservations = Mock()
        eng = self.engine(prefix_enabled=True, reservations=reservations)
        eng.add_request(self.request(nonce=7, adopted=False))
        self.assertEqual(eng.admitted, [])
        reservations.reserve.assert_not_called()
        args = eng._lwd_send_prefix_ready.call_args.args
        self.assertEqual(args[:3], ("r", 0, 7))

    def test_happy_path_reserve_admit_ready(self):
        reservations = Mock()
        reservations.reserve.return_value = 32
        eng = self.engine(prefix_enabled=True, reservations=reservations)
        eng.add_request(self.request(nonce=7))
        self.assertEqual(len(eng.admitted), 1)
        self.assertEqual(
            eng._lwd_send_prefix_ready.call_args.args, ("r", 32, 7)
        )


class TestTruncateGroups(unittest.TestCase):
    """reserve_hit 的到达钳制配套:组布局按钳制边界截前缀块。"""

    def setUp(self):
        self.blocks = [NS(is_null=False) for _ in range(3)]
        self.manager = make_manager()
        self.reservations = LwdPrefixReservations(self.manager)

    def test_truncate_to_hit_prefix(self):
        groups = (list(self.blocks),)
        self.assertEqual(
            self.reservations.truncate_groups(groups, 32),
            ([self.blocks[0], self.blocks[1]],),
        )
        # 截断保持原列表不被改写
        self.assertEqual(len(groups[0]), 3)

    def test_truncate_to_zero_empties_all_groups(self):
        self.assertEqual(
            self.reservations.truncate_groups((list(self.blocks),), 0),
            ([],),
        )

    def test_truncate_preserves_null_slots_in_place(self):
        blks = [self.blocks[0], NS(is_null=True), self.blocks[2]]
        out = self.reservations.truncate_groups((blks,), 32)
        self.assertTrue(out[0][1].is_null)


def _load_prep_readmission():
    """AST 抽取 _lwd_prep_prefix_readmission(不重导 vllm 全栈)。"""
    path = CLOUD_ENGINE.parent / "lwd_cloud_mixed_scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LwdCloudMixedScheduler"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_lwd_prep_prefix_readmission"
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"logger": logging.getLogger("test_lwd_prefix_cache")}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method.name]


class TestReadmissionArrivalClamp(unittest.TestCase):
    """抢占重准入的到达钳制(§5.3):H' 不得超过 recv_rows(hash 块对齐)。

    三分支:H'≤recv 不钳;H'>recv 钳到对齐下界并只钉钳制边界内的块;
    钳后 <S 走 refill(挂闸门 + 发 LwdRefillNotify + 不钉块)。
    """

    def setUp(self):
        self.blocks = [NS(is_null=False), NS(is_null=False)]
        # 重查恒命中 2 块 = 32 tokens;hash 块 8 / 调度块 16(混合组下
        # hash 块是 GCD,钳制粒度必须取调度块,否则破坏组块对齐——
        # recv=24 时按 hash 粒度钳会得 24,按调度块粒度钳得 16)
        self.manager = make_manager(
            hit_groups=(list(self.blocks),), hit=32,
            hash_block_size=8, group_block_size=16, scheduler_block_size=16,
        )
        self.reservations = LwdPrefixReservations(self.manager)
        self.prep = _load_prep_readmission()

    def scheduler(self, recv_rows, sender=None):
        return NS(
            lwd_prefix_reservations=self.reservations,
            kv_cache_manager=self.manager,
            lwd_recv_rows={"r": recv_rows},
            _lwd_refill_gate=set(),
            _lwd_refill_pending={},
            lwd_refill_sender=sender or Mock(),
        )

    def test_no_clamp_when_hit_within_arrived(self):
        sched = self.scheduler(recv_rows=100)
        req = make_request(hit_cap=0)
        self.assertTrue(self.prep(sched, req))
        # 未钳制:两块全钉,快照 32,无 refill
        self.manager.block_pool.touch.assert_called_once_with(self.blocks)
        self.assertEqual(
            self.reservations.get_computed_blocks(req),
            ((tuple(self.blocks),), 32),
        )
        sched.lwd_refill_sender.assert_not_called()
        self.assertEqual(sched._lwd_refill_gate, set())

    def test_clamp_to_arrived_pins_only_clamped_prefix(self):
        # recv=24:按 hash 块粒度(8)会钳到 24(破坏组块对齐);按调度块
        # 粒度(16)钳到 16——本用例区分两种粒度
        sched = self.scheduler(recv_rows=24)
        req = make_request(hit_cap=0)
        self.assertTrue(self.prep(sched, req))
        # H'=32 钳到 16:只钉第一块,快照 16
        self.manager.block_pool.touch.assert_called_once_with(
            [self.blocks[0]]
        )
        self.assertEqual(
            self.reservations.get_computed_blocks(req),
            (((self.blocks[0],),), 16),
        )
        sched.lwd_refill_sender.assert_not_called()

    def test_clamp_below_floor_triggers_refill(self):
        sender = Mock()
        sched = self.scheduler(recv_rows=8, sender=sender)
        req = make_request(hit_cap=32)  # S=32
        self.assertFalse(self.prep(sched, req))
        # H'=32 钳到 0(退化为无命中,不钉块)< S → refill [0, 32)
        self.manager.block_pool.touch.assert_not_called()
        self.assertEqual(
            self.reservations.get_computed_blocks(req), (((),), 0)
        )
        self.assertEqual(sched._lwd_refill_gate, {"r"})
        self.assertEqual(sched._lwd_refill_pending, {"r": 0})
        sender.assert_called_once_with("r", 0, 32)

    def test_skip_reading_request_untouched_by_clamp(self):
        sched = self.scheduler(recv_rows=0)
        req = make_request(hit_cap=0, skip=True)
        self.assertTrue(self.prep(sched, req))
        self.manager.coordinator.find_longest_cache_hit.assert_not_called()
        self.manager.block_pool.touch.assert_not_called()


class TestRefillHintPassthrough(unittest.TestCase):
    """recv hint 的 refills 透传(§5.3):hint 构造函数在 ascend 仓,
    此处按 AST 钉住云侧调用点契约(桩隔离,不导入 ascend 实现)。"""

    @classmethod
    def setUpClass(cls):
        tree = ast.parse(CLOUD_ENGINE.read_text(encoding="utf-8"))
        cls.call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "make_recv_hint"
        )

    def test_make_recv_hint_receives_refills_kwarg(self):
        kw = next((k for k in self.call.keywords if k.arg == "refills"), None)
        self.assertIsNotNone(
            kw, "make_recv_hint call must pass refills= (is_refill passthrough)"
        )
        # 与 req_ids 平行的逐条目列表推导:[item.is_refill for item in items]
        comp = kw.value
        self.assertIsInstance(comp, ast.ListComp)
        items = [NS(is_refill=False), NS(is_refill=True), NS(is_refill=False)]
        value = eval(
            compile(ast.Expression(comp), str(CLOUD_ENGINE), "eval"),
            {},
            {"items": items},
        )
        self.assertEqual(value, [False, True, False])

    def test_hint_kwarg_stays_parallel_to_req_ids(self):
        names = [k.arg for k in self.call.keywords]
        self.assertIn("req_ids", names)
        self.assertIn("refills", names)


if __name__ == "__main__":
    unittest.main()
