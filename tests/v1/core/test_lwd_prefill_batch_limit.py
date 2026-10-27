# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the DeepSeek V4 edge prefill batch limit."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace as NS


class TestPrefillBatchLimit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).resolve().parents[3]
            / "vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_scheduler.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        scheduler = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "LwdEdgeScheduler"
        )
        method = next(
            node
            for node in scheduler.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_lwd_pick_prefill_batch"
        )
        namespace = {}
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        cls.pick = staticmethod(namespace[method.name])

    @staticmethod
    def request(rid, prompt=8196, computed=0):
        return NS(
            request_id=rid, num_prompt_tokens=prompt, num_computed_tokens=computed
        )

    def runner(self, model_type="deepseek_v4", running=(), waiting=(), skipped=()):
        runner = NS(
            vllm_config=NS(model_config=NS(hf_text_config=NS(model_type=model_type))),
            _lwd_mixed=True,
            _lwd_awaiting={},
            _lwd_spec_factor=2,
            _lwd_chunk_unit=8196,
            max_num_scheduled_tokens=20000,
            max_num_running_reqs=60,
            running=list(running),
            waiting=list(waiting),
            skipped_waiting=list(skipped),
            _lwd_route={r.request_id: 0 for r in (*running, *waiting, *skipped)},
            _lwd_prefix_pending={},
        )
        runner._lwd_engine_load = lambda engine: (
            len(runner.running) + len(runner._lwd_prefix_pending),
            len(runner._lwd_awaiting),
        )
        return runner

    def test_dsv4_waiting_batch_contains_only_first_request(self):
        runner = self.runner(waiting=[self.request("a"), self.request("b")])
        picked, expected, _ = self.pick(runner)
        self.assertEqual(picked, ["a"])
        self.assertEqual(expected, {"a": 8196})
        self.assertEqual(len(runner.waiting), 2)

    def test_continuation_excludes_other_continuations_and_new_requests(self):
        runner = self.runner(
            running=[self.request("a", computed=8000), self.request("b")],
            waiting=[self.request("c")],
        )
        self.assertEqual(self.pick(runner)[:2], (["a"], {"a": 196}))

    def test_decode_does_not_consume_prefill_slot_and_skipped_has_priority(self):
        runner = self.runner(
            running=[self.request("decode", computed=8200)],
            skipped=[self.request("skipped")],
            waiting=[self.request("new")],
        )
        self.assertEqual(self.pick(runner)[:2], (["skipped"], {"skipped": 8196}))

    def test_budget_truncation_and_full_sequence_slots(self):
        runner = self.runner(
            running=[self.request("a", computed=1000)], waiting=[self.request("b")]
        )
        runner.max_num_scheduled_tokens = 102
        runner.max_num_running_reqs = 1
        # host 暂存后 cap = 整步预算(不再有 decode 预留系数):续传块
        # 被剩余额度截尾成 102
        self.assertEqual(self.pick(runner), (["a"], {"a": 102}, 102))
        runner.running[0].num_computed_tokens = 8196
        # 续传已完结 + free_seq = 0(inflight=1 占满名额):无新准入
        self.assertEqual(self.pick(runner), ([], {}, 102))
        runner.max_num_scheduled_tokens = 2
        self.assertEqual(self.pick(runner), ([], {}, 2))

    def test_other_models_keep_multi_request_batches(self):
        for model_type in ("deepseek_v3", "qwen3_5"):
            with self.subTest(model_type=model_type):
                runner = self.runner(
                    model_type,
                    waiting=[
                        self.request("a"),
                        self.request("b"),
                    ],
                )
                self.assertEqual(
                    self.pick(runner)[:2], (["a", "b"], {"a": 8196, "b": 8196})
                )

    def test_threshold_boundary_and_short_batches(self):
        for length in (7999, 8000, 8001, 8062, 8192):
            with self.subTest(length=length):
                runner = self.runner(
                    waiting=[
                        self.request("a", prompt=length),
                        self.request("b", prompt=1),
                    ]
                )
                expected = {"a": length}
                if length < 8000:
                    expected["b"] = 1
                self.assertEqual(self.pick(runner)[:2], (list(expected), expected))

    def test_short_then_long_preserves_fcfs(self):
        runner = self.runner(
            waiting=[
                self.request("short", prompt=10),
                self.request("long", prompt=8000),
                self.request("later", prompt=10),
            ]
        )
        self.assertEqual(self.pick(runner)[:2], (["short"], {"short": 10}))

    def test_long_continuation_blocks_new_short_requests(self):
        runner = self.runner(
            running=[self.request("long", prompt=8000, computed=7998)],
            waiting=[self.request("short", prompt=10)],
        )
        self.assertEqual(self.pick(runner)[:2], (["long"], {"long": 2}))

    def test_pending_long_continuation_is_not_bypassed(self):
        runner = self.runner(
            running=[
                self.request("short", prompt=20, computed=10),
                self.request("long", prompt=8000, computed=7998),
            ],
            waiting=[self.request("new", prompt=10)],
        )
        self.assertEqual(self.pick(runner)[:2], (["short"], {"short": 10}))

    def test_short_continuations_and_waiting_can_batch(self):
        runner = self.runner(
            running=[self.request("a", prompt=20, computed=10)],
            waiting=[self.request("b", prompt=10), self.request("c", prompt=10)],
        )
        self.assertEqual(
            self.pick(runner)[:2], (["a", "b", "c"], {"a": 10, "b": 10, "c": 10})
        )

    def test_prefix_hit_new_request_plans_remaining_only(self):
        # 真机事故:命中请求 num_computed=1536(12×128 块),picker 曾按
        # 全长 4106 记账,原生按剩余 2570 排程,核验当场误炸
        runner = self.runner(
            waiting=[self.request("hit", prompt=4106, computed=1536)]
        )
        self.assertEqual(self.pick(runner)[:2], (["hit"], {"hit": 2570}))

    def test_prefix_hit_chunk_capped_by_unit(self):
        runner = self.runner(
            model_type="qwen3_5",
            waiting=[self.request("hit", prompt=6000, computed=1536)],
        )
        runner._lwd_chunk_unit = 2000
        # 剩余 4468 被 chunk 单元截到 2000(与全长无关)
        self.assertEqual(self.pick(runner)[:2], (["hit"], {"hit": 2000}))

    def test_fully_computed_waiting_request_is_skipped(self):
        runner = self.runner(
            model_type="qwen3_5",
            waiting=[
                self.request("done", prompt=1536, computed=1536),
                self.request("new", prompt=10),
            ],
        )
        self.assertEqual(self.pick(runner)[:2], (["new"], {"new": 10}))


if __name__ == "__main__":
    unittest.main()
