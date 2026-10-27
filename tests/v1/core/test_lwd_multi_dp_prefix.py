# SPDX-License-Identifier: Apache-2.0
"""CPU tests for multi-DP prefix negotiation and failure cleanup."""

import ast
import logging
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[3]


class _EngineShim:
    def _lwd_shutdown_planes(self):
        self.shutdown_calls += 1

    def abort_requests(self, request_ids):
        self.native_aborted.append(list(request_ids))


def load_engine():
    path = (
        ROOT
        / "vllm/v1/lwd_control/control_edge_scheduler/lwd_edge_engine.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LwdEdgeEngineCore"
    )
    wanted = {
        "_lwd_configure_prefix_cache",
        "_lwd_cross_check_hello",
        "_lwd_handle_refill",
        "abort_requests",
    }
    methods = [
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias("annotations")], level=0
    )
    shim = ast.ClassDef(
        name="Engine",
        bases=[ast.Name(id="_EngineShim", ctx=ast.Load())],
        keywords=[], body=methods, decorator_list=[],
    )
    module = ast.Module(body=[future, shim], type_ignores=[])
    namespace = {
        "_EngineShim": _EngineShim,
        "LWD_WIRE_VERSION": 4,
        "LwdHelloNotify": object,
        "logger": logging.getLogger("test_lwd_multi_dp_prefix"),
        "deque": deque,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["Engine"]


Engine = load_engine()


def hello(engine, enabled=True, block_size=16, wire=4):
    return NS(
        engine_index=engine,
        prefix_caching=enabled,
        hash_block_size=block_size,
        prefix_alignment_tokens=block_size,
        prefix_alignment_required=False,
        prefix_use_eagle=False,
        wire_version=wire,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
    )


def make_engine(hellos, enabled=True, expected=2):
    engine = Engine()
    engine._lwd_prefix_enabled = enabled
    engine._lwd_peer_hellos = hellos
    engine._lwd_expected_engines = expected
    engine._lwd_prefix_control = Mock() if enabled else None
    engine._lwd_pending_refills = deque()
    engine.native_aborted = []
    engine.shutdown_calls = 0
    engine.vllm_config = NS(
        model_config=NS(uses_mrope=False),
        scheduler_config=NS(max_num_batched_tokens=1024, max_num_seqs=4),
    )
    engine.scheduler = NS(
        lwd_edge_build_refill=Mock(return_value=None),
        lwd_edge_abort=Mock(),
    )
    return engine


class TestMultiDpPrefixHello(unittest.TestCase):
    def test_single_dp_consistent_hello_configures_once(self):
        engine = make_engine({0: hello(0)}, expected=1)
        engine._lwd_configure_prefix_cache()
        engine._lwd_prefix_control.configure.assert_called_once_with(
            engine._lwd_peer_hellos[0]
        )
        self.assertEqual(engine.shutdown_calls, 0)

    def test_multi_dp_consistent_hellos_configure_once(self):
        engine = make_engine({0: hello(0), 1: hello(1)})
        engine._lwd_configure_prefix_cache()
        engine._lwd_prefix_control.configure.assert_called_once()
        self.assertEqual(engine.shutdown_calls, 0)

    def test_any_engine_switch_mismatch_fails(self):
        engine = make_engine({0: hello(0), 1: hello(1, enabled=False)})
        with self.assertRaisesRegex(RuntimeError, "prefix caching mismatch"):
            engine._lwd_configure_prefix_cache()
        engine._lwd_prefix_control.configure.assert_not_called()
        self.assertEqual(engine.shutdown_calls, 1)

    def test_hash_granularity_mismatch_fails(self):
        engine = make_engine({0: hello(0), 1: hello(1, block_size=32)})
        with self.assertRaisesRegex(RuntimeError, "inconsistent prefix"):
            engine._lwd_configure_prefix_cache()
        engine._lwd_prefix_control.configure.assert_not_called()

    def test_old_wire_fails_before_negotiation(self):
        engine = make_engine({0: hello(0, wire=3)}, expected=1)
        with self.assertRaisesRegex(RuntimeError, "prefix wire"):
            engine._lwd_configure_prefix_cache()
        engine._lwd_prefix_control.configure.assert_not_called()

    def test_disabled_pc_does_not_require_hash_agreement(self):
        engine = make_engine(
            {0: hello(0, enabled=False, block_size=16),
             1: hello(1, enabled=False, block_size=32)},
            enabled=False,
        )
        engine._lwd_configure_prefix_cache()
        self.assertEqual(engine.shutdown_calls, 0)

    def test_hello_engine_set_must_be_exact(self):
        engine = make_engine({0: hello(0), 1: hello(1)})
        engine._lwd_cross_check_hello()
        for hellos in ({0: hello(0), 2: hello(2)},
                       {0: hello(0), 1: hello(1), 2: hello(2)}):
            with self.subTest(engines=list(hellos)):
                engine = make_engine(hellos)
                with self.assertRaisesRegex(RuntimeError, "engine set mismatch"):
                    engine._lwd_cross_check_hello()
                self.assertEqual(engine.shutdown_calls, 1)


class TestRefillFailureCleanup(unittest.TestCase):
    def test_failed_refill_uses_full_abort_and_drops_queued_batch(self):
        engine = make_engine({0: hello(0)}, expected=1)
        queued = NS(lwd_batch=NS(batch_meta=NS(req_ids=["r"])))
        engine._lwd_pending_refills.append(queued)
        engine._lwd_handle_refill(
            NS(request_id="r", start_row=0, end_row=1)
        )
        self.assertEqual(engine.native_aborted, [["r"]])
        engine.scheduler.lwd_edge_abort.assert_called_once_with(["r"])
        self.assertEqual(len(engine._lwd_pending_refills), 0)


class TestOutputContract(unittest.TestCase):
    def test_per_engine_watermark_field_is_declared(self):
        path = ROOT / "vllm/v1/outputs.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "ModelRunnerOutput"
        )
        fields = {
            node.target.id for node in cls.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
        }
        self.assertIn("lwd_up_send_watermark", fields)
        self.assertIn("lwd_up_send_watermarks", fields)


if __name__ == "__main__":
    unittest.main()
