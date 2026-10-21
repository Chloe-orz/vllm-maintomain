# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the LWD role-scoped module policy on ``LwdConfig``."""

from vllm.config.lwd import LwdConfig


def edge(mode: str = "prefill_only", enabled: bool = True) -> LwdConfig:
    return LwdConfig(enabled=enabled, role="edge", mode=mode, edge_head_tail_layers=(0, 0))


def cloud(mode: str = "prefill_only", enabled: bool = True) -> LwdConfig:
    return LwdConfig(enabled=enabled, role="cloud", mode=mode, edge_head_tail_layers=(0, 0))


class TestDisabledOrOtherModes:
    def test_disabled_skips_nothing(self):
        config = LwdConfig(enabled=False, role="edge", mode="prefill_only")
        assert not config.skips_module("model.norm")
        assert not config.skips_module("model.visual")

    def test_head_tail_mode_skips_nothing(self):
        # In these modes the edge runs the tail segment, so the trailing norm
        # and the tower stay live on whichever side owns them.
        assert not cloud("head_tail").skips_module("model.visual")
        assert not edge("head_tail").skips_module("model.norm")

    def test_embedding_only_mode_skips_nothing(self):
        assert not cloud("embedding_only").skips_module("model.visual")
        assert not edge("embedding_only").skips_module("model.norm")


class TestEdgeRole:
    def test_skips_backbone_norm_text_layout(self):
        assert edge().skips_module("model.norm")

    def test_skips_backbone_norm_vl_layout(self):
        assert edge().skips_module("model.language_model.model.norm")

    def test_keeps_embedding_and_head(self):
        config = edge()
        assert not config.skips_module("model.embed_tokens")
        assert not config.skips_module("lm_head")
        assert not config.skips_module("language_model.lm_head")

    def test_keeps_per_layer_norms(self):
        config = edge()
        assert not config.skips_module("model.layers.5.input_layernorm")
        assert not config.skips_module("model.layers.5.post_attention_layernorm")

    def test_keeps_vision_tower_and_its_inner_norm(self):
        # The edge is the side that runs the encoder, and the tower's merger
        # has a child literally called `norm`: the path-based rule must not
        # catch it.
        config = edge()
        assert not config.skips_module("model.visual")
        assert not config.skips_module("model.visual.merger.norm")
        assert not config.skips_module("model.visual.blocks.0.norm1")


class TestCloudRole:
    def test_skips_vision_tower(self):
        assert cloud().skips_module("model.visual")

    def test_skips_vision_tower_without_prefix(self):
        assert cloud().skips_module("visual")

    def test_keeps_everything_the_language_model_needs(self):
        config = cloud()
        for path in (
            "model.language_model.model.norm",
            "model.language_model.model.embed_tokens",
            "model.language_model.model.layers.3.mlp.down_proj",
            "lm_head",
        ):
            assert not config.skips_module(path), path

    def test_does_not_skip_a_different_tower_name(self):
        # Only the Qwen3.x tower is covered today; other names stay untouched.
        assert not cloud().skips_module("model.vision_tower")
