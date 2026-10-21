# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the LWD placeholder-weight filter used during model loading."""

from vllm.model_executor.model_loader.lwd_weight_filter import (
    name_is_skipped,
    parse_skip_keys,
)

# ---------------------------------------------------------------------------
# Unit tests for parse_skip_keys
# ---------------------------------------------------------------------------


class TestParseSkipKeys:
    def test_strips_trailing_dot(self):
        assert parse_skip_keys(["model.norm.", "visual."]) == [
            ["model", "norm"],
            ["visual"],
        ]

    def test_drops_empty_names(self):
        assert parse_skip_keys(["", "model.norm."]) == [["model", "norm"]]

    def test_empty_input(self):
        assert parse_skip_keys([]) == []


# ---------------------------------------------------------------------------
# Unit tests for name_is_skipped
# ---------------------------------------------------------------------------


class TestNameIsSkipped:
    def test_no_keys_keeps_everything(self):
        assert not name_is_skipped("model.norm.weight", [])

    def test_skips_layer_tensor(self):
        keys = parse_skip_keys(["model.layers.3."])
        assert name_is_skipped("model.layers.3.mlp.down_proj.weight", keys)

    def test_does_not_skip_neighbouring_layer(self):
        # Component-wise matching: layers.3 must not swallow layers.30.
        keys = parse_skip_keys(["model.layers.3."])
        assert not name_is_skipped("model.layers.30.mlp.down_proj.weight", keys)

    def test_skips_every_tensor_of_a_layer(self):
        keys = parse_skip_keys(["model.layers.7."])
        for name in (
            "model.layers.7.self_attn.q_proj.weight",
            "model.layers.7.input_layernorm.weight",
            "model.layers.7.mlp.gate_up_proj.weight_scale",
        ):
            assert name_is_skipped(name, keys), name

    def test_skips_edge_final_norm(self):
        keys = parse_skip_keys(["model.norm."])
        assert name_is_skipped("model.norm.weight", keys)

    def test_keeps_other_norms_when_only_final_norm_trimmed(self):
        # Regression guard: the edge trims only the language model's trailing
        # norm, so per-layer norms and the vision tower's own `norm` must stay.
        keys = parse_skip_keys(["model.norm."])
        for name in (
            "model.layers.5.input_layernorm.weight",
            "model.layers.5.post_attention_layernorm.weight",
            "model.visual.merger.norm.weight",
        ):
            assert not name_is_skipped(name, keys), name

    def test_skips_visual_tower_behind_mapped_checkpoint_prefix(self):
        # VL checkpoints store the tower as model.visual.* while the vLLM
        # module is visual; component matching must still find it.
        keys = parse_skip_keys(["visual."])
        for name in (
            "model.visual.blocks.0.attn.qkv.weight",
            "visual.patch_embed.proj.weight",
            "model.visual.merger.norm.weight",
        ):
            assert name_is_skipped(name, keys), name

    def test_skips_language_model_norm_in_vl_layout(self):
        keys = parse_skip_keys(["model.language_model.model.norm."])
        assert name_is_skipped("model.language_model.model.norm.weight", keys)
        assert not name_is_skipped(
            "model.language_model.model.layers.0.input_layernorm.weight", keys
        )

    def test_keeps_embedding_and_head(self):
        keys = parse_skip_keys(["model.layers.3.", "model.norm."])
        assert not name_is_skipped("model.embed_tokens.weight", keys)
        assert not name_is_skipped("lm_head.weight", keys)

    def test_keeps_visual_when_not_trimmed(self):
        keys = parse_skip_keys(["model.norm."])
        assert not name_is_skipped("visual.blocks.0.norm1.weight", keys)


# ---------------------------------------------------------------------------
# End-to-end shape of the two deployment roles
# ---------------------------------------------------------------------------


class TestRoleLayouts:
    def test_edge_keeps_only_embedding_and_head(self):
        # Edge prefill_only: every transformer layer plus the final norm is a
        # placeholder; embed_tokens and lm_head are built.
        placeholder_names = [f"model.layers.{i}." for i in range(4)]
        placeholder_names.append("model.norm.")
        keys = parse_skip_keys(placeholder_names)

        kept, skipped = [], []
        for name in (
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.3.mlp.down_proj.weight",
            "lm_head.weight",
        ):
            (skipped if name_is_skipped(name, keys) else kept).append(name)

        assert skipped == [
            "model.norm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.3.mlp.down_proj.weight",
        ]
        assert kept == ["model.embed_tokens.weight", "lm_head.weight"]

    def test_cloud_keeps_everything_but_the_vision_tower(self):
        keys = parse_skip_keys(["visual."])
        assert name_is_skipped("model.visual.blocks.0.mlp.0.weight", keys)
        for name in (
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.1.mlp.down_proj.weight",
            "lm_head.weight",
        ):
            assert not name_is_skipped(name, keys), name
