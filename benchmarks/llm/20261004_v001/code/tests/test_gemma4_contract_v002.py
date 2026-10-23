"""Metadata checks against untouched pinned official constructors on meta.

This is not full Transformers integration or numerical/checkpoint qualification.
Only selected official class AST nodes are loaded. The checkpointing base is
nn.Module because no forward/backward runs; ACT2FN is only used for construction.
The optional kernel-replacement decorator is an identity shim. Its rotary
function dependency raises if called: no numerical oracle is implied here.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from strassen_mm.gemma4_contract_v001 import validate_config, layer_geometry, required_tensor_shapes

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "results/v6e/llm_campaign_20261003_v001/inputs_gemma4_v001/gemma4_31b.config.json"
REFERENCE = ROOT / "third_party/transformers_gemma4_v001/modeling_gemma4.py"


class Gemma4MetadataContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == "6a81841cad2b6ba06841e23c6afb5f0a27827bc12c64328ffa6338831a21267e"
        assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == "6a86e03348df5ec104703e7161de9a911137cba500a0be0f133e2850ef0bf935"
        cls.raw = json.loads(CONFIG.read_text())
        names = {"Gemma4RMSNorm", "Gemma4TextMLP", "Gemma4TextAttention", "Gemma4TextDecoderLayer"}
        nodes = [n for n in ast.parse(REFERENCE.read_text()).body if isinstance(n, ast.ClassDef) and n.name in names]
        assert {n.name for n in nodes} == names
        tree = ast.Module(body=ast.parse("from __future__ import annotations").body + nodes, type_ignores=[])
        def unexecuted_rotary(*args, **kwargs):
            raise AssertionError("Metadata checks must not execute a rotary forward")

        scope = {"use_kernelized_func": lambda *args, **kwargs: lambda cls: cls,
                 "apply_rotary_pos_emb": unexecuted_rotary, "torch": torch, "nn": torch.nn, "GradientCheckpointingLayer": torch.nn.Module,
                 "ACT2FN": {"gelu_pytorch_tanh": torch.nn.GELU(approximate="tanh")}}
        exec(compile(tree, str(REFERENCE), "exec"), scope)
        cls.OfficialLayer = scope["Gemma4TextDecoderLayer"]

    def test_every_official_layer_state_including_buffers(self):
        config = SimpleNamespace(**self.raw["text_config"])
        required = required_tensor_shapes(self.raw)
        with torch.device("meta"):
            for i in range(config.num_hidden_layers):
                official = self.OfficialLayer(config, i)
                prefix = f"model.layers.{i}."
                actual = {prefix + k: list(v.shape) for k, v in official.state_dict().items()}
                expected = {k: v for k, v in required.items() if k.startswith(prefix)}
                self.assertEqual(actual, expected)
                self.assertIn("layer_scalar", dict(official.named_buffers()))
                self.assertNotIn("layer_scalar", dict(official.named_parameters()))
                geometry = layer_geometry(self.raw, i, batch=4, sequence=1024)
                self.assertEqual(geometry["attention_scale"], official.self_attn.scaling)
                self.assertEqual(geometry["kv_heads"], config.num_attention_heads // official.self_attn.num_key_value_groups)
                for site in ("q", "k", "v", "o"):
                    linear = getattr(official.self_attn, site + "_proj")
                    if linear is None:
                        self.assertNotIn(site, geometry["projection_shapes_mkn"])
                    else:
                        self.assertEqual(geometry["projection_shapes_mkn"][site], [4096, linear.in_features, linear.out_features])

    def test_pinned_31b_projection_geometry(self):
        local = layer_geometry(self.raw, 0, batch=8, sequence=1024)
        global_ = layer_geometry(self.raw, 5, batch=8, sequence=1024)
        self.assertEqual(local["projection_shapes_mkn"], {
            "q": [8192, 5376, 8192], "k": [8192, 5376, 4096], "v": [8192, 5376, 4096],
            "o": [8192, 8192, 5376], "gateup": [8192, 5376, 43008], "down": [8192, 21504, 5376]})
        self.assertEqual(global_["projection_shapes_mkn"], {
            "q": [8192, 5376, 16384], "k": [8192, 5376, 2048], "o": [8192, 16384, 5376],
            "gateup": [8192, 5376, 43008], "down": [8192, 21504, 5376]})
        self.assertTrue(global_["shared_raw_kv"])
        self.assertEqual(local["sliding_window"], 1024)
        self.assertIsNone(global_["sliding_window"])

    def test_configuration_is_copied_and_unsupported_features_rejected(self):
        before = deepcopy(self.raw)
        c = validate_config(self.raw)
        c["rope_parameters"]["full_attention"]["rope_theta"] = 1
        c["layer_types"][0] = "full_attention"
        self.assertEqual(self.raw, before)
        for key, value in [("hidden_size_per_layer_input", 256), ("num_kv_shared_layers", 4),
                           ("enable_moe_block", True), ("use_double_wide_mlp", True),
                           ("attention_k_eq_v", False), ("attention_bias", True),
                           ("use_bidirectional_attention", "all"), ("num_global_key_value_heads", 3),
                           ("global_head_dim", 511), ("dtype", "float32"),
                           ("tie_word_embeddings", False), ("attention_dropout", .1),
                           ("hidden_size", True), ("rms_norm_eps", float("nan"))]:
            with self.subTest(key=key):
                invalid = deepcopy(self.raw)
                invalid["text_config"][key] = value
                with self.assertRaises(ValueError):
                    validate_config(invalid)
        for key, value in [("rope_type", "linear"), ("partial_rotary_factor", .3), ("factor", 2)]:
            invalid = deepcopy(self.raw)
            invalid["text_config"]["rope_parameters"]["full_attention"][key] = value
            with self.assertRaises(ValueError):
                validate_config(invalid)
        for index, batch, sequence in [(-1, 1, 512), (60, 1, 512), (0, 0, 512),
                                       (0, 1, 262145), (True, 1, 512)]:
            with self.assertRaises(ValueError):
                layer_geometry(self.raw, index, batch, sequence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
