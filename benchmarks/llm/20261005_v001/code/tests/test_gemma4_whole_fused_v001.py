"""Dense synthetic whole-layer checks; official CPU oracle, no TPU timing.

Native retains the <=2% layer qualification limit. Custom arms use the existing
projection/layer tuning eligibility limit of <=10%; this is not a predictive
quality gate and does not qualify any real-checkpoint policy for deployment.
"""
import json
import os
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import Gemma4ForCausalLM
from transformers.models.gemma4 import modeling_gemma4 as official
import test_gemma4_native_v002 as fixture
from strassen_mm import model_gemma4_fused_v001 as fused


class Gemma4WholeLayerFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(1004)
        cls.c = fixture.tiny_config(8)
        cls.c.hidden_size, cls.c.intermediate_size = 256, 384
        cls.c.num_attention_heads, cls.c.num_key_value_heads, cls.c.num_global_key_value_heads = 2, 2, 1
        cls.c.head_dim, cls.c.global_head_dim = 256, 512
        cls.model = Gemma4ForCausalLM(cls.c).to(torch.bfloat16).eval()
        cls.model.model.rotary_emb = official.Gemma4TextRotaryEmbedding(cls.c)
        with torch.no_grad():
            for name, value in cls.model.named_parameters():
                if "norm" in name:
                    value.copy_(torch.linspace(.75, 1.25, value.numel()))
            for i, layer in enumerate(cls.model.model.layers):
                layer.layer_scalar.fill_(.625 if i == 0 else 1.375)
        cls.captured = []
        hooks = [layer.register_forward_hook(lambda m, args, result: cls.captured.append((args[0].detach().clone(), result.detach().clone()))) for layer in cls.model.model.layers]
        with torch.no_grad():
            cls.model(torch.randint(1, 97, (2, 9)), use_cache=False)
        for hook in hooks:
            hook.remove()
        cls.weights = [fixture.layer_weights(layer) for layer in cls.model.model.layers]
        cls.records = []

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("STRASSEN_EXECUTION_DIR"):
            out = Path(os.environ["STRASSEN_EXECUTION_DIR"]) / "artifacts"
            out.mkdir(exist_ok=True)
            (out / "gemma4-whole-fused-checks.json").write_text(json.dumps(cls.records, indent=2) + "\n")

    def compare(self, label, actual, expected, limit):
        expected = expected.detach().float().numpy() if isinstance(expected, torch.Tensor) else expected
        error = fixture.metric(expected, actual)
        self.records.append(dict(label=label, **error, relative_l2_limit=limit, synthetic=True, cpu_interpretation=True))
        self.assertTrue(error["finite"])
        self.assertLessEqual(error["relative_l2"], limit, label + str(error))

    def test_native_official_and_xla_attention_both_layer_types(self):
        for index in range(2):
            x, expected = self.captured[index]
            x = fixture.array(x)
            fn = fused.build_layer(self.c.to_dict(), 9, batch_size=2, layer_index=index,
                                   attention_backend="explicit", capture=True)
            actual, traces = fn(x, self.weights[index])
            self.compare(f"native-explicit-layer{index}", actual, expected, .02)
            self.assertEqual(set(traces), set(fn.metadata["matrix_product_sites"]))
            self.assertEqual(len(traces), 6 if index == 0 else 5)
            if index == 1:
                self.assertNotIn("v", traces)
                self.assertTrue(fn.metadata["shared_raw_kv"])
                self.assertEqual(fn.metadata["projections"]["k"]["epilogue"]["kind"], "none")
            unfused = fused.build_layer(self.c.to_dict(), 9, batch_size=2, layer_index=index,
                attention_backend="explicit", fused_sites=())
            np.testing.assert_array_equal(np.asarray(unfused(x, self.weights[index])), np.asarray(actual))
            xla = fused.build_layer(self.c.to_dict(), 9, batch_size=2, layer_index=index)
            self.compare(f"native-xla-layer{index}", xla(x, self.weights[index]), expected, .02)
            prepared = fn.prepare_weights(self.weights[index])
            np.testing.assert_array_equal(np.asarray(fn.prepared(x, prepared)[0]), np.asarray(actual))

    def test_custom_dense_layers_both_rounding_contracts(self):
        for index in range(2):
            x, expected = self.captured[index]
            x = fixture.array(x)
            for rounding in ("model", "accumulator"):
                _, geometry, specs = fused.geometry(self.c.to_dict(), 2, 9, index, rounding=rounding)
                for implementation, depth, mode in (("cubic", 0, "outputs"), ("cubic_full", 0, "outputs"),
                                                    ("current", 1, "outputs"), ("current", 2, "products")):
                    label = f"{implementation}-S{depth}-{mode}-{rounding}-layer{index}"
                    with self.subTest(label=label):
                        policy = fused.native_policy(self.c.to_dict(), index)
                        for site in policy:
                            bn = 2*geometry["head_dim"] if specs[site].rotary else 512
                            policy[site] = dict(policy[site], implementation=implementation,
                                depth=depth, accumulator=mode, tile=[32, bn, 512])
                        fn = fused.build_layer(self.c.to_dict(), 9, policy, batch_size=2, layer_index=index,
                            rounding=rounding, attention_backend="explicit", interpret=True)
                        prepared = fn.prepare_weights(self.weights[index])
                        actual = fn.prepared(x, prepared)
                        self.compare(label, actual, expected, .1)
                        self.assertEqual(actual.dtype, jnp.bfloat16)
                        self.assertEqual(set(fn.metadata["projections"]), set(geometry["projection_shapes_mkn"]))
                    jax.clear_caches()

    def test_global_layer_rejects_fictitious_v_and_non_bf16_campaign_output(self):
        policy = fused.native_policy(self.c.to_dict(), 1)
        policy["v"] = policy["k"]
        with self.assertRaisesRegex(ValueError, "nonexistent"):
            fused.build_layer(self.c.to_dict(), 9, policy, batch_size=2, layer_index=1)
        with self.assertRaisesRegex(ValueError, "BF16"):
            fused.build_layer(self.c.to_dict(), 9, output_dtype="float32")


if __name__ == "__main__":
    unittest.main(verbosity=2)
