"""Run only in the isolated qualification environment; no CPU speed claims.

GEMMA_ADAPTER_MODULE selects a new adapter version without editing these tests.
The official Transformers/PyTorch model is the independent numerical oracle.
"""
import importlib.util
import os
from pathlib import Path
import unittest

_PATH = Path(__file__).resolve().parents[1] / "tools" / "qualify_gemma_v001.py"
_SPEC = importlib.util.spec_from_file_location("gemma_qualification_v001", _PATH)
qualification = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qualification)


class GemmaOfficialQualification(unittest.TestCase):
    def assert_checks(self, result):
        failures = {name: row for name, row in result.items() if not row["passed"]}
        self.assertFalse(failures, str(failures))

    def test_official_primitives_and_exact_window_boundary(self):
        adapter = qualification.adapter_module()
        self.assert_checks(qualification.primitive_checks(adapter))

    def test_official_whole_tiny_model_and_custom_pallas_layers(self):
        result = qualification.tiny_qualification(custom=True)
        self.assertGreater(result["sequence_length"], result["sliding_window"])
        self.assertGreater(result["hidden_size"], 256)
        self.assertGreater(result["intermediate_size"], 256)
        self.assertNotEqual(result["hidden_size"] % 256, 0)
        self.assertIsNone(result["final_logit_softcapping"])
        self.assert_checks(result["checks"])
        for algorithm in ("cubic_full", "strassen"):
            self.assertIn(f"layer_000_{algorithm}_interpret", result["checks"])
            self.assertIn(f"layer_001_{algorithm}_interpret", result["checks"])

    def test_official_actual_window512_and_softcapped_tied_head(self):
        result = qualification.tiny_qualification(long_window=True, final_softcap=3.0)
        self.assertEqual(result["sliding_window"], 512)
        self.assertEqual(result["sequence_length"], 529)
        self.assertEqual(result["layer_types"][4:6], ["sliding_attention", "full_attention"])
        self.assert_checks(result["checks"])

    def test_metric_gates_reject_large_architectural_error(self):
        import numpy as np
        reference = np.array([[2., -1., 0.], [-1., 2., 0.]], dtype=np.float32)
        identity = qualification.metrics(reference, reference, targets=[0, 1])
        swapped = qualification.metrics(reference, reference[:, ::-1], targets=[0, 1])
        self.assertTrue(identity["passed"])
        self.assertFalse(swapped["passed"])
        self.assertFalse(qualification.metrics(reference, reference * 0)["passed"])
        # Layer/hidden-state tolerance is intentionally stricter than logits.
        shifted = reference * 1.025
        self.assertTrue(qualification.metrics(reference, shifted)["passed"])
        self.assertFalse(qualification.metrics(reference, shifted, layer=True)["passed"])

    def test_nondefault_rope_is_explicitly_unsupported(self):
        _, _, Config = qualification.official_dependencies()
        config = Config(vocab_size=97, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, layer_types=["sliding_attention", "full_attention"],
            rope_scaling={"rope_type": "linear", "factor": 2.0}).to_dict()
        with self.assertRaisesRegex(ValueError, "unscaled default RoPE"):
            qualification.adapter_module().validate_config(config)


if __name__ == "__main__":
    unittest.main()
