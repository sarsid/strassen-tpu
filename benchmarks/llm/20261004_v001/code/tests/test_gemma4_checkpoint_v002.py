"""Strict loading and independent round-trip tests using tiny synthetic shards."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import jax.numpy as jnp
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import Gemma4ForCausalLM, Gemma4TextConfig
from transformers.models.gemma4 import modeling_gemma4 as official
from strassen_mm.checkpoint_gemma4_v001 import Checkpoint
from strassen_mm import model_gemma4_v001 as adapter

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("official_gemma4_loader", ROOT / "tools/load_gemma4_text_v002.py")
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)


class Gemma4CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        config = Gemma4TextConfig(vocab_size=97, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            num_global_key_value_heads=1, head_dim=16, global_head_dim=32,
            attention_k_eq_v=True, hidden_size_per_layer_input=0,
            layer_types=["sliding_attention", "full_attention"], sliding_window=8,
            max_position_embeddings=8192, final_logit_softcapping=3.0, dtype="bfloat16")
        config._attn_implementation = "eager"
        cls.model = Gemma4ForCausalLM(config).to(torch.bfloat16).eval()
        cls.model.model.rotary_emb = official.Gemma4TextRotaryEmbedding(config)
        with torch.no_grad():
            for name, value in cls.model.named_parameters():
                if "norm" in name:
                    value.copy_(torch.linspace(.75, 1.25, value.numel()))
            cls.model.model.layers[0].layer_scalar = torch.tensor([1.001], dtype=torch.float32)
            cls.model.model.layers[1].layer_scalar = torch.tensor([.625], dtype=torch.bfloat16)
        cls.raw = {"model_type": "gemma4", "dtype": "bfloat16", "tie_word_embeddings": True,
                   "text_config": config.to_dict(), "vision_config": {"metadata_only": True}}
        cls.canonical = {k: v.detach().clone() for k, v in cls.model.state_dict().items()}

    def shard_state(self):
        result = {("model.language_model." + k[len("model."):]) if k.startswith("model.") else k: v.clone()
                  for k, v in self.canonical.items()}
        result["model.vision_tower.unused_test.weight"] = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        return result

    def write_checkpoint(self, folder, state):
        (folder / "config.json").write_text(json.dumps(self.raw))
        save_file({k: v.contiguous() for k, v in state.items()}, folder / "model.safetensors")
        records = []
        for name in ("config.json", "model.safetensors"):
            data = (folder / name).read_bytes()
            records.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        manifest = dict(model_id="synthetic/gemma4-test", revision="a"*40, cache_dir=str(folder),
                        config=deepcopy(self.raw), files=records)
        path = folder / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_exact_state_round_trip_and_native_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            checkpoint = Checkpoint(self.write_checkpoint(folder, self.shard_state()))
            model = loader.load_text(folder)
            loaded = model.state_dict()
            for name, value in self.canonical.items():
                self.assertEqual(loaded[name].dtype, value.dtype, name)
                torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)
                mapped = checkpoint.tensor(name)
                np.testing.assert_array_equal(mapped.astype(np.float32), value.float().numpy())
                self.assertFalse(mapped.flags.writeable)
            self.assertEqual(model.lm_head.weight.data_ptr(), model.model.embed_tokens.weight.data_ptr())
            self.assertNotIn("v", checkpoint.layer(1))
            self.assertIn("v", checkpoint.layer(0))
            self.assertEqual(checkpoint.layer(0)["layer_scalar"].dtype, np.float32)
            for _, buffer in model.model.rotary_emb.named_buffers():
                self.assertEqual(buffer.dtype, torch.float32)
            tokens = (np.arange(34).reshape(2, 17) % 96 + 1).astype(np.int32)
            with torch.no_grad():
                reference = model(torch.tensor(tokens, dtype=torch.long), use_cache=False).logits.float().numpy()
                original = self.model(torch.tensor(tokens, dtype=torch.long), use_cache=False).logits.float().numpy()
            np.testing.assert_array_equal(reference, original)
            hidden = jnp.asarray(checkpoint.embeddings(tokens))
            for index in range(2):
                fn = adapter.build_layer(checkpoint.config, 17, batch_size=2, layer_index=index)
                hidden = fn(hidden, {k: jnp.asarray(v) for k, v in checkpoint.layer(index).items()})
            actual = np.asarray(adapter.head_logits(hidden, jnp.asarray(checkpoint.tensor("model.norm.weight")),
                                jnp.asarray(checkpoint.head()), checkpoint.config)).astype(np.float32)
            self.assertTrue(np.isfinite(actual).all())
            self.assertLessEqual(np.linalg.norm(reference-actual)/np.linalg.norm(reference), .03)

    def test_optional_tied_head_and_standalone_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            state = {k: v.clone() for k, v in self.canonical.items() if k != "lm_head.weight"}
            checkpoint = Checkpoint(self.write_checkpoint(folder, state))
            model = loader.load_text(folder)
            np.testing.assert_array_equal(checkpoint.head().astype(np.float32), model.lm_head.weight.detach().float().numpy().T)

    def test_missing_scalar_global_v_wrong_shapes_dtypes_and_alias(self):
        scalar = "model.language_model.layers.0.layer_scalar"
        mutations = [
            lambda s: s.pop(scalar),
            lambda s: s.update({"model.language_model.layers.1.self_attn.v_proj.weight": torch.zeros(32, 32, dtype=torch.bfloat16)}),
            lambda s: s.update({scalar: torch.ones(2, dtype=torch.float32)}),
            lambda s: s.update({"model.language_model.norm.weight": torch.ones(32, dtype=torch.float32)}),
            lambda s: s.update({"lm_head.weight": torch.zeros_like(s["lm_head.weight"])}),
            lambda s: s.update({"model.layers.0.layer_scalar": s[scalar].clone()}),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                state = self.shard_state()
                mutate(state)
                manifest = self.write_checkpoint(folder, state)
                with self.assertRaises(ValueError):
                    Checkpoint(manifest)
                with self.assertRaises(ValueError):
                    loader.load_text(folder)

    def test_manifest_hash_and_path_integrity(self):
        for change in ("hash", "config", "path"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                manifest = self.write_checkpoint(folder, self.shard_state())
                value = json.loads(manifest.read_text())
                if change == "hash":
                    value["files"][1]["sha256"] = "0" * 64
                elif change == "config":
                    value["config"]["text_config"]["hidden_size"] = 64
                else:
                    value["files"][1]["path"] = "../escape.safetensors"
                manifest.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    Checkpoint(manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
