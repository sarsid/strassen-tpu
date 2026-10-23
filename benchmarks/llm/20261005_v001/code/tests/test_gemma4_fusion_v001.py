"""Bounded Gemma4 BF16 fusion semantics; CPU interpretation, no speed claims.

Sparse integer products isolate layout/reduction/epilogue correctness from
Strassen's BF16 pre-add approximation. Independent epilogues use official Torch
Gemma4 RMSNorm and rotary operations. Device and dense-input checks remain due.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4 import modeling_gemma4 as official
from strassen_mm.fusion_v004 import Epilogue, reference
from strassen_mm.kernels_fused_v005 import make_projection


def choice(implementation, depth=0, mode="outputs", bn=512):
    return dict(architecture="v6e", implementation=implementation, depth=depth,
                accumulator=mode, tile=None if implementation == "native" else [32, bn, 512],
                buffers=2, output_dtype="bfloat16", vmem_limit_bytes=112*1024**2,
                compiler_options={})


def tensor(value):
    return torch.tensor(np.asarray(value, np.float32), dtype=torch.bfloat16)


def inputs(spec, seed):
    rng = np.random.default_rng(seed)
    m, k, n = 17, 513, (3*spec.head_dim if spec.rotary else 258)
    a = rng.integers(-1, 2, (m, k)).astype(np.float32)
    a[:, np.arange(k) % 64 != 0] = 0
    b = rng.integers(-1, 2, (k, n)).astype(np.float32)
    projected = torch.tensor(a.astype(np.float64) @ b.astype(np.float64), dtype=torch.float32)
    scale = jnp.asarray(np.linspace(.5, 1.5, spec.head_dim if spec.rotary else n), jnp.bfloat16)
    aux = {"scale": scale}
    if spec.rotary:
        config = Gemma4TextConfig(head_dim=256, global_head_dim=512,
            num_hidden_layers=2, layer_types=["sliding_attention", "full_attention"])
        kind = "sliding_attention" if spec.head_dim == 256 else "full_attention"
        cosine, sine = official.Gemma4TextRotaryEmbedding(config)(
            torch.zeros(1, m, 1, spec.head_dim, dtype=torch.bfloat16), torch.arange(m)[None], kind)
        aux.update(cos=jnp.asarray(cosine.float().numpy()[0, :, :spec.head_dim//2], jnp.bfloat16),
                   sin=jnp.asarray(sine.float().numpy()[0, :, :spec.head_dim//2], jnp.bfloat16))
    else:
        aux["residual"] = jnp.asarray(rng.normal(size=(m, n)), jnp.bfloat16)
    return (m, k, n), jnp.asarray(a, jnp.bfloat16), jnp.asarray(b, jnp.bfloat16), projected, aux


def oracle(projected, aux, spec):
    x = projected.to(torch.bfloat16) if spec.rounding == "model" else projected
    norm = official.Gemma4RMSNorm(spec.head_dim if spec.rotary else x.shape[-1], eps=spec.eps).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(tensor(aux["scale"]))
        if spec.rotary:
            value = norm(x.reshape(1, x.shape[0], -1, spec.head_dim))
            cosine = tensor(aux["cos"])
            sine = tensor(aux["sin"])
            cosine = torch.cat((cosine, cosine), -1)[None]
            sine = torch.cat((sine, sine), -1)[None]
            if spec.rounding == "accumulator":
                cosine, sine = cosine.float(), sine.float()
            value = official.apply_rotary_pos_emb(value, cosine, sine, unsqueeze_dim=2).reshape(x.shape)
        else:
            value = norm(x) + tensor(aux["residual"])
    return value.to(torch.bfloat16).float().numpy()


class Gemma4Fusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest() == "6a86e03348df5ec104703e7161de9a911137cba500a0be0f133e2850ef0bf935"
        torch.set_num_threads(1)
        cls.records = []

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("STRASSEN_EXECUTION_DIR"):
            out = Path(os.environ["STRASSEN_EXECUTION_DIR"]) / "artifacts"
            out.mkdir(exist_ok=True)
            (out / "gemma4-fusion-checks.json").write_text(json.dumps(cls.records, indent=2) + "\n")

    def compare(self, label, actual, expected):
        value = np.asarray(actual, np.float32)
        self.assertTrue(np.isfinite(value).all())
        relative = float(np.linalg.norm(value-expected) / max(np.linalg.norm(expected), 1e-30))
        self.records.append(dict(label=label, relative_l2=relative, max_abs=float(abs(value-expected).max()),
                                 atol=.002, rtol=.002, scope="CPU interpretation; sparse exact products"))
        np.testing.assert_allclose(value, expected, atol=.002, rtol=.002, err_msg=label)

    def specs(self):
        return [Epilogue("norm_residual_add", norm="gemma4"),
                Epilogue("qk_norm_rope", norm="gemma4", head_dim=256),
                Epilogue("qk_norm_rope", norm="gemma4", head_dim=512)]

    def test_reference_against_official_epilogues_both_rounding_contracts(self):
        for index, base in enumerate(self.specs()):
            for rounding in ("model", "accumulator"):
                spec = replace(base, rounding=rounding)
                shape, a, b, projected, aux = inputs(spec, 1004 + index)
                actual = jax.jit(lambda x, operands: reference(x, spec, jnp.bfloat16, **operands))(jnp.asarray(projected.numpy()), aux)
                self.compare(f"reference-{spec.kind}-{spec.head_dim}-{rounding}", actual, oracle(projected, aux, spec))

    def test_sparse_products_all_recursive_schedules(self):
        choices = [("native", 0, "outputs"), ("cubic", 0, "outputs"), ("cubic_full", 0, "outputs")]
        choices += [("current", depth, mode) for depth in (1, 2) for mode in ("products", "outputs")]
        for index, spec in enumerate(self.specs()):
            shape, a, b, projected, aux = inputs(spec, 1004 + index)
            expected = oracle(projected, aux, spec)
            for impl, depth, mode in choices:
                with self.subTest(epilogue=spec.kind, head_dim=spec.head_dim, impl=impl, depth=depth, mode=mode):
                    fn = make_projection(choice(impl, depth, mode, 2*spec.head_dim if spec.rotary else 512), shape, spec, interpret=True)
                    actual = jax.jit(fn.prepared)(a, fn.prepare_weights(b), **aux)
                    self.compare(f"{impl}-{depth}-{mode}-{spec.kind}-{spec.head_dim}", actual, expected)
                    self.assertEqual(actual.dtype, jnp.bfloat16)
                    self.assertEqual(fn.metadata["epilogue"]["norm"], "gemma4")
                    self.assertEqual(fn.metadata["kernel_version"], "kernels_fused_v005")
                jax.clear_caches()

    def test_zero_direct_scale_cannot_act_like_gemma3_offset(self):
        projected = jnp.ones((3, 258), jnp.float32)
        spec = Epilogue("norm_residual_add", norm="gemma4")
        residual = jnp.full((3, 258), .25, jnp.bfloat16)
        zero = jnp.zeros((258,), jnp.bfloat16)
        actual = reference(projected, spec, jnp.bfloat16, scale=zero, residual=residual)
        np.testing.assert_array_equal(actual, residual)
        legacy = reference(projected, replace(spec, norm="gemma"), jnp.bfloat16, scale=zero, residual=residual)
        self.assertGreater(float(np.max(abs(np.asarray(legacy, np.float32)-np.asarray(actual, np.float32)))), .5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
