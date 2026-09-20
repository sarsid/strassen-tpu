"""N8 CPU interpreter equivalence across real packing and padding boundaries.

Small integer operands make the seven/eight/full product algebra exact. An
independent NumPy reference then applies the declared BF16 projection and
activation boundaries. These tests are correctness checks, not TPU timings.
N9 uses a real official Transformers reference rather than duplicated math.
"""
import unittest

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from strassen_mm.kernels_n8_v001 import make_epilogue


TILE = (16, 256, 256)
BF16 = ml_dtypes.bfloat16


def data(kind, seed):
    rng = np.random.default_rng(seed)
    shape = (17, 259, 514 if kind == "swiglu" else 257)
    m, k, n = shape
    a = rng.integers(-1, 2, (m, k)).astype(BF16)
    b = rng.integers(-1, 2, (k, n)).astype(BF16)
    r = rng.integers(-3, 4, (m, n)).astype(BF16) if kind == "residual" else None
    return shape, a, b, r


def independent_reference(a, b, residual, kind):
    projection = (a.astype(np.float32) @ b.astype(np.float32)).astype(BF16)
    if kind == "residual":
        return (projection.astype(np.float32) + residual.astype(np.float32)).astype(BF16)
    gate, up = np.split(projection, 2, axis=1)
    gate64 = gate.astype(np.float64)
    silu = (gate64 / (1.0 + np.exp(-gate64))).astype(BF16)
    return (silu.astype(np.float32) * up.astype(np.float32)).astype(BF16)


def arm_configurations():
    yield "native", dict(algorithm="native", fused=False, packed=False, variant="plain")
    for algorithm in ("cubic_full", "cubic_quadrant", "strassen"):
        variant = "plain" if algorithm == "cubic_full" else "interleaved"
        yield algorithm + "_unpacked", dict(algorithm=algorithm, variant=variant, fused=False)
        yield algorithm + "_packed", dict(algorithm=algorithm, variant=variant, fused=False, packed=True)
        yield algorithm + "_fused", dict(algorithm=algorithm, variant=variant, fused=True)
    yield "strassen_early", dict(algorithm="strassen", variant="interleaved", fused=True, early=True)


class N8Applications(unittest.TestCase):
    def check_equivalence(self, kind, seed):
        shape, a_np, b_np, r_np = data(kind, seed)
        expected = independent_reference(a_np, b_np, r_np, kind).astype(np.float32)
        a, b = jnp.asarray(a_np), jnp.asarray(b_np)
        args = (a, b) if r_np is None else (a, b, jnp.asarray(r_np))
        for name, parameters in arm_configurations():
            with self.subTest(kind=kind, arm=name):
                fn = make_epilogue(shape=shape, tile=TILE, kind=kind, interpret=True, **parameters)
                got = jax.jit(fn)(*args)
                self.assertEqual(got.dtype, jnp.bfloat16)
                self.assertEqual(got.shape, expected.shape)
                np.testing.assert_array_equal(np.asarray(got, dtype=np.float32), expected)

    def test_swiglu_packed_unpacked_fused_early_with_tails_and_multiple_k_panels(self):
        self.check_equivalence("swiglu", 318)

    def test_residual_quadrant_finalization_with_tails_and_multiple_k_panels(self):
        self.check_equivalence("residual", 319)

    def test_prepared_inputs_reproduce_full_call_and_accumulators_reset(self):
        for kind in ("swiglu", "residual"):
            with self.subTest(kind=kind):
                shape, a_np, b_np, r_np = data(kind, 320)
                fn = make_epilogue("strassen", shape, TILE, kind=kind, fused=True,
                                   early=True, variant="interleaved", interpret=True)
                args = (jnp.asarray(a_np), jnp.asarray(b_np))
                if r_np is not None:
                    args += (jnp.asarray(r_np),)
                operands = jax.jit(fn.prepare)(*args)
                full = jax.jit(fn)(*args)
                prepared = jax.jit(fn.prepared)(*operands)
                np.testing.assert_array_equal(np.asarray(prepared, dtype=np.float32),
                                              np.asarray(full, dtype=np.float32))
                self.assertEqual(operands[0].shape, (32, 512))
                self.assertEqual(operands[1].shape, (512, 768 if kind == "swiglu" else 512))
                self.assertFalse(np.asarray(operands[0][17:, :], dtype=np.float32).any())
                self.assertFalse(np.asarray(operands[0][:, 259:], dtype=np.float32).any())
                # Reuse the same compiled kernel with a distinct activation.
                # A missing first-K-panel reset would retain the prior result.
                reversed_a = (-a_np.astype(np.float32)).astype(BF16)
                next_args = (jnp.asarray(reversed_a), args[1]) + args[2:]
                next_operands = jax.jit(fn.prepare)(*next_args)
                next_result = jax.jit(fn.prepared)(*next_operands)
                expected = independent_reference(reversed_a, b_np, r_np, kind)
                np.testing.assert_array_equal(np.asarray(next_result, dtype=np.float32),
                                              expected.astype(np.float32))

    def test_unsupported_schedule_or_precision_cannot_be_mislabeled(self):
        for parameters in (
            dict(algorithm="strassen", fused=True, variant="plain"),
            dict(algorithm="cubic_full", fused=True, early=True),
            dict(algorithm="native", fused=True),
            dict(algorithm="unknown", fused=True),
        ):
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    make_epilogue(shape=(17, 259, 514), tile=TILE, kind="swiglu",
                                   interpret=True, **parameters)
        fn = make_epilogue("cubic_full", (17, 259, 257), TILE, kind="residual", fused=True, interpret=True)
        with self.assertRaisesRegex(ValueError, "BF16"):
            fn.prepare(jnp.ones((17, 259), jnp.bfloat16), jnp.ones((259, 257), jnp.bfloat16),
                       jnp.ones((17, 257), jnp.float32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
