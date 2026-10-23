"""CPU Pallas-interpreter checks; these are not TPU performance evidence."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from strassen_mm.kernels_v001 import VARIANTS, make_matmul


class KernelContracts(unittest.TestCase):
    def test_invalid_contracts(self):
        for algorithm, shape, tile, variant in (
            ("unknown", (16, 256, 256), (16, 256, 256), "plain"),
            ("strassen", (16, 256, 256), (8, 256, 256), "plain"),
            ("strassen", (16, 256, 256), (16, 128, 256), "plain"),
            ("cubic_full", (16, 256, 256), (16, 256, 256), "interleaved"),
            ("native", (0, 256, 256), None, "plain"),
            ("native", (16, 256, 256), None, "not_a_variant"),
        ):
            with self.subTest(algorithm=algorithm, shape=shape, tile=tile, variant=variant):
                with self.assertRaises(ValueError):
                    make_matmul(algorithm, shape, tile, variant=variant, interpret=True)

    def test_input_dtype_conversion_is_not_hidden(self):
        fn = make_matmul("native", (3, 5, 7))
        with self.assertRaisesRegex(ValueError, "BF16"):
            fn(jnp.ones((3, 5), jnp.float32), jnp.ones((5, 7), jnp.float32))

    def test_padding_is_explicit_and_reversible(self):
        fn = make_matmul("strassen", (17, 257, 129), (16, 256, 256), interpret=True)
        self.assertEqual(fn.metadata["padded_shape_mkn"], [32, 512, 256])
        a = jnp.ones((17, 257), jnp.bfloat16)
        b = jnp.ones((257, 129), jnp.bfloat16)
        ap, bp = fn.prepare(a, b)
        self.assertEqual(ap.shape, (32, 512))
        self.assertEqual(bp.shape, (512, 256))
        np.testing.assert_array_equal(np.asarray(ap[:17, :257]), np.asarray(a))
        self.assertEqual(float(jnp.sum(ap[17:, :])), 0.0)
        self.assertEqual(float(jnp.sum(ap[:, 257:])), 0.0)
        self.assertEqual(float(jnp.sum(bp[257:, :])), 0.0)
        self.assertEqual(float(jnp.sum(bp[:, 129:])), 0.0)
        self.assertEqual(fn.finish(jnp.ones((32, 256), jnp.float32)).shape, (17, 129))

    def test_native_uses_original_dimensions(self):
        fn = make_matmul("native", (3, 5, 7), (256, 256, 256), interpret=True)
        self.assertEqual(fn.metadata["padded_shape_mkn"], [3, 5, 7])
        self.assertIsNone(fn.metadata["tile_bm_bn_bk"])
        a = jnp.arange(15).reshape(3, 5).astype(jnp.bfloat16)
        b = jnp.arange(35).reshape(5, 7).astype(jnp.bfloat16)
        expected = np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32)
        got = jax.jit(fn)(a, b)
        self.assertEqual(got.dtype, jnp.float32)
        np.testing.assert_array_equal(np.asarray(got), expected)


class KernelAlgebra(unittest.TestCase):
    def test_exact_integer_products_cover_quadrants_and_k_accumulation(self):
        # Integer values keep BF16 pre-adds and FP32 products exact, separating
        # indexing/algebra bugs from Strassen's expected numerical error. The
        # second shape crosses every grid axis and requires partial-tile pad.
        rng = np.random.default_rng(187)
        cases = [(16, 256, 256), (17, 259, 257)]
        arms = [("cubic_full", "plain")]
        arms += [(algorithm, variant) for algorithm in ("cubic_quadrant", "strassen")
                 for variant in VARIANTS]
        for shape in cases:
            m, k, n = shape
            a_np = rng.integers(-2, 3, size=(m, k)).astype(np.float32)
            b_np = rng.integers(-2, 3, size=(k, n)).astype(np.float32)
            a, b = jnp.asarray(a_np, jnp.bfloat16), jnp.asarray(b_np, jnp.bfloat16)
            expected = a_np @ b_np
            for algorithm, variant in arms:
                with self.subTest(shape=shape, algorithm=algorithm, variant=variant):
                    fn = make_matmul(algorithm, shape, (16, 256, 256),
                                     variant=variant, interpret=True)
                    got = jax.jit(fn)(a, b)
                    self.assertEqual(got.dtype, jnp.float32)
                    np.testing.assert_array_equal(np.asarray(got), expected)

    def test_random_bf16_strassen_has_bounded_nonzero_rounding_error(self):
        # N4 owns stress/error studies. This check ensures that our precision
        # contract really includes BF16 input-combination rounding.
        rng = np.random.default_rng(23)
        a = jnp.asarray(rng.normal(size=(16, 256)), jnp.bfloat16)
        b = jnp.asarray(rng.normal(size=(256, 256)), jnp.bfloat16)
        ref = np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                fn = make_matmul("strassen", (16, 256, 256), (16, 256, 256),
                                 variant=variant, interpret=True)
                out = np.asarray(jax.jit(fn)(a, b), dtype=np.float64)
                rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
                self.assertGreater(rel, 0.0)
                self.assertLess(rel, 0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
