"""Numerical equivalence, chunk boundaries, and unchanged benchmark contracts."""
import ast
import math
from pathlib import Path
import sys
import unittest
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from gemma4_error_metrics_v001 import chunked_error, CHUNK_ELEMENTS


def old_error(r, v):
    r = r.astype(jnp.float32); v = v.astype(jnp.float32); d = v - r
    return dict(finite=bool(jnp.all(jnp.isfinite(v))),
                relative_l2=float(jnp.linalg.norm(d) / jnp.maximum(jnp.linalg.norm(r), 1e-30)),
                max_abs=float(jnp.max(jnp.abs(d))),
                maxnorm_relative=float(jnp.max(jnp.abs(d)) / jnp.maximum(jnp.max(jnp.abs(r)), 1e-30)))


class Checks(unittest.TestCase):
    def check_pair(self, r, v, chunk=257):
        actual = chunked_error(r, v, jax=jax, jnp=jnp, chunk_elements=chunk)
        expected = old_error(r, v)
        self.assertEqual(actual['finite'], expected['finite'])
        for key in ('relative_l2', 'max_abs', 'maxnorm_relative'):
            np.testing.assert_allclose(actual[key], expected[key], rtol=2e-6, atol=1e-12, equal_nan=True)

    def test_bf16_and_fp32_tail(self):
        rng = np.random.default_rng(7331)
        for dtype in (jnp.bfloat16, jnp.float32):
            r = jnp.asarray(rng.normal(size=(37, 113)), dtype)
            v = (r.astype(jnp.float32) + jnp.asarray(rng.normal(scale=.03, size=r.shape))).astype(dtype)
            self.check_pair(r, v)
            self.check_pair(r, v, 8192)

    def test_zero_and_tiny_denominator(self):
        for r, v in [(jnp.zeros(599), jnp.zeros(599)),
                     (jnp.zeros(599), jnp.ones(599)),
                     (jnp.full(599, 1e-15), jnp.full(599, 2e-15))]:
            self.check_pair(r, v)

    def test_nonfinite_and_tail_max(self):
        r = jnp.ones(599)
        for value in (math.nan, math.inf, -math.inf, 10.):
            self.check_pair(r, r.at[-1].set(value))

    def test_every_tile_bounded(self):
        sizes = []
        class RecordingJax:
            device_get = staticmethod(jax.device_get)
            @staticmethod
            def jit(fn):
                compiled = jax.jit(fn)
                def call(r, v):
                    sizes.append(r.size)
                    return compiled(r, v)
                return call
        r = jnp.ones(1003, dtype=jnp.bfloat16)
        result = chunked_error(r, r.at[-1].set(2), jax=RecordingJax, jnp=jnp, chunk_elements=128)
        self.assertEqual(sum(sizes), r.size)
        self.assertLessEqual(max(sizes), 128)
        self.assertEqual(result['max_abs'], 1.)
        self.assertEqual(CHUNK_ELEMENTS, 1048576)

    def test_invalid_input(self):
        for r, v, chunk in [(jnp.ones(2),jnp.ones(3),1),
                            (jnp.ones(0),jnp.ones(0),1),
                            (jnp.ones(2),jnp.ones(2),0)]:
            with self.assertRaises(ValueError):chunked_error(r,v,jax=jax,jnp=jnp,chunk_elements=chunk)

    def test_scientific_and_timing_functions_unchanged(self):
        def functions(version):
            module = ast.parse((ROOT / f'tools/gemma4_batch_stage_{version}.py').read_text())
            return {n.name:ast.dump(n, include_attributes=False) for n in module.body if isinstance(n, ast.FunctionDef)}
        before, after = functions('v003'), functions('v004')
        self.assertEqual(set(before), set(after))
        for name in before:
            if name not in ('error','main'):self.assertEqual(before[name],after[name],name)


if __name__ == '__main__': unittest.main()
