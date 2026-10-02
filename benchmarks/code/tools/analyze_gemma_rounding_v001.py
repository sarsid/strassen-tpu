"""Replay captured real Gemma RoPE inputs locally; no TPU fix/performance claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from strassen_mm import model_gemma_v002 as gm


def measure(reference, actual):
    reference = np.asarray(reference, np.float64)
    actual = np.asarray(actual, np.float64)
    delta = actual - reference
    return dict(relative_l2=float(np.linalg.norm(delta.ravel()) / np.linalg.norm(reference.ravel())),
                max_abs=float(np.max(np.abs(delta))), unequal_elements=int(np.count_nonzero(delta)),
                elements=int(reference.size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    args = parser.parse_args()
    out = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
    out.mkdir()
    arrays = np.load(args.capture, allow_pickle=False)
    rows = []
    for layer in (0, 5, 47):
        cosine = arrays[f'rope_{layer:03}_cos_input'][0, :, None, :]
        sine = arrays[f'rope_{layer:03}_sin_input'][0, :, None, :]
        theta, factor = (10000., 1.) if layer == 0 else (1000000., 8.)
        for kind in ('q', 'k'):
            x = arrays[f'rope_{layer:03}_{kind}_input'][0].transpose(1, 0, 2)
            ref = arrays[f'rope_{layer:03}_{kind}_output'][0].transpose(1, 0, 2)
            half = np.concatenate((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), -1)
            def bf(v):
                return v.astype(ml_dtypes.bfloat16).astype(np.float32)
            left, right = x * cosine, half * sine
            row = dict(layer=layer, kind=kind,
                numpy_separate_bf16_products=measure(ref, bf(bf(left) + bf(right))),
                numpy_single_final_bf16_round=measure(ref, bf(left + right)))
            xj, cj, sj = (jnp.asarray(v, jnp.bfloat16) for v in (x, cosine, sine))
            def simple(a, c, s):
                return ((a*c).astype(a.dtype) + (gm.rotate_half(a)*s).astype(a.dtype)).astype(a.dtype)
            def barrier(a, c, s):
                left = jax.lax.optimization_barrier((a*c).astype(a.dtype))
                right = jax.lax.optimization_barrier((gm.rotate_half(a)*s).astype(a.dtype))
                return (left+right).astype(a.dtype)
            def explicit_round(a, c, s):
                def rounded(v):
                    return jax.lax.reduce_precision(v, exponent_bits=8, mantissa_bits=7)
                left = rounded(a.astype(jnp.float32)*c.astype(jnp.float32))
                right = rounded(gm.rotate_half(a).astype(jnp.float32)*s.astype(jnp.float32))
                return (left+right).astype(a.dtype)
            for name, fn in [('captured_trig', simple), ('barrier', barrier), ('explicit_round', explicit_round)]:
                for compiled in (False, True):
                    with jax.disable_jit(not compiled):
                        result = jax.jit(fn)(xj, cj, sj)
                        row[f'cpu_{name}_{"jit" if compiled else "eager"}'] = measure(ref, result)
            for compiled in (False, True):
                with jax.disable_jit(not compiled):
                    result = jax.jit(lambda v: gm.rope(v, theta, factor))(xj)
                    row[f'cpu_original_{"jit" if compiled else "eager"}'] = measure(ref, result)
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = dict(scope='CPU replay of actual 12B activations; candidate boundaries are not yet TPU-qualified',
                  capture=str(args.capture), capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
                  jax_version=jax.__version__, backend=jax.default_backend(), rows=rows)
    (out / 'rounding.json').write_text(json.dumps(result, indent=2)+'\n')
    # Establish the original reference arithmetic independently of the JAX compiler.
    assert all(row['numpy_separate_bf16_products']['unequal_elements'] == 0 for row in rows)
    assert all(row['numpy_single_final_bf16_round']['unequal_elements'] > 0 for row in rows)


if __name__ == '__main__':
    main()
