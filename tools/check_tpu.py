"""Confirm a single TPU AND that Pallas actually lowers on this image.

Sentinels are assembled at runtime rather than written as literals: a Python
traceback echoes the source lines it failed on, so a driver grepping for a
literal sentinel matches the printed source of the very program that failed
and concludes success.

The device check alone is not enough. Colab's image drifts, and a JAX whose
Mosaic serialization is newer than the VM's libtpu fails only at the first
Pallas compile -- after a session, an upload and a benchmark's first arm have
already been spent. This fails in seconds instead.
"""
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent if HERE.name == "tools" else HERE
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import jax, jax.numpy as jnp

import mosaic_compat


compat_enabled = mosaic_compat.enable_v7_compat()

devices = jax.devices()
print("DEVICES", devices)
print("JAX", jax.__version__)
try:
    import jaxlib
    print("JAXLIB", jaxlib.__version__)
except Exception as e:
    print("JAXLIB unknown:", e)
try:
    from importlib import metadata
    print("LIBTPU", metadata.version("libtpu"))
except Exception:
    try:
        from importlib import metadata
        print("LIBTPU", metadata.version("libtpu-nightly"))
    except Exception as e:
        print("LIBTPU unknown:", e)

backend = jax.lib.xla_bridge.get_backend()
print("PLATFORM_VERSION", getattr(backend, "platform_version", "unknown"))
print("MOSAIC_COMPAT", json.dumps(
    mosaic_compat.compatibility_info(compat_enabled), sort_keys=True))

assert jax.default_backend() == "tpu" and len(devices) == 1, f"not single-TPU: {devices}"

# Smallest possible Pallas kernel: does Mosaic deserialize through the same
# compatibility path used by the real kernels?
from jax.experimental import pallas as pl
def _k(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 2.0
x = jnp.ones((8, 128), jnp.float32)
try:
    out = pl.pallas_call(_k, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32))(x)
    jax.block_until_ready(out)
    print("PALLAS" + "_" + "OK")
except Exception as error:
    print("PALLAS" + "_" + "FAIL", type(error).__name__, str(error)[:200])
    raise SystemExit(3)
print("TPU" + "_" + "OK")
