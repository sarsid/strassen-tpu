"""Narrow Mosaic serialization compatibility for the measured Colab image.

JAX 0.7.2 normally emits Mosaic TPU serialization version 8.  The Colab v5e
backend used by this project accepts at most version 7.  JAX already supports
downgrading a lowered module, but its Cloud-TPU age check does not identify
this Colab image.  Apply the same version-7 target explicitly for the two
libtpu 0.0.21 patch releases observed in our sessions.

This deliberately uses a private JAX hook and is therefore guarded by exact
JAX and libtpu versions.  Unknown environments must fail visibly instead of
silently inheriting a compatibility override that has not been qualified.
"""

from __future__ import annotations

from importlib import metadata

import jax


SUPPORTED_JAX = "0.7.2"
SUPPORTED_LIBTPU = frozenset(("0.0.21", "0.0.21.1"))
TARGET_IR_VERSION = 7


def libtpu_version() -> str | None:
    try:
        return metadata.version("libtpu")
    except metadata.PackageNotFoundError:
        return None


def enable_v7_compat() -> bool:
    """Force Mosaic IR v7 on the two qualified Colab runtime versions."""
    version = libtpu_version()
    if jax.__version__ != SUPPORTED_JAX or version not in SUPPORTED_LIBTPU:
        return False

    from jax._src import tpu_custom_call  # pylint: disable=g-import-not-at-top

    original = getattr(
        tpu_custom_call,
        "_strassen_original_get_ir_version",
        tpu_custom_call.get_ir_version,
    )
    tpu_custom_call._strassen_original_get_ir_version = original

    def get_ir_version_v7(ctx):
        version_from_jax = original(ctx)
        return (
            TARGET_IR_VERSION
            if version_from_jax is None
            else min(version_from_jax, TARGET_IR_VERSION)
        )

    tpu_custom_call.get_ir_version = get_ir_version_v7
    return True


def compatibility_info(enabled: bool) -> dict[str, object]:
    return {
        "jax": jax.__version__,
        "libtpu": libtpu_version(),
        "supported_jax": SUPPORTED_JAX,
        "supported_libtpu": sorted(SUPPORTED_LIBTPU),
        "override_enabled": enabled,
        "target_ir_version": TARGET_IR_VERSION if enabled else None,
    }
