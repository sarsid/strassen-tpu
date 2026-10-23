"""Configure the adopted exact profile in a fresh process before JAX imports."""
import os
import sys
from pathlib import Path
import parent_stage_v001 as original

PARENT = original.PARENT
install_transport = original.install_transport


def configure(output):
    if any(name in sys.modules for name in ('benchmark_qwen3_32b_layer',
           'benchmark_qwen3_32b_streamed_inference', 'strassen_pallas')):
        raise RuntimeError('Configure the exact parent profile in a fresh process')
    # Remove inherited experimental panel overrides and explicitly select the
    # same kernel/Native memory settings as the reproduced parent workload.
    os.environ.update(QWEN3_PRODUCT_PANELS='', QWEN3_STRASSEN_POLICY='up_only',
                      STRASSEN_VMEM_PROFILE='max48', STRASSEN_MANAGE_CEILING='0',
                      LIBTPU_INIT_ARGS='--xla_tpu_use_enhanced_launch_barrier=true')
    return original.configure(Path(output))
