"""Run the in-kernel exact-panel gate under a 96 MiB scoped-VMEM ceiling.

Sets the scoped-VMEM flag before anything imports JAX; strassen_pallas then
respects the existing option instead of installing its 48 MiB default.
"""

import os

os.environ["STRASSEN_VMEM_PROFILE"] = "max48"
existing = os.environ.get("LIBTPU_INIT_ARGS", "").strip()
flag = "--xla_tpu_scoped_vmem_limit_kib=98304"
if "--xla_tpu_scoped_vmem_limit_kib=" not in existing:
    os.environ["LIBTPU_INIT_ARGS"] = f"{existing} {flag}".strip()

import benchmark_checkpoint_panel_kernel as gate

gate.main()
