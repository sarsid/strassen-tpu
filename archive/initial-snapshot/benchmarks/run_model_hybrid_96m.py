"""Run the hybrid full-checkpoint gate (calibration text) at 96 MiB scoped VMEM."""

import os

os.environ["STRASSEN_VMEM_PROFILE"] = "max48"
existing = os.environ.get("LIBTPU_INIT_ARGS", "").strip()
flag = "--xla_tpu_scoped_vmem_limit_kib=98304"
if "--xla_tpu_scoped_vmem_limit_kib=" not in existing:
    os.environ["LIBTPU_INIT_ARGS"] = f"{existing} {flag}".strip()

import benchmark_checkpoint_model_hybrid as gate

gate.main()
