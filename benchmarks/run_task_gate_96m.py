"""Run the predeclared task-level gate at the 96 MiB scoped-VMEM ceiling."""

import os

os.environ["STRASSEN_VMEM_PROFILE"] = "max48"
existing = os.environ.get("LIBTPU_INIT_ARGS", "").strip()
flag = "--xla_tpu_scoped_vmem_limit_kib=98304"
if "--xla_tpu_scoped_vmem_limit_kib=" not in existing:
    os.environ["LIBTPU_INIT_ARGS"] = f"{existing} {flag}".strip()

import benchmark_checkpoint_task_gate as gate

gate.main()
