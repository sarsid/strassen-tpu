"""Native layer timing at the 96 MiB scoped-VMEM ceiling."""

import os

os.environ["STRASSEN_VMEM_PROFILE"] = "max48"
os.environ["LAYER_VMEM_MODE"] = "96m"
existing = os.environ.get("LIBTPU_INIT_ARGS", "").strip()
flag = "--xla_tpu_scoped_vmem_limit_kib=98304"
if "--xla_tpu_scoped_vmem_limit_kib=" not in existing:
    os.environ["LIBTPU_INIT_ARGS"] = f"{existing} {flag}".strip()

import benchmark_native_layer_vmem as screen

screen.main()
