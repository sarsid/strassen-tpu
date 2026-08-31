"""Native layer timing at the 48 MiB scoped-VMEM ceiling (max48 default)."""

import os

os.environ["STRASSEN_VMEM_PROFILE"] = "max48"
os.environ["LAYER_VMEM_MODE"] = "48m"

import benchmark_native_layer_vmem as screen

screen.main()
