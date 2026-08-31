"""Run the native VMEM sensitivity screen with no scoped-VMEM flag."""

import os

os.environ.pop("CONTROL_SCOPED_VMEM_KIB", None)

import benchmark_native_vmem_sensitivity as screen

screen.main()
