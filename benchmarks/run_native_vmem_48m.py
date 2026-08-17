"""Run the native VMEM sensitivity screen under the max48 scoped-VMEM flag."""

import os

os.environ["CONTROL_SCOPED_VMEM_KIB"] = "49152"

import benchmark_native_vmem_sensitivity as screen

screen.main()
