"""Run the 34 MiB spatial scoped-VMEM profile in a fresh TPU kernel."""

import os
import shlex


args = [
    arg for arg in shlex.split(os.environ.get("LIBTPU_INIT_ARGS", ""))
    if not arg.startswith("--xla_tpu_scoped_vmem_limit_kib=")
]
args.append("--xla_tpu_scoped_vmem_limit_kib=34816")
os.environ["LIBTPU_INIT_ARGS"] = " ".join(args)
os.environ["STRASSEN_VMEM_PROFILE"] = "spatial34"

import benchmark_vmem_profile_common as runner  # noqa: E402


runner.main()
