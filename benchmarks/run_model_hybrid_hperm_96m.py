"""Hybrid depth gate with the global hidden permutation + exact up panel."""

import os
from pathlib import Path

os.environ["STRASSEN_VMEM_PROFILE"] = "max48"
os.environ["HYBRID_HPERM_FILE"] = "/content/hybrid_global_hperm.npy"
existing = os.environ.get("LIBTPU_INIT_ARGS", "").strip()
flag = "--xla_tpu_scoped_vmem_limit_kib=98304"
if "--xla_tpu_scoped_vmem_limit_kib=" not in existing:
    os.environ["LIBTPU_INIT_ARGS"] = f"{existing} {flag}".strip()

import benchmark_checkpoint_model_hybrid as gate

gate.OUTPUT = Path(
    "/content/results/strassen_checkpoint_model_hybrid_hperm.jsonl"
)
gate.main()
