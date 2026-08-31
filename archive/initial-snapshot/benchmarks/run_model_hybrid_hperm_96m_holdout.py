"""Frozen holdout of the global-hperm hybrid depth gate (disjoint corpus)."""

import os
from pathlib import Path

os.environ["STRASSEN_VMEM_PROFILE"] = "max48"
os.environ["HYBRID_HPERM_FILE"] = "/content/hybrid_global_hperm.npy"
os.environ["HYBRID_IPERM_FILE"] = "/content/results/hybrid_iperms.npy"
os.environ["HYBRID_BUDGET_FILE"] = "/content/results/hybrid_budgets.npy"
existing = os.environ.get("LIBTPU_INIT_ARGS", "").strip()
flag = "--xla_tpu_scoped_vmem_limit_kib=98304"
if "--xla_tpu_scoped_vmem_limit_kib=" not in existing:
    os.environ["LIBTPU_INIT_ARGS"] = f"{existing} {flag}".strip()

import benchmark_checkpoint_layer as layer_bench
import benchmark_checkpoint_model_hybrid as gate

gate.OUTPUT = Path(
    "/content/results/strassen_checkpoint_model_hybrid_hperm_holdout.jsonl"
)
layer_bench.TEXTS = (
    "Rain moved across the valley in silver bands while the old stone bridge "
    "held its place above the river and the road disappeared into mist.",
    "Economic measurements summarize many individual choices, so a useful "
    "forecast states its assumptions and gives a range rather than one number.",
    "The telescope gathered faint light for several hours. After calibration, "
    "the image revealed a distant structure that no single exposure could show.",
    "A legal rule can be simple to state yet difficult to apply because facts, "
    "precedent, language, and the purpose of the rule must be considered together.",
    "During rehearsal the actors changed the pace of the scene. A longer pause "
    "made the final line quieter, clearer, and more surprising to the audience.",
    "An algorithm is useful only in a setting: input sizes, hardware, precision, "
    "memory traffic, and surrounding operations decide whether its theory pays.",
    "The recipe began with ordinary ingredients, but temperature and timing "
    "changed their texture. The cook wrote down each adjustment for the next trial.",
    "Maps select details for a purpose. A subway diagram, a weather chart, and a "
    "topographic survey describe the same region through very different questions.",
)

gate.main()
