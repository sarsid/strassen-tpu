"""Run one stage of the promoted Qwen3 inference experiment."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = (
    HERE.parent.parent
    if HERE.name == "qwen3" and HERE.parent.name == "experiments"
    else HERE
)
for path in (REPOSITORY_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# The promoted configuration is platform specific.  Hardcoding the v5e tile
# meant the documented reproduction path could not reproduce the published
# v6e result: Trillium's winner uses the full 5120 contraction depth and a
# raised kernel budget, and routes o and down on their own tuned tiles.
PROFILES = {
    "v5e": {
        "product_tile": "2048,2048,512",
        "cubic_tile": "2048,2048,512",
        "qk_tile": "2048,1024,512",
        "site_tiles": "o:1024,2560,512;down:2048,1024,1024",
        "extended_sites": "o,down",
        "strassen_limit_mib": "47",
        "cubic_limit_mib": "48",
        "scoped_vmem_kib": "49152",
    },
    "v6e": {
        "product_tile": "2048,1024,5120",
        "cubic_tile": "2048,2048,512",
        "qk_tile": "2048,1024,5120",
        "site_tiles": "o:2048,512,8192;down:2048,2560,1024",
        "extended_sites": "o,down",
        "strassen_limit_mib": "120",
        "cubic_limit_mib": "124",
        # Deliberately the value XLA prefers: raising it alongside the kernel
        # budget costs native XLA 37% at the layer and 4% on a bare GEMM.
        "scoped_vmem_kib": "49152",
    },
}

STAGES = {
    "tune": "benchmark_qwen3_scaling_tile_tune",
    "synthetic": "benchmark_qwen3_32b_product_aware_inference",
    "layer": "benchmark_qwen3_32b_full_layer_product_inference",
    "streamed": "benchmark_qwen3_32b_streamed_inference",
    "quality": "benchmark_qwen3_streamed_natural_gate",
}


def default_output_dir() -> str:
    return "/content/runs" if Path("/content").is_dir() else "runs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("8b", "14b", "32b"))
    parser.add_argument("--stage", choices=tuple(STAGES))
    parser.add_argument("--device", choices=tuple(PROFILES), default=None,
                        help="platform profile; selects the promoted tiles")
    parser.add_argument("--tile", help="override the gate/up tile, bm,bn,bk")
    parser.add_argument("--fused-qk", action="store_true",
                        help="route q and k through the qk_norm_rope epilogue")
    parser.add_argument("--output-dir")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        config_path = Path("/content/job.json")
        if not config_path.is_file():
            config_path = Path("job.json")
    config = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    model = args.model or config.get("model")
    stage = args.stage or config.get("stage")
    output_dir = args.output_dir or config.get("output_dir") or default_output_dir()
    device = args.device or config.get("device") or "v5e"
    if device not in PROFILES:
        parser.error(f"unknown device {device!r}; choose {tuple(PROFILES)}")
    profile = dict(PROFILES[device])
    if args.tile or config.get("tile"):
        profile["product_tile"] = args.tile or config["tile"]
    fused_qk = bool(args.fused_qk or config.get("fused_qk"))
    if model not in ("8b", "14b", "32b"):
        parser.error("set --model or provide model in job.json")
    if stage not in STAGES:
        parser.error("set --stage or provide stage in job.json")

    os.environ["QWEN3_MODEL"] = model
    os.environ["QWEN3_PRODUCT_TILE"] = profile["product_tile"]
    os.environ["QWEN3_CUBIC_TILE"] = profile["cubic_tile"]
    os.environ["QWEN3_QK_TILE"] = profile["qk_tile"]
    os.environ["QWEN3_SITE_TILES"] = profile["site_tiles"]
    os.environ["QWEN3_EXTENDED_SITES"] = profile["extended_sites"]
    os.environ["QWEN3_STRASSEN_LIMIT_MIB"] = profile["strassen_limit_mib"]
    os.environ["QWEN3_CUBIC_LIMIT_MIB"] = profile["cubic_limit_mib"]
    os.environ["QWEN3_SCOPED_VMEM_KIB"] = profile["scoped_vmem_kib"]
    os.environ["QWEN3_FUSED_QK"] = "1" if fused_qk else "0"
    os.environ["QWEN3_OUTPUT_SUFFIX"] = ""
    os.environ["QWEN3_STREAM_PRODUCT_AWARE"] = "1"
    os.environ["STRASSEN_OUTPUT_DIR"] = output_dir

    if args.dry_run:
        print(json.dumps({
            "model": model,
            "stage": stage,
            "module": STAGES[stage],
            "output_dir": output_dir,
            "device": device,
            "fused_qk": fused_qk,
            **{k: v for k, v in profile.items()},
        }, sort_keys=True))
        return

    module = importlib.import_module(STAGES[stage])
    module.main()


if __name__ == "__main__":
    main()
