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
    if model not in ("8b", "14b", "32b"):
        parser.error("set --model or provide model in job.json")
    if stage not in STAGES:
        parser.error("set --stage or provide stage in job.json")

    os.environ["QWEN3_MODEL"] = model
    os.environ["QWEN3_PRODUCT_TILE"] = "2048,2048,512"
    os.environ["QWEN3_CUBIC_TILE"] = "2048,2048,512"
    os.environ["QWEN3_OUTPUT_SUFFIX"] = ""
    os.environ["QWEN3_STREAM_PRODUCT_AWARE"] = "1"
    os.environ["STRASSEN_OUTPUT_DIR"] = output_dir

    if args.dry_run:
        print(json.dumps({
            "model": model,
            "stage": stage,
            "module": STAGES[stage],
            "output_dir": output_dir,
            "tile": [2048, 2048, 512],
        }, sort_keys=True))
        return

    module = importlib.import_module(STAGES[stage])
    module.main()


if __name__ == "__main__":
    main()
