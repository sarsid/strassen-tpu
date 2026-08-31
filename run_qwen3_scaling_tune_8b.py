"""Pin the 8b study point and run the fixed-budget tile tuner."""

import os

os.environ["QWEN3_MODEL"] = "8b"

import benchmark_qwen3_scaling_tile_tune as tune


if __name__ == "__main__":
    tune.main()
