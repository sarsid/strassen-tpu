"""Registered synthetic gate/up qualification for the 32b study point.

Tiles pinned from the fixed-budget tuner (strassen_qwen3_32b_scaling_tiles).
"""

import os

os.environ["QWEN3_MODEL"] = "32b"
os.environ["QWEN3_PRODUCT_TILE"] = "2048,2048,512"
os.environ["QWEN3_CUBIC_TILE"] = "2048,2048,512"
os.environ["QWEN3_OUTPUT_SUFFIX"] = "_campaign"

import benchmark_qwen3_32b_product_aware_inference as benchmark


if __name__ == "__main__":
    benchmark.main()
