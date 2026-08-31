"""Natural-corpus streamed quality gate for the 32b study point."""

import os

os.environ["QWEN3_MODEL"] = "32b"
os.environ["QWEN3_PRODUCT_TILE"] = "2048,2048,512"
os.environ["QWEN3_CUBIC_TILE"] = "2048,2048,512"

import benchmark_qwen3_streamed_natural_gate as benchmark


if __name__ == "__main__":
    benchmark.main()
