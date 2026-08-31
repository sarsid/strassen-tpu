"""Streamed all-layer product-aware inference gate for the 8b study point."""

import os

os.environ["QWEN3_MODEL"] = "8b"
os.environ["QWEN3_STREAM_PRODUCT_AWARE"] = "1"
os.environ["QWEN3_PRODUCT_TILE"] = "2048,2048,512"
os.environ["QWEN3_CUBIC_TILE"] = "2048,2048,512"
os.environ["QWEN3_OUTPUT_SUFFIX"] = ""

import benchmark_qwen3_32b_streamed_inference as benchmark


if __name__ == "__main__":
    benchmark.main()
