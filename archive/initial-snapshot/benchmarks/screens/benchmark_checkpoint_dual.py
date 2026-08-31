"""Real-checkpoint screen of the dual seven-product classical formula."""

from pathlib import Path

import benchmark_checkpoint_permutations as benchmark


benchmark.OUTPUT = Path("/content/results/strassen_checkpoint_dual.jsonl")
benchmark.FORMULA_VARIANT = "dual"


if __name__ == "__main__":
    benchmark.main()
