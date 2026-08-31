"""Terminal-layer follow-up to the uniform checkpoint schedule frontier."""

from pathlib import Path

import benchmark_checkpoint_schedules as schedules


schedules.OUTPUT = Path(
    "/content/results/strassen_checkpoint_terminal.jsonl"
)
schedules.SCHEDULES = {
    "first_1": {"both": frozenset({0}), "up": frozenset()},
    "last_1": {"both": frozenset({31}), "up": frozenset()},
    "last_2": {"both": frozenset({30, 31}), "up": frozenset()},
    "last_4": {"both": frozenset(range(28, 32)), "up": frozenset()},
    "last_8": {"both": frozenset(range(24, 32)), "up": frozenset()},
}


if __name__ == "__main__":
    schedules.main()
