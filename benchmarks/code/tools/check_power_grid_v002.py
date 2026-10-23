"""Active grid checks including revised chart layout and descriptive exports."""
from pathlib import Path
import sys
import unittest


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root / "tests")]
    modules = (
        "test_power_midpoint_grid_v001",
        "test_power_grid_v001",
        "test_plot_power_grid_v002",
        "test_power_grid_audit_v002",
        "test_summarize_power_grid_v001",
    )
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
