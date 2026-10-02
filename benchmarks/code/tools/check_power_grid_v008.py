"""Active grid checks including the final 3D camera and unchanged evidence contracts."""
from pathlib import Path
import sys
import unittest


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "src"), str(root / "tests")]
    modules = (
        "test_power_midpoint_grid_v001",
        "test_power_grid_v001",
        "test_plot_power_grid_v006",
        "test_plot_power_grid_v005.TickPresentation",
        "test_plot_power_grid_v004.NearsetScopeRegression",
        "test_plot_power_grid_v003.FrontmatterPresentation",
        "test_plot_power_grid_v002.PowerGridLayoutTests",
        "test_power_grid_audit_v004",
        "test_power_grid_audit_v003",
        "test_summarize_power_grid_v002",
    )
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
