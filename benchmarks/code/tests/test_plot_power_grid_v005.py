"""Compact latency tick presentation; data and logarithmic scales are retained."""
import ast
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot = load("plot_power_grid_v005_tests", ROOT / "tools/plot_power_grid_v005.py")
original = load("plot_protocol_v004_for_v005", ROOT / "tests/test_plot_power_grid_v004.py")


class ScientificProtocolUnchanged(original.ScientificProtocolUnchanged):
    def setUp(self):
        patch = mock.patch.object(original.original, "plot", plot)
        patch.start()
        self.addCleanup(patch.stop)


class TickPresentation(unittest.TestCase):
    def test_only_detailed_tick_presentation_changed(self):
        def definitions(version):
            tree = ast.parse((ROOT / f"tools/plot_power_grid_{version}.py").read_text())
            return {n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        before, after = definitions("v004"), definitions("v005")
        class RemovePresentationCall(ast.NodeTransformer):
            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "format_latency_ticks":
                    return None
                return self.generic_visit(node)
        for name in before:
            node = RemovePresentationCall().visit(after[name])
            self.assertEqual(ast.dump(before[name], include_attributes=False), ast.dump(node, include_attributes=False), name)

    def test_narrow_and_broad_ranges_keep_log_data_limits_and_readable_labels(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.addCleanup(plt.close, "all")
        for values in ([.224, .253, .293], [42., 66., 370.], [3., 80., 1800.]):
            with self.subTest(values=values):
                fig, ax = plt.subplots(figsize=(5.3, 4))
                points = ax.scatter(values, [0, 1, 2])
                ax.set_xscale("log")
                before_limits = ax.get_xlim()
                before_data = points.get_offsets().copy()
                plot.format_latency_ticks(ax)
                fig.canvas.draw()
                self.assertEqual(ax.get_xscale(), "log")
                self.assertEqual(ax.get_xlim(), before_limits)
                self.assertEqual(points.get_offsets().tolist(), before_data.tolist())
                renderer = fig.canvas.get_renderer()
                visible = [t.label1 for t in ax.xaxis.get_major_ticks() if before_limits[0] <= t.get_loc() <= before_limits[1]]
                self.assertLessEqual(len(visible), 6)
                self.assertGreaterEqual(len(visible), 1)
                boxes = sorted([label.get_window_extent(renderer) for label in visible], key=lambda b: b.x0)
                for left, right in zip(boxes, boxes[1:]):
                    self.assertLess(left.x1 + 4, right.x0)
                self.assertTrue(all("\\mathdefault" not in label.get_text() and "e" not in label.get_text().lower() for label in visible))


if __name__ == "__main__":
    unittest.main()
