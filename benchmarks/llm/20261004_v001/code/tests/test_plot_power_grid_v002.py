"""Layout-only regression checks plus unchanged scientific protocol fixtures.

All labels/positions below are synthetic QA data, never experimental results.
No scientific output or benchmark invocation is created by these tests.
"""
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


plot = load("plot_power_grid_v002_tests", ROOT / "tools/plot_power_grid_v002.py")
original_tests = load("original_plot_tests", ROOT / "tests/test_plot_power_grid_v001.py")


class ScientificProtocolUnchanged(original_tests.PowerGridPlotTests):
    """Run the frozen v001 fixture suite against v002 without editing v001."""
    def setUp(self):
        patch = mock.patch.object(original_tests, "plot", plot)
        patch.start()
        self.addCleanup(patch.stop)


class PowerGridLayoutTests(unittest.TestCase):
    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
        self.addCleanup(plt.close, "all")
        self.note = ("LAYOUT QA ONLY - PLACEHOLDER POSITIONS, NOT EXPERIMENT RESULTS. "
                     "Pointwise paired 95% bootstrap intervals; no multiplicity correction. "
                     "Native_default is the library default; native is its frozen tuned choice. ") * 3

    def assert_page_fits(self, fig):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        outer = fig.bbox
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            bounds = ax.get_tightbbox(renderer)
            self.assertGreaterEqual(bounds.x0, outer.x0, ax.get_title())
            self.assertGreaterEqual(bounds.y0, outer.y0, ax.get_title())
            self.assertLessEqual(bounds.x1, outer.x1, ax.get_title())
            self.assertLessEqual(bounds.y1, outer.y1, ax.get_title())
            footer = fig.texts[-1].get_window_extent(renderer)
            header = fig._suptitle.get_window_extent(renderer)
            self.assertGreater(bounds.y0, footer.y1)
            self.assertLess(bounds.y1, header.y0)
        for text in (fig._suptitle, fig.texts[-1]):
            bounds = text.get_window_extent(renderer)
            self.assertGreaterEqual(bounds.x0, outer.x0)
            self.assertLessEqual(bounds.x1, outer.x1)

    def test_scientific_definitions_are_ast_identical_to_v001(self):
        def definitions(version):
            tree = ast.parse((ROOT / f"tools/plot_power_grid_{version}.py").read_text())
            return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        original, current = definitions("v001"), definitions("v002")
        for name in original:
            if name not in {"Book", "slice_atlas"}:
                self.assertEqual(original[name], current[name], name)

    def test_long_shape_forest_labels_and_long_footer_fit(self):
        fig, ax = self.plt.subplots(figsize=(12, 8.2), layout="constrained")
        ax.scatter([1] * 20, range(20))
        ax.set_yticks(range(20), ["131072 × 2048 × 2048  n=30"] * 20, fontsize=8)
        ax.set(xlabel="native_default mean / strassen mean", ylabel="Shape (M × N × K)", xscale="log")
        result = plot.fit_page_layout(fig, "LAYOUT QA ONLY - long forest labels", self.note, "speedups")
        self.assertGreaterEqual(result["footer_lines"], 3)
        self.assert_page_fits(fig)

    def test_long_native_labels_fit_both_panels_without_collision(self):
        fig, axes = self.plt.subplots(1, 2, figsize=(12, 8.2), layout="constrained")
        for ax in axes:
            ax.imshow([[1, 2, 3, 4]] * 20, aspect="auto")
            ax.set_yticks(range(20), ["131072 × 2048 × 2048"] * 20, fontsize=7)
            ax.set_xticks(range(4), ["default", "vmem_32m", "vmem_48m", "vmem_64m"], rotation=25, ha="right", fontsize=8)
            ax.set_title("Complete device call")
        plot.fit_page_layout(fig, "LAYOUT QA ONLY - native options", self.note, "native_tuning")
        self.assert_page_fits(fig)
        renderer = fig.canvas.get_renderer()
        self.assertLess(axes[0].get_tightbbox(renderer).x1, axes[1].get_tightbbox(renderer).x0)

    def test_twenty_four_tile_labels_fit_and_titles_do_not_collide(self):
        fig, axes = self.plt.subplots(2, 2, figsize=(14, 13), layout="constrained", gridspec_kw={"height_ratios": [2, 1]})
        for row, count in zip(axes, (24, 6)):
            for ax in row:
                ax.scatter([1] * count, range(count))
                ax.set_yticks(range(count), ["1024/2048/1536"] * count, fontsize=8)
                ax.set(xlabel="Candidate latency / frozen family winner", xscale="log", title="Confirmation shortlist (colored=call; gray=prepared)")
        plot.fit_page_layout(fig, "LAYOUT QA ONLY - tile labels", self.note, "tiles")
        self.assert_page_fits(fig)
        renderer = fig.canvas.get_renderer()
        self.assertLess(axes[0, 0].get_tightbbox(renderer).x1, axes[0, 1].get_tightbbox(renderer).x0)

    def test_four_by_three_atlas_and_colorbar_fit(self):
        fig, axes = self.plt.subplots(4, 3, figsize=(12, 13.5), layout="constrained")
        for ax in axes.flat:
            plot.axes_log_lattice(ax)
            ax.tick_params(labelsize=7)
            ax.set_title("K=131,072 · native/strassen", fontsize=9)
        fig.colorbar(self.plt.cm.ScalarMappable(cmap="RdBu"), ax=list(axes.flat),
                     location="right", fraction=.022, pad=.025, shrink=.82,
                     label="log₂(reference/candidate), capped at ±2")
        plot.fit_page_layout(fig, "LAYOUT QA ONLY - 4x3 slices", self.note, "slice_atlas")
        self.assert_page_fits(fig)

    def test_render_width_wrapping_preserves_words_and_ascii_dashes(self):
        fig = self.plt.figure(figsize=(6, 4))
        text = "Wide words WWWWWW repeated — " * 25
        wrapped = plot.wrap_canvas_text(fig, text, fontsize=8)
        self.assertEqual(wrapped.split(), text.replace("—", "-").split())
        from matplotlib.font_manager import FontProperties
        renderer = fig.canvas.get_renderer()
        for line in wrapped.splitlines():
            width, _, _ = renderer.get_text_width_height_descent(line, FontProperties(size=8), ismath=False)
            self.assertLessEqual(width, fig.bbox.width * .93)


if __name__ == "__main__":
    unittest.main()
