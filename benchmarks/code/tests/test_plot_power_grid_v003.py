"""Methods/provenance presentation QA; fixtures are synthetic and temporary."""
import ast
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot = load("plot_power_grid_v003_tests", ROOT / "tools/plot_power_grid_v003.py")
original = load("plot_protocol_v001_for_v003", ROOT / "tests/test_plot_power_grid_v001.py")


class ScientificProtocolUnchanged(original.PowerGridPlotTests):
    def setUp(self):
        patch = mock.patch.object(original, "plot", plot)
        patch.start()
        self.addCleanup(patch.stop)


class FrontmatterPresentation(unittest.TestCase):
    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
        self.addCleanup(plt.close, "all")

    def fake_book(self, directory, evidence=None):
        book = plot.Book.__new__(plot.Book)
        book.plt, book.ev, book.output = self.plt, evidence, directory
        book.figures = []
        (directory / "figures").mkdir()
        book.pdf = SimpleNamespace(savefig=mock.Mock())
        return book

    def test_all_nonpresentation_definitions_match_v002_exactly(self):
        def definitions(version):
            tree = ast.parse((ROOT / f"tools/plot_power_grid_{version}.py").read_text())
            return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        before, after = definitions("v002"), definitions("v003")
        for name in before:
            if name not in {"Book", "generate", "interactive"}:
                self.assertEqual(before[name], after[name], name)

    def test_methods_derive_counts_precision_and_rounds_from_fixture(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_methods_qa_") as temp:
            screen, confirm, manifest = original.fixture(Path(temp))
            ev = plot.Evidence(screen, confirm, manifest)
            pages = plot.methods_sections(ev)
            text = "\n".join(body for _, sections in pages for _, body in sections)
            self.assertIn("1 exploratory shapes; 1 reserved holdouts", text)
            self.assertIn("Screen: 3 measured rounds", text)
            self.assertIn("Confirmation: 3 rounds", text)
            self.assertIn("Inputs: not recorded", text)
            self.assertNotIn("Screen: 7", text)
            self.assertIn("global tile optima", text)
            self.assertIn("one Strassen level per tile-panel", text)
            self.assertIn("SHA256:", text)

    def test_frontmatter_fits_and_page_number_does_not_change_metadata_title(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_methods_qa_") as temp:
            parent = Path(temp)
            screen, confirm, manifest = original.fixture(parent)
            ev = plot.Evidence(screen, confirm, manifest)
            output = parent / "layout_only"
            output.mkdir()
            book = self.fake_book(output, ev)
            captured = []
            book.pdf.savefig.side_effect = lambda fig: captured.append(fig)
            plot.frontmatter_pages(book)
            self.assertEqual(len(book.figures), 2)
            self.assertEqual([item["page"] for item in book.figures], [1, 2])
            self.assertEqual(book.figures[0]["title"], "Methods and reading guide")
            self.assertEqual(book.figures[1]["pdf"], "chartbook.pdf#page=2")
            for page, fig in enumerate(captured, 1):
                self.assertTrue(fig._suptitle.get_text().startswith(f"{page}. "))
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                footer = fig.texts[-1].get_window_extent(renderer)
                header = fig._suptitle.get_window_extent(renderer)
                for ax in fig.axes:
                    bbox = ax.get_tightbbox(renderer)
                    self.assertGreaterEqual(bbox.x0, 0)
                    self.assertLessEqual(bbox.x1, fig.bbox.width)
                    self.assertGreater(bbox.y0, footer.y1)
                    self.assertLess(bbox.y1, header.y0)
                # Top row body must end before the lower row headings.
                for index in (0, 1):
                    upper = fig.axes[index].texts[0].get_window_extent(renderer)
                    lower = fig.axes[index+2]._left_title.get_window_extent(renderer)
                    self.assertGreater(upper.y0, lower.y1)
            fig, axes = book.figure()
            book.save(fig, "first_actual_chart", "Original chart title", category="overview")
            self.assertTrue(fig._suptitle.get_text().startswith("3. Original chart title"))
            self.assertEqual(book.figures[-1]["title"], "Original chart title")
            self.assertEqual(book.figures[-1]["pdf"], "chartbook.pdf#page=3")

    def test_interactive_color_note_present_without_changing_data_logic(self):
        source = (ROOT / "tools/plot_power_grid_v003.py").read_text()
        self.assertIn("Color scale capped at ±2 in log₂ units; hover for exact ratios.", source)
        # Remove only the added presentation annotation; compare the remaining
        # interactive function AST to the already reviewed v002 implementation.
        def interactive_ast(version):
            tree = ast.parse((ROOT / f"tools/plot_power_grid_{version}.py").read_text())
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "interactive")
            fn.body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                       and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "add_annotation")]
            return ast.dump(fn, include_attributes=False)
        self.assertEqual(interactive_ast("v002"), interactive_ast("v003"))


if __name__ == "__main__":
    unittest.main()
