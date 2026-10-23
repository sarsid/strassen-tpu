"""Regression QA for timing/projection scopes and interactive presentation."""
import ast
import csv
import importlib.util
import json
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


plot = load("plot_power_grid_v004_tests", ROOT / "tools/plot_power_grid_v004.py")
prior = load("plot_power_grid_v003_reference", ROOT / "tools/plot_power_grid_v003.py")
original = load("plot_protocol_v001_for_v004", ROOT / "tests/test_plot_power_grid_v001.py")


class ScientificProtocolUnchanged(original.PowerGridPlotTests):
    def setUp(self):
        patch = mock.patch.object(original, "plot", plot)
        patch.start()
        self.addCleanup(patch.stop)


class NearsetScopeRegression(unittest.TestCase):
    def test_other_definitions_match_v003_exactly(self):
        def definitions(version):
            tree = ast.parse((ROOT / f"tools/plot_power_grid_{version}.py").read_text())
            return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        before, after = definitions("v003"), definitions("v004")
        for name in before:
            if name not in {"nearset_pages", "interactive"}:
                self.assertEqual(before[name], after[name], name)

    def test_populated_panels_and_csv_keep_both_scopes_and_qualification(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.addCleanup(plt.close, "all")
        with tempfile.TemporaryDirectory(prefix="synthetic_nearset_qa_") as temp:
            parent = Path(temp)
            screen, confirm, manifest = original.fixture(parent)
            report_path = confirm / "confirmation.json"
            report = json.loads(report_path.read_text())
            sid = "unit_m256_n512_k128"
            qualification = "Discrete tested tuples only; axis projections do not certify their Cartesian product."
            definitions = ("descriptive_1_percent", "descriptive_3_percent", "descriptive_5_percent",
                           "pointwise_ci95_certified_5_percent")
            for family in ("cubic", "strassen"):
                for scope, count in (("call", 2), ("prepared_kernel", 3)):
                    report["by_shape"][sid][family]["near_sets"][scope] = {
                        definition: {"scope": qualification,
                                     "tuples_bm_bn_bk": [[256, 512, 256*i] for i in range(1, count+1)],
                                     "axis_values": {"bm": [256], "bn": [512], "bk": [256*i for i in range(1, count+1)]}}
                        for definition in definitions}
            original.dump(report_path, report)
            ev = plot.Evidence(screen, confirm, manifest)
            output = parent / "qa_only"
            output.mkdir()
            (output / "tables").mkdir()
            (output / "figures").mkdir()
            book = plot.Book.__new__(plot.Book)
            book.ev, book.plt, book.output, book.figures = ev, plt, output, []
            captured = []
            book.pdf = SimpleNamespace(savefig=lambda fig: captured.append(fig))
            plot.nearset_pages(book)
            self.assertEqual(len(captured), 2)
            for fig, count in zip(captured, (2, 3)):
                self.assertEqual(len(fig.axes), 2)
                for ax in fig.axes:
                    self.assertEqual(len(ax.lines), 4)
                    for line in ax.lines:
                        self.assertEqual(list(line.get_ydata()), [count])
                        self.assertEqual(list(line.get_xdata()), [0])
            with (output / "tables/near_sets.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 16)
            self.assertEqual({r["scope"] for r in rows}, {"call", "prepared_kernel"})
            self.assertEqual({r["projection_scope"] for r in rows}, {qualification})
            self.assertEqual({r["family"] for r in rows}, {"cubic", "strassen"})
            for row in rows:
                self.assertEqual(len(json.loads(row["tuples_bm_bn_bk"])), 2 if row["scope"] == "call" else 3)

    def test_interactive_data_and_switching_unchanged_with_separate_controls(self):
        import plotly.graph_objects as go
        with tempfile.TemporaryDirectory(prefix="synthetic_interactive_qa_") as temp:
            parent = Path(temp)
            screen, confirm, manifest = original.fixture(parent)
            ev = plot.Evidence(screen, confirm, manifest)
            book = SimpleNamespace(ev=ev, output=parent)
            figures = []
            with mock.patch.object(go.Figure, "write_html", autospec=True,
                                   side_effect=lambda fig, *a, **k: figures.append(fig)):
                prior.interactive(book)
                plot.interactive(book)
            before, after = [fig.to_plotly_json() for fig in figures]
            self.assertEqual(before["data"], after["data"])
            old_buttons = before["layout"]["updatemenus"][0]["buttons"]
            new_buttons = after["layout"]["updatemenus"][0]["buttons"]
            self.assertEqual(len(new_buttons), 6)
            for old, new in zip(old_buttons, new_buttons):
                self.assertEqual(old["args"][0], new["args"][0])
                self.assertEqual(old["args"][1]["title"], new["args"][1]["title.text"])
            layout = after["layout"]
            self.assertEqual(layout["legend"]["orientation"], "h")
            self.assertLess(layout["legend"]["y"], 0)
            self.assertGreater(layout["coloraxis"]["colorbar"]["x"], 1)
            self.assertEqual(layout["title"]["y"], .97)
            self.assertGreater(layout["updatemenus"][0]["y"], layout["annotations"][0]["y"])
            self.assertGreater(layout["annotations"][0]["y"], 1)


if __name__ == "__main__":
    unittest.main()
