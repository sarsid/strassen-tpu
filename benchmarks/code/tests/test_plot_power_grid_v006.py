"""Explicit 3D camera prevents viewport-edge tick clipping; data are unchanged."""
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


plot = load("plot_power_grid_v006_tests", ROOT / "tools/plot_power_grid_v006.py")
prior = load("plot_power_grid_v005_reference", ROOT / "tools/plot_power_grid_v005.py")
original = load("plot_protocol_v001_for_v006", ROOT / "tests/test_plot_power_grid_v001.py")


class ScientificProtocolUnchanged(original.PowerGridPlotTests):
    def setUp(self):
        patch = mock.patch.object(original, "plot", plot)
        patch.start()
        self.addCleanup(patch.stop)


class CameraPresentation(unittest.TestCase):
    def test_camera_is_only_definition_change(self):
        def definitions(version):
            tree = ast.parse((ROOT / f"tools/plot_power_grid_{version}.py").read_text())
            return {n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
        before, after = definitions("v005"), definitions("v006")
        class RemoveCamera(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                node.keywords = [k for k in node.keywords if k.arg != "camera"]
                return node
        for name in before:
            node = RemoveCamera().visit(after[name]) if name == "interactive" else after[name]
            self.assertEqual(ast.dump(before[name], include_attributes=False), ast.dump(node, include_attributes=False), name)

    def test_complete_interactive_figure_identical_except_camera(self):
        import plotly.graph_objects as go
        with tempfile.TemporaryDirectory(prefix="synthetic_camera_qa_") as temp:
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
            camera = after["layout"]["scene"].pop("camera")
            self.assertEqual(camera, {"eye": {"x": 1.55, "y": 1.55, "z": 1.55}})
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
