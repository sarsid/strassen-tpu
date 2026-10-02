"""Offline geometry, memory-contract and immutable-output checks (stdlib only)."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


GENERATOR = Path(__file__).resolve().parents[1] / "tools/build_power_midpoint_grid_v001.py"
SPEC = importlib.util.spec_from_file_location("power_midpoint_grid", GENERATOR)
grid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grid)


class PowerMidpointGridTests(unittest.TestCase):
    def test_dimensions_are_exact_integer_midpoints_and_endpoints(self):
        values = grid.dimensions()
        self.assertEqual(len(values), 34)
        self.assertEqual(len(values) ** 3, 39304)
        self.assertEqual(values[:8], (1, 2, 3, 4, 6, 8, 12, 16))
        self.assertEqual(values[-3:], (65536, 98304, 131072))
        self.assertEqual(values, tuple(sorted(set(values))))
        self.assertNotIn(1.5, values)
        self.assertIn(384, values)
        self.assertNotIn(640, values)

    def test_memory_lower_bound_preserves_axis_roles(self):
        self.assertEqual(grid.resident_lower_bound_bytes(2, 3, 5), 74)
        self.assertEqual(grid.resident_lower_bound_bytes(131072, 131072, 131072), 128 * grid.GIB)
        self.assertEqual(grid.resident_lower_bound_bytes(16384, 16384, 16384), 2 * grid.GIB)
        self.assertEqual(grid.pair_buffer_estimate(2, 3, 5, [(1, 1, 1)]),
                         2 * 50 + 2 * 24 + grid.REFERENCE_RESERVE_BYTES)
        with self.assertRaises(ValueError):
            grid.resident_lower_bound_bytes(1.5, 3, 5)

    def test_memory_counts_distinct_prepared_shapes_once(self):
        # Shape3^3: original inputs36; padded4^3 input64/output64.
        expected = 36 + 64 + 64 + 2 * 64 + grid.REFERENCE_RESERVE_BYTES
        self.assertEqual(grid.pair_buffer_estimate(3, 3, 3, [(2, 2, 2), (4, 4, 4)]), expected)
        # Introducing padded6^3 reserves BOTH prepared input pairs.
        expected_two = 36 + 144 + 64 + 144 + 2 * 144 + grid.REFERENCE_RESERVE_BYTES
        self.assertEqual(grid.pair_buffer_estimate(3, 3, 3, [(2, 2, 2), (6, 6, 6)]), expected_two)

    def test_sample_has_disjoint_holdouts_domain_edges_and_boundary_flags(self):
        manifest = grid.sampled_manifest()
        self.assertEqual(len(manifest["exploratory_shape_ids"]), 60)
        self.assertEqual(len(manifest["holdout_shape_ids"]), 12)
        self.assertFalse(set(manifest["exploratory_shape_ids"]) & set(manifest["holdout_shape_ids"]))
        self.assertEqual(manifest["holdout_history_check_scope"]["status"],
                         "no_exact_geometry_overlap_in_checked_configs")
        shapes = manifest["shapes"]
        self.assertEqual(len({(row["m"], row["n"], row["k"]) for row in shapes}), 72)
        self.assertEqual(sum(not row["on_lattice"] for row in shapes), 10)
        self.assertTrue(all(row["conservative_pair_buffer_bytes"] <= grid.BUDGET_BYTES for row in shapes))
        self.assertEqual(sum(row["family"] == "square_scale" for row in shapes), 14)
        self.assertEqual(sum(row["family"] == "interior_rectangle" for row in shapes), 6)
        for axis in ("m", "n", "k"):
            self.assertEqual(max(row[axis] for row in shapes), 131072)
        self.assertIn((2048, 2048, 2048), {(r["m"], r["n"], r["k"]) for r in shapes if r["stage"] == "exploratory"})
        self.assertIn((16384, 16384, 16384), {(r["m"], r["n"], r["k"]) for r in shapes if r["stage"] == "exploratory"})
        holdouts = {(r["m"], r["n"], r["k"]) for r in shapes if r["stage"] == "holdout"}
        self.assertTrue({(192, 192, 192), (3072, 3072, 1536), (6144, 6144, 3072)} <= holdouts)
        self.assertFalse({(s, s, s) for s in (1536, 3072, 6144)} & holdouts)

    def test_history_check_rejects_known_observed_geometry(self):
        with self.assertRaisesRegex(ValueError, "overlap checked historical geometry"):
            grid.check_holdout_history([{"m": 1536, "n": 1536, "k": 1536}])

    def test_output_lattice_order_cardinality_hashes_and_refusal_to_overwrite(self):
        with tempfile.TemporaryDirectory() as parent:
            output = Path(parent) / "new-output"
            # One exact-alignment planning tile keeps this I/O test inexpensive;
            # the default palette's memory contract is tested above.
            summary = grid.generate(output, tiles=[(256, 256, 256)])
            with (output / "full_lattice_preflight.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 39304)
            points = [(int(row["m"]), int(row["n"]), int(row["k"])) for row in rows]
            self.assertEqual(points, sorted(set(points)))
            self.assertEqual(points[0], (1, 1, 1))
            self.assertEqual(points[-1], (131072, 131072, 131072))
            self.assertEqual(sum(r["sample_stage"] != "preflight_only" for r in rows), 62)
            self.assertEqual(sum(summary["lattice_planning_budget_counts"].values()), 39304)
            manifest = json.loads((output / "artifact_manifest.json").read_text())
            for name, expected in manifest["artifacts_sha256"].items():
                self.assertEqual(grid.hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                grid.generate(output, tiles=[(256, 256, 256)])
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
