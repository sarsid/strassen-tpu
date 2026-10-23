"""Protocol guard tests; no JAX backend or TPU allocation is used."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "tools/benchmark_strassen_optimized_v001.py"
SPEC = importlib.util.spec_from_file_location("optimized_ablation_runner", SOURCE)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class PlanTests(unittest.TestCase):
    def args(self, *extra):
        return RUNNER.parser().parse_args(["--output-dir", "/tmp/not-used-by-protocol-tests", *extra])

    def test_complete_default_ablation_and_fixed_tiles(self):
        plan = RUNNER.make_plan(self.args())
        self.assertEqual(len(plan["shapes"]), 8)
        self.assertEqual(len(plan["candidates"]), 9)
        for shape in plan["shapes"]:
            bound = RUNNER.bound_candidates(plan, shape)
            self.assertEqual([c["candidate_id"] for c in bound[:3]], ["native_default", "native_vmem_64m", "old_strassen"])
            self.assertTrue(all(c["tile"] == shape["tile_bm_bn_bk"] for c in bound[2:]))
            self.assertTrue(all(c["tile"] is None for c in bound[:2]))
        largest = next(s for s in plan["shapes"] if s["m"] == s["n"] == s["k"] == 16384)
        self.assertEqual(largest["tile_bm_bn_bk"], [2048, 2048, 512])
        self.assertLess(RUNNER.memory_estimate(largest, RUNNER.bound_candidates(plan, largest))["estimated_live_device_bytes"], 8*2**30)

    def test_schedule_candidates_are_opt_in_registered_and_unique(self):
        plan = RUNNER.make_plan(self.args("--include-schedule-candidates"))
        self.assertEqual(len(plan["candidates"]), 14)
        extras = plan["candidates"][9:]
        self.assertEqual(len({tuple(c["product_order"]) for c in extras}), 5)
        for candidate in extras:
            self.assertNotEqual(tuple(candidate["product_order"]), RUNNER.PRODUCT_ORDER)
            self.assertEqual(candidate["variant"], "optimized")
            self.assertEqual(candidate["traversal"], "mnk")
            self.assertIn("SOPT-003", candidate["optimization_ids"])

    def test_interpreter_defaults_small_and_rejects_tpu_size(self):
        plan = RUNNER.make_plan(self.args("--interpret-correctness"))
        self.assertEqual([(s["m"], s["n"], s["k"]) for s in plan["shapes"]], list(RUNNER.INTERPRET_SHAPES))
        with self.assertRaisesRegex(ValueError, "CPU correctness interpretation"):
            RUNNER.make_plan(self.args("--interpret-correctness", "--shape", "8192,8192,8192"))

    def test_shape_tile_override_preserves_public_axis_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"shapes.json"
            path.write_text(json.dumps({"shapes": [{"m":17,"n":257,"k":259,"tile_bm_bn_bk":[16,512,256]}]}))
            plan = RUNNER.make_plan(self.args("--shapes-json", str(path)))
            shape = plan["shapes"][0]
            self.assertEqual(shape["id"], "ablation_m17_n257_k259")
            self.assertEqual(RUNNER.bound_candidates(plan, shape)[2]["tile"], [16,512,256])

    def test_invalid_geometry_or_tile_is_not_silently_rounded(self):
        with self.assertRaisesRegex(ValueError, "unique shapes"):
            RUNNER.make_plan(self.args("--shape","17,257,259","--shape","17,257,259"))
        with self.assertRaisesRegex(ValueError, "align"):
            RUNNER.make_plan(self.args("--tile","17,256,256"))

    def test_prior_run_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("plan.json","summary.json","results.jsonl","source_manifest.json","environment.json")
            for name in names: (root/name).write_text("{}\n")
            (root/"artifact_manifest.json").write_text(json.dumps({"sha256":{name:RUNNER.base.digest_file(root/name) for name in names}}))
            RUNNER.verify_artifact_hashes(root)
            (root/"plan.json").write_text('{"tampered":true}\n')
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                RUNNER.verify_artifact_hashes(root)


if __name__ == "__main__":
    unittest.main()
