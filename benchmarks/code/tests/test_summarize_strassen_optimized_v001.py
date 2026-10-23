"""Synthetic schema/unit fixtures only; never measured benchmark evidence."""
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import statistics
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "tools/summarize_strassen_optimized_v001.py"
SPEC = importlib.util.spec_from_file_location("summarize_optimized_test", SOURCE)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def seal(root):
    files = {str(p.relative_to(root)): SUMMARY.sha256(p) for p in root.rglob("*")
             if p.is_file() and p.name != "artifact_manifest.json"}
    write_json(root / "artifact_manifest.json", {"sha256": files, "sealed_utc": "synthetic-test"})


def fixture(root, phase="screen", seed=101, cpu=False, failed=None, parent=None):
    """Write a tiny explicitly synthetic sealed schema, with no kernel execution."""
    root.mkdir()
    tile = [16, 256, 256]
    shape = {"id": "synthetic_m17_n257_k259", "m": 17, "n": 257, "k": 259,
             "tile_bm_bn_bk": tile}
    candidates = [
        {"candidate_id": "native_default", "implementation": "native", "variant": "plain", "tile": None, "compiler_options": {}},
        {"candidate_id": "native_vmem_64m", "implementation": "native", "variant": "plain", "tile": None,
         "compiler_options": {"xla_tpu_scoped_vmem_limit_kib": 65536}},
        {"candidate_id": "old_strassen", "implementation": "kernels_v002", "variant": "interleaved_output_accumulator", "tile": tile, "compiler_options": {}},
        {"candidate_id": "new_optimized", "implementation": "strassen_optimized", "variant": "optimized", "tile": tile,
         "compiler_options": {}, "product_order": [4, 6, 5, 2, 7, 3, 1], "traversal": "mnk"},
    ]
    plan = {"shape_order": "M,N,K", "tile_order": "BM,BN,BK", "scopes": list(SUMMARY.SCOPES),
            "shapes": [shape], "candidates": candidates, "interpret_correctness": cpu}
    write_json(root / "plan.json", plan)
    devices = [{"platform": "cpu" if cpu else "tpu", "kind": "SYNTHETIC UNIT FIXTURE", "id": 0}]
    identity = {"allocation_id": "synthetic-not-an-allocation", "device_kind": "SYNTHETIC UNIT FIXTURE", "devices": devices}
    environment = {"identity": identity, "devices": devices, "backend": "cpu" if cpu else "tpu",
                   "device_count": 1, "local_device_count": 1, "process_count": 1, "interpretation_only": cpu}
    write_json(root / "environment.json", environment)
    (root / "source_snapshot").mkdir()
    source = root / "source_snapshot/synthetic.py"
    source.write_text("# Synthetic unit fixture; no measurements\n")
    write_json(root / "source_manifest.json", {"sha256": {"source_snapshot/synthetic.py": SUMMARY.sha256(source)}})
    repeats = 30 if phase == "confirm" else 7
    group = {"shape": shape, "group_id": shape["id"] + "__fixed_ablation", "candidates": candidates,
             "scopes": list(SUMMARY.SCOPES), "timing": {"repeats": repeats},
             "input": {"distribution": "gaussian", "seed": seed}}
    write_json(root / "planned_cases.json", [group])
    events = []

    def emit(event, **fields):
        events.append({"event": event, "sequence": len(events) + 1, "phase": phase, **fields})

    emit("run_start", seed=seed, repeats=repeats, interpretation_only=cpu,
         plan_sha256=SUMMARY.sha256(root / "plan.json"), execution_plan=[group])
    values = {"native_default": 2., "native_vmem_64m": 1.8, "old_strassen": 2.5, "new_optimized": 1.5}
    samples = {cid: [v * (1 + r / 100) for r in range(repeats)] for cid, v in values.items()}
    means = {cid: statistics.mean(v) for cid, v in samples.items()}
    cases = []
    for candidate in candidates:
        cid = candidate["candidate_id"]
        for scope in SUMMARY.SCOPES:
            context = {**candidate, "shape_id": shape["id"], "scope": scope, "group_id": group["group_id"],
                       "shape_mnk": [17, 257, 259], "shape_mkn": [17, 259, 257],
                       "distribution": "gaussian", "seed": seed}
            active = cid != failed
            if active:
                for r, elapsed in enumerate(samples[cid]):
                    emit("sample", **context, round=r, elapsed_ms=elapsed)
            comparisons = []
            if active:
                for ref in SUMMARY.REFERENCES:
                    if ref in (cid, failed):
                        continue
                    speedup = means[ref] / means[cid]
                    comparisons.append({"reference_candidate_id": ref, "group_id": group["group_id"], "scope": scope,
                                        "valid_numerical_comparison": True, "screen_results_are_not_confirmation": phase != "confirm",
                                        "paired_rounds": list(range(repeats)), "speedup_ratio_of_means": speedup,
                                        "time_reduction_fraction": 1 - 1 / speedup,
                                        "speedup_ci95": [speedup * .9, speedup * 1.1],
                                        "method": "SYNTHETIC unit interval, not a real bootstrap result"})
            case = {**context, "status": "ok" if active else "compile_error",
                    "interpretation_only": cpu, "numerically_eligible": active,
                    "eligible_for_speedup_claim": active and not cpu, "correctness": {"pass": active},
                    "timing": {"sample_count": repeats, "mean_ms": means[cid]} if active else {"sample_count": 0},
                    "comparisons": comparisons, "error_message": None if active else "Synthetic test compilation failure"}
            cases.append(case)
            emit("case_result", **case)
    summary = {"phase": phase, "completed": True, "plan_only": False, "interpretation_only": cpu,
               "seed": seed, "repeats": repeats, "planned_shapes": 1, "planned_candidates": len(candidates),
               "case_status_counts": {status: sum(c["status"] == status for c in cases) for status in {c["status"] for c in cases}},
               "error": None}
    emit("run_complete", **summary)
    (root / "results.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    write_json(root / "summary.json", summary)
    if parent:
        write_json(root / "screen_provenance.json", {
            "path": str(parent), "plan_sha256": SUMMARY.sha256(parent / "plan.json"),
            "results_sha256": SUMMARY.sha256(parent / "results.jsonl"),
            "source_manifest_sha256": SUMMARY.sha256(parent / "source_manifest.json")})
    seal(root)
    return root


def edit_events(root, edit):
    path = root / "results.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    edit(events)
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    seal(root)


class SummaryContracts(unittest.TestCase):
    def test_time_reduction_direction_and_monotone_interval_conversion(self):
        point, ci = SUMMARY.time_reduction_percent(2., [1.5, 2.5])
        self.assertEqual(point, 50.)
        self.assertAlmostEqual(ci[0], 100 / 3)
        self.assertEqual(ci[1], 60.)
        point, ci = SUMMARY.time_reduction_percent(.5, [.4, .6])
        self.assertEqual(point, -100.)
        self.assertEqual(ci[0], -150.)
        self.assertAlmostEqual(ci[1], -200 / 3)
        self.assertEqual(SUMMARY.time_reduction_percent(1., [1., 1.]), (0., [0., 0.]))

    def test_invalid_intervals_are_rejected(self):
        for point, interval in ((0., [1., 2.]), (math.nan, [1., 2.]),
                                (1., [2., 1.]), (1., [0., 1.]), (1., [1., math.inf]), (1., [1.])):
            with self.assertRaises(ValueError):
                SUMMARY.time_reduction_percent(point, interval)

    def test_raw_means_pair_direction_and_descriptive_minimum(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_summary_unit_") as td:
            ev = SUMMARY.Evidence(fixture(Path(td) / "screen"), "screen")
            report = SUMMARY.summarize_run(ev)
            rows = {r["candidate_id"]: r for r in report["shape_tables"][0]["rows"]}
            self.assertAlmostEqual(rows["new_optimized"]["comparisons"]["old_strassen"]["time_reduction_percent"], 40.)
            self.assertAlmostEqual(rows["new_optimized"]["comparisons"]["native_default"]["time_reduction_percent"], 25.)
            self.assertEqual(report["shape_tables"][0]["descriptive_lowest_mean_candidate"], "new_optimized")
            self.assertEqual(report["failures_and_ineligible_cases"], [])
            self.assertEqual(rows["old_strassen"]["comparisons"]["old_strassen"]["pointwise_result"], "self_reference")

    def test_cpu_and_non_tpu_evidence_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_summary_unit_") as td:
            cpu = fixture(Path(td) / "cpu", cpu=True)
            with self.assertRaisesRegex(ValueError, "CPU correctness"):
                SUMMARY.Evidence(cpu)
            fake = fixture(Path(td) / "non_tpu")
            env = SUMMARY.read_json(fake / "environment.json")
            env["backend"] = "cpu"
            write_json(fake / "environment.json", env)
            seal(fake)
            with self.assertRaisesRegex(ValueError, "one TPU"):
                SUMMARY.Evidence(fake)

    def test_cpu_terminal_row_cannot_enter_hardware_summary(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_summary_unit_") as td:
            root = fixture(Path(td) / "screen")
            edit_events(root, lambda es: next(e for e in es if e["event"] == "case_result").update(interpretation_only=True))
            with self.assertRaisesRegex(ValueError, "CPU result"):
                SUMMARY.Evidence(root)

    def test_failed_arm_is_visible_but_never_compared_or_selected(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_summary_unit_") as td:
            ev = SUMMARY.Evidence(fixture(Path(td) / "screen", failed="new_optimized"))
            report = SUMMARY.summarize_run(ev)
            self.assertEqual(len(report["failures_and_ineligible_cases"]), 2)
            row = next(r for r in report["shape_tables"][0]["rows"] if r["candidate_id"] == "new_optimized")
            self.assertFalse(row["eligible"])
            self.assertTrue(all(not c["available"] for c in row["comparisons"].values()))
            self.assertIsNone(report["shape_tables"][0]["descriptive_lowest_mean_new_candidate"])
            text = SUMMARY.markdown({"runs": [report]})
            self.assertIn("compile_error", text)
            self.assertIn("Synthetic test compilation failure", text)

    def test_missing_and_failed_eligibility_flags_filter_results(self):
        good = {"status": "ok", "numerically_eligible": True, "eligible_for_speedup_claim": True, "correctness": {"pass": True}}
        self.assertTrue(SUMMARY.eligible(good))
        for field, value in (("interpretation_only", True), ("status", "numerical_failure"),
                             ("eligible_for_speedup_claim", False), ("correctness", {"pass": False})):
            self.assertFalse(SUMMARY.eligible(dict(good, **{field: value})))
        self.assertFalse(SUMMARY.eligible({"status": "ok"}))

    def test_seal_and_raw_mean_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_summary_unit_") as td:
            root = fixture(Path(td) / "screen")
            (root / "plan.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                SUMMARY.Evidence(root)
            other = fixture(Path(td) / "other")
            edit_events(other, lambda es: next(e for e in es if e["event"] == "sample").update(elapsed_ms=900.))
            with self.assertRaisesRegex(ValueError, "mean disagrees"):
                SUMMARY.Evidence(other)

    def test_wrong_comparison_scope_or_point_estimate_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_summary_unit_") as td:
            for i, changes in enumerate(({"scope": "prepared_kernel"}, {"speedup_ratio_of_means": 99.})):
                root = fixture(Path(td) / str(i))
                def mutate(events):
                    row = next(e for e in events if e["event"] == "case_result" and e["candidate_id"] == "new_optimized" and e["scope"] == "call")
                    row["comparisons"][0].update(changes)
                edit_events(root, mutate)
                with self.assertRaisesRegex(ValueError, "Comparison"):
                    SUMMARY.summarize_run(SUMMARY.Evidence(root))

    def test_phase_binding_requires_fresh_seed_and_exact_parent(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_summary_unit_") as td:
            screen = fixture(Path(td) / "screen")
            confirm = fixture(Path(td) / "confirm", phase="confirm", seed=102, parent=screen)
            left, right = SUMMARY.Evidence(screen, "screen"), SUMMARY.Evidence(confirm, "confirm")
            SUMMARY.bind_phases(left, right)
            self.assertEqual(left.repeats, 7)
            self.assertEqual(right.repeats, 30)
            same_seed = fixture(Path(td) / "same_seed", phase="confirm", seed=101, parent=screen)
            with self.assertRaisesRegex(ValueError, "cohort mismatch"):
                SUMMARY.bind_phases(left, SUMMARY.Evidence(same_seed))
            right.identity = copy.deepcopy(right.identity)
            right.identity["allocation_id"] = "different-synthetic-allocation"
            with self.assertRaisesRegex(ValueError, "cohort mismatch"):
                SUMMARY.bind_phases(left, right)

    def test_output_creation_is_exclusive_and_cpu_emits_no_output(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_summary_unit_") as td:
            screen = fixture(Path(td) / "screen")
            output = Path(td) / "output"
            argv = ["--screen", str(screen), "--output-dir", str(output)]
            with redirect_stdout(io.StringIO()):
                SUMMARY.main(argv)
            initial = (output / "findings.json").read_bytes()
            with self.assertRaises(FileExistsError):
                SUMMARY.main(argv)
            self.assertEqual((output / "findings.json").read_bytes(), initial)
            cpu = fixture(Path(td) / "cpu", cpu=True)
            rejected = Path(td) / "cpu_output"
            with self.assertRaises(ValueError):
                SUMMARY.main(["--screen", str(cpu), "--output-dir", str(rejected)])
            self.assertFalse(rejected.exists())


if __name__ == "__main__":
    unittest.main()
