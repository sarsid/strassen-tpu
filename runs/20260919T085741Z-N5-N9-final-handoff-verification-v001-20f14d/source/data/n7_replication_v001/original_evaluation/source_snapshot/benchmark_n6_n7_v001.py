"""Real TPU profiling (N6) and frozen-selector held-out evaluation (N7).

This runner reuses the frozen numerical/timing mechanics, never fits a selector,
and never changes an N5/N7 candidate selection. Profile wall times, device trace
durations, compiler diagnostics and analytic roofline estimates are separate.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

from strassen_mm import benchmark_v001 as base
from strassen_mm import benchmark_n5_v001 as tuning
from strassen_mm import selector_v001 as selection_rule


class EvidenceJournal(tuning.TuningJournal):
    def __init__(self, directory, phase):
        super().__init__(directory, phase)
        self.results, self.samples, self.captures = [], [], []

    def emit(self, event, **fields):
        super().emit(event, **fields)
        # Parent receives its own keyword dictionary. Normalize this retained
        # copy independently so offline summaries use public IDs too.
        stored = dict(fields)
        group = self.groups.get(stored.get("group_id"))
        if group:
            names = {arm["arm_id"]: arm["public_arm_id"] for arm in group["arms"]}
            if "arm_id" in stored:
                stored["arm_id"] = names.get(stored["arm_id"], stored["arm_id"])
            if stored.get("comparisons"):
                stored["comparisons"] = [dict(item, reference_arm=names.get(item["reference_arm"], item["reference_arm"]))
                                          for item in stored["comparisons"]]
        if event == "case_result":
            self.results.append(stored)
        elif event == "sample":
            self.samples.append(stored)
        elif event == "profile_capture":
            self.captures.append(stored)


def load_confirmed_rows(path):
    directory = selection_rule.artifacts(path)
    selection_rule.verify_seal(directory)
    summary = selection_rule.load(directory / "summary.json")
    if summary.get("phase") != "N5-confirm" or summary.get("completed") is not True:
        raise ValueError("N6 needs a complete N5 confirmation")
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines() if line.strip()]
    return directory, [row for row in rows if row.get("event") == "case_result" and row.get("scope") == "call"]


def representative_shapes(rows, choices):
    by_shape = {}
    for row in rows:
        by_shape.setdefault(row["shape_id"], {})[row["arm_id"]] = row
    categories = {"win": [], "tie": [], "loss": []}
    for shape_id, arms in by_shape.items():
        cubic, strassen = arms.get("cubic_selected"), arms.get("strassen_selected")
        if not selection_rule.passing(cubic) or not selection_rule.passing(strassen):
            continue
        pair = next((item for item in strassen.get("comparisons", []) if item.get("reference_arm") == "cubic_selected"), None)
        if not pair or not pair.get("valid_numerical_comparison"):
            continue
        lo, hi = pair["speedup_ci95"]
        category = "win" if lo > 1 else "loss" if hi < 1 else "tie"
        shape = choices["by_shape"][shape_id]["shape_mkn"]
        categories[category].append({"shape_id": shape_id, "shape_mkn": shape, "category": category,
                                     "paired_comparison": pair})
    chosen = []
    for category, values in categories.items():
        ordered = sorted(values, key=lambda row: (math.prod(row["shape_mkn"]), row["shape_id"]))
        if ordered:
            chosen.append(ordered[(len(ordered) - 1) // 2])
    return {"representatives": chosen, "available_class_counts": {key: len(value) for key, value in categories.items()},
            "absent_classes": [key for key, value in categories.items() if not value],
            "rule": "lower median by M*K*N then shape ID within each available eligible Strassen-vs-selected-cubic class"}


def selection_arms(row):
    arms = [tuning.make_arm(dict(selection_rule.NATIVE), "native")]
    for family in ("cubic", "strassen"):
        candidate = row.get(family)
        if candidate is None:
            candidate = {"candidate_id": family + "_no_eligible_candidate", "algorithm": "cubic_full" if family == "cubic" else "strassen",
                         "variant": "plain", "tile": [256, 256, 256], "predeclared_error": "unsupported",
                         "error_message": "No eligible independently screened candidate; no replacement."}
        arms.append(tuning.make_arm(candidate, family, family + "_selected"))
    return arms


def evaluation_groups(campaign, manifest, choices, rule):
    experiment = campaign["experiments"]["N7-evaluate"]
    shapes = {shape["id"]: shape for shape in manifest["shapes"]}
    groups, decisions = [], []
    for shape in tuning.shape_order([shapes[key] for key in experiment["shape_ids"]]):
        shape_mkn = [shape[axis] for axis in ("m", "k", "n")]
        row = choices["by_shape"][shape["id"]]
        if row["shape_mkn"] != shape_mkn:
            raise ValueError("N7 alternative selection shape mismatch")
        decision = selection_rule.choose(rule, shape_mkn, input_scope=experiment["input_scope"])
        decisions.append({"shape_id": shape["id"], **decision})
        arms = selection_arms(row)
        arms.append({**decision["choice"], "family": "selector", "arm_id": "selector", "public_arm_id": "selector"})
        groups.append({"group_id": shape["id"] + "__frozen_rule", "shape": shape, "arms": arms,
                       "inputs": [{"distribution": experiment["distribution"], "seed": experiment["seed"]}],
                       "timing": experiment["timing"], "scopes": ["call", "prepared_kernel"],
                       "tile": next(arm["tile"] for arm in arms if arm.get("tile") is not None)})
    return groups, decisions


def analyze_trace(trace, expected_steps):
    """Extract one actual TPU module track; never sum nested scope events."""
    events = trace["traceEvents"]
    processes = {event["pid"]: event.get("args", {}).get("name", "") for event in events
                 if event.get("ph") == "M" and event.get("name") == "process_name"}
    device_pids = [pid for pid, name in processes.items() if name.startswith("/device:TPU:")]
    if len(device_pids) != 1:
        raise ValueError("Trace must contain one identifiable actual TPU process")
    pid = device_pids[0]
    threads = {(event["pid"], event["tid"]): event.get("args", {}).get("name", "") for event in events
               if event.get("ph") == "M" and event.get("name") == "thread_name"}
    module_tids = [tid for (process, tid), name in threads.items() if process == pid and name == "XLA Modules"]
    if len(module_tids) != 1:
        raise ValueError("Missing or ambiguous TPU XLA Modules track")
    modules = sorted([event for event in events if event.get("ph") == "X" and event.get("pid") == pid
                      and event.get("tid") == module_tids[0]], key=lambda event: event["ts"])
    if len(modules) != expected_steps or any(event.get("dur", 0) <= 0 for event in modules):
        raise ValueError(f"Expected {expected_steps} positive-duration TPU modules; observed {len(modules)}")
    if any(left["ts"] + left["dur"] > right["ts"] + .05 for left, right in zip(modules, modules[1:])):
        raise ValueError("TPU module intervals overlap; refusing a serialized timing interpretation")
    values = [event["dur"] / 1000 for event in modules]
    return {"device_process": processes[pid], "module_samples_ms": values, "module_mean_ms": statistics.mean(values),
            "module_names": dict(Counter(event["name"] for event in modules)),
            "device_tracks": sorted(name for (process, _), name in threads.items() if process == pid),
            "possible_drop_indicators": sorted({event.get("name", "") for event in events
                if any(word in event.get("name", "").lower() for word in ("dropped", "overflow", "truncated"))}),
            "scope": "Actual device module elapsed durations in trace; not MXU/vector utilization or overlap counters."}


def roofline_estimate(metadata, policy):
    m, k, n = metadata["shape_mkn"]
    mp, kp, nn = metadata["padded_shape_mkn"]
    is_strassen = metadata["algorithm"] == "strassen"
    dot_work = 2 * mp * kp * nn * (7 / 8 if is_strassen else 1)
    compulsory = 2 * (m * k + k * n) + 4 * m * n
    tile = metadata.get("tile_bm_bn_bk")
    tile_bytes, vector_ops = None, None
    if tile:
        bm, bn, bk = tile
        tiles = (mp // bm) * (nn // bn) * (kp // bk)
        tile_bytes = tiles * 2 * (bm * bk + bk * bn) + 4 * mp * nn
        vector_ops = tiles * ((5 * (bm * bk + bk * bn) / 4 + 3 * bm * bn) if is_strassen else bm * bn)
    compute_ms = dot_work / policy["bf16_peak_flops_per_second"] * 1000
    bandwidth_ms = compulsory / policy["hbm_peak_bytes_per_second"] * 1000
    return {"useful_classical_flops": 2 * m * k * n, "estimated_mxu_dot_flops": dot_work,
            "compulsory_original_input_output_bytes": compulsory,
            "compulsory_arithmetic_intensity": dot_work / compulsory,
            "unshared_output_tile_traffic_estimate_bytes": tile_bytes,
            "source_level_vector_elementwise_ops_estimate": vector_ops,
            "ideal_dot_compute_ms": compute_ms, "ideal_compulsory_hbm_ms": bandwidth_ms,
            "optimistic_roofline_lower_bound_ms": max(compute_ms, bandwidth_ms),
            "vendor_source": policy["source"], "vendor_values": policy,
            "limitations": "Analytic optimistic estimates, not measured traffic/utilization. Excludes vector timing, extra traffic, padding/layout copies, launch costs and nonideal array occupancy. Tile traffic assumes no sharing across output tiles; actual reuse is unknown."}


def compiler_diagnostics(executable, directory, metadata, roofline_policy):
    directory.mkdir(parents=True, exist_ok=False)
    record = {"scope": "Compiler estimates and static code; not runtime memory peaks or counters.",
              "roofline": roofline_estimate(metadata, roofline_policy)}
    try:
        memory = executable.memory_analysis()
        keys = ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes",
                "generated_code_size_in_bytes", "host_argument_size_in_bytes", "host_output_size_in_bytes",
                "host_temp_size_in_bytes", "host_alias_size_in_bytes")
        record["compiled_memory_analysis"] = {key: int(getattr(memory, key)) for key in keys if hasattr(memory, key)}
        record["compiled_memory_repr"] = str(memory)
    except Exception as error:
        record["compiled_memory_unavailable"] = str(error)
    try:
        record["compiled_cost_analysis"] = base.json_safe(executable.cost_analysis())
    except Exception as error:
        record["compiled_cost_unavailable"] = str(error)
    try:
        text = executable.as_text()
        with (directory / "compiled_hlo.txt").open("x") as handle:
            handle.write(text or "")
        record["compiled_hlo_sha256"] = base.digest_file(directory / "compiled_hlo.txt")
    except Exception as error:
        record["compiled_hlo_unavailable"] = str(error)
    base.exclusive_json(directory / "diagnostics.json", record)
    return record


class ProfileRunner(tuning.TuningRunner):
    def run_profile_group(self, group):
        estimate = tuning.group_memory_estimate(group)
        self.journal.emit("memory_preflight", group_id=group["group_id"], **estimate)
        if estimate["estimated_live_device_bytes"] > self.campaign["memory"]["estimated_live_device_budget_gib"] * 1024 ** 3:
            self.journal.emit("profile_unavailable", group_id=group["group_id"], reason="memory_preflight", estimate=estimate)
            return
        policy = self.campaign["experiments"]["N6"]
        shape = tuple(group["shape"][axis] for axis in ("m", "k", "n"))
        a_host, b_host = base.generate_inputs(shape, **group["inputs"][0])
        entries = self.compile_entries(group, a_host, b_host)
        before = {**group, "group_id": group["group_id"] + "__before"}
        after = {**group, "group_id": group["group_id"] + "__after"}
        for item in (before, after):
            self.journal.groups[item["group_id"]] = item
        try:
            self.execute_case(before, before["inputs"][0], entries, a_host, b_host)
            a, b = base.jax.device_put(a_host), base.jax.device_put(b_host)
            base.jax.block_until_ready((a, b))
            before_results = {row["arm_id"]: row for row in self.journal.results
                              if row.get("group_id") == before["group_id"] and row.get("scope") == "call"}
            live = []
            for entry in entries:
                if "error" not in entry and selection_rule.passing(before_results.get(entry["public_arm_id"])):
                    live.append(entry)
                else:
                    self.journal.emit("profile_unavailable", group_id=group["group_id"], arm_id=entry["arm_id"],
                                      reason="compile_or_before_profile_numerical_failure")
            for entry in live:
                diagnostics = compiler_diagnostics(entry["executable"], self.journal.directory / "compiler" / group["shape"]["id"] / entry["public_arm_id"],
                                                    entry["metadata"], policy["roofline"])
                self.journal.emit("compiler_diagnostics", group_id=group["group_id"], arm_id=entry["arm_id"], diagnostics=diagnostics)
            for block in range(policy["capture_blocks"]):
                order = list(reversed(live)) if block % 2 else live
                for entry in order:
                    self.check_deadline()
                    name = entry["public_arm_id"]
                    target = self.journal.directory / "profiles" / group["shape"]["id"] / f"block-{block}-{name}"
                    detail = {"group_id": group["group_id"], "arm_id": entry["arm_id"], "block": block}
                    samples, failure, analysis = [], None, None
                    try:
                        output = entry["executable"](a, b)
                        output.block_until_ready()
                        output.delete()
                        options = base.jax.profiler.ProfileOptions()
                        options.host_tracer_level = policy["profiler"]["host_tracer_level"]
                        options.python_tracer_level = policy["profiler"]["python_tracer_level"]
                        options.enable_hlo_proto = policy["profiler"]["enable_hlo_proto"]
                        options.advanced_configuration = {"tpu_trace_mode": policy["profiler"]["tpu_trace_mode"]}
                        with base.jax.profiler.trace(target, create_perfetto_trace=True, profiler_options=options):
                            for step in range(policy["capture_steps"]):
                                with base.jax.profiler.StepTraceAnnotation(name, step_num=step):
                                    start = time.perf_counter_ns()
                                    output = entry["executable"](a, b)
                                    output.block_until_ready()
                                    samples.append((time.perf_counter_ns() - start) / 1e6)
                                    output.delete()
                        traces = list(target.rglob("perfetto_trace.json.gz"))
                        xplanes = list(target.rglob("*.xplane.pb"))
                        if len(traces) != 1 or not xplanes:
                            raise ValueError("Profiler did not produce one Perfetto trace plus XPlane data")
                        trace = json.loads(gzip.decompress(traces[0].read_bytes()))
                        analysis = analyze_trace(trace, policy["capture_steps"])
                    except Exception as error:
                        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
                    files = [{"path": str(path.relative_to(self.journal.directory)), "bytes": path.stat().st_size,
                              "sha256": base.digest_file(path)} for path in sorted(target.rglob("*")) if path.is_file()]
                    self.journal.emit("profile_capture", **detail, status="ok" if analysis else "diagnostic_failure", files=files,
                                      instrumented_wall_samples_ms=samples, device_analysis=analysis, error=failure,
                                      scope="Profiler-instrumented observations; ordinary latency is measured before/after separately.")
            del a, b
            self.execute_case(after, after["inputs"][0], entries, a_host, b_host)
        finally:
            del entries, a_host, b_host
            base.jax.clear_caches()
            base.gc.collect()


def evaluate_regret(journal, groups, decisions):
    result = []
    by_decision = {row["shape_id"]: row for row in decisions}
    for group in groups:
        group_rows = [row for row in journal.results if row.get("group_id") == group["group_id"]]
        for scope in group["scopes"]:
            rows = {row["arm_id"]: row for row in group_rows if row["scope"] == scope}
            selected = rows.get("selector")
            eligible = {name: row for name, row in rows.items() if name != "selector" and selection_rule.passing(row)}
            fastest = min(eligible, key=lambda name: eligible[name]["timing"]["mean_ms"]) if eligible else None
            regret = (selected["timing"]["mean_ms"] / eligible[fastest]["timing"]["mean_ms"]
                      if selected and selection_rule.passing(selected) and fastest else None)
            paired = {}
            chosen_samples = [row for row in journal.samples if row.get("group_id") == group["group_id"]
                              and row.get("scope") == scope and row.get("arm_id") == "selector"]
            for name, row in eligible.items():
                other_samples = [item for item in journal.samples if item.get("group_id") == group["group_id"]
                                 and item.get("scope") == scope and item.get("arm_id") == name]
                paired[name] = base.paired_comparison(other_samples, chosen_samples, group["inputs"][0]["seed"])
            result.append({"shape_id": group["shape"]["id"], "shape_mkn": [group["shape"][axis] for axis in ("m", "k", "n")],
                           "scope": scope, "decision": by_decision[group["shape"]["id"]],
                           "selector_status": selected.get("status") if selected else "missing", "fastest_eligible_measured_alternative": fastest,
                           "selector_time_over_fastest_alternative": regret, "paired_selector_speedups": paired,
                           "observed_means_ms": {name: (row.get("timing") or {}).get("mean_ms") for name, row in rows.items()},
                           "regret_scope": "Descriptive ratio to fastest eligible alternative in these confirmation rounds, not an oracle outside the bounded search."})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--phase", choices=("N6", "N7-evaluate"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-identity", type=Path, required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--selector", type=Path)
    parser.add_argument("--max-wall-seconds", type=float, default=21600)
    args = parser.parse_args(argv)
    if args.max_wall_seconds <= 0 or (args.phase == "N6" and args.confirmation is None) or (args.phase == "N7-evaluate" and args.selector is None):
        parser.error("Positive wall budget, N6 --confirmation, and N7 --selector are required")
    args.campaign = args.campaign.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir
    journal = EvidenceJournal(out, args.phase)
    planned, done, decisions, reps = [], [], [], None
    started, completed, failure = time.monotonic(), False, None
    try:
        campaign = selection_rule.load(args.campaign)
        if base.digest_file(args.campaign) != selection_rule.CONFIG_SHA256:
            raise ValueError("Campaign differs from frozen N5-N9 design")
        experiment = campaign["experiments"][args.phase]
        shape_path = args.campaign.parent / experiment.get("shape_manifest", campaign["shape_manifest"])
        manifest = selection_rule.load(shape_path)
        source_manifest = base.snapshot_sources(out, args.campaign, shape_path, args.campaign.parent / campaign["distribution_manifest"])
        choices = selection_rule.load(args.selection)
        base.exclusive_json(out / "selection_input.json", choices)
        rule, frozen_hash = None, None
        if args.phase == "N6":
            confirmation_dir, confirmation_rows = load_confirmed_rows(args.confirmation)
            source_identity = selection_rule.load(confirmation_dir / "environment.json")["identity"]
            if base.digest_file(args.selection) != selection_rule.load(confirmation_dir / "selection_input_provenance.json")["sha256"]:
                raise ValueError("N6 selections differ from the measured N5 confirmation")
            reps = representative_shapes(confirmation_rows, choices)
            base.exclusive_json(out / "representatives.json", reps)
            base.exclusive_json(out / "confirmation_provenance.json", {"directory": str(confirmation_dir),
                "results_sha256": base.digest_file(confirmation_dir / "results.jsonl"), "artifact_manifest_sha256": base.digest_file(confirmation_dir / "artifact_manifest.json")})
        else:
            rule = selection_rule.load(args.selector)
            frozen_hash = base.digest_file(args.selector)
            if rule.get("source_hashes", {}).get("heldout_reservation") != base.digest_file(shape_path):
                raise ValueError("Heldout manifest differs from frozen selector reserve")
            tuning.verify_heldout_selector(rule, manifest, experiment["shape_ids"])
            if choices.get("phase") != "N7-screen" or choices.get("frozen_selector_sha256") != frozen_hash:
                raise ValueError("N7 alternatives must be screened after this exact selector was frozen")
            base.exclusive_json(out / "frozen_selector_input.json", rule)
        base.exclusive_json(out / "input_provenance.json", {"selection_sha256": base.digest_file(args.selection),
            "frozen_selector_sha256": frozen_hash, "read_before_any_compilation_or_timing": True})
        journal.emit("run_start", allocation_id=args.allocation_id, source_manifest=source_manifest, argv=sys.argv, frozen_selector_sha256=frozen_hash)
        if "jax" in sys.modules:
            raise RuntimeError("Use a fresh process; JAX was imported before fixed environment setup")
        os.environ.update(base.FIXED_ENVIRONMENT)
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        from strassen_mm.kernels_v002 import make_matmul, enable_qualified_mosaic_v7_compat
        base.jax, base.jnp, base.np, base.ml_dtypes, base.make_matmul = jax, jnp, np, ml_dtypes, make_matmul
        environment = base.capture_environment(args, campaign, enable_qualified_mosaic_v7_compat())
        base.exclusive_json(out / "environment.json", environment)
        journal.emit("identity_check", **base.verify_identity(environment, args.expected_identity, campaign))
        if choices.get("environment_identity") != environment["identity"]:
            raise ValueError("Selections and this execution are from different cohorts")
        if args.phase == "N6":
            if source_identity != environment["identity"]:
                raise ValueError("N6 profiling and N5 confirmation machine identities differ")
            shape_lookup = {shape["id"]: shape for shape in manifest["shapes"]}
            for representative in reps["representatives"]:
                shape = shape_lookup[representative["shape_id"]]
                arms = selection_arms(choices["by_shape"][shape["id"]])
                planned.append({"group_id": shape["id"] + "__profile", "shape": shape, "arms": arms,
                                "inputs": [{"distribution": "gaussian", "seed": campaign["experiments"]["N5-confirm"]["seed"]}],
                                "timing": "confirm", "scopes": ["call"], "tile": next(arm["tile"] for arm in arms if arm.get("tile") is not None)})
        else:
            tuning.verify_heldout_selector(rule, manifest, experiment["shape_ids"], environment["identity"])
            tuning.verify_selection(choices, campaign, manifest, "N7-confirm", environment["identity"])
            planned, decisions = evaluation_groups(campaign, manifest, choices, rule)
            base.exclusive_json(out / "selector_decisions.json", decisions)
        journal.groups = {group["group_id"]: group for group in planned}
        base.exclusive_json(out / "planned_cases.json", planned)
        runner = ProfileRunner(args, campaign, journal) if args.phase == "N6" else tuning.TuningRunner(args, campaign, journal)
        for index, group in enumerate(planned):
            runner.check_deadline()
            print(f"{args.phase} [{index+1}/{len(planned)}] {group['group_id']}", flush=True)
            journal.emit("group_start", group_id=group["group_id"], index=index + 1, total=len(planned))
            if args.phase == "N6":
                runner.run_profile_group(group)
            else:
                runner.run_group(group)
            done.append(group["group_id"])
            journal.emit("group_complete", group_id=group["group_id"])
        if args.phase == "N7-evaluate":
            base.exclusive_json(out / "selector_evaluation.json", {"cases": evaluate_regret(journal, planned, decisions),
                "scope": "Frozen performance policy on reserved Gaussian-input geometries. No model-quality or universal accuracy guarantee."})
        completed = True
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        journal.emit("run_error", **failure)
        print(f"{args.phase} stopped: {error}", file=sys.stderr, flush=True)
    finally:
        expected_captures = len(planned) * 3 * 2 if args.phase == "N6" else None
        summary = {"phase": args.phase, "completed": completed, "status": "completed" if completed else "failed_or_interrupted",
                   "finished_utc": base.utc_now(), "wall_seconds": time.monotonic() - started, "completed_groups": done,
                   "planned_group_count": len(planned), "not_completed_group_ids": [group["group_id"] for group in planned if group["group_id"] not in done],
                   "case_status_counts": dict(journal.status_counts), "error": failure,
                   "profile_expected_captures": expected_captures, "profile_successful_captures": sum(row["status"] == "ok" for row in journal.captures),
                   "profile_attempted_captures": len(journal.captures),
                   "scientific_scope": "Execution completion does not imply positive speedups, full trace coverage, certified accuracy or model quality."}
        journal.emit("run_complete", **summary)
        journal.close()
        base.exclusive_json(out / "summary.json", summary)
        hashes = {str(path.relative_to(out)): base.digest_file(path) for path in sorted(out.rglob("*")) if path.is_file()}
        base.exclusive_json(out / "artifact_manifest.json", {"sha256": hashes, "sealed_utc": base.utc_now()})
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
