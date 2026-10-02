"""Equal-attempt independent tuning and fresh confirmation on one v5e cohort.

N5-screen explores every preregistered candidate, preserving failures. N5-confirm
loads immutable selections from that screen and measures selected classical,
Strassen and native kernels in a new execution. Both complete-call and prepared
kernel scopes are required. N7-screen/confirm reuse the same protocol on a
separately frozen held-out shape set, only after a frozen selector exists.

This is a Gaussian-input performance study, not a shape-only guarantee of
numerical fidelity. In particular, the earlier cancellation failures remain
valid evidence; this runner never relaxes correctness gates to select a winner.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from strassen_mm import benchmark_v001 as base


PHASES = ("N5-screen", "N5-confirm", "N7-screen", "N7-confirm")
FAMILIES = ("cubic", "strassen")
INTERNAL_ARMS = {"cubic": "cubic_basic", "strassen": "strassen_basic", "native": "native_xla"}


def resolve_input_path(campaign_path, argument, configured):
    if argument is not None:
        return argument.resolve()
    if configured:
        return (campaign_path.parent / configured).resolve()
    return None


def validate_candidate_space(experiment):
    families = experiment["candidate_families"]
    if set(families) != set(FAMILIES) or not families["cubic"]:
        raise ValueError("candidate_families must contain nonempty cubic and strassen lists")
    if len(families["cubic"]) != len(families["strassen"]):
        raise ValueError("classical and Strassen must receive equal candidate attempt budgets")
    names = set()
    for family in FAMILIES:
        for item in families[family]:
            key = item["candidate_id"]
            if key in names:
                raise ValueError(f"duplicate candidate id: {key}")
            names.add(key)
            allowed = {"cubic_full", "cubic_quadrant"} if family == "cubic" else {"strassen"}
            if item["algorithm"] not in allowed:
                raise ValueError(f"candidate algorithm does not belong to family {family}")
            tile = item["tile"]
            if len(tile) != 3 or any(type(x) is not int or x <= 0 for x in tile):
                raise ValueError("candidate tile must contain three positive integers")
            if item["algorithm"] == "cubic_full" and item["variant"] not in ("plain", "output_accumulator"):
                raise ValueError("full-tile cubic has no meaningful per-panel product-interleaving variant")
    return len(families["cubic"])


def shape_order(shapes):
    return sorted(shapes, key=lambda s: (s.get("stage") == "anchor",
                                        2 * (s["m"] * s["k"] + s["k"] * s["n"])
                                        + 8 * s["m"] * s["n"], s["id"]))


def make_arm(candidate, family, public_id=None):
    return {**copy.deepcopy(candidate), "family": family,
            "arm_id": INTERNAL_ARMS[family],
            "public_arm_id": public_id or ("native_xla" if family == "native" else candidate["candidate_id"])}


def build_groups(campaign, manifest, phase, selection=None):
    experiment = campaign["experiments"][phase]
    shapes = {item["id"]: item for item in manifest["shapes"]}
    chosen = shape_order([shapes[key] for key in experiment["shape_ids"]])
    seed = experiment.get("seed", campaign["seed"])
    input_case = {"distribution": experiment.get("distribution", "gaussian"), "seed": seed}
    if input_case["distribution"] != "gaussian":
        raise ValueError("v001 performance tuning is preregistered for the Gaussian cohort")
    native = experiment.get("native", {"candidate_id": "native", "algorithm": "native",
                                        "variant": "plain", "tile": None})
    groups = []
    screen = phase.endswith("screen")
    if screen:
        budget = validate_candidate_space(experiment)
    for shape_index, shape in enumerate(chosen):
        if screen:
            rng = base.np.random.Generator(base.np.random.PCG64(seed + shape_index * 7919))
            shuffled = {family: [experiment["candidate_families"][family][int(i)]
                                 for i in rng.permutation(budget)] for family in FAMILIES}
            batches = []
            for index in range(budget):
                arms = [make_arm(shuffled[family][index], family) for family in FAMILIES]
                if index == 0:
                    arms.append(make_arm(native, "native"))
                batches.append((f"candidate_pair_{index:03d}", arms))
        else:
            if selection is None or shape["id"] not in selection.get("by_shape", {}):
                raise ValueError(f"confirmation selection missing shape: {shape['id']}")
            row = selection["by_shape"][shape["id"]]
            if row.get("shape_mkn") != [shape["m"], shape["k"], shape["n"]]:
                raise ValueError("confirmation shape differs from selected shape")
            arms = [make_arm(native, "native")]
            for family in FAMILIES:
                candidate = row.get(family)
                if candidate is None:
                    candidate = {"candidate_id": f"{family}_no_eligible_candidate",
                                 "algorithm": "cubic_full" if family == "cubic" else "strassen",
                                 "variant": "plain", "tile": [256, 256, 256],
                                 "predeclared_error": "unsupported",
                                 "error_message": "No eligible screen candidate; no adaptive replacement."}
                arms.append(make_arm(candidate, family, f"{family}_selected"))
            batches = [("fresh_confirmation", arms)]
        for suffix, arms in batches:
            groups.append({"group_id": f"{shape['id']}__{suffix}", "shape": shape,
                           "arms": arms, "inputs": [dict(input_case)],
                           "timing": experiment["timing"], "scopes": ["call", "prepared_kernel"],
                           "tile": next(a["tile"] for a in arms if a["tile"] is not None)})
    return groups


def compact_selection_result(fields):
    correctness = fields.get("correctness") or {}
    return {key: fields.get(key) for key in (
        "shape_id", "shape_mkn", "arm_id", "candidate_id", "family", "algorithm",
        "variant", "tile", "scope", "status", "timing", "group_id", "case_id",
    )} | {"correctness_pass": correctness.get("pass", False),
          "relative_l2": correctness.get("relative_l2"),
          "reference_scope": correctness.get("reference_scope")}


class TuningJournal(base.Journal):
    def __init__(self, directory, phase):
        super().__init__(directory, phase)
        self.groups = {}
        self.selection_rows = []

    def emit(self, event, **fields):
        group = self.groups.get(fields.get("group_id"))
        if group:
            shape = group["shape"]
            fields.setdefault("shape_id", shape["id"])
            fields.setdefault("shape_mkn", [shape["m"], shape["k"], shape["n"]])
            names = {arm["arm_id"]: arm["public_arm_id"] for arm in group["arms"]}
            arm = next((a for a in group["arms"] if a["arm_id"] == fields.get("arm_id")), None)
            if arm:
                for key in ("candidate_id", "family", "algorithm", "variant", "tile"):
                    fields.setdefault(key, arm.get(key))
                fields["arm_id"] = arm["public_arm_id"]
            if fields.get("comparisons"):
                fields["comparisons"] = [dict(item, reference_arm=names.get(item["reference_arm"], item["reference_arm"]))
                                         for item in fields["comparisons"]]
        if event == "case_result":
            self.selection_rows.append(compact_selection_result(fields))
            fields.setdefault("fidelity_scope", "Gaussian-input gate only; shape does not establish universal numerical accuracy")
        super().emit(event, **fields)


def select_winners(rows, campaign, manifest, phase):
    experiment = campaign["experiments"][phase]
    budget = validate_candidate_space(experiment)
    required_samples = campaign["timing"][experiment["timing"]]["repeats"]
    shapes = {row["id"]: row for row in manifest["shapes"]}
    by_shape = {}
    for shape_id in experiment["shape_ids"]:
        shape = shapes[shape_id]
        outcomes = {"shape_mkn": [shape["m"], shape["k"], shape["n"]],
                    "attempt_counts": {}, "eligible_candidate_counts": {}, "candidate_attempts": {}}
        shape_rows = [row for row in rows if row.get("shape_id") == shape_id]
        for family in (*FAMILIES, "native"):
            candidates = ([experiment.get("native", {"candidate_id": "native", "algorithm": "native", "variant": "plain", "tile": None})]
                          if family == "native" else experiment["candidate_families"][family])
            eligible, attempts = [], []
            for candidate in candidates:
                observed = {row["scope"]: row for row in shape_rows
                            if row.get("family") == family and row.get("candidate_id") == candidate["candidate_id"]}
                qualifies = len(observed) == 2 and all(
                    row.get("status") == "ok" and row.get("correctness_pass")
                    and (row.get("timing") or {}).get("sample_count") == required_samples
                    and isinstance((row.get("timing") or {}).get("mean_ms"), (int, float))
                    and math.isfinite(row["timing"]["mean_ms"]) and row["timing"]["mean_ms"] > 0
                    for row in observed.values())
                attempt = {**candidate, "eligible": qualifies,
                           "scope_status": {scope: observed.get(scope, {}).get("status", "not_run")
                                            for scope in ("call", "prepared_kernel")},
                           "call_mean_ms": (observed.get("call", {}).get("timing") or {}).get("mean_ms"),
                           "prepared_mean_ms": (observed.get("prepared_kernel", {}).get("timing") or {}).get("mean_ms")}
                attempts.append(attempt)
                if qualifies:
                    call = observed["call"]
                    eligible.append({**candidate, "mean_ms": call["timing"]["mean_ms"],
                                     "prepared_mean_ms": observed["prepared_kernel"]["timing"]["mean_ms"],
                                     "relative_l2": call["relative_l2"], "reference_scope": call["reference_scope"],
                                     "screen_record_id": {"group_id": call["group_id"], "case_id": call["case_id"],
                                                          "arm_id": call["arm_id"], "scope": "call"},
                                     "selected_on": "minimum complete-call arithmetic mean among two-scope eligible candidates"})
            outcomes[family] = min(eligible, key=lambda item: (item["mean_ms"], item["candidate_id"])) if eligible else None
            outcomes["attempt_counts"][family] = len(attempts)
            outcomes["eligible_candidate_counts"][family] = len(eligible)
            outcomes["candidate_attempts"][family] = attempts
        outcomes["selection_status"] = ("selected" if all(outcomes[f] is not None for f in FAMILIES)
                                         else "no_eligible_candidate_in_one_or_more_families")
        by_shape[shape_id] = outcomes
    return {"schema_version": 1, "stage": "screen", "phase": phase,
            "qualification": "Gaussian-input performance selection; not a universal numerical guarantee",
            "selection_scope": "call", "required_eligible_scopes": ["call", "prepared_kernel"],
            "required_samples_per_scope": required_samples, "budget_per_family_per_shape": budget,
            "tie_break": "candidate_id lexical order", "by_shape": by_shape}


def verify_selection(selection, campaign, manifest, phase, identity):
    expected_screen = phase.replace("confirm", "screen")
    if selection.get("phase") != expected_screen or selection.get("stage") != "screen":
        raise ValueError("confirmation requires a selection from its corresponding screen phase")
    if selection.get("environment_identity") != identity:
        raise ValueError("screen and confirmation identities differ; start a separate cohort")
    screen = campaign["experiments"][expected_screen]
    validate_candidate_space(screen)
    known = {family: {c["candidate_id"]: c for c in screen["candidate_families"][family]}
             for family in FAMILIES}
    for shape_id in campaign["experiments"][phase]["shape_ids"]:
        if shape_id not in selection.get("by_shape", {}):
            raise ValueError(f"selection is missing required shape {shape_id}")
        for family in FAMILIES:
            chosen = selection["by_shape"][shape_id].get(family)
            if chosen is None:
                continue
            expected = known[family].get(chosen.get("candidate_id"))
            if expected is None or any(chosen.get(key) != expected[key] for key in ("algorithm", "variant", "tile")):
                raise ValueError("selected candidate is outside the preregistered search space")
    return True


def verify_heldout_selector(selector, manifest, shape_ids, identity=None):
    """Reject training-shape reuse or a silently changed held-out reserve."""
    training = {tuple(shape) for shape in selector.get("training_shapes", [])}
    reserved = {tuple(shape) for shape in selector.get("reserved_heldout_shapes", [])}
    if not training or not reserved:
        raise ValueError("frozen selector must declare training and reserved held-out geometries")
    shapes = {shape["id"]: tuple(shape[axis] for axis in ("m", "k", "n"))
              for shape in manifest["shapes"]}
    requested = {shapes[key] for key in shape_ids}
    if requested & training:
        raise ValueError("held-out phase includes a selector training geometry")
    if not requested.issubset(reserved):
        raise ValueError("held-out phase includes geometry outside the frozen reserve")
    if identity is not None and selector.get("source_identity") != identity:
        raise ValueError("selector training and held-out evaluation identities differ")
    return True


def group_memory_estimate(group):
    shape = tuple(group["shape"][axis] for axis in ("m", "k", "n"))
    m, k, n = shape
    original_bytes = 2 * (m * k + k * n)
    prepared, largest_pad, largest_output = {}, original_bytes, 4 * m * n
    for arm in group["arms"]:
        if arm["algorithm"] == "native":
            continue
        bm, bn, bk = arm["tile"]
        mp, kp, nn = (((x + b - 1) // b) * b for x, b in zip(shape, (bm, bk, bn)))
        amount = 2 * (mp * kp + kp * nn)
        largest_pad = max(largest_pad, amount)
        largest_output = max(largest_output, 4 * mp * nn)
        if (mp, kp, nn) != shape:
            prepared[(mp, kp, nn)] = amount
    estimate = original_bytes + largest_pad + sum(prepared.values()) + 2 * largest_output + 8 * 128 * 128
    return {"estimated_live_device_bytes": estimate, "original_input_bytes": original_bytes,
            "distinct_prepared_input_bytes": sum(prepared.values()),
            "largest_padded_input_bytes": largest_pad,
            "reserved_output_and_temporary_bytes": 2 * largest_output,
            "estimate_is_not_measured_peak": True, "unmeasured": "executable/runtime peak memory"}


class TuningRunner(base.Runner):
    def compile_entries(self, group, a, b):
        shape = tuple(group["shape"][axis] for axis in ("m", "k", "n"))
        entries = []
        for arm in group["arms"]:
            self.check_deadline()
            context = {"group_id": group["group_id"], "arm_id": arm["arm_id"]}
            if arm.get("predeclared_error"):
                entries.extend({**context, **arm, "scope": scope, "error": arm["predeclared_error"],
                                "error_message": arm["error_message"]} for scope in group["scopes"])
                continue
            try:
                fn = base.make_matmul(arm["algorithm"], shape, arm["tile"], variant=arm["variant"],
                                      interpret=False, vmem_limit_bytes=(None if arm["algorithm"] == "native" else
                                          self.campaign["memory"]["kernel_vmem_limit_mib"] * 1024**2))
            except Exception as error:
                self.emit_error(context, error, "compile")
                entries.extend({**context, **arm, "scope": scope, "error": base.error_status(error),
                                "error_message": str(error)} for scope in group["scopes"])
                continue
            for scope in group["scopes"]:
                self.check_deadline()
                entry = {**context, **arm, "scope": scope, "fn": fn, "metadata": fn.metadata}
                started = time.perf_counter_ns()
                try:
                    if scope == "call":
                        specs = (base.jax.ShapeDtypeStruct(a.shape, a.dtype), base.jax.ShapeDtypeStruct(b.shape, b.dtype))
                        executable = base.jax.jit(fn).lower(*specs).compile()
                    else:
                        mp, kp, nn = fn.metadata["padded_shape_mkn"]
                        specs = (base.jax.ShapeDtypeStruct((mp, kp), a.dtype), base.jax.ShapeDtypeStruct((kp, nn), b.dtype))
                        executable = base.jax.jit(fn.kernel).lower(*specs).compile()
                    entry["executable"] = executable
                    self.journal.emit("compilation", **context, scope=scope, status="ok",
                                      compile_ms=(time.perf_counter_ns() - started) / 1e6,
                                      kernel_metadata=fn.metadata)
                except Exception as error:
                    entry.update(error=base.error_status(error), error_message=str(error))
                    self.emit_error({**context, "scope": scope}, error, "compile")
                entries.append(entry)
        return entries

    def run_group(self, group):
        estimate = group_memory_estimate(group)
        self.journal.emit("memory_preflight", group_id=group["group_id"], **estimate)
        if estimate["estimated_live_device_bytes"] > self.campaign["memory"]["estimated_live_device_budget_gib"] * 1024**3:
            for arm in group["arms"]:
                for scope in group["scopes"]:
                    self.journal.emit("case_result", group_id=group["group_id"], **group["inputs"][0],
                                      arm_id=arm["arm_id"], scope=scope, status="skipped_memory_preflight",
                                      eligible_for_speedup_claim=False, memory_estimate=estimate)
            return
        shape = tuple(group["shape"][axis] for axis in ("m", "k", "n"))
        entries = None
        try:
            a, b = base.generate_inputs(shape, **group["inputs"][0])
            entries = self.compile_entries(group, a, b)
            self.execute_case(group, group["inputs"][0], entries, a, b)
            del a, b
        finally:
            del entries
            base.jax.clear_caches()
            base.gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-identity", type=Path, required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selector", type=Path)
    parser.add_argument("--max-wall-seconds", type=float, default=21600)
    args = parser.parse_args(argv)
    if args.max_wall_seconds <= 0:
        parser.error("--max-wall-seconds must be positive")
    args.campaign = args.campaign.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir
    journal = TuningJournal(out, args.phase)
    start, planned, done, completed, error_summary = time.monotonic(), [], [], False, None
    selection_output = None
    try:
        campaign = json.loads(args.campaign.read_text())
        experiment = campaign["experiments"][args.phase]
        shape_path = (args.campaign.parent / experiment.get("shape_manifest", campaign["shape_manifest"])).resolve()
        distribution_path = (args.campaign.parent / campaign.get("distribution_manifest", "distributions_v1.json")).resolve()
        manifest = json.loads(shape_path.read_text())
        if campaign["precision"]["native_precision"] != "DEFAULT" or campaign["device"]["target"] != "v5e":
            raise ValueError("N5/N7 v001 requires the qualified DEFAULT BF16 v5e contract")
        source_manifest = base.snapshot_sources(out, args.campaign, shape_path, distribution_path)
        selection_path = resolve_input_path(args.campaign, args.selection, experiment.get("selection_file"))
        selection = None
        if args.phase.endswith("confirm"):
            if selection_path is None:
                raise ValueError("confirmation requires --selection or configured selection_file")
            selection = json.loads(selection_path.read_text())
            base.exclusive_json(out / "selection_input.json", selection)
            base.exclusive_json(out / "selection_input_provenance.json", {
                "source_path": str(selection_path), "sha256": base.digest_file(selection_path)})
        selector_path = resolve_input_path(args.campaign, args.selector, experiment.get("selector_frozen_file"))
        selector_hash = None
        if args.phase.startswith("N7"):
            if selector_path is None:
                raise ValueError("held-out evaluation requires a previously frozen selector before any timing")
            selector = json.loads(selector_path.read_text())
            verify_heldout_selector(selector, manifest, experiment["shape_ids"])
            selector_hash = base.digest_file(selector_path)
            base.exclusive_json(out / "frozen_selector_input.json", selector)
            base.exclusive_json(out / "selector_input_provenance.json", {
                "source_path": str(selector_path), "sha256": selector_hash,
                "read_before_any_compilation_or_timing": True,
                "heldout_shape_manifest_sha256": base.digest_file(shape_path)})
        journal.emit("run_start", allocation_id=args.allocation_id, source_manifest=source_manifest,
                     argv=sys.argv, max_wall_seconds=args.max_wall_seconds,
                     selection_sha256=base.digest_file(selection_path) if selection_path else None,
                     frozen_selector_sha256=selector_hash)
        if "jax" in sys.modules:
            raise RuntimeError("run in a fresh Python process before importing JAX")
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
        if args.phase.startswith("N7"):
            verify_heldout_selector(selector, manifest, experiment["shape_ids"], environment["identity"])
        if selection is not None:
            verify_selection(selection, campaign, manifest, args.phase, environment["identity"])
            if args.phase.startswith("N7") and selection.get("frozen_selector_sha256") != selector_hash:
                raise ValueError("N7 screen and confirmation require the identical frozen selector")
        planned = build_groups(campaign, manifest, args.phase, selection)
        journal.groups = {group["group_id"]: group for group in planned}
        base.exclusive_json(out / "planned_cases.json", planned)
        runner = TuningRunner(args, campaign, journal)
        print(f"{args.phase}: {len(planned)} groups; fixed v5e cohort {args.allocation_id}", flush=True)
        for index, group in enumerate(planned):
            runner.check_deadline()
            print(f"[{index + 1}/{len(planned)}] {group['group_id']} starting", flush=True)
            journal.emit("group_start", group_id=group["group_id"], index=index + 1, total=len(planned))
            runner.run_group(group)
            done.append(group["group_id"])
            journal.emit("group_complete", group_id=group["group_id"], index=index + 1, total=len(planned))
            print(f"[{index + 1}/{len(planned)}] complete; {dict(journal.status_counts)}", flush=True)
        if args.phase.endswith("screen"):
            selection_output = select_winners(journal.selection_rows, campaign, manifest, args.phase)
            selection_output.update(created_utc=base.utc_now(), environment_identity=environment["identity"],
                              campaign_sha256=base.digest_file(args.campaign),
                              shape_manifest_sha256=base.digest_file(shape_path),
                              source_manifest_sha256=base.digest_file(out / "source_manifest.json"),
                              frozen_selector_sha256=selector_hash)
        completed = True
    except BaseException as error:
        error_summary = {"type": type(error).__name__, "message": str(error),
                         "status": base.error_status(error, "execute")}
        journal.emit("run_error", **error_summary, traceback=traceback.format_exc())
        print(f"{args.phase} stopped: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    finally:
        summary = {"phase": args.phase, "completed": completed,
                   "status": "completed" if completed else "failed_or_interrupted",
                   "finished_utc": base.utc_now(), "wall_seconds": time.monotonic() - start,
                   "completed_groups": done, "planned_group_count": len(planned),
                   "not_completed_group_ids": [group["group_id"] for group in planned if group["group_id"] not in done],
                   "case_status_counts": dict(journal.status_counts), "error": error_summary,
                   "scientific_scope": "bounded fair-attempt Gaussian-input tuning/confirmation; no universal accuracy claim",
                   "screen_vs_confirm": "separate executions; screens are not confirmation"}
        journal.emit("run_complete", **summary)
        journal.close()
        base.exclusive_json(out / "summary.json", summary)
        if completed and selection_output is not None:
            selection_output["screen_results_sha256"] = base.digest_file(out / "results.jsonl")
            selection_output["results_hash_scope"] = "complete sealed journal including run_complete"
            base.exclusive_json(out / "selections.json", selection_output)
        hashes = {str(path.relative_to(out)): base.digest_file(path) for path in sorted(out.rglob("*")) if path.is_file()}
        base.exclusive_json(out / "artifact_manifest.json", {"sha256": hashes, "sealed_utc": base.utc_now()})
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
