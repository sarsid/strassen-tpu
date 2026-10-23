"""Immutable power/midpoint-grid tuning and independent confirmation on one v5e.

Native tiles are compiler-managed: native candidates vary only explicit per-compile
options. Custom candidates vary tiles with a fixed implementation per family.
Screen winners are frozen before confirmation. Near-optimal tuple sets are discrete,
not assertions that the Cartesian box between their extrema is fast.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

from strassen_mm import benchmark_v001 as base
from strassen_mm import benchmark_n5_v001 as n5

PHASES = ("GRID-smoke", "GRID-screen", "GRID-confirm")
FAMILIES = ("native", "cubic", "strassen")
CANDIDATE_FIELDS = ("candidate_id", "algorithm", "variant", "tile", "compiler_options")
NATIVE_OPTION = "xla_tpu_scoped_vmem_limit_kib"


def candidate_spec(candidate):
    return {key: copy.deepcopy(candidate.get(key, {} if key == "compiler_options" else None))
            for key in CANDIDATE_FIELDS}


def selection_policy(experiment):
    return {"top_k": experiment.get("shortlist_top_k", 3),
            "near_fraction": experiment.get("near_optimal_fraction", .05),
            "max_confirm_per_family": experiment.get("shortlist_max_per_family", 6),
            **experiment.get("selection_policy", {})}


def validate_candidate_space(experiment):
    families = experiment["candidate_families"]
    if set(families) != set(FAMILIES) or any(not families[f] for f in FAMILIES):
        raise ValueError("candidate_families requires nonempty native, cubic, strassen lists")
    if len(families["cubic"]) != len(families["strassen"]):
        raise ValueError("custom families require equal candidate attempt budgets")
    names, native_defaults = set(), 0
    for family in FAMILIES:
        for candidate in families[family]:
            item = candidate_spec(candidate)
            name = item["candidate_id"]
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("candidate IDs must be nonempty and globally unique")
            names.add(name)
            options = item["compiler_options"]
            if not isinstance(options, dict):
                raise ValueError("compiler_options must be an object")
            if family == "native":
                if item["algorithm"] != "native" or item["variant"] != "plain" or item["tile"] is not None:
                    raise ValueError("native requires plain algorithm=native, tile=null; tiles are compiler-managed")
                if set(options) - {NATIVE_OPTION}:
                    raise ValueError("unregistered native compiler option")
                if options and (type(options[NATIVE_OPTION]) is not int or options[NATIVE_OPTION] <= 0):
                    raise ValueError("native scoped VMEM option must be a positive integer KiB value")
                native_defaults += not options
                continue
            algorithm, variant, alignment = (("cubic_full", "output_accumulator", (8, 128, 128))
                                              if family == "cubic" else
                                              ("strassen", "interleaved_output_accumulator", (16, 256, 256)))
            if item["algorithm"] != algorithm or item["variant"] != variant or options:
                raise ValueError("custom families require their fixed variant and no compiler_options")
            tile = item["tile"]
            if not isinstance(tile, (list, tuple)) or len(tile) != 3 or any(
                type(x) is not int or x <= 0 or x % a for x, a in zip(tile, alignment)
            ):
                raise ValueError(f"{family} tile BM,BN,BK must have alignments {alignment}")
    if native_defaults != 1:
        raise ValueError("exactly one native default candidate with empty compiler_options is required")
    policy = selection_policy(experiment)
    top, cap = policy.get("top_k", 3), policy.get("max_confirm_per_family", 6)
    near = policy.get("near_fraction", .05)
    if type(top) is not int or type(cap) is not int or not 1 <= top <= cap:
        raise ValueError("selection policy requires 1 <= top_k <= max_confirm_per_family")
    if not isinstance(near, (int, float)) or not math.isfinite(near) or near < 0:
        raise ValueError("near_fraction must be finite and nonnegative")
    return {family: len(families[family]) for family in FAMILIES}


def native_default(experiment):
    return next(c for c in experiment["candidate_families"]["native"] if not c.get("compiler_options", {}))


def make_arm(candidate, family, *, baseline=False):
    # The inherited comparison engine recognizes these two baseline names.
    arm_id = ("native_xla" if family == "native" and not candidate.get("compiler_options", {}) else
              "cubic_basic" if family == "cubic" and baseline else
              f"{family}__{candidate['candidate_id']}")
    return {**candidate_spec(candidate), "family": family, "arm_id": arm_id}


def build_groups(campaign, manifest, phase, selection=None):
    experiment = campaign["experiments"][phase]
    screen = campaign["experiments"].get("GRID-screen", experiment)
    validate_candidate_space(screen)
    shapes = {s["id"]: s for s in manifest["shapes"]}
    chosen = n5.shape_order([shapes[key] for key in experiment["shape_ids"]])
    seed = experiment.get("seed", campaign["seed"])
    distribution = experiment.get("distribution", "gaussian")
    if distribution != "gaussian":
        raise ValueError("this performance phase requires the preregistered Gaussian cohort")
    groups = []
    for index, shape in enumerate(chosen):
        batches = []
        if phase != "GRID-confirm":
            space = experiment.get("candidate_families", screen["candidate_families"])
            validate_candidate_space({**screen, "candidate_families": space})
            shuffled = copy.deepcopy(space)
            rng = random.Random(seed + index * 7919)
            for family in FAMILIES:
                rng.shuffle(shuffled[family])
            for i in range(max(map(len, shuffled.values()))):
                arms = [make_arm(shuffled[f][i], f, baseline=f == "cubic")
                        for f in FAMILIES if i < len(shuffled[f])]
                batches.append((f"candidate_batch_{i:03d}", arms))
        else:
            if selection is None or shape["id"] not in selection.get("by_shape", {}):
                raise ValueError(f"missing frozen selection for {shape['id']}")
            row = selection["by_shape"][shape["id"]]
            headline = []
            for family in FAMILIES:
                winner = row[family].get("winner")
                if winner:
                    headline.append(make_arm(winner, family, baseline=True))
            default = native_default(screen)
            if default["candidate_id"] not in {a["candidate_id"] for a in headline}:
                headline.append(make_arm(default, "native"))
            batches = [("headline_confirmation", headline)]
            # Keep buffers bounded and compare each finalist to its frozen winner
            # in the same timed group. Repeated references are never pooled.
            for family in FAMILIES:
                winner = row[family].get("winner")
                if not winner:
                    continue
                for candidate in row[family]["confirmation_candidates"]:
                    if candidate["candidate_id"] == winner["candidate_id"]:
                        continue
                    if family == "native" and candidate["candidate_id"] == default["candidate_id"]:
                        continue  # Already paired with the native winner in headline.
                    batches.append((f"finalist_{family}_{candidate['candidate_id']}",
                                    [make_arm(winner, family, baseline=True), make_arm(candidate, family)]))
        for suffix, arms in batches:
            if len({a["arm_id"] for a in arms}) != len(arms):
                raise ValueError("duplicate internal arm IDs in one group")
            groups.append({"group_id": f"{shape['id']}__{suffix}", "shape": shape, "arms": arms,
                           "inputs": [{"distribution": distribution, "seed": seed}],
                           "timing": experiment["timing"], "scopes": ["call", "prepared_kernel"],
                           "tile": next((a["tile"] for a in arms if a["tile"] is not None), [256, 256, 256])})
    return groups


class GridJournal(base.Journal):
    def __init__(self, directory, phase):
        super().__init__(directory, phase)
        self.groups, self.selection_rows, self.samples = {}, [], {}

    def emit(self, event, **fields):
        group = self.groups.get(fields.get("group_id"))
        if group:
            fields.setdefault("shape_id", group["shape"]["id"])
            fields.setdefault("shape_mkn", [group["shape"][d] for d in ("m", "k", "n")])
            arm = next((a for a in group["arms"] if a["arm_id"] == fields.get("arm_id")), None)
            if arm:
                for key in (*CANDIDATE_FIELDS, "family"):
                    fields.setdefault(key, arm.get(key))
        if event == "sample":
            key = (fields["shape_id"], fields["candidate_id"], fields["scope"])
            self.samples.setdefault((fields["group_id"], *key), []).append({k: fields[k] for k in ("round", "elapsed_ms")})
        if event == "case_result":
            row = n5.compact_selection_result(fields)
            row["compiler_options"] = fields.get("compiler_options", {})
            self.selection_rows.append(row)
            fields.setdefault("fidelity_scope", "Gaussian-input gate only; not a universal accuracy guarantee")
        super().emit(event, **fields)


def eligible_rows(rows, shape_id, family, candidate_id, repeats, group_id=None):
    observed = [r for r in rows if r.get("shape_id") == shape_id and r.get("family") == family
                and r.get("candidate_id") == candidate_id
                and (group_id is None or r.get("group_id") == group_id)]
    # Duplicate results cannot silently replace one another.
    by_scope = {r["scope"]: r for r in observed}
    if len(observed) != 2 or set(by_scope) != {"call", "prepared_kernel"}:
        return None
    for row in observed:
        timing = row.get("timing") or {}
        value = timing.get("mean_ms")
        if not (row.get("status") == "ok" and row.get("correctness_pass")
                and timing.get("sample_count") == repeats and isinstance(value, (int, float))
                and math.isfinite(value) and value > 0):
            return None
    return by_scope


def tile_projection(candidates):
    tuples = [list(c["tile"]) for c in candidates if c.get("tile") is not None]
    return {"tuples_bm_bn_bk": tuples,
            "axis_values": {axis: sorted({t[i] for t in tuples}) for i, axis in enumerate(("bm", "bn", "bk"))},
            "scope": "Discrete tested tuples only; axis projections do not certify their Cartesian product."}


def select_winners(rows, campaign, manifest, phase="GRID-screen"):
    experiment = campaign["experiments"][phase]
    counts = validate_candidate_space(experiment)
    repeats = campaign["timing"][experiment["timing"]]["repeats"]
    policy = selection_policy(experiment)
    shapes = {s["id"]: s for s in manifest["shapes"]}
    result = {}
    for shape_id in experiment["shape_ids"]:
        shape = shapes[shape_id]
        row = {"shape_mkn": [shape[d] for d in ("m", "k", "n")]}
        for family in FAMILIES:
            eligible, attempts = [], []
            for candidate in experiment["candidate_families"][family]:
                observed = eligible_rows(rows, shape_id, family, candidate["candidate_id"], repeats)
                candidate_rows = [r for r in rows if r.get("shape_id") == shape_id and r.get("family") == family
                                  and r.get("candidate_id") == candidate["candidate_id"]]
                attempts.append({**candidate_spec(candidate), "eligible": observed is not None,
                                 "scope_status": {scope: next((r.get("status") for r in candidate_rows if r.get("scope") == scope), "not_run")
                                                  for scope in ("call", "prepared_kernel")}})
                if observed:
                    eligible.append({**candidate_spec(candidate),
                                     "mean_ms": observed["call"]["timing"]["mean_ms"],
                                     "prepared_mean_ms": observed["prepared_kernel"]["timing"]["mean_ms"]})
            eligible.sort(key=lambda c: (c["mean_ms"], c["candidate_id"]))
            winner = eligible[0] if eligible else None
            near = [c for c in eligible if c["mean_ms"] <= winner["mean_ms"] * (1 + policy["near_fraction"])]
            desired = {c["candidate_id"] for c in eligible[:policy["top_k"]] + near}
            shortlist = [c for c in eligible if c["candidate_id"] in desired][:policy["max_confirm_per_family"]]
            confirmed_ids = {c["candidate_id"] for c in shortlist}
            row[family] = {"winner": winner, "confirmation_candidates": shortlist,
                           "screen_eligible_candidates": eligible, "candidate_attempts": attempts,
                           "screen_near_candidates": [{**c, "scheduled_for_confirmation": c["candidate_id"] in confirmed_ids}
                                                      for c in near],
                           "screen_near_tile_projection": tile_projection(near),
                           "near_status": "descriptive screen threshold; confirmation required"}
        result[shape_id] = row
    return {"schema_version": 1, "phase": phase, "stage": "screen", "by_shape": result,
            "selection_policy": policy, "candidate_attempt_budgets": counts,
            "required_samples_per_scope": repeats, "selection_scope": "call",
            "required_eligible_scopes": ["call", "prepared_kernel"],
            "headline_policy": "Frozen screen winner per family; do not reselect confirmation minimum",
            "native_tuning_scope": "Per-compile scoped VMEM options; tiles remain compiler-managed"}


def verify_selection(selection, campaign, manifest, identity):
    if selection.get("phase") != "GRID-screen" or selection.get("stage") != "screen":
        raise ValueError("confirmation requires a GRID-screen selection")
    if selection.get("environment_identity") != identity:
        raise ValueError("screen and confirmation environment identities differ")
    experiment = campaign["experiments"]["GRID-screen"]
    validate_candidate_space(experiment)
    shapes = {s["id"]: [s[d] for d in ("m", "k", "n")] for s in manifest["shapes"]}
    for shape_id in campaign["experiments"]["GRID-confirm"]["shape_ids"]:
        row = selection.get("by_shape", {}).get(shape_id)
        if not row or row.get("shape_mkn") != shapes[shape_id]:
            raise ValueError("frozen selection shape missing or changed")
        for family in FAMILIES:
            allowed = {c["candidate_id"]: candidate_spec(c) for c in experiment["candidate_families"][family]}
            candidates = row[family]["confirmation_candidates"]
            ids = [c["candidate_id"] for c in candidates]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate frozen confirmation candidate")
            winner = row[family].get("winner")
            if winner and winner["candidate_id"] not in ids:
                raise ValueError("frozen winner must be confirmed")
            for candidate in candidates + ([winner] if winner else []):
                if candidate_spec(candidate) != allowed.get(candidate["candidate_id"]):
                    raise ValueError("frozen candidate differs from preregistered candidate")
    return True


def confirmation_report(journal, selection, campaign):
    repeats = campaign["timing"][campaign["experiments"]["GRID-confirm"]["timing"]]["repeats"]
    report = {"schema_version": 1, "reference_policy": "Frozen screen winner, never confirmation minimum",
              "confidence_scope": "Pointwise paired bootstrap 95% intervals; not simultaneous or untested-tile guarantees",
              "by_shape": {}}
    for shape_id, selected in selection["by_shape"].items():
        if shape_id not in campaign["experiments"]["GRID-confirm"]["shape_ids"]:
            continue
        shape_report = {"shape_mkn": selected["shape_mkn"]}
        for family in FAMILIES:
            winner = selected[family]["winner"]
            candidates = list(selected[family]["confirmation_candidates"])
            default = native_default(campaign["experiments"]["GRID-screen"])
            if family == "native" and default["candidate_id"] not in {c["candidate_id"] for c in candidates}:
                candidates.append(default)
            family_report = {"frozen_winner": winner, "candidates": [], "near_sets": {}}
            for candidate in candidates:
                is_headline = (winner and candidate["candidate_id"] == winner["candidate_id"]) or (
                    family == "native" and candidate["candidate_id"] == default["candidate_id"])
                suffix = "headline_confirmation" if is_headline else f"finalist_{family}_{candidate['candidate_id']}"
                group_id = f"{shape_id}__{suffix}"
                reference = (eligible_rows(journal.selection_rows, shape_id, family, winner["candidate_id"], repeats, group_id)
                             if winner else None)
                observed = eligible_rows(journal.selection_rows, shape_id, family, candidate["candidate_id"], repeats, group_id)
                item = {**candidate_spec(candidate), "confirmed_eligible": observed is not None,
                        "reference_available": reference is not None, "group_id": group_id, "scopes": {}}
                for scope in ("call", "prepared_kernel"):
                    if reference and observed:
                        left = journal.samples.get((group_id, shape_id, winner["candidate_id"], scope), [])
                        right = journal.samples.get((group_id, shape_id, candidate["candidate_id"], scope), [])
                        pair = base.paired_comparison(left, right, campaign["seed"] + 711)
                        if pair:
                            lo, hi = pair["speedup_ci95"]
                            item["scopes"][scope] = {
                                "mean_ms": observed[scope]["timing"]["mean_ms"],
                                "latency_ratio_to_frozen_winner": 1 / pair["speedup_ratio_of_means"],
                                "latency_ratio_ci95": [1 / hi, 1 / lo],
                                "certified_within_5_percent": 1 / lo <= 1.05,
                                "method": pair["method"]}
                family_report["candidates"].append(item)
            for scope in ("call", "prepared_kernel"):
                measured = [c for c in family_report["candidates"] if scope in c["scopes"]]
                family_report["near_sets"][scope] = {
                    f"descriptive_{percent}_percent": tile_projection([
                        c for c in measured if c["scopes"][scope]["latency_ratio_to_frozen_winner"] <= 1 + percent / 100])
                    for percent in (1, 3, 5)}
                family_report["near_sets"][scope]["pointwise_ci95_certified_5_percent"] = tile_projection([
                    c for c in measured if c["scopes"][scope]["certified_within_5_percent"]])
            shape_report[family] = family_report
        # Explicit headline contrasts include tuned native even when it differs
        # from the inherited runner's native_xla baseline name.
        headline_group = f"{shape_id}__headline_confirmation"
        headline = [(f, selected[f]["winner"]) for f in FAMILIES if selected[f]["winner"]]
        default = native_default(campaign["experiments"]["GRID-screen"])
        if default["candidate_id"] not in {c["candidate_id"] for _, c in headline}:
            headline.append(("native", default))
        contrasts = []
        for family, candidate in headline:
            observed = eligible_rows(journal.selection_rows, shape_id, family, candidate["candidate_id"], repeats, headline_group)
            for ref_family, reference_candidate in headline:
                if reference_candidate["candidate_id"] == candidate["candidate_id"]:
                    continue
                reference = eligible_rows(journal.selection_rows, shape_id, ref_family, reference_candidate["candidate_id"], repeats, headline_group)
                if not observed or not reference:
                    continue
                for scope in ("call", "prepared_kernel"):
                    pair = base.paired_comparison(
                        journal.samples.get((headline_group, shape_id, reference_candidate["candidate_id"], scope), []),
                        journal.samples.get((headline_group, shape_id, candidate["candidate_id"], scope), []), campaign["seed"] + 811)
                    if pair:
                        contrasts.append({"candidate_id": candidate["candidate_id"], "family": family,
                                          "reference_candidate_id": reference_candidate["candidate_id"],
                                          "reference_family": ref_family, "scope": scope,
                                          "group_id": headline_group, **pair})
        shape_report["headline_comparisons"] = contrasts
        report["by_shape"][shape_id] = shape_report
    return report


class GridRunner(n5.TuningRunner):
    def compile_entries(self, group, a, b):
        shape = tuple(group["shape"][d] for d in ("m", "k", "n"))
        entries = []
        for arm in group["arms"]:
            self.check_deadline()
            context = {"group_id": group["group_id"], "arm_id": arm["arm_id"]}
            try:
                fn = base.make_matmul(arm["algorithm"], shape, arm["tile"], variant=arm["variant"],
                                      interpret=False, vmem_limit_bytes=(None if arm["algorithm"] == "native" else
                                          self.campaign["memory"]["kernel_vmem_limit_mib"] * 1024**2))
            except Exception as error:
                self.emit_error(context, error, "compile")
                entries.extend({**context, **arm, "scope": s, "error": base.error_status(error),
                                "error_message": str(error)} for s in group["scopes"])
                continue
            for scope in group["scopes"]:
                self.check_deadline()
                entry = {**context, **arm, "scope": scope, "fn": fn,
                         "metadata": {**fn.metadata, "compiler_options": arm.get("compiler_options", {})}}
                started = time.perf_counter_ns()
                try:
                    if scope == "call":
                        specs = (base.jax.ShapeDtypeStruct(a.shape, a.dtype), base.jax.ShapeDtypeStruct(b.shape, b.dtype))
                        lowered = base.jax.jit(fn).lower(*specs)
                    else:
                        mp, kp, nn = fn.metadata["padded_shape_mkn"]
                        specs = (base.jax.ShapeDtypeStruct((mp, kp), a.dtype), base.jax.ShapeDtypeStruct((kp, nn), b.dtype))
                        lowered = base.jax.jit(fn.kernel).lower(*specs)
                    options = arm.get("compiler_options", {})
                    entry["executable"] = lowered.compile(compiler_options=options) if options else lowered.compile()
                    self.journal.emit("compilation", **context, scope=scope, status="ok",
                                      compile_ms=(time.perf_counter_ns() - started) / 1e6,
                                      kernel_metadata=entry["metadata"], compiler_options=options)
                except Exception as error:
                    unsupported = bool(arm.get("compiler_options")) and any(
                        word in str(error).lower() for word in ("unknown", "unrecognized", "unsupported", "not supported", "no such"))
                    entry.update(error="unsupported_compiler_option" if unsupported else base.error_status(error), error_message=str(error))
                    self.emit_error({**context, "scope": scope}, error, "compile")
                entries.append(entry)
        return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-identity", type=Path, required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--max-wall-seconds", type=float, default=21600)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0:
        parser.error("--max-wall-seconds must be finite and positive")
    args.campaign = args.campaign.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out, start = args.output_dir, time.monotonic()
    journal = GridJournal(out, args.phase)
    planned, done, completed, error_summary, selection_output, confirmation = [], [], False, None, None, None
    try:
        campaign = json.loads(args.campaign.read_text())
        experiment = campaign["experiments"][args.phase]
        shape_path = (args.campaign.parent / experiment.get("shape_manifest", campaign["shape_manifest"])).resolve()
        distribution_path = (args.campaign.parent / campaign.get("distribution_manifest", "distributions_v1.json")).resolve()
        manifest = json.loads(shape_path.read_text())
        validate_candidate_space(campaign["experiments"]["GRID-screen"])
        if campaign["precision"]["native_precision"] != "DEFAULT" or campaign["device"]["target"] != "v5e":
            raise ValueError("runner requires DEFAULT BF16 qualified single-v5e contract")
        source_manifest = base.snapshot_sources(out, args.campaign, shape_path, distribution_path)
        selection = None
        if args.phase == "GRID-confirm":
            selection_path = n5.resolve_input_path(args.campaign, args.selection, experiment.get("selection_file"))
            if selection_path is None:
                raise ValueError("confirmation requires a frozen --selection")
            selection = json.loads(selection_path.read_text())
            if selection.get("campaign_sha256") != base.digest_file(args.campaign):
                raise ValueError("campaign changed after screen selection")
            if selection.get("shape_manifest_sha256") != base.digest_file(shape_path):
                raise ValueError("shape manifest changed after screen selection")
            source_results = selection_path.parent / "results.jsonl"
            if not source_results.is_file() or base.digest_file(source_results) != selection.get("screen_results_sha256"):
                raise ValueError("screen journal missing or inconsistent with frozen selection")
            screen_source_path = selection_path.parent / "source_manifest.json"
            if not screen_source_path.is_file() or base.digest_file(screen_source_path) != selection.get("source_manifest_sha256"):
                raise ValueError("screen source manifest missing or changed")
            screen_sources = json.loads(screen_source_path.read_text())["sha256"]
            python_hashes = lambda sources: {name: digest for name, digest in sources.items() if name.startswith("source_snapshot/")}
            if python_hashes(screen_sources) != python_hashes(source_manifest["sha256"]):
                raise ValueError("source files changed between screening and confirmation")
            screen_rows = [json.loads(line) for line in source_results.read_text().splitlines()]
            replay_rows = [dict(n5.compact_selection_result(row), compiler_options=row.get("compiler_options", {}))
                           for row in screen_rows if row.get("event") == "case_result"]
            replay = select_winners(replay_rows, campaign, manifest)
            if any(selection.get(key) != replay[key] for key in ("by_shape", "selection_policy")):
                raise ValueError("frozen selection differs from deterministic replay of sealed screen journal")
            base.exclusive_json(out / "selection_input.json", selection)
            base.exclusive_json(out / "selection_input_provenance.json", {
                "source_path": str(selection_path), "sha256": base.digest_file(selection_path)})
        journal.emit("run_start", allocation_id=args.allocation_id, source_manifest=source_manifest,
                     argv=sys.argv if argv is None else argv, max_wall_seconds=args.max_wall_seconds)
        if "jax" in sys.modules:
            raise RuntimeError("run in a fresh process before importing JAX")
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
        if selection is not None:
            verify_selection(selection, campaign, manifest, environment["identity"])
        planned = build_groups(campaign, manifest, args.phase, selection)
        journal.groups = {g["group_id"]: g for g in planned}
        base.exclusive_json(out / "planned_cases.json", planned)
        runner = GridRunner(args, campaign, journal)
        for index, group in enumerate(planned):
            runner.check_deadline()
            print(f"[{index + 1}/{len(planned)}] {group['group_id']}", flush=True)
            journal.emit("group_start", group_id=group["group_id"], index=index + 1, total=len(planned))
            runner.run_group(group)
            done.append(group["group_id"])
            journal.emit("group_complete", group_id=group["group_id"], index=index + 1, total=len(planned))
        if args.phase == "GRID-screen":
            selection_output = select_winners(journal.selection_rows, campaign, manifest)
            selection_output.update(created_utc=base.utc_now(), environment_identity=environment["identity"],
                                    campaign_sha256=base.digest_file(args.campaign),
                                    shape_manifest_sha256=base.digest_file(shape_path),
                                    source_manifest_sha256=base.digest_file(out / "source_manifest.json"))
        elif args.phase == "GRID-confirm":
            confirmation = confirmation_report(journal, selection, campaign)
        completed = True
    except BaseException as error:
        error_summary = {"type": type(error).__name__, "message": str(error), "status": base.error_status(error, "execute")}
        journal.emit("run_error", **error_summary, traceback=traceback.format_exc())
        print(f"{args.phase} stopped: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    finally:
        summary = {"phase": args.phase, "completed": completed, "status": "completed" if completed else "failed_or_interrupted",
                   "finished_utc": base.utc_now(), "wall_seconds": time.monotonic() - start,
                   "completed_groups": done, "planned_group_count": len(planned),
                   "not_completed_group_ids": [g["group_id"] for g in planned if g["group_id"] not in done],
                   "case_status_counts": dict(journal.status_counts), "error": error_summary,
                   "scientific_scope": "Sampled-shape Gaussian study; one Strassen level per tile; no generalization guarantee",
                   "screen_vs_confirm": "Separate executions; headline choices frozen in screen"}
        journal.emit("run_complete", **summary)
        journal.close()
        base.exclusive_json(out / "summary.json", summary)
        if completed and selection_output is not None:
            selection_output["screen_results_sha256"] = base.digest_file(out / "results.jsonl")
            base.exclusive_json(out / "selections.json", selection_output)
        if completed and confirmation is not None:
            base.exclusive_json(out / "confirmation.json", confirmation)
        hashes = {str(p.relative_to(out)): base.digest_file(p) for p in sorted(out.rglob("*")) if p.is_file()}
        base.exclusive_json(out / "artifact_manifest.json", {"sha256": hashes, "sealed_utc": base.utc_now()})
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
