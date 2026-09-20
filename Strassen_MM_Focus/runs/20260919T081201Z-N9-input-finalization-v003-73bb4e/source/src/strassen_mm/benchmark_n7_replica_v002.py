"""Fresh-v5e fixed-policy replication with explicit canonical smoke evidence.

Version 002 separates the launcher-copied expected identity from the full
canonical smoke artifacts; exact identity bytes must agree before reuse.

No allocation, release, selection fitting or retuning. Original selector and
alternative-selection bytes remain unchanged. Current runtime must match its
own newly qualified smoke evidence; explicit cross-cohort validation allows
only disclosed allocation/host/boot identity changes. Root orchestration owns
the requirement that original-cohort N9 finishes before this process runs.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from strassen_mm import benchmark_v001 as base
from strassen_mm import benchmark_n5_v001 as tuning
from strassen_mm import benchmark_n6_n7_v002 as evaluation
from strassen_mm import selector_v002 as selection_rule

PHASE = "N7-replicate"
ALLOWED_IDENTITY_CHANGES = frozenset({"allocation_id", "colab_endpoint", "hostname", "host_id", "boot_id"})
REUSED_MODULES = ("kernels_v002.py", "benchmark_v001.py", "benchmark_n5_v001.py", "benchmark_n6_n7_v002.py", "selector_v002.py")
PRECISION_CONTRACT = {"input_dtype": "bfloat16", "output_dtype": "float32", "accumulation_dtype": "float32", "dot_precision": "DEFAULT"}


def verify_completed_run(path, phase, count):
    directory = selection_rule.artifacts(path)
    seal_sha = selection_rule.verify_seal(directory)
    summary = selection_rule.load(directory / "summary.json")
    if summary.get("phase") != phase or summary.get("completed") is not True or summary.get("status") != "completed":
        raise ValueError(f"Expected a completed {phase} run")
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines() if line.strip()]
    results = [row for row in rows if row.get("event") == "case_result"]
    if len(results) != count:
        raise ValueError(f"Expected {count} {phase} result rows, observed {len(results)}")
    keys = [(row.get("group_id"), row.get("distribution"), row.get("seed"), row.get("arm_id"), row.get("scope")) for row in results]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate {phase} case_result keys")
    for row in results:
        metadata = row.get("kernel_metadata")
        if metadata is not None and any(metadata.get(key) != value for key, value in PRECISION_CONTRACT.items()):
            raise ValueError(f"{phase} evidence has a different arithmetic contract")
    environment = selection_rule.load(directory / "environment.json")
    return directory, environment, results, {"directory": str(directory), "phase": phase,
        "result_count": len(results), "seal_sha256": seal_sha,
        "results_sha256": base.digest_file(directory / "results.jsonl"),
        "environment_sha256": base.digest_file(directory / "environment.json")}



def qualify_canonical_smoke(smoke_environment_path, expected_identity_path):
    """Link the launcher's copied identity to the separately sealed smoke run."""
    smoke_environment_path = Path(smoke_environment_path).resolve()
    expected_identity_path = Path(expected_identity_path).resolve()
    if smoke_environment_path.name != "environment.json":
        raise ValueError("--smoke-environment must name the canonical smoke environment.json")
    if smoke_environment_path.read_bytes() != expected_identity_path.read_bytes():
        raise ValueError("Canonical smoke environment and launcher-copied expected identity bytes differ")
    smoke_dir, smoke_environment, smoke_results, smoke_provenance = verify_completed_run(
        smoke_environment_path.parent, "smoke", 24)
    if smoke_dir / "environment.json" != smoke_environment_path:
        raise ValueError("--smoke-environment must name the canonical smoke environment.json")
    if any(row.get("status") != "ok" or not (row.get("correctness") or {}).get("pass") for row in smoke_results):
        raise ValueError("New-cohort smoke did not pass every numerical/compilation case")
    identity_link = {"status": "byte_identical",
        "canonical_smoke_environment": str(smoke_environment_path),
        "launcher_expected_identity": str(expected_identity_path),
        "sha256": base.digest_file(smoke_environment_path)}
    return smoke_dir, smoke_environment, smoke_results, smoke_provenance, identity_link


def validate_setup_smoke(setup, smoke_environment):
    """Qualify new smoke against its own recorded setup, never the old cohort."""
    current = smoke_environment.get("identity", {})
    required = {"colab_endpoint", "hostname", "boot_id", "versions", "devices"}
    missing = sorted(key for key in required if key not in setup or key not in current)
    if missing:
        raise ValueError(f"New setup/smoke identity fields unavailable: {missing}")
    differences = {key: {"setup": setup[key], "smoke": current[key]} for key in sorted(required) if setup[key] != current[key]}
    if differences:
        raise ValueError("New setup and smoke disagree: " + json.dumps(differences, sort_keys=True))
    if not setup.get("colab_endpoint") or not setup.get("boot_id"):
        raise ValueError("New setup must expose its logical allocation and boot ID")
    if current.get("allocation_id") != setup["colab_endpoint"]:
        raise ValueError("Smoke allocation ID does not match its setup endpoint")
    if setup.get("backend") != "tpu" or setup.get("device_count") != 1:
        raise ValueError("New setup is not a single TPU allocation")
    return {"status": "matched", "fields": sorted(required), "setup_allocation": setup["colab_endpoint"]}


def validate_compatibility(original, current, precision):
    """Keep frozen old identity intact and separately validate compatibility."""
    expected_precision = {"input_dtype": "bfloat16", "output_dtype": "float32", "accumulator_dtype": "float32",
                          "pre_add_dtype": "bfloat16", "native_precision": "DEFAULT", "reference_dtype": "float32"}
    if any(precision.get(key) != value for key, value in expected_precision.items()):
        raise ValueError("Replication requires the frozen BF16/FP32/default precision contract")
    for label, environment in (("original", original), ("current", current)):
        if environment.get("qualified_single_v5e") is not True or environment.get("backend") != "tpu":
            raise ValueError(f"{label} environment is not qualified single v5e")
        if any(environment.get(key) != 1 for key in ("device_count", "local_device_count", "process_count")):
            raise ValueError(f"{label} environment does not have one device and one process")
    old, new = original["identity"], current["identity"]
    if not old.get("allocation_id") or not new.get("allocation_id") or old["allocation_id"] == new["allocation_id"]:
        raise ValueError("Fresh-cohort replication requires a distinct recorded allocation ID")
    required = ("device_kind", "device_id", "process_index", "jax_version", "jaxlib_version", "libtpu_version", "numpy_version",
                "versions", "runtime_flags", "mosaic_compatibility", "devices")
    if any(key not in old or key not in new for key in required):
        raise ValueError("A required software/device compatibility field is missing")
    changes = {key: {"original": old.get(key), "replica": new.get(key)} for key in sorted(set(old) | set(new)) if old.get(key) != new.get(key)}
    forbidden = {key: value for key, value in changes.items() if key not in ALLOWED_IDENTITY_CHANGES}
    if forbidden:
        raise ValueError("Incompatible replica environment: " + json.dumps(forbidden, sort_keys=True))
    return {"status": "compatible_new_v5e_cohort", "original_identity": old, "replica_identity": new,
            "allowed_changed_fields": sorted(ALLOWED_IDENTITY_CHANGES), "observed_identity_changes": changes,
            "matched_compatibility_fields": sorted(key for key in set(old) | set(new) if key not in changes),
            "precision_contract": precision, "runtime_identity_scope": "Separate logical allocations; physical chip serial unavailable.",
            "analysis_policy": "Compare within-cohort paired results and report cohorts separately; do not pool raw timings."}


def verify_original_policy(original_dir, original_environment, choices, rule, choices_path, rule_path):
    provenance = selection_rule.load(original_dir / "input_provenance.json")
    choices_sha, rule_sha = base.digest_file(choices_path), base.digest_file(rule_path)
    if provenance.get("selection_sha256") != choices_sha or provenance.get("frozen_selector_sha256") != rule_sha:
        raise ValueError("Provided choices/rule are not the exact bytes used by original N7 evaluation")
    identity = original_environment["identity"]
    if choices.get("environment_identity") != identity or rule.get("source_identity") != identity:
        raise ValueError("Original choices/rule do not belong to the original N7 evaluation cohort")
    if choices.get("phase") != "N7-screen" or choices.get("frozen_selector_sha256") != rule_sha:
        raise ValueError("Original alternatives were not screened against this exact frozen selector")
    return {"selection_sha256": choices_sha, "selector_sha256": rule_sha,
            "preservation": "Original bytes, identities, prototypes, parameters and selected tiles are not changed."}


def verify_reused_source(original_dir):
    manifest = selection_rule.load(original_dir / "source_manifest.json")
    package = Path(__file__).resolve().parent
    result = {}
    for name in REUSED_MODULES:
        expected = manifest["sha256"].get("source_snapshot/" + name)
        current = base.digest_file(package / name)
        if expected is None or current != expected:
            raise ValueError(f"Replication reused module differs from original execution: {name}")
        result[name] = current
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--phase", choices=(PHASE,), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--expected-identity", type=Path, required=True, help="Launcher-copied new smoke identity")
    parser.add_argument("--smoke-environment", type=Path, required=True, help="Canonical new smoke artifacts/environment.json with complete adjacent seal/results")
    parser.add_argument("--setup-identity", type=Path, required=True, help="New allocation's setup identity.json")
    parser.add_argument("--original-evaluation", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=21600)
    args = parser.parse_args(argv)
    if args.max_wall_seconds <= 0:
        parser.error("Wall-time budget must be positive")
    args.campaign = args.campaign.resolve()
    args.expected_identity = args.expected_identity.resolve()
    args.smoke_environment = args.smoke_environment.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir
    journal = evaluation.EvidenceJournal(out, PHASE)
    planned, done, completed, failure = [], [], False, None
    started = time.monotonic()
    try:
        campaign = selection_rule.load(args.campaign)
        if base.digest_file(args.campaign) != selection_rule.CONFIG_SHA256:
            raise ValueError("Frozen N5-N9 campaign changed")
        shape_path = args.campaign.parent / campaign["heldout_shape_manifest"]
        if base.digest_file(shape_path) != selection_rule.HELDOUT_SHA256:
            raise ValueError("Held-out reserve changed")
        manifest = selection_rule.load(shape_path)
        source_manifest = base.snapshot_sources(out, args.campaign, shape_path, args.campaign.parent / campaign["distribution_manifest"])
        original_dir, original_environment, original_results, original_provenance = verify_completed_run(args.original_evaluation, "N7-evaluate", 128)
        choices, rule = selection_rule.load(args.selection), selection_rule.load(args.selector)
        preservation = verify_original_policy(original_dir, original_environment, choices, rule, args.selection, args.selector)
        source_hashes = verify_reused_source(original_dir)
        # Preserve the exact supplied original bytes, not a reserialized object.
        for source, name in ((args.selection, "original_selections.json"), (args.selector, "original_selector.json")):
            with (out / name).open("xb") as handle:
                handle.write(source.read_bytes())
        smoke_dir, smoke_environment, smoke_results, smoke_provenance, smoke_identity_link = qualify_canonical_smoke(
            args.smoke_environment, args.expected_identity)
        setup = selection_rule.load(args.setup_identity)
        setup_match = validate_setup_smoke(setup, smoke_environment)
        base.exclusive_json(out / "original_environment.json", original_environment)
        base.exclusive_json(out / "new_setup_identity.json", setup)
        base.exclusive_json(out / "new_smoke_environment.json", smoke_environment)
        base.exclusive_json(out / "replication_inputs.json", {"original_evaluation": original_provenance,
            "new_smoke": smoke_provenance, "smoke_expected_identity_link": smoke_identity_link,
            "setup_identity_sha256": base.digest_file(args.setup_identity),
            "preserved_policy": preservation, "reused_source_sha256": source_hashes,
            "setup_smoke_match": setup_match, "lifecycle_scope": "Root orchestration schedules this only after original-cohort N9 completion; this runner does not allocate or release machines."})
        journal.emit("run_start", allocation_id=args.allocation_id, source_manifest=source_manifest,
                     original_allocation=original_environment["identity"]["allocation_id"], preserved_policy=preservation, argv=sys.argv)
        if "jax" in sys.modules:
            raise RuntimeError("Use a fresh process before fixed runtime setup")
        os.environ.update(base.FIXED_ENVIRONMENT)
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        from strassen_mm.kernels_v002 import make_matmul, enable_qualified_mosaic_v7_compat
        base.jax, base.jnp, base.np, base.ml_dtypes, base.make_matmul = jax, jnp, np, ml_dtypes, make_matmul
        environment = base.capture_environment(args, campaign, enable_qualified_mosaic_v7_compat())
        base.exclusive_json(out / "environment.json", environment)
        journal.emit("new_cohort_identity_check", **base.verify_identity(environment, args.expected_identity, campaign))
        compatibility = validate_compatibility(original_environment, environment, campaign["precision"])
        base.exclusive_json(out / "cohort_compatibility.json", compatibility)
        journal.emit("cross_cohort_compatibility", **compatibility)
        tuning.verify_heldout_selector(rule, manifest, campaign["experiments"]["N7-evaluate"]["shape_ids"], original_environment["identity"])
        tuning.verify_selection(choices, campaign, manifest, "N7-confirm", original_environment["identity"])
        planned, decisions = evaluation.evaluation_groups(campaign, manifest, choices, rule)
        expected = sum(len(group["arms"]) * len(group["scopes"]) * len(group["inputs"]) for group in planned)
        if len(planned) != 16 or expected != 128:
            raise ValueError("Fixed-policy replication must preserve 16 held-out groups and 128 results")
        base.exclusive_json(out / "planned_cases.json", planned)
        base.exclusive_json(out / "selector_decisions.json", decisions)
        journal.groups = {group["group_id"]: group for group in planned}
        runner = tuning.TuningRunner(args, campaign, journal)
        for index, group in enumerate(planned):
            runner.check_deadline()
            print(f"{PHASE} [{index+1}/16] {group['group_id']}", flush=True)
            journal.emit("group_start", group_id=group["group_id"], index=index + 1, total=16)
            runner.run_group(group)
            done.append(group["group_id"])
            journal.emit("group_complete", group_id=group["group_id"])
        if len(journal.results) != 128:
            raise ValueError("Fixed-policy replication result count differs from 128")
        base.exclusive_json(out / "selector_evaluation.json", {"cases": evaluation.evaluate_regret(journal, planned, decisions),
            "scope": "Fresh-v5e fixed-policy replication; no retuning, unchanged original policy/alternatives, separate cohort timings.",
            "original_policy_hashes": preservation})
        completed = True
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        journal.emit("run_error", **failure)
        print(f"{PHASE} stopped: {error}", file=sys.stderr, flush=True)
    finally:
        summary = {"phase": PHASE, "completed": completed, "status": "completed" if completed else "failed_or_interrupted",
                   "finished_utc": base.utc_now(), "wall_seconds": time.monotonic() - started,
                   "completed_groups": done, "planned_group_count": len(planned),
                   "not_completed_group_ids": [group["group_id"] for group in planned if group["group_id"] not in done],
                   "case_status_counts": dict(journal.status_counts), "error": failure,
                   "scientific_scope": "Fresh v5e cohort with preserved choices/rule, 16 held-out shapes, same arithmetic and software contract; no retuning or raw-timing pooling."}
        journal.emit("run_complete", **summary)
        journal.close()
        base.exclusive_json(out / "summary.json", summary)
        hashes = {str(path.relative_to(out)): base.digest_file(path) for path in sorted(out.rglob("*")) if path.is_file()}
        base.exclusive_json(out / "artifact_manifest.json", {"sha256": hashes, "sealed_utc": base.utc_now()})
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())

