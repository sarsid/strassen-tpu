"""Post-hoc device diagnostics for the two largest N5 joint wins.

This supplements the original median-volume N6 representatives. It reuses all
frozen MM choices and valid device collection, accepts no N7 outcome data, and
does not fit or update a selector. It is not an independent confirmation.
"""
from __future__ import annotations
import argparse
from functools import partial
import math
from pathlib import Path

from . import benchmark_n6_n7_v002 as original
from . import benchmark_n6_recovery_v001 as recovery

base = original.base
selection = original.selection_rule
SUPPLEMENT = "n6_large_joint_wins_v001"
RULE = "Two largest M*K*N among N5-confirm eligible complete-call Strassen CI95 lower>1 versus both native_xla and cubic_selected; descending volume, ascending shape_id tie-break."


def large_joint_wins(rows, choices):
    if choices.get("phase") != "N5-screen":
        raise ValueError("Supplement accepts only original N5 screening choices")
    by_shape = {}
    for row in rows:
        if row.get("phase") != "N5-confirm" or row.get("scope") != "call":
            raise ValueError("Supplement accepts only N5-confirm complete-call outcomes")
        arms = by_shape.setdefault(row["shape_id"], {})
        if row["arm_id"] in arms:
            raise ValueError("Duplicate N5 confirmation arm")
        arms[row["arm_id"]] = row
    eligible = []
    for shape_id, arms in by_shape.items():
        required = ("native_xla", "cubic_selected", "strassen_selected")
        if any(not selection.passing(arms.get(arm)) or (arms[arm].get("timing") or {}).get("sample_count") != 30 for arm in required):
            continue
        strassen = arms["strassen_selected"]
        if not all(selection.confidence(strassen, reference, 1.0) for reference in required[:2]):
            continue
        shape = list(selection.check_shape(choices["by_shape"][shape_id]["shape_mkn"]))
        pairs = {pair["reference_arm"]: pair for pair in strassen["comparisons"]}
        eligible.append({"shape_id": shape_id, "shape_mkn": shape,
                         "category": "post_hoc_joint_win",
                         "paired_comparison": pairs["cubic_selected"],
                         "native_paired_comparison": pairs["native_xla"]})
    eligible.sort(key=lambda item: (-math.prod(item["shape_mkn"]), item["shape_id"]))
    if len(eligible) < 2:
        raise ValueError("Fewer than two eligible joint wins; no substitute shapes")
    return {"representatives": eligible[:2], "available_class_counts": {"joint_win": len(eligible)},
            "eligible_shapes_in_volume_order": [item["shape_id"] for item in eligible],
            "absent_classes": [], "rule": RULE, "supplement_id": SUPPLEMENT,
            "post_hoc": True, "independent_confirmation": False}


def validate_inputs(args):
    if base.digest_file(args.campaign) != selection.CONFIG_SHA256:
        raise ValueError("Original campaign bytes changed")
    confirmation, rows = original.load_confirmed_rows(args.confirmation)
    choices = selection.load(args.selection)
    if base.digest_file(args.selection) != selection.load(confirmation / "selection_input_provenance.json")["sha256"]:
        raise ValueError("Supplement choices differ from frozen N5 confirmation")
    hashes = selection.load(confirmation / "source_manifest.json")["sha256"]
    for name in recovery.CORE_FILES:
        if base.digest_file(Path(__file__).parent / name) != hashes["source_snapshot/" + name]:
            raise ValueError("Frozen N5/N6 dependency changed: " + name)
    representatives = large_joint_wins(rows, choices)
    prior = selection.artifacts(args.prior_profile)
    prior_hash = selection.verify_seal(prior)
    prior_summary = selection.load(prior / "summary.json")
    if (prior_summary.get("phase") != "N6" or prior_summary.get("completed") is not True
            or prior_summary.get("profile_successful_captures") != 18
            or prior_summary.get("profile_attempted_captures") != 18):
        raise ValueError("Supply the successful original three-representative N6 recovery")
    if selection.load(prior / "input_provenance.json")["selection_sha256"] != base.digest_file(args.selection):
        raise ValueError("Prior N6 used different MM choices")
    identity = selection.load(confirmation / "environment.json")["identity"]
    if selection.load(prior / "environment.json")["identity"] != identity:
        raise ValueError("Prior N6 and N5 confirmation have different identities")
    prior_confirmation = selection.load(prior / "confirmation_provenance.json")
    if (prior_confirmation["results_sha256"] != base.digest_file(confirmation / "results.jsonl")
            or prior_confirmation["artifact_manifest_sha256"] != base.digest_file(confirmation / "artifact_manifest.json")):
        raise ValueError("Prior N6 used different confirmation evidence")
    return {"prior_profile": str(prior), "prior_profile_artifact_manifest_sha256": prior_hash,
            "confirmation": str(confirmation), "confirmation_results_sha256": base.digest_file(confirmation / "results.jsonl"),
            "confirmation_artifact_manifest_sha256": base.digest_file(confirmation / "artifact_manifest.json"),
            "selection_sha256": base.digest_file(args.selection), "source_identity": identity,
            "representatives": representatives}


class SupplementJournal(original.EvidenceJournal):
    def emit(self, event, **fields):
        fields.setdefault("supplement_id", SUPPLEMENT)
        fields.setdefault("post_hoc_diagnostic", True)
        super().emit(event, **fields)


class SupplementProfileRunner(recovery.RecoveryProfileRunner):
    def __init__(self, args, campaign, journal, *, provenance):
        identity = selection.load(journal.directory / "environment.json")["identity"]
        if identity != provenance["source_identity"]:
            raise ValueError("Supplement cannot change N5/N6 cohort")
        if selection.load(journal.directory / "representatives.json") != provenance["representatives"]:
            raise ValueError("Supplement representatives changed between validation and execution")
        effective = recovery.profiling_override(campaign)
        design = {"schema_version": 1, "supplement_id": SUPPLEMENT,
                  "post_hoc_diagnostic": True, "independent_confirmation": False,
                  "selection_rule": RULE, "provenance": provenance,
                  "original_campaign_sha256": base.digest_file(args.campaign),
                  "profile_override": {"original_tpu_trace_mode": "TRACE_ONLY",
                                       "effective_tpu_trace_mode": recovery.MODE, "device_tracer_level": 1},
                  "planned_shapes": 2, "arms_per_shape": 3, "capture_blocks_per_arm": 2,
                  "synchronized_calls_per_capture": 8, "expected_device_captures": 12,
                  "ordinary_before_after": {"warmups": 5, "paired_repeats": 30, "scope": "call"},
                  "intent": "Check whether the two largest confirmed joint wins also have shorter profiled device modules; retain original ranking reversals.",
                  "limits": "Post-hoc selection on N5 outcomes; no N7 inputs, retuning, selector update, independent replication, utilization or overlap inference."}
        base.exclusive_json(journal.directory / "supplementary_design.json", design)
        journal.emit("supplementary_design", **design)
        # Keep the already qualified capture method, including strict device
        # coverage and constructor restoration, while bypassing its recovery-
        # specific original-three-shape initialization.
        recovery.OriginalProfileRunner.__init__(self, args, effective, journal)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("N6",), required=True)
    for name in ("campaign", "output-dir", "expected-identity", "selection", "confirmation", "prior-profile"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=21600)
    args = parser.parse_args(argv)
    if args.max_wall_seconds <= 0:
        parser.error("Positive wall budget required")
    provenance = validate_inputs(args)
    forwarded = ["--phase", "N6"]
    for name in ("campaign", "output_dir", "expected_identity", "selection", "confirmation", "allocation_id", "max_wall_seconds"):
        forwarded.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    previous = original.ProfileRunner, original.representative_shapes, original.EvidenceJournal
    try:
        original.ProfileRunner = partial(SupplementProfileRunner, provenance=provenance)
        original.representative_shapes = large_joint_wins
        original.EvidenceJournal = SupplementJournal
        return original.main(forwarded)
    finally:
        original.ProfileRunner, original.representative_shapes, original.EvidenceJournal = previous


if __name__ == "__main__":
    raise SystemExit(main())
