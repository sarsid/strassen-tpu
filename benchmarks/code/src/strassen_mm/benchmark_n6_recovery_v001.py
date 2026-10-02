"""Recover N6 device traces without changing frozen MM experiments.

The original campaign accidentally requested the unsupported TRACE_ONLY value.
This wrapper records an explicit, local profiling-only override and reuses the
untouched v002 runner. It cannot run N7, select new kernels or change a cohort.
"""
from __future__ import annotations
import argparse
import copy
from functools import partial
from pathlib import Path

from . import benchmark_n6_n7_v002 as original

base = original.base
selection = original.selection_rule
OriginalProfileRunner = original.ProfileRunner
MODE = "TRACE_COMPUTE_AND_SYNC"
CORE_FILES = ("benchmark_v001.py", "benchmark_n5_v001.py", "benchmark_n6_n7_v002.py",
              "kernels_v001.py", "kernels_v002.py", "selector_v002.py")


def profiling_override(campaign):
    """Return a copy; preserve all choices, timings, gates and original bytes."""
    if campaign["experiments"]["N6"]["profiler"]["tpu_trace_mode"] != "TRACE_ONLY":
        raise ValueError("Recovery applies only to the recorded TRACE_ONLY mistake")
    effective = copy.deepcopy(campaign)
    effective["experiments"]["N6"]["profiler"]["tpu_trace_mode"] = MODE
    return effective


def enabled_device_options(constructor):
    options = constructor()
    options.device_tracer_level = 1
    return options


def validate_original(args):
    directory = selection.artifacts(args.original_profile)
    seal_hash = selection.verify_seal(directory)
    summary = selection.load(directory / "summary.json")
    if summary.get("phase") != "N6" or summary.get("completed") is not True:
        raise ValueError("Recovery requires the explicit completed original N6 execution")
    if summary.get("profile_attempted_captures") != 18 or summary.get("profile_successful_captures") != 0:
        raise ValueError("This recovery is bounded to the recorded 18 failed captures")
    if base.digest_file(args.campaign) != selection.CONFIG_SHA256:
        raise ValueError("Original campaign bytes changed")
    provenance = selection.load(directory / "input_provenance.json")
    if base.digest_file(args.selection) != provenance["selection_sha256"]:
        raise ValueError("Recovery selections differ from original N6")
    confirmation = selection.artifacts(args.confirmation)
    expected = selection.load(directory / "confirmation_provenance.json")
    if (base.digest_file(confirmation / "results.jsonl") != expected["results_sha256"]
            or base.digest_file(confirmation / "artifact_manifest.json") != expected["artifact_manifest_sha256"]):
        raise ValueError("Recovery confirmation differs from original N6")
    hashes = selection.load(directory / "source_manifest.json")["sha256"]
    for name in CORE_FILES:
        if base.digest_file(Path(__file__).parent / name) != hashes["source_snapshot/" + name]:
            raise ValueError("Frozen N6 dependency changed: " + name)
    return directory, seal_hash


class RecoveryProfileRunner(OriginalProfileRunner):
    def __init__(self, args, campaign, journal, *, original_profile, original_seal_hash):
        original_profile = Path(original_profile)
        for name in ("environment.json", "representatives.json", "planned_cases.json"):
            old = selection.load(original_profile / name)
            new = selection.load(journal.directory / name)
            if name == "environment.json":
                old, new = old["identity"], new["identity"]
            if old != new:
                raise ValueError("Recovery must preserve original N6 " + name)
        effective = profiling_override(campaign)
        override = {
            "schema_version": 1, "recovery_version": "benchmark_n6_recovery_v001",
            "original_profile_directory": str(original_profile),
            "original_profile_artifact_manifest_sha256": original_seal_hash,
            "original_campaign_sha256": base.digest_file(args.campaign),
            "original_mode": "TRACE_ONLY", "effective_mode": MODE,
            "device_tracer_level": 1,
            "changes": {"experiments.N6.profiler.tpu_trace_mode": {"from": "TRACE_ONLY", "to": MODE},
                        "ProfileOptions.device_tracer_level": {"from": "implicit runtime default", "to": 1}},
            "reason": "TRACE_ONLY was an erroneous unsupported value; archived traces contain host events only.",
            "same_representatives_choices_inputs_gates_timings_and_cohort": True,
            "references": ["https://docs.jax.dev/en/latest/profiling.html",
                           "strassen-tpu/evidence/validation/2026-09-11-v5e-profiling/plan.json"],
            "qualification": "Device trace coverage remains mandatory; no utilization-counter or speedup inference from profiler wall times."
        }
        base.exclusive_json(journal.directory / "profile_recovery_override.json", override)
        journal.emit("profile_recovery_override", **override)
        super().__init__(args, effective, journal)

    def run_profile_group(self, group):
        # The original method constructs ProfileOptions internally. Temporarily
        # wrap only that constructor; the returned object remains the real JAX
        # options type. A fresh serialized process owns this profiling session.
        profiler = base.jax.profiler
        constructor = profiler.ProfileOptions
        before = len(self.journal.captures)
        try:
            profiler.ProfileOptions = partial(enabled_device_options, constructor)
            super().run_profile_group(group)
        finally:
            profiler.ProfileOptions = constructor
        captures = self.journal.captures[before:]
        expected = 3 * self.campaign["experiments"]["N6"]["capture_blocks"]
        if len(captures) != expected or any(row["status"] != "ok" for row in captures):
            raise RuntimeError("N6 recovery has incomplete device trace coverage; stopping before further groups")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("N6",), required=True)
    for name in ("campaign", "output-dir", "expected-identity", "selection", "confirmation", "original-profile"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=21600)
    args = parser.parse_args(argv)
    if args.max_wall_seconds <= 0:
        parser.error("Positive wall budget required")
    directory, seal_hash = validate_original(args)
    forwarded = ["--phase", "N6"]
    for name in ("campaign", "output_dir", "expected_identity", "selection", "confirmation", "allocation_id", "max_wall_seconds"):
        forwarded.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    previous = original.ProfileRunner
    try:
        original.ProfileRunner = partial(RecoveryProfileRunner, original_profile=directory,
                                         original_seal_hash=seal_hash)
        return original.main(forwarded)
    finally:
        original.ProfileRunner = previous


if __name__ == "__main__":
    raise SystemExit(main())
