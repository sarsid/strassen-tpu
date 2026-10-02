"""N4 supplement: actual checkpoint B, synthetic Gaussian BF16 A.

The core benchmark_v001 source remains unchanged. This separate runner reuses
its identity, compilation, numerical-reference, result-journal and sealing
helpers. It performs no timing and cannot create an inference-quality claim.
The public checkpoint artifacts must already be present at the preregistered
local manifest path; this module performs no network downloads.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

from strassen_mm import benchmark_v001 as base


EXPECTED_TENSORS = {
    "model.layers.0.self_attn.q_proj.weight": [2048, 1024],
    "model.layers.0.mlp.down_proj.weight": [1024, 3072],
}


def load_campaign(path):
    supplement = json.loads(path.read_text())
    parent_path = (path.parent / supplement["base_campaign"]).resolve()
    parent = json.loads(parent_path.read_text())
    campaign = copy.deepcopy(parent)
    for key, value in supplement.items():
        if key == "experiments":
            campaign[key].update(value)
        else:
            campaign[key] = value
    return campaign, supplement, parent_path


def verified_file(directory, relative, expected_sha256):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"weight artifact path is missing or leaves its directory: {relative}")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError(f"invalid expected SHA256: {relative}")
    actual = base.digest_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ValueError(f"weight artifact hash mismatch: {relative}")
    return path


def load_verified_weights(manifest_path, expected_model, expected_revision=None):
    """Verify raw artifacts and preserve exact BF16 bits while transposing B."""
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model_id") != expected_model:
        raise ValueError("checkpoint model_id does not match supplement")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", str(manifest.get("revision", ""))):
        raise ValueError("checkpoint revision must be a pinned 40-hex commit")
    if expected_revision is not None and manifest["revision"] != expected_revision:
        raise ValueError("checkpoint revision does not match the frozen supplement")
    if sys.byteorder != "little":
        raise RuntimeError("raw BF16 supplement loader is qualified only for little-endian hosts")
    directory = manifest_path.parent
    verified_file(directory, manifest["config_path"], manifest["config_sha256"])
    verified_file(directory, manifest["license_path"], manifest["license_sha256"])
    names = [item.get("name") for item in manifest.get("tensors", [])]
    if len(names) != len(set(names)) or set(names) != set(EXPECTED_TENSORS):
        raise ValueError("manifest must contain exactly the two preregistered layer-0 tensors")
    weights = {}
    provenance = copy.deepcopy(manifest)
    provenance["manifest_sha256"] = base.digest_file(manifest_path)
    provenance["verification_utc"] = base.utc_now()
    provenance["input_semantics"] = {
        "a": "synthetic Gaussian BF16 activation-shaped values",
        "b": "actual checkpoint weight transpose, exact stored BF16 values",
        "not_measured": "real model activations, end-to-end inference, or model quality",
    }
    for original, enriched in zip(manifest["tensors"], provenance["tensors"]):
        name = original["name"]
        shape = EXPECTED_TENSORS[name]
        if original.get("shape") != shape or original.get("dtype") != "BF16":
            raise ValueError(f"unexpected stored tensor shape or dtype: {name}")
        if original.get("byte_order") != "little":
            raise ValueError(f"unsupported tensor byte order: {name}")
        path = verified_file(directory, original["path"], original["sha256"])
        expected_bytes = 2 * math.prod(shape)
        if path.stat().st_size != expected_bytes or original.get("bytes") != expected_bytes:
            raise ValueError(f"tensor byte count mismatch: {name}")
        stored = base.np.fromfile(path, dtype="<u2").reshape(shape).view(base.ml_dtypes.bfloat16)
        b = stored.T.copy()
        if not base.np.isfinite(b).all():
            raise ValueError(f"nonfinite checkpoint BF16 values: {name}")
        weights[name] = b
        enriched["verified_sha256"] = base.digest_file(path)
        enriched["b_shape_kn"] = list(b.shape)
        enriched["b_sha256_after_transpose"] = hashlib.sha256(memoryview(b.view(base.np.uint8))).hexdigest()
        enriched["transformation"] = "stored [N,K] weight transposed and copied contiguously to B[K,N]; no numerical cast"
    return weights, provenance


class WeightJournal(base.Journal):
    def __init__(self, directory, shapes=None, weights_provenance=None):
        super().__init__(directory, "N4")
        self.by_shape = {item["id"]: item for item in (shapes or {}).get("shapes", [])}
        self.provenance = weights_provenance or {}

    def emit(self, event, **fields):
        fields.setdefault("supplement", "real_weights_v1")
        if event == "case_result":
            fields["numerical_eligible"] = fields.get("status") == "ok"
            fields["eligible_for_speedup_claim"] = False
        if event in ("case_start", "case_result", "correctness"):
            shape_id = fields.get("shape_id") or fields.get("group_id", "").split("__", 1)[0]
            shape = self.by_shape.get(shape_id)
            if shape:
                fields.setdefault("weight_tensor", shape["tensor_name"])
                fields.setdefault("weight_model_id", self.provenance["model_id"])
                fields.setdefault("weight_revision", self.provenance["revision"])
                fields.setdefault("activation_source", "synthetic_gaussian_bf16")
        super().emit(event, **fields)


class RealWeightRunner(base.Runner):
    def __init__(self, args, campaign, journal, weights):
        super().__init__(args, campaign, journal)
        self.weights = weights

    def run_group(self, group):
        shape = tuple(group["shape"][axis] for axis in ("m", "k", "n"))
        m, k, n = shape
        tensor_name = group["shape"]["tensor_name"]
        b_host = self.weights[tensor_name]
        if b_host.shape != (k, n):
            raise ValueError(f"actual tensor and planned matrix shape disagree: {tensor_name}")
        estimate = base.memory_estimate(shape, group["tile"], group["arms"], group["scopes"])
        self.journal.emit("memory_preflight", group_id=group["group_id"], **estimate)
        if estimate["estimated_live_device_bytes"] > self.campaign["memory"]["estimated_live_device_budget_gib"] * 1024**3:
            for input_case in group["inputs"]:
                for arm in group["arms"]:
                    self.journal.emit("case_result", group_id=group["group_id"], **input_case,
                                      arm_id=arm["arm_id"], scope="call", status="skipped_memory_preflight",
                                      eligible_for_speedup_claim=False, memory_estimate=estimate)
            return
        entries = None
        try:
            for input_case in group["inputs"]:
                self.check_deadline()
                if input_case["distribution"] != "real_weight_gaussian_activation":
                    raise ValueError("unsupported supplement activation distribution")
                rng = base.np.random.Generator(base.np.random.PCG64(input_case["seed"]))
                a_host = (rng.standard_normal((m, k), dtype=base.np.float32)
                          / base.np.float32(math.sqrt(k))).astype(base.ml_dtypes.bfloat16)
                if not base.np.isfinite(a_host).all():
                    raise ValueError("nonfinite generated activation values")
                if entries is None:
                    entries = self.compile_entries(group, a_host, b_host)
                self.execute_case(group, input_case, entries, a_host, b_host)
                del a_host
                base.gc.collect()
        finally:
            del entries
            base.jax.clear_caches()
            base.gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--phase", choices=("N4",), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-identity", type=Path, required=True)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=21600)
    args = parser.parse_args(argv)
    if args.max_wall_seconds <= 0:
        parser.error("--max-wall-seconds must be positive")
    args.campaign = args.campaign.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir
    # The journal covers setup failures too; populate tensor provenance only
    # after verification, retaining its exclusively created file descriptor.
    journal = WeightJournal(out)
    started = time.monotonic()
    completed, planned, done, error_summary = False, [], [], None
    try:
        campaign, supplement, parent_path = load_campaign(args.campaign)
        shape_path = (args.campaign.parent / supplement["shape_manifest"]).resolve()
        weights_path = (args.campaign.parent / supplement["weights_manifest"]).resolve()
        shapes = json.loads(shape_path.read_text())
        if campaign["precision"]["native_precision"] != "DEFAULT" or campaign["device"]["target"] != "v5e":
            raise ValueError("supplement requires the qualified v5e DEFAULT BF16 precision contract")
        source_manifest = base.snapshot_sources(out, args.campaign, shape_path, weights_path)
        with (out / "config_snapshot" / parent_path.name).open("xb") as handle:
            handle.write(parent_path.read_bytes())
        base.exclusive_json(out / "config_snapshot" / "effective_campaign.json", campaign)
        base.exclusive_json(out / "supplement_provenance.json", {
            "base_campaign_path": str(parent_path), "base_campaign_sha256": base.digest_file(parent_path),
            "effective_campaign_sha256": base.digest_file(out / "config_snapshot" / "effective_campaign.json"),
            "source_manifest": source_manifest, "weights_manifest_sha256": base.digest_file(weights_path),
            "core_campaign_unchanged": True,
        })
        journal.emit("run_start", allocation_id=args.allocation_id, source_manifest=source_manifest,
                     argv=sys.argv, supplement="real_weights_v1", max_wall_seconds=args.max_wall_seconds)
        if "jax" in sys.modules:
            raise RuntimeError("run in a fresh Python process before importing JAX")
        os.environ.update(base.FIXED_ENVIRONMENT)
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        from strassen_mm.kernels_v001 import make_matmul, enable_qualified_mosaic_v7_compat
        base.jax, base.jnp, base.np, base.ml_dtypes, base.make_matmul = jax, jnp, np, ml_dtypes, make_matmul
        environment = base.capture_environment(args, campaign, enable_qualified_mosaic_v7_compat())
        base.exclusive_json(out / "environment.json", environment)
        journal.emit("identity_check", **base.verify_identity(environment, args.expected_identity, campaign))
        weights, provenance = load_verified_weights(
            weights_path, supplement["expected_model_id"], supplement["expected_revision"])
        base.exclusive_json(out / "weights_provenance.json", provenance)
        journal.by_shape = {item["id"]: item for item in shapes["shapes"]}
        journal.provenance = provenance
        planned = base.planned_groups(campaign, shapes, "N4")
        expected_rows = sum(len(g["arms"]) * len(g["inputs"]) * len(g["scopes"]) for g in planned)
        if len(planned) != 4 or expected_rows != 48:
            raise ValueError("supplement must retain its preregistered four groups and 48 result rows")
        base.exclusive_json(out / "planned_cases.json", planned)
        runner = RealWeightRunner(args, campaign, journal, weights)
        print(f"N4 real-weight supplement: 4 groups, 48 cases; {provenance['model_id']}@{provenance['revision']}", flush=True)
        for index, group in enumerate(planned):
            runner.check_deadline()
            print(f"[{index + 1}/4] {group['group_id']} starting", flush=True)
            journal.emit("group_start", group_id=group["group_id"], index=index + 1, total=4)
            runner.run_group(group)
            done.append(group["group_id"])
            journal.emit("group_complete", group_id=group["group_id"], index=index + 1, total=4)
            print(f"[{index + 1}/4] complete; {dict(journal.status_counts)}", flush=True)
        completed = True
    except BaseException as error:
        error_summary = {"type": type(error).__name__, "message": str(error),
                         "status": base.error_status(error, "execute")}
        journal.emit("run_error", **error_summary, traceback=traceback.format_exc())
        print(f"N4 real-weight supplement stopped: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    finally:
        summary = {"phase": "N4", "supplement": "real_weights_v1", "completed": completed,
                   "status": "completed" if completed else "failed_or_interrupted",
                   "finished_utc": base.utc_now(), "wall_seconds": time.monotonic() - started,
                   "completed_groups": done, "planned_group_count": len(planned),
                   "not_completed_group_ids": [g["group_id"] for g in planned if g["group_id"] not in done],
                   "case_status_counts": dict(journal.status_counts), "error": error_summary,
                   "scientific_success": "Numerical eligibility only; not model-quality or inference evidence.",
                   "n4_scope": "actual checkpoint B with synthetic Gaussian A; call correctness only, no timing"}
        journal.emit("run_complete", **summary)
        journal.close()
        base.exclusive_json(out / "summary.json", summary)
        hashes = {str(p.relative_to(out)): base.digest_file(p) for p in sorted(out.rglob("*")) if p.is_file()}
        base.exclusive_json(out / "artifact_manifest.json", {"sha256": hashes, "sealed_utc": base.utc_now()})
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
