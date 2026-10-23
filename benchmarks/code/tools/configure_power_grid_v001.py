"""Create the sampled-grid campaign from the exclusive geometry manifest.

This is configuration generation only. It never allocates or benchmarks a TPU.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


CUBIC_TILES = [
    (16, 128, 128), (64, 256, 256), (128, 512, 512),
    (256, 256, 256), (384, 384, 384), (512, 512, 256),
    (512, 512, 512), (768, 768, 512), (768, 768, 768),
    (1024, 1024, 512), (1024, 1024, 1024), (1536, 1536, 512),
    (2048, 2048, 512), (256, 1024, 512), (1024, 256, 512),
    (512, 1024, 512), (1024, 512, 512), (512, 1536, 512),
    (1536, 512, 512), (512, 1024, 1024), (1024, 512, 1024),
    (1024, 2048, 512), (2048, 1024, 512), (1024, 1024, 1536),
]
STRASSEN_TILES = [
    (16, 256, 256) if t == (16, 128, 128) else
    (384, 512, 512) if t == (384, 384, 384) else t
    for t in CUBIC_TILES
]


def candidates():
    result = {}
    for family, tiles, algorithm, variant in (
        ("cubic", CUBIC_TILES, "cubic_full", "output_accumulator"),
        ("strassen", STRASSEN_TILES, "strassen", "interleaved_output_accumulator"),
    ):
        assert len(tiles) == len(set(tiles)) == 24
        result[family] = [
            {"candidate_id": f"{family}_{bm}_{bn}_{bk}", "algorithm": algorithm,
             "variant": variant, "tile": [bm, bn, bk]}
            for bm, bn, bk in tiles
        ]
    result["native"] = [
        {"candidate_id": "native_default", "algorithm": "native", "variant": "plain",
         "tile": None, "compiler_options": {}},
        *[{"candidate_id": f"native_vmem_{mib}m", "algorithm": "native", "variant": "plain",
           "tile": None, "compiler_options": {"xla_tpu_scoped_vmem_limit_kib": mib * 1024}}
          for mib in (32, 48, 64)],
    ]
    return result


def build_config(base, manifest, manifest_name):
    config = copy.deepcopy(base)
    config.update(campaign_id="power_midpoint_sample_v001", seed=20260920,
                  status="preregistered_pending_hardware_smoke",
                  shape_manifest=manifest_name,
                  distribution_manifest="distributions_v1.json")
    families = candidates()
    ids = manifest["exploratory_shape_ids"]
    assert len(ids) == len(set(ids))
    assert not set(ids).intersection(manifest["holdout_shape_ids"])
    shapes = {row["id"]: row for row in manifest["shapes"]}
    # Smoke exercises tiny, odd, ordinary and midpoint matrix extents. It is
    # explicitly part of the exploratory cohort, never held-out evidence.
    smoke_ids = []
    for target in ((3, 3, 3), (512, 512, 512), (768, 768, 768)):
        matches = [key for key in ids if tuple(shapes[key][axis] for axis in ("m", "n", "k")) == target]
        if matches:
            smoke_ids.append(matches[0])
    if not smoke_ids:
        smoke_ids = ids[:3]
    common = {"candidate_families": families, "distribution": "gaussian"}
    config["experiments"] = {
        "GRID-smoke": {**copy.deepcopy(common), "shape_ids": smoke_ids,
                       "timing": "smoke", "seed": 20260920},
        "GRID-screen": {**copy.deepcopy(common), "shape_ids": ids,
                        "timing": "screen", "seed": 20260921,
                        "shortlist_top_k": 3, "shortlist_max_per_family": 6,
                        "near_optimal_fraction": 0.05},
        "GRID-confirm": {**copy.deepcopy(common), "shape_ids": ids,
                         "timing": "confirm", "seed": 20260922},
    }
    config["grid_study"] = {
        "public_coordinate_order": ["m", "n", "k"],
        "kernel_coordinate_order": ["m", "k", "n"],
        "matrix_operation": "A[M,K] @ B[K,N] -> C[M,N]",
        "full_lattice_execution": False,
        "reserved_holdout_shape_ids": manifest["holdout_shape_ids"],
        "holdout_policy": "No timing or accuracy observation until a later mathematical predictor is frozen.",
        "screen_sets_relative_tolerance": [0.01, 0.03, 0.05],
        "confirm_reference": "Frozen screen winner per family; no confirmation reselection for headline claims.",
        "near_set_policy": "Actual measured tile triples, not all combinations of marginal min/max ranges.",
        "native_tuning": "Compiler-managed tiles; bounded per-executable VMEM compiler-option screen with explicit library default.",
        "unsupported_native_options": "Retain compile failures; never claim an unaccepted option was tuned.",
        "custom_tuning": "24 tile attempts per family, fixed documented implementation variant; failures count.",
        "scope": "One-level tile-local Strassen, single v5e, resident Gaussian BF16 operands and FP32 output.",
        "new_cohort": True,
    }
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / "configs/campaign_v1.json").read_text())
    manifest = json.loads(args.manifest.read_text())
    config = build_config(base, manifest, args.manifest.name)
    with args.output.open("x") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"config": str(args.output), "exploratory_shapes": len(manifest["exploratory_shape_ids"]),
                      "reserved_holdout_shapes": len(manifest["holdout_shape_ids"]),
                      "custom_candidates_per_family": 24, "native_candidates": 4}))


if __name__ == "__main__":
    main()
