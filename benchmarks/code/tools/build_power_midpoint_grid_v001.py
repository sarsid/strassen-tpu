#!/usr/bin/env python3
"""Create an immutable, data-independent geometry plan; never execute a kernel.

The complete lattice is a preflight inventory, not a request to benchmark every
point. Named M/N/K fields are authoritative; older benchmark tuples use M/K/N.
Only the exploratory sample may be used to develop a performance model. The
holdout must remain unmeasured until that model and its parameters are frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path


SCHEMA_VERSION = 1
GIB = 1024 ** 3
BUDGET_BYTES = 8 * GIB
REFERENCE_RESERVE_BYTES = 8 * 128 * 128
# Planning inputs, not an assertion that these tiles compile or fit TPU VMEM.
# Runtime preflight must use the campaign's actual simultaneously resident arms.
PLANNING_TILES = (
    (256, 256, 256), (384, 384, 256), (512, 512, 256),
    (768, 768, 256), (1024, 512, 512), (512, 1024, 512),
    (1024, 1024, 512), (1536, 1536, 768),
    (2048, 1024, 512), (1024, 2048, 512), (2048, 2048, 1024),
)
CSV_FIELDS = (
    "shape_id", "m", "n", "k", "m_index", "n_index", "k_index",
    "useful_cubic_flops", "lower_bound_resident_bytes",
    "conservative_pair_buffer_bytes", "planning_budget_bytes",
    "planning_budget_status", "sample_stage", "sample_family",
)


def dimensions():
    """Unique integer endpoints and true midpoints for inclusive i=0..16.

    The i=0 midpoint is exactly 1.5 and is omitted, never rounded.
    """
    values = set()
    for exponent in range(17):
        lower, upper = 2 ** exponent, 2 ** (exponent + 1)
        values.update((lower, upper))
        if (lower + upper) % 2 == 0:
            values.add((lower + upper) // 2)
    return tuple(sorted(values))


def _check_shape(m, n, k):
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in (m, n, k)):
        raise ValueError("Matrix extents must be positive integers")


def resident_lower_bound_bytes(m, n, k):
    """BF16 A[M,K], BF16 B[K,N], one FP32 C[M,N]; no extra storage."""
    _check_shape(m, n, k)
    return 2 * m * k + 2 * k * n + 4 * m * n


def _checked_tiles(tiles):
    checked = tuple(tuple(tile) for tile in tiles)
    if not checked:
        raise ValueError("At least one planning tile is required")
    for tile in checked:
        if len(tile) != 3:
            raise ValueError("Tiles must contain BM, BN, BK")
        _check_shape(*tile)
    return checked


def pair_buffer_estimate(m, n, k, tiles=PLANNING_TILES):
    """Worst two-custom-arm prepared-buffer estimate over an explicit palette.

    Mirrors the existing N5 group estimate: original BF16 inputs + largest
    padded input pair + one input pair per DISTINCT nonoriginal padded shape
    + two largest padded FP32 outputs + 128 KiB reference reserve. Includes
    same-tile pairs. Excludes executable/runtime peaks, on-chip VMEM and host
    memory; this is neither a measured peak nor a feasibility guarantee.
    """
    _check_shape(m, n, k)
    tiles = _checked_tiles(tiles)
    original_inputs = 2 * (m * k + k * n)
    padded = {}
    for bm, bn, bk in tiles:
        mp = ((m + bm - 1) // bm) * bm
        np = ((n + bn - 1) // bn) * bn
        kp = ((k + bk - 1) // bk) * bk
        padded[(mp, np, kp)] = (2 * (mp * kp + kp * np), 4 * mp * np)
    largest_estimate = 0
    for shape_a, shape_b in itertools.combinations_with_replacement(padded, 2):
        input_a, output_a = padded[shape_a]
        input_b, output_b = padded[shape_b]
        prepared_inputs = sum(padded[shape][0] for shape in {shape_a, shape_b}
                              if shape != (m, n, k))
        estimate = (original_inputs + max(original_inputs, input_a, input_b)
                    + prepared_inputs + 2 * max(4 * m * n, output_a, output_b)
                    + REFERENCE_RESERVE_BYTES)
        largest_estimate = max(largest_estimate, estimate)
    return largest_estimate


def shape_id(m, n, k):
    return f"grid_m{m}_n{n}_k{k}"


def check_holdout_history(holdouts):
    """Check exact geometry against the declared, bounded config inventory."""
    root = Path(__file__).resolve().parents[1]
    paths = [root / "configs/shapes_v1.json", root / "configs/shapes_heldout_v1.json"]
    for folder in ("generated_n8_v001", "generated_n9_v001"):
        found = sorted((root / "configs" / folder).glob("*.json"))
        if not found:
            raise ValueError(f"Missing historical configuration inventory: {folder}")
        paths.extend(found)
    historical = set()

    def visit(value):
        if isinstance(value, dict):
            if all(isinstance(value.get(axis), int) for axis in ("m", "n", "k")):
                historical.add(tuple(value[axis] for axis in ("m", "n", "k")))
            mkn = value.get("shape_mkn")
            if isinstance(mkn, list) and len(mkn) == 3 and all(isinstance(v, int) for v in mkn):
                historical.add((mkn[0], mkn[2], mkn[1]))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    hashes = {}
    for path in paths:
        content = path.read_bytes()
        hashes[str(path.relative_to(root))] = hashlib.sha256(content).hexdigest()
        visit(json.loads(content))
    overlap = {(row["m"], row["n"], row["k"]) for row in holdouts} & historical
    if overlap:
        raise ValueError(f"Reserved holdouts overlap checked historical geometry: {sorted(overlap)}")
    return {
        "status": "no_exact_geometry_overlap_in_checked_configs",
        "files_sha256": hashes,
        "unique_historical_geometries": len(historical),
        "parsed_geometry_fields": ["named integer m,n,k", "shape_mkn lists"],
        "limitation": "Only these configuration files were checked; archived runs and other projects were not exhaustively searched. This is not a claim of globally never-measured geometry.",
    }


def sampled_manifest(tiles=PLANNING_TILES):
    """Sixty exploratory and twelve untouched validation geometries."""
    tiles = _checked_tiles(tiles)
    lattice = set(dimensions())
    shapes = []
    seen = set()

    def add(m, n, k, family, stage="exploratory"):
        if (m, n, k) in seen:
            raise ValueError(f"Duplicate sample geometry {(m, n, k)}")
        seen.add((m, n, k))
        estimate = pair_buffer_estimate(m, n, k, tiles)
        if estimate > BUDGET_BYTES:
            raise ValueError(f"Sample exceeds planning budget: {(m, n, k)}: {estimate}")
        shapes.append({"id": shape_id(m, n, k), "m": m, "n": n, "k": k,
                       "family": family, "stage": stage,
                       "on_lattice": all(value in lattice for value in (m, n, k)),
                       "lower_bound_resident_bytes": resident_lower_bound_bytes(m, n, k),
                       "conservative_pair_buffer_bytes": estimate,
                       "useful_cubic_flops": 2 * m * n * k})

    for side in (1, 3, 16, 64, 128, 256, 384, 512, 768, 1024, 2048,
                 4096, 8192, 16384):
        add(side, side, side, "square_scale")
    for point in ((64, 32768, 512), (32768, 64, 512),
                  (4096, 16384, 256), (16384, 4096, 256),
                  (1536, 8192, 32768), (8192, 1536, 32768)):
        add(*point, family="interior_rectangle")
    for axis in ("m", "n", "k"):
        for extent in (8, 256, 16384, 32768, 65536, 131072):
            point = {"m": 2048, "n": 2048, "k": 2048, axis: extent}
            add(**point, family=f"axis_{axis}")
    for point in itertools.permutations((512, 4096, 16384)):
        add(*point, family="aspect_permutation")
    for point in itertools.permutations((768, 3072, 12288)):
        add(*point, family="midpoint_permutation")
    # These intentionally lie outside the power/midpoint lattice. Keep their
    # identity visible so padding evidence cannot be mistaken for lattice data.
    for side in (1023, 1025, 4095, 4097):
        add(side, side, side, "off_lattice_square_boundary")
    for extent in (2047, 2049):
        add(extent, 4096, 4096, "off_lattice_m_boundary")
    for extent in (8191, 8193):
        add(2048, 2048, extent, "off_lattice_k_boundary")
        add(2048, extent, 4096, "off_lattice_n_boundary")
    for side in (192, 12288):
        add(side, side, side, "heldout_square", "holdout")
    for point in ((3072, 3072, 1536), (6144, 6144, 3072)):
        add(*point, family="heldout_intermediate_rectangle", stage="holdout")
    for axis in ("m", "n", "k"):
        for extent in (49152, 98304):
            point = {"m": 2048, "n": 2048, "k": 2048, axis: extent}
            add(**point, family=f"heldout_axis_{axis}", stage="holdout")
    for point in ((1536, 6144, 24576), (24576, 6144, 1536)):
        add(*point, family="heldout_rectangle", stage="holdout")
    exploratory = [row["id"] for row in shapes if row["stage"] == "exploratory"]
    holdout = [row["id"] for row in shapes if row["stage"] == "holdout"]
    if (len(exploratory), len(holdout)) != (60, 12):
        raise AssertionError("Unexpected sample cardinality")
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": "power_midpoint_sample_v001",
        "shape_order": ["m", "n", "k"],
        "legacy_benchmark_tuple_order": ["m", "k", "n"],
        "tile_order": ["bm", "bn", "bk"],
        "selection_basis": "Geometry only; no new latency or correctness observations used.",
        "execution_status": "planned_unmeasured",
        "holdout_policy": "Do not time or fit holdouts until the mathematical model, parameters, and tile-selection rule are frozen.",
        "holdout_history_check_scope": check_holdout_history([row for row in shapes if row["stage"] == "holdout"]),
        "exploratory_shape_ids": exploratory,
        "holdout_shape_ids": holdout,
        "shapes": shapes,
    }


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def generate(output_dir, tiles=PLANNING_TILES):
    """Exclusively create output_dir; existing directories always fail."""
    tiles = _checked_tiles(tiles)
    manifest = sampled_manifest(tiles)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    values = dimensions()
    dimension_data = {
        "schema_version": SCHEMA_VERSION,
        "formula": "union {2^i, (2^i + 2^(i+1))/2, 2^(i+1)} for integer i in [0,16]",
        "excluded_fractional_midpoints": [{"i": 0, "exact_value": "3/2", "reason": "matrix dimensions must be integers; not rounded"}],
        "values": values, "count": len(values), "largest_endpoint": values[-1],
        "full_lattice_count": len(values) ** 3,
    }
    samples = {(row["m"], row["n"], row["k"]): row for row in manifest["shapes"]}
    budget_counts = Counter()
    grid_path = output / "full_lattice_preflight.csv"
    with grid_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for (mi, m), (ni, n), (ki, k) in itertools.product(enumerate(values), repeat=3):
            estimate = pair_buffer_estimate(m, n, k, tiles)
            status = "within_planning_budget" if estimate <= BUDGET_BYTES else "over_planning_budget"
            budget_counts[status] += 1
            sample = samples.get((m, n, k), {})
            writer.writerow({"shape_id": shape_id(m, n, k), "m": m, "n": n, "k": k,
                             "m_index": mi, "n_index": ni, "k_index": ki,
                             "useful_cubic_flops": 2 * m * n * k,
                             "lower_bound_resident_bytes": resident_lower_bound_bytes(m, n, k),
                             "conservative_pair_buffer_bytes": estimate,
                             "planning_budget_bytes": BUDGET_BYTES,
                             "planning_budget_status": status,
                             "sample_stage": sample.get("stage", "preflight_only"),
                             "sample_family": sample.get("family", "")})
    summary = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "Full lattice for analytical inventory and visualization; only the sample is an experiment plan.",
        "execution_status": "No TPU execution or performance measurements performed by this generator.",
        "dimension_count": len(values), "full_lattice_count": len(values) ** 3,
        "csv_coordinate_order": ["m", "n", "k"],
        "csv_sort_order": "ascending lexicographic M,N,K; K varies fastest",
        "exploratory_count": len(manifest["exploratory_shape_ids"]),
        "holdout_count": len(manifest["holdout_shape_ids"]),
        "off_lattice_sample_count": sum(not row["on_lattice"] for row in manifest["shapes"]),
        "sample_family_counts": dict(sorted(Counter(row["family"] for row in manifest["shapes"]).items())),
        "planning_budget_bytes": BUDGET_BYTES,
        "lattice_planning_budget_counts": dict(budget_counts),
        "planning_tiles_bm_bn_bk": tiles,
        "memory_estimate_contract": {
            "lower_bound": "2*M*K + 2*K*N + 4*M*N bytes for BF16 inputs and one FP32 output",
            "conservative_pair": "Maximum over two custom planning tiles: original input bytes + largest padded input bytes + distinct nonoriginal prepared input bytes + 2*largest padded FP32 output bytes +128KiB reference reserve",
            "estimate_is_not_measured_peak": True,
            "excluded": ["executable storage", "runtime peak allocations", "on-chip VMEM", "host memory"],
            "runtime_requirement": "Recompute with actual concurrent candidate arms; planning budget classification is not hardware feasibility.",
        },
        "largest_cube": {"m": values[-1], "n": values[-1], "k": values[-1],
                         "lower_bound_resident_bytes": resident_lower_bound_bytes(values[-1], values[-1], values[-1])},
        "largest_exploratory_square": 16384,
        "analysis_requirements": [
            "Show all three M,N,K coordinate orientations and separate off-lattice points.",
            "Keep small latency-dominated shapes and failed/unsupported candidates visible.",
            "Report useful FLOPs, padding, input/output footprint, arithmetic precision and numerical errors.",
            "Native XLA is compiler-tiled: do not label an exposed manual tile search as native tuning.",
            "Freeze a model before observing holdouts; do not interpolate a full measured volume from this sparse sample.",
        ],
    }
    _write_json(output / "dimensions.json", dimension_data)
    _write_json(output / "sampled_shapes.json", manifest)
    _write_json(output / "analytical_summary.json", summary)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(output.iterdir()) if path.is_file()}
    _write_json(output / "artifact_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifacts_sha256": hashes,
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--planning-tiles-json", type=Path,
                        help="Optional JSON list of [BM,BN,BK] tiles for memory estimates only")
    args = parser.parse_args()
    tiles = (json.loads(args.planning_tiles_json.read_text())
             if args.planning_tiles_json else PLANNING_TILES)
    summary = generate(args.output_dir, tiles)
    print(json.dumps({key: summary[key] for key in
                      ("full_lattice_count", "exploratory_count", "holdout_count")}))


if __name__ == "__main__":
    main()
