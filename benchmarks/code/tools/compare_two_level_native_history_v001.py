"""Read preserved measurements; compare Native with tuned Strassen without TPU work."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ["STRASSEN_PROJECT_ROOT"])
OUT = Path(os.environ["STRASSEN_EXECUTION_DIR"]) / "artifacts"
OLD = ROOT / "runs/20260921-llm-large-shapes-cohort-v001/phases/20260921-llm-large-shapes-cohort-v001-grid-confirm-5af514"
NEW = ROOT / "runs/20260921-two-level-tune-cohort-v001/phases/20260921-two-level-tune-cohort-v001-level-tune-confirm-60ae50"
PROVENANCE = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_phase(path):
    assert json.loads((path / "completion.json").read_text())["status"] == "completed"
    manifest = json.loads((path / "artifact-manifest.json").read_text())["sha256"]
    for name in ("artifacts/results.jsonl", "artifacts/environment.json"):
        assert digest(path / name) == manifest[name], name
        PROVENANCE[str((path / name).relative_to(ROOT))] = digest(path / name)
    rows = [json.loads(line) for line in (path / "artifacts/results.jsonl").read_text().splitlines()]
    env = json.loads((path / "artifacts/environment.json").read_text())
    assert env["qualified_single_v5e"]
    return rows, env["identity"]


def summarize(rows, shape, arm, expected_seeds):
    cases = [r for r in rows if r.get("event") == "case_result" and
             r.get("shape_id") == shape and r.get("arm_id") == arm and r.get("scope") == "call"]
    assert len(cases) == expected_seeds
    samples, errors = [], []
    for case in cases:
        assert case["status"] == "ok" and case["correctness"]["pass"]
        meta = case["kernel_metadata"]
        assert (meta["input_dtype"], meta["accumulation_dtype"], meta["output_dtype"], meta["dot_precision"]) == ("bfloat16", "float32", "float32", "DEFAULT")
        assert not meta["padding_required"]
        raw = [r for r in rows if r.get("event") == "sample" and r.get("case_id") == case["case_id"] and r.get("arm_id") == arm and r.get("scope") == "call"]
        assert len(raw) == case["timing"]["sample_count"] == 30
        assert sorted(r["round"] for r in raw) == list(range(30))
        values = [r["elapsed_ms"] for r in raw]
        assert all(math.isfinite(v) and v > 0 for v in values)
        assert math.isclose(statistics.fmean(values), case["timing"]["mean_ms"], rel_tol=1e-12)
        samples.extend(values)
        errors.append(case["correctness"]["relative_l2"])
    return {"mean_ms": statistics.fmean(samples), "sample_count": len(samples),
            "seeds": [r["seed"] for r in cases], "relative_l2_mean": statistics.fmean(errors),
            "relative_l2_range": [min(errors), max(errors)],
            "shape_mkn": cases[0]["kernel_metadata"]["shape_mkn"],
            "tile_bm_bn_bk": cases[0]["kernel_metadata"]["tile_bm_bn_bk"],
            "samples_ms": samples}


def main():
    OUT.mkdir()
    old, old_env = load_phase(OLD)
    new, new_env = load_phase(NEW)
    differences = {k: {"historical": old_env.get(k), "current": new_env.get(k)}
                   for k in old_env.keys() | new_env.keys() if old_env.get(k) != new_env.get(k)}
    assert set(differences) == {"allocation_id", "colab_endpoint", "boot_id", "host_id", "hostname"}
    native = summarize(old, "qwen_3_8b_gate_up_concat_m16384", "native_xla", 1)
    qwen = {arm: summarize(new, "qwen_gate_up_m16384", arm, 3) for arm in ("one_level", "two_level")}
    square = {arm: summarize(new, "square_12288", arm, 3) for arm in ("one_level", "two_level")}
    assert all(v["shape_mkn"] == native["shape_mkn"] == [16384, 4096, 24576] for v in qwen.values())
    for value in qwen.values():
        value["historical_native_speedup"] = native["mean_ms"] / value["mean_ms"]
        value["historical_native_latency_reduction_pct"] = 100 * (1 - value["mean_ms"] / native["mean_ms"])

    # Retain the rejected square evidence and its BF16-output source contract.
    legacy = ROOT.parent / "strassen-tpu/archive/initial-snapshot"
    legacy_files = ["evidence/strassen_v5e_best_square_12k_16k.jsonl",
                    "evidence/strassen_v5e_classical_large.jsonl",
                    "benchmarks/benchmark_common.py", "benchmarks/benchmark_strassen.py"]
    legacy_values = []
    for name in legacy_files:
        source = legacy / name
        target = OUT / "legacy_square_evidence" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        PROVENANCE[str(source)] = digest(source)
        if source.suffix == ".jsonl":
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            case = next(r for r in rows if r.get("kind") == "performance" and r.get("size") == 12288)
            mean = statistics.fmean(case["native"]["samples_ms"])
            assert math.isclose(mean, case["native"]["mean_ms"], rel_tol=1e-12)
            legacy_values.append({"source": name, "mean_ms": mean, "sample_count": len(case["native"]["samples_ms"])})
    result = {"scope": "Historical comparison only; no new TPU measurements",
              "qwen": {"native_historical": native, **qwen}, "square_current": square,
              "square_legacy_excluded": {"measurements": legacy_values, "reason": "BF16 output and constant timing inputs; current study uses FP32 output and Gaussian inputs. Full legacy allocation/software identity unavailable."},
              "identity_differences": differences, "source_sha256": PROVENANCE,
              "limitations": ["Native and current Strassen measurements use different allocations, seeds and interleaving; no paired Native confidence interval is valid.", "Gaussian matrix multiplication only; not end-to-end LLM inference.", "Matching input/output dtypes do not imply matching accuracy; two-level BF16 pre-adds increase error.", "No qualified historical FP32-output Native measurement found for square 12288."]}
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Tuned Strassen versus historical Native", "", "No new TPU execution. Exact Qwen shape: (16384 x 4096) @ (4096 x 24576).", "", "| Implementation | Mean ms | Timing samples | Relative L2 | Lower latency vs historical Native |", "|---|---:|---:|---:|---:|"]
    for label, data in [("Native XLA, older run", native), ("One-level, newly tuned", qwen["one_level"]), ("Two-level, newly tuned", qwen["two_level"])]:
        reduction = f'{data["historical_native_latency_reduction_pct"]:.2f}%' if "historical_native_latency_reduction_pct" in data else "—"
        lines.append(f'| {label} | {data["mean_ms"]:.6f} | {data["sample_count"]} | {data["relative_l2_mean"]:.3g} | {reduction} |')
    lines += ["", "The two-level result suggests a 1.291x historical speedup. This is promising, but Native must be rerun alongside the selected kernels on the same allocation for a controlled claim.", "", "Both modern runs use one v5e device, JAX/jaxlib 0.7.2, libtpu 0.0.21.1, matching runtime image/flags, BF16 inputs and FP32 output/accumulation. Timings include the device call (padding/crop if needed), excluding compilation and host transfer. Neither shape needs padding. Native has 30 rounds on one seed; each current Strassen arm has 30 rounds on each of three fresh seeds. Native uses default compiler options.", "", "Relative L2 uses the same host FP32 reference convention on quantized BF16 operands, but different input seeds. Native is substantially more accurate. The speed comparison is not an accuracy-equivalence result.", "", "## Square 12288", "", f'Current one-level: {square["one_level"]["mean_ms"]:.6f} ms; two-level: {square["two_level"]["mean_ms"]:.6f} ms. Older Native measurements are {min(r["mean_ms"] for r in legacy_values):.3f}–{max(r["mean_ms"] for r in legacy_values):.3f} ms, but return BF16 rather than FP32 and use constant inputs. Retained as context only; excluded from the matched table, figure and speedup claims.', "", "## Evidence", ""]
    lines += [f'- `{name}` (SHA256 `{sha}`)' for name, sha in PROVENANCE.items()]
    (OUT / "COMPARISON.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    vals = [native["mean_ms"], qwen["one_level"]["mean_ms"], qwen["two_level"]["mean_ms"]]
    bars = ax.bar(["Native XLA\nHistorical run", "Strassen 1 level\nNew tuned run", "Strassen 2 levels\nNew tuned run"], vals, color=["#64748b", "#2563eb", "#0d9488"], width=.60)
    ax.bar_label(bars, labels=[f"{v:.2f} ms" for v in vals], padding=6, fontsize=12)
    ax.axhline(min(vals), color="#0d9488", linestyle="--", linewidth=1)
    ax.set_ylim(0, 23)
    ax.set_ylabel("Mean device-call latency (ms); lower is better")
    ax.set_title("Qwen gate/up: historical Native comparison\n(16384 × 4096) @ (4096 × 24576)", pad=15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(alpha=.2)
    fig.supxlabel("Different TPU allocations and seeds; provisional comparison.\nBF16 inputs / FP32 output. Native is substantially more accurate.", fontsize=10)
    fig.savefig(OUT / "native_comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"status": "complete", "qwen_means_ms": vals, "two_level_latency_reduction_pct": qwen["two_level"]["historical_native_latency_reduction_pct"], "square_legacy_excluded": legacy_values, "timing_samples_replayed": 390, "new_tpu_executions": 0}))


if __name__ == "__main__":
    main()
