#!/usr/bin/env python3
"""Plot sealed optimization findings without re-estimating or pooling results.

Reads the findings.json and sibling artifact_manifest.json produced by
summarize_strassen_optimized_v001.py. Each phase/allocation receives independent
figures. Failed or ineligible cases are annotations, never zero-latency bars.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


VERSION = "plot_strassen_optimized_v002"
REFERENCE_LABELS = {"old_strassen": "Frozen Strassen", "native_default": "Native default", "native_vmem_64m": "Native 64 MiB"}
REFERENCE_COLORS = {"old_strassen": "#4c5560", "native_default": "#277dad", "native_vmem_64m": "#8659a6"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def read_findings(path):
    path = Path(path).resolve()
    seal_path = path.parent / "artifact_manifest.json"
    seal = json.loads(seal_path.read_text())
    require(seal.get("sha256", {}).get(path.name) == sha256(path), "Findings file is absent from or differs from its seal")
    for name, expected in seal["sha256"].items():
        artifact = (path.parent / name).resolve()
        require(artifact.is_relative_to(path.parent) and artifact.is_file(), f"Invalid sealed summary path: {name}")
        require(sha256(artifact) == expected, f"Summary artifact hash mismatch: {name}")
    report = json.loads(path.read_text())
    require(report.get("version") == "summarize_strassen_optimized_v001", "Unsupported findings schema")
    require(isinstance(report.get("runs"), list) and report["runs"], "Findings contain no run")
    seen = set()
    for run in report["runs"]:
        require(run["phase"] in ("screen", "confirm"), "Only distinct screen/confirm TPU cohorts are supported")
        require(run["rounds"] in (7, 30), "Unexpected per-arm timing count")
        require(run["phase"] != "confirm" or run["rounds"] == 30, "Confirmation requires 30 rounds")
        identity = run["provenance"]["identity"]
        devices = identity.get("devices", [])
        require(identity.get("allocation_id") and len(devices) == 1 and devices[0].get("platform") == "tpu", "Findings do not identify a single TPU allocation")
        key = (run["phase"], identity["allocation_id"])
        require(key not in seen, "Duplicate phase/allocation would make plot provenance ambiguous")
        seen.add(key)
        shape_ids = set()
        require(bool(run["shape_tables"]), "Run contains no shape table")
        for shape in run["shape_tables"]:
            require(shape["shape_id"] not in shape_ids and shape["scope"] == "call", "Duplicate shape or mixed timing scope")
            shape_ids.add(shape["shape_id"])
            for field in ("shape_mnk", "tile_bm_bn_bk"):
                values = shape[field]
                require(len(values) == 3 and all(type(v) is int and v > 0 for v in values), f"Invalid {field}")
            candidates = {row["candidate_id"]: row for row in shape["rows"]}
            require(len(candidates) == len(shape["rows"]) and set(REFERENCE_LABELS) <= candidates.keys(), "Duplicate candidate or missing reference")
            for row in shape["rows"]:
                require(type(row["eligible"]) is bool, "Invalid eligibility flag")
                if row["eligible"]:
                    require(row["status"] == "ok" and finite(row["mean_ms"]) and row["mean_ms"] > 0
                            and row["sample_count"] == run["rounds"] and (row.get("correctness") or {}).get("pass") is True,
                            "An ineligible or incomplete mean was marked eligible")
                for ref in REFERENCE_LABELS:
                    comparison = row["comparisons"][ref]
                    if not comparison["available"]:
                        continue
                    require(row["eligible"] and candidates[ref]["eligible"], "Ineligible candidate/reference has a plotted comparison")
                    point = comparison["time_reduction_percent"]
                    interval = comparison["time_reduction_ci95_percent"]
                    require(finite(point) and len(interval) == 2 and all(finite(v) for v in interval) and interval[0] <= interval[1], "Invalid recorded time-reduction interval")
                    expected = 100 * (1 - row["mean_ms"] / candidates[ref]["mean_ms"])
                    require(math.isclose(point, expected, rel_tol=1e-9, abs_tol=1e-9), "Time reduction does not match the named reference mean")
                    require(comparison["paired_sample_count"] == run["rounds"], "Comparison has incomplete pairing")
    return report, {"findings_path": str(path), "findings_sha256": sha256(path),
                    "summary_seal_sha256": sha256(seal_path), "summarizer_sha256": report["summarizer_sha256"]}


def candidate_label(row):
    label = row["label"]
    if row["candidate_id"].startswith("new_optimized_schedule_"):
        alias = row["candidate_id"].removeprefix("new_optimized_schedule_").replace("_", " ")
        label = f"Optimized: {alias}"
    if row.get("hybrid") and "hybrid" not in label.lower():
        label += " (hybrid)"
    if row.get("native_only_fallback"):
        label += "\n(native-only fallback)"
    return textwrap.fill(label, 30, break_long_words=False, break_on_hyphens=False)


def arm_color(row):
    cid = row["candidate_id"]
    if cid in REFERENCE_COLORS:
        return REFERENCE_COLORS[cid]
    if row.get("hybrid"):
        return "#bd7340"
    if cid in ("new_optimized", "new_optimized_nmk"):
        return "#138a7a"
    return "#89aaa4"


def shape_title(shape):
    dims = " × ".join(f"{v:,}" for v in shape["shape_mnk"])
    tile = " × ".join(f"{v:,}" for v in shape["tile_bm_bn_bk"])
    return f"M,N,K = {dims}\nCustom tile BM,BN,BK = {tile}"


def phase_label(run):
    name = "Screening" if run["phase"] == "screen" else "Fresh confirmation"
    return name + (" — incomplete run" if not run["completed"] else "")


def configure_axis(ax, shape):
    labels = [candidate_label(row) for row in shape["rows"]]
    ax.set_yticks(range(len(labels)), labels, fontsize=8.5)
    ax.set_ylim(len(labels) - 0.45, -0.6)
    ax.set_title(shape_title(shape), fontsize=10.5, pad=12, loc="left")
    ax.grid(axis="x", alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))


def make_figure(shapes):
    nrows = (len(shapes) + 1) // 2
    height = max(7, nrows * (3.0 + max(len(s["rows"]) for s in shapes) * 0.18) + 2.0)
    fig, axes = plt.subplots(nrows, 2, figsize=(16.5, height), squeeze=False)
    fig.subplots_adjust(left=0.16, right=0.975, bottom=0.17, top=0.84, wspace=0.58, hspace=0.50)
    axes = list(axes.flat)
    for ax in axes[len(shapes):]:
        ax.set_visible(False)
    return fig, axes


def add_context(fig, run, title, note):
    identity = run["provenance"]["identity"]
    fig.suptitle(f"{phase_label(run)} · {title}", x=0.04, y=0.977, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.04, 0.932, f"{run['rounds']} paired rounds per eligible arm · {identity.get('device_kind', 'TPU')} · seed {run['provenance']['seed']}", fontsize=10, ha="left")
    fig.text(0.04, 0.905, f"Allocation: {identity['allocation_id']}", fontsize=8.5, color="#4c5560", ha="left")
    fig.text(0.04, 0.103, textwrap.fill(note, 160), fontsize=9, va="top", color="#273b47")
    scope = ("Complete device call: preparation + matmul + finish; excludes compilation and host transfer.\n"
             "Fixed matched-tile Gaussian ablation. Phases, allocations and timing scopes remain separate. Lowest means are descriptive, not a selection policy.")
    fig.text(0.04, 0.025, scope, fontsize=8.2, va="bottom", color="#52606a")


def timing_figure(run, shapes):
    fig, axes = make_figure(shapes)
    for ax, shape in zip(axes, shapes):
        configure_axis(ax, shape)
        eligible = [row["mean_ms"] for row in shape["rows"] if row["eligible"]]
        maximum = max(eligible) if eligible else 1.0
        ax.set_xlim(0, maximum * 1.38)
        for y, row in enumerate(shape["rows"]):
            if row["eligible"]:
                ax.barh(y, row["mean_ms"], height=0.64, color=arm_color(row),
                        hatch="//" if row.get("hybrid") else None, edgecolor="white", linewidth=0.5)
                ax.text(row["mean_ms"] + maximum * 0.018, y, f"{row['mean_ms']:.4g} ms", va="center", fontsize=8.5)
            else:
                status = row["status"] if row["status"] != "ok" else "ineligible"
                detail = f"{status}; n={row['sample_count']}"
                if finite(row.get("mean_ms")):
                    detail += f"; recorded {row['mean_ms']:.4g} ms"
                ax.text(0.02, y, textwrap.fill(detail, 44), transform=ax.get_yaxis_transform(), va="center", fontsize=8, color="#a43b38")
        ax.set_xlabel("Arithmetic mean complete-call latency (ms)", fontsize=9)
    add_context(fig, run, "measured complete-call latency",
                "Bars show eligible arithmetic means; the findings do not supply confidence intervals for individual mean latencies. Failed/ineligible arms are annotated without a latency bar. Hatched arms combine a Strassen core with native tails.")
    return fig


def comparison_figure(run, shapes):
    fig, axes = make_figure(shapes)
    references = list(REFERENCE_LABELS)
    for ax, shape in zip(axes, shapes):
        configure_axis(ax, shape)
        limits = [0.0]
        unavailable = 0
        for y, row in enumerate(shape["rows"]):
            available = 0
            for offset, ref in zip((-0.21, 0., 0.21), references):
                comparison = row["comparisons"][ref]
                if not comparison["available"]:
                    unavailable += 1
                    continue
                available += 1
                point = comparison["time_reduction_percent"]
                lo, hi = comparison["time_reduction_ci95_percent"]
                limits.extend((point, lo, hi))
                position = y + offset
                color = REFERENCE_COLORS[ref]
                self_reference = row["candidate_id"] == ref
                if not self_reference:
                    # Draw intervals independently: a bootstrap interval need not contain its point estimate.
                    ax.hlines(position, lo, hi, color=color, linewidth=1.4)
                    ax.vlines((lo, hi), position - 0.045, position + 0.045, color=color, linewidth=1)
                ax.plot(point, position, "o", markersize=4.0, markeredgewidth=1,
                        markeredgecolor=color, markerfacecolor="white" if self_reference else color)
            if not available:
                status = row["status"] if row["status"] != "ok" else "comparison unavailable"
                ax.text(0.025, y, status, transform=ax.get_yaxis_transform(), va="center", color="#a43b38", fontsize=8)
        lo, hi = min(limits), max(limits)
        margin = max(2.0, (hi-lo) * 0.08)
        ax.set_xlim(lo-margin, hi+margin)
        ax.axvline(0, color="#68737d", linestyle=":", linewidth=1)
        ax.set_xlabel("Time reduction vs reference (%) · positive = faster", fontsize=9)
        if unavailable:
            ax.text(0.99, 1.015, f"{unavailable} unavailable comparisons", transform=ax.transAxes, ha="right", fontsize=7.5, color="#a43b38")
    legend = [Line2D([], [], color=REFERENCE_COLORS[ref], marker="o", markersize=4, label=f"vs {REFERENCE_LABELS[ref]}") for ref in references]
    fig.subplots_adjust(top=0.79)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.61, 0.879), ncol=3, frameon=False, fontsize=9)
    add_context(fig, run, "paired time reduction",
                "Dots and lines use the recorded point estimates and pointwise 95% paired-bootstrap intervals; they are not multiplicity-adjusted. Hollow self-reference dots are identities, not confidence intervals. Missing comparisons are not plotted as zero.")
    return fig


def generate(findings, output_dir):
    findings = Path(findings).resolve()
    report, provenance = read_findings(findings)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    figures = []
    for run_index, run in enumerate(report["runs"], 1):
        for start in range(0, len(run["shape_tables"]), 4):
            shapes = run["shape_tables"][start:start+4]
            page = start//4 + 1
            for kind, make in (("complete_call_latency", timing_figure), ("paired_time_reduction", comparison_figure)):
                fig = make(run, shapes)
                prefix = f"{run['phase']}_{run_index:02}_{page:02}_{kind}"
                paths = []
                for suffix in ("png", "svg"):
                    path = out / f"{prefix}.{suffix}"
                    fig.savefig(path, dpi=165, bbox_inches="tight", pad_inches=0.25, facecolor="white")
                    paths.append(path.name)
                plt.close(fig)
                figures.append({"id": prefix, "phase": run["phase"], "allocation_id": run["provenance"]["identity"]["allocation_id"],
                                "scope": "call", "kind": kind, "rounds": run["rounds"],
                                "shape_ids": [s["shape_id"] for s in shapes], "shape_order": "M,N,K",
                                "eligible_complete_call_cases": sum(r["eligible"] for s in shapes for r in s["rows"]),
                                "ineligible_complete_call_cases": sum(not r["eligible"] for s in shapes for r in s["rows"]),
                                "files": paths})
    require(sha256(findings) == provenance["findings_sha256"], "Findings changed during plotting")
    (out/"figures.json").write_text(json.dumps(figures, indent=2)+"\n")
    notes = ["# Strassen optimization static figures", "", report["interpretation"], "",
             "Only complete-call results are plotted. Individual latency confidence intervals are unavailable in this schema; paired comparison intervals are reused without re-estimation.", "",
             "Each phase/allocation has separate figures. Failed/ineligible arms have textual status annotations; they never become zero-valued latency bars. Hatched latency bars identify the hybrid Strassen/native-tail implementation.", ""]
    for item in figures:
        notes.append(f"- {item['id']}: {', '.join(item['files'])}")
    (out/"README.md").write_text("\n".join(notes)+"\n")
    manifest = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": sha256(__file__), "input": provenance, "figure_count": len(figures),
                "new_measurements": False, "interval_method": "Use summarizer's recorded pointwise paired intervals; no re-estimation",
                "sha256": {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}}
    with (out/"artifact_manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False); stream.write("\n")
    return {"output_dir": str(out), "figure_count": len(figures), "file_count": len(manifest["sha256"])+1}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--findings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.findings, args.output_dir)))


if __name__ == "__main__":
    main()
