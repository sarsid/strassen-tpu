#!/usr/bin/env python3
"""Plot the sealed larger-LLM-shape evidence report; never execute MM kernels.

Requires the completed cohort's audited comparisons.json produced by the
independent summary tool. Dependencies: Python 3.10+, NumPy, Matplotlib.
Only PNG/PDF/JSON/Markdown outputs; a new output directory is mandatory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

VERSION = "plot_llm_mini_v001"
FAMILIES = ("native", "cubic", "strassen")
LABELS = {"native": "Native XLA", "cubic": "Tuned cubic", "strassen": "Tuned one-level Strassen"}
COLORS = {"native": "#3478AC", "cubic": "#DB8B32", "strassen": "#278C78"}
MODELS = (
    ("Qwen/Qwen3-8B", "Qwen3-8B", 4096, 12288),
    ("mistralai/Mistral-7B-v0.3", "Mistral-7B-v0.3", 4096, 14336),
    ("google/gemma-3-12b-pt", "Gemma3-12B · text backbone", 3840, 15360),
)
PROJECTIONS = ("gate_up_concat", "down")
M_VALUES = (2048, 8192, 16384)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def load_report(directory):
    directory = Path(directory).resolve()
    if (directory / "artifacts/comparisons.json").is_file():
        directory /= "artifacts"
    seal_path = directory / "artifact-manifest.json"
    seal = read_json(seal_path)
    if not {"comparisons.json", "audit.json"}.issubset(seal.get("sha256", {})):
        raise ValueError("The report seal must cover comparisons.json and audit.json")
    for name, digest in seal["sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha256(path) != digest:
            raise ValueError("Report artifact seal mismatch: " + name)
    data = read_json(directory / "comparisons.json")
    audit = read_json(directory / "audit.json")
    if audit.get("passed") is not True or audit.get("issues"):
        raise ValueError("The source evidence audit did not pass")
    coverage = data["coverage"]["GRID-confirm"]
    if coverage["groups"] != 18 or coverage["repeats"] != 30:
        raise ValueError("Expected a completed 18-group, 30-repeat confirmation")
    rows = data["complete_call_rows"]
    if len(rows) != 18 or len({row["shape_id"] for row in rows}) != 18:
        raise ValueError("Expected exactly 18 unique confirmed shape records")
    by_model = {}
    exclusions = []
    for model_id, _, hidden, intermediate in MODELS:
        group = [r for r in rows if r["model_id"] == model_id]
        if len(group) != 6:
            raise ValueError("Expected six confirmed shapes for " + model_id)
        if {(r["projection"], r["m"]) for r in group} != {(p, m) for p in PROJECTIONS for m in M_VALUES}:
            raise ValueError("Projection/M coverage differs from the frozen plan")
        group.sort(key=lambda r: (PROJECTIONS.index(r["projection"]), r["m"]))
        for row in group:
            if row.get("scope") != "call" or row.get("repeats") != 30:
                raise ValueError("Rows must describe complete-call confirmation with 30 rounds")
            expected = (hidden, 2 * intermediate) if row["projection"] == "gate_up_concat" else (intermediate, hidden)
            if (row["k"], row["n"]) != expected:
                raise ValueError("Actual model-derived dimensions changed")
            for family in FAMILIES:
                flag = row.get(family + "_eligible")
                if type(flag) is not bool:
                    raise ValueError("Every family requires an explicit eligibility boolean")
                if flag and (not positive(row.get(family + "_call_ms")) or
                             not positive(row.get(family + "_prepared_kernel_ms"))):
                    raise ValueError("Eligible family lacks finite positive timing in both scopes")
                if not flag:
                    exclusions.append({"shape_id": row["shape_id"], "model_id": model_id,
                        "family": family, "candidate": row.get(family + "_candidate"),
                        "status_by_scope": row.get(family + "_status_by_scope"),
                        "observed_ms_by_scope": row.get(family + "_observed_ms_by_scope"),
                        "row_refs": row.get(family + "_row_refs")})
            for reference in ("native", "cubic"):
                key = "strassen_vs_" + reference
                if row["strassen_eligible"] and row[reference + "_eligible"]:
                    point, low, high = (row.get(key), row.get(reference + "_ci_low"), row.get(reference + "_ci_high"))
                    if not all(positive(x) for x in (point, low, high)) or low > high:
                        raise ValueError("Eligible comparison lacks a valid paired CI")
                    if not math.isclose(point, row[reference + "_call_ms"] / row["strassen_call_ms"], rel_tol=1e-8):
                        raise ValueError("Reported speedup differs from the complete-call means")
        by_model[model_id] = group
    return directory, data, by_model, exclusions


def short_status(row, family):
    statuses = row.get(family + "_status_by_scope") or {}
    text = " ".join(str(v).lower() for v in statuses.values()) if isinstance(statuses, dict) else str(statuses).lower()
    if "oom" in text:
        return "OOM"
    if "numerical" in text:
        return "GATE"
    if "unsupported" in text:
        return "UNSUP."
    if not row.get(family + "_candidate"):
        return "NO PICK"
    return "INVALID"


def group_label(row):
    site = "Gate/up" if row["projection"] == "gate_up_concat" else "Down"
    return f"{site}\nM = {row['m']:,}"


def save_figure(fig, output, name):
    for extension in ("png", "pdf"):
        with (output / f"{name}.{extension}").open("xb") as stream:
            fig.savefig(stream, format=extension, dpi=180, facecolor="white")


def style_axis(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E4E9EF", linewidth=.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#BFC8D1")
    axis.tick_params(colors="#344454", labelsize=9)


def latency_figure(by_model, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np
    fig, axes = plt.subplots(3, 1, figsize=(13.2, 12.0))
    fig.subplots_adjust(top=.875, bottom=.115, left=.08, right=.975, hspace=.65)
    fig.suptitle("Larger LLM-derived shapes · complete-call latency", x=.08, y=.975,
                 ha="left", fontsize=19, fontweight="bold", color="#172D43")
    fig.text(.08, .944, "Synthetic BF16 matrix multiplication on one v5e cohort. Lower is faster.", fontsize=11, color="#516579")
    handles = [Patch(facecolor=COLORS[f], label=LABELS[f]) for f in FAMILIES]
    handles.append(Line2D([0], [0], color="#172D43", linewidth=1.4, label="Lowest eligible measured mean"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.075, .927), ncol=4, frameon=False, fontsize=9)
    winners = []
    for axis, (model_id, title, hidden, intermediate) in zip(axes, MODELS):
        rows = by_model[model_id]
        x = np.arange(6)
        width = .235
        maximum = max((r[f + "_call_ms"] for r in rows for f in FAMILIES if r[f + "_eligible"]), default=1)
        axis.set_ylim(0, maximum * 1.27)
        for i, family in enumerate(FAMILIES):
            for j, row in enumerate(rows):
                position = x[j] + (i - 1) * width
                if row[family + "_eligible"]:
                    value = row[family + "_call_ms"]
                    axis.bar(position, value, width=.215, color=COLORS[family], zorder=3)
                    label = f"{value:.3f}" if value < 1 else f"{value:.2f}" if value < 10 else f"{value:.1f}"
                    axis.text(position, value + maximum * .022, label, ha="center", va="bottom", fontsize=8, color="#25394C")
                else:
                    axis.plot(position, .025, marker="x", color="#B0443E", transform=axis.get_xaxis_transform(), clip_on=False)
                    axis.text(position, .065, short_status(row, family), color="#B0443E", fontsize=6.6,
                              ha="center", va="bottom", rotation=90, transform=axis.get_xaxis_transform())
        for j, row in enumerate(rows):
            available = [(row[f + "_call_ms"], f) for f in FAMILIES if row[f + "_eligible"]]
            if available:
                value = min(v for v, _ in available)
                families = [f for v, f in available if v == value]
                axis.hlines(value, j - .37, j + .37, colors="#172D43", linewidth=1.3, zorder=5)
                winners.append({"shape_id": row["shape_id"], "mean_ms": value,
                                "families": families, "scope": "measured minimum among eligible displayed arms; not a significance claim"})
        axis.axvline(2.5, color="#CAD3DD", linewidth=.8, linestyle=":")
        axis.set_title(f"{title}     H = {hidden:,} · I = {intermediate:,}", loc="left", fontsize=12, fontweight="bold", pad=13)
        axis.set_xticks(x, [group_label(r) for r in rows])
        axis.set_ylabel("Mean latency (ms)", fontsize=10)
        style_axis(axis)
    fig.text(.08, .067, "Bars: 30 fresh confirmation rounds; cubic and Strassen use independently selected screen winners.", fontsize=10, color="#516579")
    fig.text(.08, .048, "The horizontal marker identifies the lowest mean, not a statistically supported win. See the paired-CI figure.", fontsize=10, color="#516579")
    fig.text(.08, .029, "Padding and trimming are included; host transfer/compilation are excluded. × means missing/ineligible, never zero latency.", fontsize=9, color="#516579")
    save_figure(fig, output, "latency_by_model")
    plt.close(fig)
    return winners


def speedup_figure(by_model, output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(3, 1, figsize=(12.4, 12.0), sharex=True)
    fig.subplots_adjust(top=.88, bottom=.13, left=.215, right=.97, hspace=.55)
    fig.suptitle("Does Strassen beat each alternative?", x=.07, y=.975, ha="left", fontsize=19, fontweight="bold", color="#172D43")
    fig.text(.07, .946, "Complete-call speedup = alternative mean / Strassen mean. Values above 1 favor Strassen.", fontsize=10.4, color="#516579")
    fig.legend(handles=[Line2D([0], [0], marker="o", color=COLORS["native"], label="Native / Strassen"),
                        Line2D([0], [0], marker="s", color=COLORS["cubic"], label="Cubic / Strassen")],
               loc="upper left", bbox_to_anchor=(.065, .932), ncol=2, frameon=False, fontsize=10)
    lows, highs = [1.0], [1.0]
    classifications = Counter()
    for axis, (model_id, title, _, _) in zip(axes, MODELS):
        rows = by_model[model_id]
        for j, row in enumerate(rows):
            for reference, delta, marker in (("native", -.13, "o"), ("cubic", .13, "s")):
                key, y = "strassen_vs_" + reference, j + delta
                if not (row["strassen_eligible"] and row[reference + "_eligible"]):
                    axis.text(.985, y, "Ineligible / missing " + reference, transform=axis.get_yaxis_transform(),
                              ha="right", va="center", fontsize=7.5, color="#B0443E")
                    classifications[reference + "_ineligible"] += 1
                    continue
                point, low, high = row[key], row[reference + "_ci_low"], row[reference + "_ci_high"]
                lows.append(low); highs.append(high)
                axis.hlines(y, low, high, color=COLORS[reference], linewidth=1.7)
                axis.plot([low, high], [y, y], linestyle="none", marker="|", markersize=6, color=COLORS[reference])
                axis.plot(point, y, marker=marker, color=COLORS[reference], markersize=5)
                classifications[reference + ("_win" if low > 1 else "_loss" if high < 1 else "_inconclusive")] += 1
        axis.axvline(1, color="#24394B", linewidth=1, linestyle="--", zorder=0)
        axis.set_ylim(5.65, -.65)
        axis.set_yticks(range(6), [group_label(r).replace("\n", " · ") for r in rows])
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=13)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", color="#E4E9EF", linewidth=.7)
        axis.tick_params(labelsize=9)
    axes[-1].set_xlim(max(0, min(lows) * .9), max(highs) * 1.08)
    axes[-1].set_xlabel("Alternative / Strassen latency ratio", fontsize=11)
    fig.text(.07, .075, "Intervals: paired bootstrap 95% CIs, 2,000 resamples of the same 30 rounds; no multiplicity correction.", fontsize=10, color="#516579")
    fig.text(.07, .055, "Entire CI above 1: evidence of a Strassen win; below 1: loss; crossing 1: inconclusive, not equivalence.", fontsize=10, color="#516579")
    fig.text(.07, .035, "These are fixed Gaussian matrix inputs and model-derived shapes, not end-to-end LLM speed or quality results.", fontsize=9, color="#516579")
    save_figure(fig, output, "strassen_speedup_ci")
    plt.close(fig)
    return dict(classifications)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    directory, report, by_model, exclusions = load_report(args.report_dir)
    output = args.output_dir.resolve()
    if output.is_relative_to(directory):
        raise ValueError("Output must be outside the immutable input report directory")
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
                               "axes.labelcolor": "#344454", "figure.facecolor": "white", "savefig.facecolor": "white"})
    winners = latency_figure(by_model, output)
    classifications = speedup_figure(by_model, output)
    write_json(output / "plot_data.json", {"complete_call_rows": report["complete_call_rows"],
        "measured_mean_minima": winners, "pointwise_ci_classifications": classifications,
        "excluded_from_latency_bars": exclusions, "source_report_provenance": report.get("provenance"),
        "source_report_coverage": report["coverage"],
        "source_report_audit": read_json(directory / "audit.json")})
    write_json(output / "provenance.json", {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(__file__), "report_directory": str(directory),
        "report_artifact_manifest_sha256": sha256(directory / "artifact-manifest.json"),
        "comparisons_sha256": sha256(directory / "comparisons.json"), "arguments": {k: str(v) for k,v in vars(args).items()},
        "python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
        "input_contract": "Sealed evidence summary of completed cohort; raw rounds, screen choices and recorded CIs are audited upstream.",
        "scope": "Synthetic BF16 MM; no actual checkpoint values, model quality, end-to-end LLM timing, or held-out shape claim.",
        "families": LABELS, "shape_count": 18, "confirmation_rounds": 30,
        "eligibility": "Both scopes must pass the upstream numerical and complete-round gates; failing timings remain in plot_data.json but are not represented as valid bars.",
        "mean_marker": "Minimum among eligible measured means only; separate CI figure states statistical comparisons.",
        "prepared_kernel": "Not charted here; its recorded mean and eligibility are retained in plot_data.json."})
    with (output / "README.md").open("x") as stream:
        stream.write("# Larger model-derived MM shape figures\n\n"
            "`latency_by_model.png` and `.pdf` group six shapes per model and compare complete-call means. "
            "The dark horizontal marker is the smallest eligible mean, not a significance claim.\n\n"
            "`strassen_speedup_ci.png` and `.pdf` show separate Native/Strassen and Cubic/Strassen comparisons. "
            "Values above one favor Strassen; only an entire pointwise 95% CI above one supports a win. "
            "Intervals crossing one are inconclusive, not proof of equality. No multiplicity correction is applied.\n\n"
            "Gate/up means the concatenated pure matrix projection; down is an independent synthetic matrix input. "
            "All inputs are Gaussian BF16. These are not full-model speed or quality measurements. "
            "Padding/trimming is included; input generation, host transfer, compilation and reference calculation are excluded.\n\n"
            f"{len(exclusions)} family/shape entries were excluded from valid latency bars; every omission and its source status "
            "is retained in `plot_data.json`. Missing or failing values are marked, never shown as zero-latency bars.\n\n"
            "The source evidence report is authenticated by its artifact seal. `provenance.json` records input hashes, "
            "plot source hash and plotting-library versions.\n")
    write_json(output / "artifact_manifest.json", {"schema_version": 1, "sha256": {
        p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(json.dumps({"completed": True, "output_dir": str(output), "figures": 2, "shape_count": 18}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
