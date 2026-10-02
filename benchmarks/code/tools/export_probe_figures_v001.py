#!/usr/bin/env python3
"""Export immutable static pictures; keep target-grid and auxiliary probes apart."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
import numpy as np

INK = "#172c40"
SUBTLE = "#52677b"
WIN, LOSS, TIE = "#098879", "#cf5b58", "#8b98a5"
COLORS = {"win": WIN, "loss": LOSS, "inconclusive": TIE}
NATIVE = "#657a94"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": SUBTLE, "ytick.color": SUBTLE,
                     "axes.edgecolor": "#b7c6d2", "svg.fonttype": "none",
                     "savefig.facecolor": "white", "figure.facecolor": "white"})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def allowed_dimension(value):
    if type(value) is not int or value < 1:
        return False
    while value % 2 == 0:
        value //= 2
    return value in (1, 3)


def on_grid(row):
    return all(allowed_dimension(row[axis]) for axis in ("m", "n", "k"))


def saved(row):
    return 100 * (1 - row["strassen_ms"] / row["native_ms"])


def classification(row):
    return row.get("native_classification", "inconclusive")


def legend_items():
    return [Line2D([], [], marker="o", linestyle="", markerfacecolor=COLORS[name],
                   markeredgecolor="white", markersize=9, label=label)
            for name, label in (("win", "Strassen faster"), ("loss", "Native faster"),
                                ("inconclusive", "CI overlaps parity"))]


def subtitle(fig, title, detail):
    fig.text(.055, .956, title, fontsize=21, weight="bold", va="top")
    fig.text(.055, .909, detail, fontsize=11, color=SUBTLE, va="top")


def footnote(fig, note, allocation):
    fig.text(.055, .045, note, fontsize=9, color=SUBTLE, va="bottom")
    fig.text(.055, .019, f"Complete-call BF16 / FP32 output · {allocation}",
             fontsize=8, color=SUBTLE, va="bottom")


def format_extent(exponent):
    return f"{2 ** exponent:,}"


def setup_slice(ax, x_limits, y_limits):
    ax.set(xlim=x_limits, ylim=y_limits, xlabel="Output side S = M = N (log₂ scale)",
           ylabel="Reduction dimension K (log₂ scale)")
    xt = [0, 2, 4, 6, 8, 10, 12, 14] if x_limits[0] < 0 else [2, 6, 8, 10, 12, 14]
    yt = [0, 4, 8, 10, 12, 14, 16, 17] if y_limits[0] < 0 else [2, 6, 8, 10, 12, 14, 16, 17]
    xt = [t for t in xt if x_limits[0] <= t <= x_limits[1]]
    yt = [t for t in yt if y_limits[0] <= t <= y_limits[1]]
    ax.set_xticks(xt, [format_extent(t) for t in xt], rotation=0)
    ax.set_yticks(yt, [format_extent(t) for t in yt])
    ax.grid(color="#e6edf2", linewidth=.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def slice_points(ax, rows, *, values=True, small=False):
    points = sorted((row for row in rows if on_grid(row) and row["m"] == row["n"]),
                    key=lambda row: (row["m"], row["k"]))
    for row in points:
        x, y = math.log2(row["m"]), math.log2(row["k"])
        ax.scatter(x, y, s=70 if small else 90, c=COLORS[classification(row)],
                   edgecolors="white", linewidths=1.1, zorder=3)
        # The complete current slice has tightly spaced long-K columns. Keep
        # every marker, but label fixed representative points without overlap.
        selected_label = (len(points) <= 20 or row["m"] == row["k"]
                          or (row["m"], row["k"]) in {
                              (384, 16384), (1536, 16384), (1536, 131072),
                              (2048, 32768), (3072, 65536), (16384, 1536)})
        if values and selected_label:
            offset = (-9, 0) if row["m"] == 1536 else (8, 0)
            if len(points) > 20 and (row["m"], row["k"]) in {
                    (384, 16384), (2048, 32768), (3072, 65536)}:
                offset = (0, 11)
            if len(points) > 20 and row["m"] == row["k"] and row["m"] >= 4096:
                offset = (0, 11)
            ha = "center" if offset[0] == 0 else "right" if offset[0] < 0 else "left"
            ax.annotate(f"{saved(row):+.1f}%", (x, y), xytext=offset,
                        textcoords="offset points", fontsize=7.8 if small else 8.5,
                        ha=ha, va="center", color=INK, zorder=4,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.92, pad=.25))
    return points


def grid_slice(rows, allocation):
    fig = plt.figure(figsize=(13.6, 8.8))
    ax = fig.add_axes([.085, .18, .77, .655])
    subtitle(fig, "Target-grid slice: square outputs (M = N)",
             "Only dimensions 2ⁱ or 3 × 2ⁱ. Selected labels show Strassen time saved relative to the frozen tuned-native winner.")
    setup_slice(ax, (1, 15.15), (1, 17.75))
    selected = slice_points(ax, rows)
    ax.add_patch(Rectangle((12, 12), 2.75, 5.35, fill=False, edgecolor=WIN,
                           linestyle=(0, (5, 4)), linewidth=1.3, zorder=1))
    fig.text(.875, .76, "8 / 8 wins\nin outlined\nregion", fontsize=10,
             color=WIN, va="top", linespacing=1.6)
    ax.text(3.5, 14.8, "27 measured points\n21 wins · 3 losses · 3 inconclusive",
            fontsize=12, linespacing=1.6, color=INK)
    ax.text(3.5, 12.0, "Empty space = unmeasured.\nThe outline is a descriptive hypothesis,\nnot a validated decision boundary.",
            fontsize=10, linespacing=1.6, color=SUBTLE)
    fig.legend(handles=legend_items(), loc="lower center", bbox_to_anchor=(.5, .074),
               ncol=3, frameon=False, fontsize=10)
    footnote(fig, "Colors use pointwise paired 95% CIs; no correction for multiple comparisons. Off-grid probes and holdouts are excluded.", allocation)
    assert len(selected) == 27
    return fig


def geometry_scatter(rows, allocation):
    points = [row for row in rows if on_grid(row)]
    core = [row for row in points if min(row[a] for a in ("m", "n", "k")) >= 4096]
    fig = plt.figure(figsize=(13.6, 9.2))
    subtitle(fig, "The measured target grid in M, N and K", 
             "96 measured shapes · 53 Strassen wins · 24 native wins · 19 inconclusive. No off-grid points are shown.")
    ax = fig.add_axes([.055, .10, .68, .75], projection="3d")
    norm = TwoSlopeNorm(vmin=-70, vcenter=0, vmax=15)
    marker = {"win": "o", "loss": "X", "inconclusive": "s"}
    for kind in COLORS:
        group = [row for row in points if classification(row) == kind]
        sc = ax.scatter(*[[math.log2(row[a]) for row in group] for a in ("m", "n", "k")],
                        c=[saved(row) for row in group], cmap="RdYlGn", norm=norm,
                        s=49, marker=marker[kind], edgecolor="#254158", linewidth=.45,
                        depthshade=False, alpha=.95)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticks([2, 6, 10, 14, 17])
        axis.set_ticklabels([format_extent(e) for e in (2, 6, 10, 14, 17)], fontsize=8)
        axis.pane.fill = False
    ax.set(xlabel="M (log₂ scale)", ylabel="N (log₂ scale)", zlabel="K (log₂ scale)",
           xlim=(1, 17.7), ylim=(1, 17.7), zlim=(1, 17.7))
    ax.view_init(elev=22, azim=-54)
    ax.set_box_aspect((1, 1, 1))
    cube = list(itertools.product((12, 17.35), repeat=3))
    for a, b in itertools.combinations(cube, 2):
        if sum(a[i] != b[i] for i in range(3)) == 1:
            ax.plot(*zip(a, b), color=WIN, linestyle=(0, (4, 5)), alpha=.45, linewidth=.8)
    cax = fig.add_axes([.785, .40, .021, .31])
    bar = fig.colorbar(sc, cax=cax, ticks=[-70, -40, -10, 0, 5, 10, 15])
    bar.set_label("Strassen time saved (%)", fontsize=10, labelpad=10)
    fig.text(.75, .81, "Observed large-shape region", fontsize=12, weight="bold")
    fig.text(.75, .735, "M, N, K ≥ 4,096:\n19 / 19 measured points win",
             fontsize=10.5, linespacing=1.6)
    fig.text(.75, .26, "Circle: Strassen wins\nCross: native wins\nSquare: CI overlaps parity",
             fontsize=10, linespacing=1.7)
    fig.text(.75, .15, "Dashed box marks a hypothesis.\nIt makes no claim about\nunmeasured grid points.",
             fontsize=10, color=SUBTLE, linespacing=1.5)
    footnote(fig, "Colors show 100 × (1 − Strassen time / native time). Classifications use pointwise paired 95% CIs, not simultaneous guarantees.", allocation)
    assert len(points) == 96 and len(core) == 19 and all(classification(row) == "win" for row in core)
    return fig


def boundary_picture(rows, allocation):
    selected = {row["m"]: row for row in rows if row["m"] == row["n"] == row["k"]
                and row["m"] in (8191, 8192, 8193)}
    assert set(selected) == {8191, 8192, 8193}
    points = [selected[s] for s in (8191, 8192, 8193)]
    fig = plt.figure(figsize=(13.6, 8.2))
    subtitle(fig, "Separate boundary probe around the 8,192 square", 
             "Auxiliary diagnostic only: 8,191 and 8,193 are outside the requested power/midpoint grid.")
    left = fig.add_axes([.08, .19, .385, .61])
    right = fig.add_axes([.57, .19, .365, .61])
    x = np.arange(3)
    for offset, key, color, label in ((-.19, "native_ms", NATIVE, "Tuned native"),
                                    (.19, "strassen_ms", "#0b9885", "Strassen")):
        bars = left.bar(x + offset, [row[key] for row in points], width=.34,
                        color=color, label=label, zorder=3)
        left.bar_label(bars, fmt="%.2f", padding=5, fontsize=10)
    left.set(ylabel="Mean complete-call time (ms)", ylim=(0, 9.1))
    left.set_title("Absolute timings", loc="left", fontsize=13, pad=16, weight="bold")
    left.legend(loc="upper left", frameon=False, ncol=2, fontsize=10)
    values = [saved(row) for row in points]
    lower = [100 * (1 - 1 / row["native_ci_low"]) for row in points]
    upper = [100 * (1 - 1 / row["native_ci_high"]) for row in points]
    right.axhline(0, color=SUBTLE, linewidth=1, linestyle="--")
    for i, row in enumerate(points):
        right.errorbar(i, values[i], yerr=[[values[i] - lower[i]], [upper[i] - values[i]]],
                       fmt="o", color=COLORS[classification(row)], markersize=11,
                       elinewidth=1.8, capsize=6, zorder=4)
        right.annotate(f"{values[i]:+.1f}%", (i, values[i]), xytext=(0, 13),
                       textcoords="offset points", ha="center", fontsize=13,
                       color=COLORS[classification(row)], weight="bold")
    right.set(ylabel="Strassen time saved relative to native (%)", ylim=(-48, 15), xlim=(-.6, 2.6))
    right.set_title("Relative time, with paired 95% CI", loc="left", fontsize=13, pad=16, weight="bold")
    for ax in (left, right):
        ax.set_xticks(x, ["8,191\nOff grid", "8,192\nTarget grid", "8,193\nOff grid"], fontsize=11)
        ax.set_xlabel("Square side S (M = N = K)", labelpad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6edf2", zorder=0)
        ax.set_axisbelow(True)
    footnote(fig, "Positive = less time; negative = more time. Percent = 100 × (1 − Strassen / native). Screen winners are frozen before confirmation.", allocation)
    return fig


def comparison_picture(previous, current, previous_allocation, current_allocation):
    fig = plt.figure(figsize=(16.4, 8.6))
    subtitle(fig, "The same target-grid slice, kept separate by probe", 
             "M = N; only dimensions 2ⁱ or 3 × 2ⁱ. Identical axes and classifications; measurements are never averaged across allocations.")
    for i, (rows, label, allocation) in enumerate(((previous, "Previous probe", previous_allocation),
                                                (current, "Current probe", current_allocation))):
        ax = fig.add_axes([.065 + .48 * i, .18, .41, .60])
        setup_slice(ax, (-.55, 15.55), (-.55, 17.75))
        points = slice_points(ax, rows, small=True)
        c = Counter(classification(row) for row in points)
        ax.set_title(f"{label} · {len(points)} measured points\n{c['win']} wins / {c['loss']} losses / {c['inconclusive']} inconclusive",
                     loc="left", fontsize=12, pad=17, linespacing=1.55)
        ax.text(.0, -.215, allocation, transform=ax.transAxes, fontsize=8, color=SUBTLE)
    fig.legend(handles=legend_items(), loc="lower center", bbox_to_anchor=(.5, .062),
               ncol=3, frameon=False, fontsize=10)
    fig.text(.055, .025, "Selected labels: Strassen time saved (%). Colors: pointwise paired 95% CI. Empty space is unmeasured; no interpolation or pooled estimate.",
             fontsize=9, color=SUBTLE)
    return fig


def load_previous(path):
    data = json.loads(path.read_text())
    rows = data if isinstance(data, list) else data.get("rows", data.get("complete_call_rows"))
    if not isinstance(rows, list):
        raise ValueError("Previous normalized input requires rows or complete_call_rows")
    allocation = data.get("allocation_id", "Previous allocation") if isinstance(data, dict) else "Previous allocation"
    return rows, allocation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {args.out_dir}")
    data = json.loads(args.current.read_text())
    manifest = json.loads(args.manifest.read_text())
    rows = data["complete_call_rows"]
    metadata = {row["id"]: row for row in manifest["shapes"]}
    for row in rows:
        if on_grid(row) != metadata[row["shape_id"]]["on_lattice"]:
            raise ValueError("Literal lattice membership disagrees with frozen metadata")
        if row["shape_id"] in manifest["holdout_shape_ids"]:
            raise ValueError("Holdout measurement leaked into plot input")
    if len(rows) != 120 or data["historical_measurements_pooled"] or data["holdouts_measured"]:
        raise ValueError("Unexpected current cohort provenance or count")
    allocation = data["allocation_id"]
    figures = [("current_grid_square_output_slice", grid_slice(rows, allocation)),
               ("current_grid_geometry_3d", geometry_scatter(rows, allocation)),
               ("current_off_grid_8192_boundary", boundary_picture(rows, allocation))]
    if args.previous:
        previous, prior_allocation = load_previous(args.previous)
        figures.append(("previous_current_grid_square_output_comparison",
                        comparison_picture(previous, rows, prior_allocation, allocation)))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, figure in figures:
        for extension in ("png", "svg"):
            with (args.out_dir / f"{name}.{extension}").open("xb") as stream:
                figure.savefig(stream, format=extension, dpi=200)
        plt.close(figure)
    report = {
        "scope": "Static confirmed complete-call pictures; target grid and auxiliary boundary probes separated",
        "cohort_id": data["cohort_id"], "allocation_id": allocation,
        "figure_count": len(figures), "current_rows": len(rows),
        "current_on_grid_rows": sum(on_grid(row) for row in rows),
        "current_off_grid_rows": sum(not on_grid(row) for row in rows),
        "percentage_definition": "100*(1-strassen_ms/native_ms)",
        "comparison_basis": "Frozen screen winner per family, independently confirmed; native is tuned native",
        "ci_scope": "pointwise paired95%; no simultaneous guarantee", "historical_measurements_pooled": False,
        "source_sha256": {str(path.resolve()): sha(path) for path in
                          [Path(__file__), args.current, args.manifest] + ([args.previous] if args.previous else [])},
        "artifacts_sha256": {path.name: sha(path) for path in sorted(args.out_dir.iterdir())},
    }
    with (args.out_dir / "figure_provenance.json").open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
