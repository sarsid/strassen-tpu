#!/usr/bin/env python3
"""Make an offline scientific gallery from frozen shape-model fit artifacts.

This reads predictions only: it never fits a model or executes a matrix kernel.
Outputs are exclusively created. Matplotlib's noninteractive Agg backend is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
import numpy as np


FAMILIES = ("native", "cubic", "strassen")
COLORS = {"native": "#2878ad", "cubic": "#e58a24", "strassen": "#148879"}
LABELS = {"native": "Native XLA", "cubic": "Cubic", "strassen": "Strassen, one level"}
CONTEXT = (
    "Retrospective shape-held-out development CV; reserved shapes are not a test result here.\n"
    "Complete device call, excluding compilation and host transfer; observed means: 7 screening rounds."
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive(value, field):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be finite and positive: {value!r}")
    return value


def dimensions(row):
    values = tuple(row[k] for k in ("m", "n", "k"))
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
        raise ValueError(f"Expected positive integer public M,N,K: {values!r}")
    return values


def close(a, b, what):
    if not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"Inconsistent {what}: {a!r} != {b!r}")


class Evidence:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.paths = {name: self.directory / name for name in
                      ("model.json", "validation.json", "predictions.json", "full_fit_predictions.json")}
        self.input_hashes = {name: digest(path) for name, path in self.paths.items()}
        self.model, self.validation, self.predictions, self.full_fit = (
            json.loads(self.paths[name].read_text()) for name in self.paths
        )
        self.candidates = {c["candidate_id"]: c for c in self.model["candidates"]}
        if len(self.candidates) != len(self.model["candidates"]):
            raise ValueError("Duplicate candidate specifications")
        if set(c["family"] for c in self.candidates.values()) != set(FAMILIES):
            raise ValueError("Candidate catalog must contain native, cubic and Strassen families")
        self.selections = self.validation["selections"]
        self.by_shape = {r["shape_id"]: r for r in self.selections}
        if not self.by_shape or len(self.by_shape) != len(self.selections):
            raise ValueError("Empty or duplicate selection shape IDs")
        if len(self.by_shape) != 60 or Counter(c["family"] for c in self.candidates.values()) != {"native": 4, "cubic": 24, "strassen": 24}:
            raise ValueError("This study requires 60 exploratory shapes and 4/24/24 family candidates")
        if "training_shape_ids" in self.model and set(self.model["training_shape_ids"]) != set(self.by_shape):
            raise ValueError("Final model training shapes and validation inventory differ")
        self.index = self._validate_predictions(self.predictions, out_of_fold=True)
        self.full_index = self._validate_predictions(self.full_fit, out_of_fold=False)
        if set(self.index) != set(self.full_index):
            raise ValueError("In-sample and OOF action inventories differ")
        self.shape_rows = defaultdict(list)
        for row in self.predictions:
            self.shape_rows[row["shape_id"]].append(row)
            close(row["observed_ms"], self.full_index[(row["shape_id"], row["candidate_id"])]["observed_ms"],
                  "in-sample versus OOF observed latency")
        for sid, row in self.by_shape.items():
            dimensions(row)
            if {r["candidate_id"] for r in self.shape_rows[sid]} != set(self.candidates):
                raise ValueError(f"Incomplete candidate catalog for {sid}")
            selected = self.index[(sid, row["selected_candidate"])]
            if row["selected_family"] != selected["family"]:
                raise ValueError(f"Selected candidate family mismatch for {sid}")
            best = min(r["observed_ms"] for r in self.shape_rows[sid])
            for field, value in (("observed_selected_ms", selected["observed_ms"]),
                                 ("predicted_selected_ms", selected["predicted_ms"]),
                                 ("observed_best_ms", best),
                                 ("regret", selected["observed_ms"] / best)):
                close(row[field], value, f"{sid}: {field}")
            oracle = self.index[(sid, row["oracle_candidate"])]
            close(oracle["observed_ms"], best, f"{sid}: observed catalog minimum")
            if row["oracle_family"] != oracle["family"]:
                raise ValueError(f"Oracle candidate family mismatch for {sid}")
        self.baselines, self.baseline_notes = self._baselines()

    def _validate_predictions(self, rows, out_of_fold):
        if not isinstance(rows, list) or not rows:
            raise ValueError("Predictions must be a nonempty JSON array")
        index = {}
        for row in rows:
            sid, cid = row["shape_id"], row["candidate_id"]
            if sid not in self.by_shape or cid not in self.candidates or (sid, cid) in index:
                raise ValueError(f"Unexpected or duplicate prediction key {(sid, cid)!r}")
            if row["family"] != self.candidates[cid]["family"]:
                raise ValueError(f"Prediction family mismatch for {cid}")
            if dimensions(row) != dimensions(self.by_shape[sid]):
                raise ValueError(f"M,N,K mismatch for {sid}")
            for field in ("predicted_ms", "observed_ms"):
                positive(row[field], field)
            if out_of_fold and row["fold"] != self.by_shape[sid]["fold"]:
                raise ValueError(f"Multiple folds for shape {sid}")
            index[(sid, cid)] = row
        if len(index) != len(self.by_shape) * len(self.candidates):
            raise ValueError("Predictions do not cover every shape and candidate")
        return index

    def _baselines(self):
        baseline = {"Nested selector": [r["regret"] for r in self.selections]}
        notes = []
        if "native_default" in self.candidates:
            baseline["Always default Native XLA"] = [
                self.index[(r["shape_id"], "native_default")]["observed_ms"] / r["observed_best_ms"]
                for r in self.selections]
            for row, value in zip(self.selections, baseline["Always default Native XLA"]):
                if "native_default_regret" in row:
                    close(row["native_default_regret"], value, "default-native baseline regret")
        if all("training_fixed_native_regret" in r for r in self.selections):
            values = [positive(r["training_fixed_native_regret"], "training_fixed_native_regret") for r in self.selections]
            if min(values) < 1 - 1e-9:
                raise ValueError("Training-fixed-native regret below one")
            baseline["Training-selected fixed native configuration"] = values
        keys = sorted({k for r in self.selections for k in r.get("baseline_regrets", {})})
        labels = {"training_fixed_native": "Training-selected fixed native configuration"}
        for key in keys:
            if key == "native_default":
                continue
            if all(key in r.get("baseline_regrets", {}) for r in self.selections):
                values = [positive(r["baseline_regrets"][key], key) for r in self.selections]
                if min(values) < 1 - 1e-9:
                    raise ValueError(f"Baseline regret below one: {key}")
                baseline[labels.get(key, key.replace("_", " "))] = values
            else:
                notes.append(f"Omitted incomplete per-shape baseline series: {key}")
        if ("training_fixed_native" in self.validation.get("metrics", {})
                and "training_fixed_native" not in keys
                and "Training-selected fixed native configuration" not in baseline):
            notes.append("Training-fixed-native summary exists, but its per-shape regrets were not supplied; no CDF inferred.")
        return baseline, notes


def short_number(x, _position=None):
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:g}"


def style_axis(ax):
    ax.grid(True, alpha=0.2, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def regret_axis(ax, maximum):
    if maximum <= 3:
        ax.set_xscale("linear")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    else:
        ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(short_number))
    ax.xaxis.set_minor_formatter(NullFormatter())


class Gallery:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "figures").mkdir()
        self.figures = []

    def save(self, fig, name, title, caption, *, analysis="out_of_fold"):
        fig.suptitle(title, x=0.05, y=0.975, ha="left", fontsize=16, fontweight="bold")
        fig.text(0.05, 0.12, textwrap.fill(caption, 155), ha="left", va="top", fontsize=9, color="#303e4b")
        fig.text(0.05, 0.017, CONTEXT, ha="left", va="bottom", fontsize=8, color="#56616b")
        entry = {"id": name, "title": title, "caption": caption, "analysis": analysis, "files": {}}
        for suffix in ("png", "svg"):
            path = self.directory / "figures" / f"{name}.{suffix}"
            fig.savefig(path, dpi=155, bbox_inches="tight", pad_inches=0.22, facecolor="white")
            entry["files"][suffix] = str(path.relative_to(self.directory))
        plt.close(fig)
        self.figures.append(entry)


def prediction_accuracy(ev, gallery):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.24, top=0.83, wspace=0.28)
    for family, ax in zip(FAMILIES, axes):
        rows = [r for r in ev.predictions if r["family"] == family]
        obs = np.array([r["observed_ms"] for r in rows]); pred = np.array([r["predicted_ms"] for r in rows])
        lo, hi = min(obs.min(), pred.min()) * 0.8, max(obs.max(), pred.max()) * 1.2
        ax.scatter(obs, pred, s=12, alpha=0.45, color=COLORS[family], edgecolors="none", rasterized=True)
        ax.plot([lo, hi], [lo, hi], color="#414141", linewidth=1, label="Exact prediction")
        ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
               title=f"{LABELS[family]} · {len(rows):,} actions", xlabel="Observed screen mean (ms)")
        ax.text(0.04, 0.96, f"Median factor error: {np.exp(np.median(np.abs(np.log(pred / obs)))):.2f}×",
                transform=ax.transAxes, va="top", fontsize=9)
        style_axis(ax)
    axes[0].set_ylabel("Out-of-fold predicted complete-call latency (ms)")
    gallery.save(fig, "01_oof_prediction_accuracy", "Does the cost formula predict held-out latency?",
                 "Every point is one held-out shape and candidate. All candidate timings for a test shape were excluded from its model fit. Both axes use logarithmic scales.")


def residual_volume(ev, gallery):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.24, top=0.83, wspace=0.13)
    for family, ax in zip(FAMILIES, axes):
        rows = [r for r in ev.predictions if r["family"] == family]
        x = [sum(math.log2(v) for v in dimensions(r)) for r in rows]
        y = [math.log2(r["predicted_ms"] / r["observed_ms"]) for r in rows]
        ax.scatter(x, y, s=12, color=COLORS[family], alpha=0.4, edgecolors="none", rasterized=True)
        ax.axhline(0, color="#414141", linewidth=1)
        ax.set(title=LABELS[family], xlabel="log₂(M × N × K)")
        style_axis(ax)
    axes[0].set_ylabel("log₂(predicted latency / observed latency)")
    gallery.save(fig, "02_oof_residual_vs_volume", "Where does prediction error depend on matrix size?",
                 "Zero is an exact prediction; +1 means twice the measured latency and −1 means half. Volume alone does not describe aspect ratio or padding, so several geometries share an x coordinate.")


def regret_geometry(ev, gallery):
    groups = defaultdict(list)
    for row in ev.selections:
        groups[row["design_family"]].append(row)
    names = sorted(groups)
    fig, ax = plt.subplots(figsize=(13, max(7, len(names) * 0.36 + 2)))
    fig.subplots_adjust(left=0.26, right=0.96, bottom=0.22, top=0.85)
    for i, name in enumerate(names):
        rows = sorted(groups[name], key=lambda r: r["shape_id"])
        offsets = np.linspace(-0.18, 0.18, len(rows)) if len(rows) > 1 else [0]
        for row, offset in zip(rows, offsets):
            ax.scatter(row["regret"], i + offset, color=COLORS[row["selected_family"]], s=35, edgecolor="white", linewidth=0.5)
        ax.plot([np.median([r["regret"] for r in rows])]*2, [i-0.22, i+0.22], color="#222222", linewidth=2)
    ax.axvline(1, color="#444444", linewidth=1)
    ax.axvline(1.05, color="#888888", linestyle=":", linewidth=1)
    ax.set_yticks(range(len(names)), [f"{name.replace('_', ' ')} (n={len(groups[name])})" for name in names])
    ax.set(xlabel="Selected latency / observed catalog-best latency")
    regret_axis(ax, max(r["regret"] for r in ev.selections))
    for family in FAMILIES:
        ax.scatter([], [], c=COLORS[family], label=LABELS[family], s=35)
    ax.legend(loc="upper right", fontsize=8)
    style_axis(ax)
    gallery.save(fig, "03_oof_regret_by_geometry", "How costly are selection mistakes across geometries?",
                 "One point per held-out shape; color is the selected algorithm family, black marks the group median, and the dotted line marks 5% regret. The reference is the noisy seven-round minimum over the measured catalog.")


def regret_cdf(ev, gallery):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.25, top=0.83)
    palette = ["#7545a3", "#2878ad", "#61676c", "#e58a24"]
    for i, (name, values) in enumerate(ev.baselines.items()):
        x = np.sort(values); y = np.arange(1, len(x)+1) / len(x)
        gm = math.exp(sum(math.log(v) for v in x)/len(x))
        ax.step(np.r_[1, x], np.r_[0, y], where="post", linewidth=2, color=palette[i % len(palette)],
                label=f"{name}: gmean {gm:.3f}×; within 5%: {np.mean(x <= 1.05):.0%}")
    ax.axvline(1.05, color="#777777", linestyle=":", linewidth=1)
    ax.set(ylim=(0, 1.04), xlabel="Selected latency / observed catalog-best latency", ylabel="Fraction of held-out shapes at or below this regret")
    regret_axis(ax, max(v for values in ev.baselines.values() for v in values))
    ax.legend(loc="lower right", fontsize=9)
    style_axis(ax)
    gallery.save(fig, "04_oof_regret_cdf", "Does the formula improve on simple selection rules?",
                 "Each complete series contains the same shapes. Default-native regret is reconstructed from measured native_default rows. Training-selected baselines appear only when complete per-shape regrets are supplied; no baseline CDF is inferred from summary metrics.")


def geometry_3d(ev, gallery, *, regret):
    fig = plt.figure(figsize=(12, 8.2)); ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.05, right=0.89 if regret else 0.96, bottom=0.19, top=0.85)
    points = np.array([[math.log2(v) for v in dimensions(r)] for r in ev.selections])
    if regret:
        values = np.array([r["regret"] for r in ev.selections]); vmax = max(1.05, float(values.max()))
        artist = ax.scatter(*points.T, c=values, norm=LogNorm(vmin=1, vmax=vmax), cmap="magma_r", s=48, edgecolor="#333333", linewidth=0.35, depthshade=False)
        bar = fig.colorbar(artist, ax=ax, pad=0.10, shrink=0.65)
        bar.set_label("Observed selection regret (×)")
        ticks = sorted(set([1., vmax] + [v for v in (1.05, 1.1, 1.25, 1.5, 2, 3, 5, 10) if 1 < v < vmax]))
        if len(ticks) > 7: ticks = [1., *ticks[1:-1:2], vmax]
        bar.set_ticks(ticks); bar.set_ticklabels([f"{v:.3g}" for v in ticks]); bar.ax.minorticks_off()
    else:
        for family, marker in zip(FAMILIES, ("o", "^", "s")):
            mask = np.array([r["selected_family"] == family for r in ev.selections])
            ax.scatter(*points[mask].T, color=COLORS[family], marker=marker, s=48, edgecolor="white", linewidth=0.45,
                       depthshade=False, label=f"{LABELS[family]} ({int(mask.sum())})")
        ax.legend(loc="upper left", fontsize=9)
    ax.set(xlabel="log₂ M", ylabel="log₂ N", zlabel="log₂ K")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis): axis.labelpad = 12
    extent = math.ceil(float(points.max()))
    ticks = [v for v in range(0, extent+1, 4) if extent-v >= 2]
    ticks.append(extent)
    for axis_name in ("x", "y", "z"):
        getattr(ax, f"set_{axis_name}lim")(-0.5, extent+0.5)
        getattr(ax, f"set_{axis_name}ticks")(ticks)
    ax.view_init(elev=24, azim=134); ax.set_box_aspect((1, 1, 0.9))
    name = "06_oof_regret_3d" if regret else "05_oof_selected_family_3d"
    title = "Where in M,N,K does the formula incur regret?" if regret else "Which algorithm does the held-out formula select?"
    gallery.save(fig, name, title,
                 "Axes use public M,N,K order, each on a log₂ scale. Only measured exploratory geometries are shown; this scatter is not a measured decision boundary or a claim about every point in the surrounding lattice. Nearby boundary probes can overlap in projection.")


def square_curves(ev, gallery):
    ids = sorted((sid for sid, r in ev.by_shape.items() if len(set(dimensions(r))) == 1), key=lambda sid: ev.by_shape[sid]["m"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.25, top=0.83, wspace=0.15)
    for family, ax in zip(FAMILIES, axes):
        x, obs, pred = [], [], []
        for sid in ids:
            rows = [r for r in ev.shape_rows[sid] if r["family"] == family]
            x.append(ev.by_shape[sid]["m"]); obs.append(min(r["observed_ms"] for r in rows)); pred.append(min(r["predicted_ms"] for r in rows))
        ax.plot(x, obs, "o-", color=COLORS[family], markersize=4, label="Observed family minimum")
        ax.plot(x, pred, "x--", color="#4b4b4b", markersize=5, label="Predicted family minimum (OOF)")
        ax.set(xscale="log", yscale="log", title=LABELS[family], xlabel="Square side: M = N = K")
        ax.legend(fontsize=8, loc="upper left"); style_axis(ax)
    axes[0].set_ylabel("Complete-call latency (ms)")
    gallery.save(fig, "07_oof_square_family_minima", "Does the formula track the lower latency envelope on squares?",
                 "Each family minimum is taken across that family's tested configurations. Predicted and observed minima can come from different candidates. Lines connect sampled sizes only; adjacent 1023/1024/1025 and 4095/4096/4097 probes are retained.")


def worst_regrets(ev, gallery):
    rows = sorted(ev.selections, key=lambda r: (-r["regret"], r["shape_id"]))[:10]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1, 1.2]})
    fig.subplots_adjust(left=0.24, right=0.97, bottom=0.21, top=0.83, wspace=0.16)
    y = np.arange(len(rows)); labels = [f"{r['m']} × {r['n']} × {r['k']}\n{r['selected_candidate']}" for r in rows]
    axes[0].barh(y, [r["regret"]-1 for r in rows], left=1, color=[COLORS[r["selected_family"]] for r in rows], height=0.62)
    for i, r in enumerate(rows): axes[0].text(r["regret"], i, f"  {r['regret']:.3f}×", va="center", fontsize=8)
    xmax = max(r["regret"] for r in rows); axes[0].set_xlim(1, 1+max(0.06, xmax-1)*1.27)
    axes[0].set_yticks(y, labels, fontsize=9); axes[0].set_xlabel("Observed selected / catalog best")
    axes[0].set_title("Largest held-out selection regrets")
    for offset, key, label, color in ((-0.22, "observed_best_ms", "Observed catalog best", "#626970"),
                                    (0, "observed_selected_ms", "Observed chosen action", "#7545a3"),
                                    (0.22, "predicted_selected_ms", "Predicted chosen action", "#ba9bd3")):
        axes[1].scatter([r[key] for r in rows], y+offset, s=28, color=color, label=label)
    axes[1].set(xscale="log", xlabel="Complete-call latency (ms)", title="Cost estimate versus observed cost")
    axes[1].set_yticks(y, []); axes[1].legend(loc="lower right", fontsize=8)
    for ax in axes:
        ax.set_ylim(len(rows)-0.5, -0.6); style_axis(ax)
    gallery.save(fig, "08_oof_worst_regrets", "Which choices need the most attention?",
                 "Rows are ordered by held-out regret, not by absolute runtime. Left labels identify the chosen candidate; exact oracle IDs and all predictions remain in the source JSON. The observed catalog minimum is screening-based and subject to selection noise.")


def in_sample_comparison(ev, gallery):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.25, top=0.82)
    data, ticks, colors = [], [], []
    for family in FAMILIES:
        for rows, mode, color in ((ev.predictions, "Held-out", COLORS[family]), (ev.full_fit, "In-sample", "#cccccc")):
            data.append([abs(math.log2(r["predicted_ms"]/r["observed_ms"])) for r in rows if r["family"] == family])
            ticks.append(f"{LABELS[family]}\n{mode}"); colors.append(color)
    result = ax.boxplot(data, tick_labels=ticks, patch_artist=True, widths=0.55, showfliers=True,
                        flierprops={"markersize": 2, "alpha": 0.2}, medianprops={"color": "#222222"})
    for box, color in zip(result["boxes"], colors): box.set_facecolor(color)
    ax.set_ylabel("Absolute log₂(predicted / observed latency)"); ax.set_xlabel("Smaller error is better; in-sample scores reuse training outcomes")
    style_axis(ax)
    gallery.save(fig, "09_in_sample_vs_oof", "How much better does apparent training fit look?",
                 "Colored boxes use held-out predictions; gray boxes use the final model refit on all screening outcomes. Boxes show medians and interquartile ranges, with conventional 1.5-IQR whiskers. The gray series is descriptive training fit, never validation.",
                 analysis="explicit_in_sample_comparison")


def write_gallery(ev, gallery):
    esc = html.escape
    cards = []
    for item in gallery.figures:
        cards.append(f'<article id="{esc(item["id"])}"><h2>{esc(item["title"])}</h2>'
                     f'<a href="{esc(item["files"]["svg"])}"><img loading="lazy" src="{esc(item["files"]["png"])}" alt="{esc(item["title"])}"></a>'
                     f'<p>{esc(item["caption"])}</p><p class="downloads"><a href="{esc(item["files"]["png"])}">PNG</a> · '
                     f'<a href="{esc(item["files"]["svg"])}">Vector SVG</a></p></article>')
    notes = "".join(f"<li>{esc(note)}</li>" for note in ev.baseline_notes)
    metric_json = esc(json.dumps(ev.validation.get("metrics", {}), indent=2, sort_keys=True))
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shape cost model — development validation</title><style>
body{{font:16px/1.5 system-ui,sans-serif;margin:0;background:#eef1f4;color:#182a38}}main{{max-width:1300px;margin:auto;padding:32px}}
h1{{font-size:32px;line-height:1.2}}h2{{font-size:23px}}header,article{{background:white;padding:24px 30px;margin-bottom:24px;border-radius:8px}}
img{{width:100%;height:auto;display:block}}a{{color:#175a8b}}p{{max-width:100ch}}.scope{{border-left:4px solid #7545a3;padding-left:16px}}pre{{overflow:auto;background:#f2f4f6;padding:16px;font-size:13px}}
.downloads{{font-size:14px}}nav{{display:flex;flex-wrap:wrap;gap:12px}}@media(max-width:600px){{main{{padding:12px}}header,article{{padding:16px}}}}</style></head><body><main>
<header><h1>Can matrix dimensions predict the right algorithm and tile?</h1>
<p>{len(ev.by_shape)} exploratory shapes · {len(ev.candidates)} tested candidate specifications · {len(ev.predictions):,} held-out predictions.</p>
<p class="scope"><strong>Retrospective development cross-validation.</strong> Whole shapes are held out. The reserved geometries are not an evaluated test set here.
The target is warmed complete-device-call latency, including device padding and cropping, excluding compilation and host transfers. Observed screening means use seven rounds.
The best observed screened candidate is a noisy finite-catalog reference, not a confirmed optimum.</p>
<p>Each static 3D plot uses M, N, K in that order on log₂ axes. Candidate figures retain exact measured configurations; no interpolation of unmeasured timings is implied.
Native XLA configurations are compiler settings, with no manually selected tile.</p>
<nav>{''.join(f'<a href="#{esc(x["id"])}">{i+1}. {esc(x["title"])}</a>' for i,x in enumerate(gallery.figures))}</nav>
<details><summary>Reported validation metrics</summary><pre>{metric_json}</pre></details>
{'<ul>'+notes+'</ul>' if notes else ''}<p><a href="audit.json">Input and output hashes</a> · <a href="figures.json">Figure inventory</a></p></header>
{''.join(cards)}<footer><p>This gallery is self-contained and makes no network requests. PNGs and vector SVGs are local files.
Confirmation timings are not plotted or silently substituted for missing candidate outcomes.</p></footer></main></body></html>'''
    (gallery.directory / "index.html").write_text(page)


def generate(artifacts, output_dir):
    ev = Evidence(artifacts)
    gallery = Gallery(output_dir)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "svg.fonttype": "none", "savefig.facecolor": "white"})
    prediction_accuracy(ev, gallery)
    residual_volume(ev, gallery)
    regret_geometry(ev, gallery)
    regret_cdf(ev, gallery)
    geometry_3d(ev, gallery, regret=False)
    geometry_3d(ev, gallery, regret=True)
    square_curves(ev, gallery)
    worst_regrets(ev, gallery)
    in_sample_comparison(ev, gallery)
    (gallery.directory / "figures.json").write_text(json.dumps(gallery.figures, indent=2)+"\n")
    write_gallery(ev, gallery)
    if any(digest(path) != ev.input_hashes[name] for name, path in ev.paths.items()):
        raise ValueError("An input artifact changed during plotting; output is not sealed")
    files = sorted(p for p in gallery.directory.rglob("*") if p.is_file())
    audit = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
             "analysis": "retrospective shape-held-out development cross-validation; no new measurements",
             "time_scope": "call", "screen_rounds": 7, "shapes": len(ev.by_shape),
             "candidates": len(ev.candidates), "out_of_fold_predictions": len(ev.predictions),
             "full_fit_predictions": len(ev.full_fit), "figure_count": len(gallery.figures),
             "family_candidates": dict(Counter(c["family"] for c in ev.candidates.values())),
             "shape_folds": dict(Counter(str(r["fold"]) for r in ev.selections)),
             "baseline_series": list(ev.baselines), "notes": ev.baseline_notes,
             "source_file": str(Path(__file__).resolve()), "source_sha256": digest(__file__),
             "input_files": {name: {"path": str(path), "sha256": digest(path)} for name, path in ev.paths.items()},
             "output_files": {str(p.relative_to(gallery.directory)): {"sha256": digest(p), "bytes": p.stat().st_size} for p in files}}
    (gallery.directory / "audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    return {"output_dir": str(gallery.directory), "figure_count": len(gallery.figures),
            "output_count_including_audit": len(files)+1, "notes": ev.baseline_notes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.artifacts, args.output_dir)))


if __name__ == "__main__":
    main()
