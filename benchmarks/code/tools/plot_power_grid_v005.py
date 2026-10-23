#!/usr/bin/env python3
"""Audit and plot GRID-screen / GRID-confirm without executing any kernels.

Outputs are exclusive: a new directory, scientific PNG/PDF chartbook, CSV data,
offline HTML gallery, optional self-contained Plotly 3D, and provenance JSON.
Headline comparisons use only frozen screen winners in headline confirmation
groups. Repeated winner controls in shortlist groups are never pooled with them.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import textwrap


VERSION = "plot_power_grid_v005"
FAMILIES = ("native", "cubic", "strassen")
SCOPES = ("call", "prepared_kernel")
SCOPE_LABELS = {"call": "Complete device call", "prepared_kernel": "Prepared kernel"}
COLORS = {"native": "#2472B4", "cubic": "#DB8A24", "strassen": "#168572"}
PAIRS = (("native", "strassen"), ("cubic", "strassen"), ("native", "cubic"),
         ("native_default", "strassen"), ("native_default", "native"))
CI_NOTE = "Pointwise paired 95% bootstrap intervals; no multiplicity correction. >1 favors denominator."


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dimensions():
    return sorted({2 ** i for i in range(18)} | {3 * 2 ** (i - 1) for i in range(1, 17)})


def finite(value, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not positive or value > 0))


def eligible(row):
    return bool(row and row.get("status") == "ok"
                and (row.get("correctness") or {}).get("pass")
                and finite((row.get("timing") or {}).get("mean_ms"), True))


def mean_ms(row):
    return (row.get("timing") or {}).get("mean_ms") if row else None


def row_key(row):
    return tuple(row.get(k) for k in ("group_id", "case_id", "candidate_id", "scope"))


def flat(row, prefix=""):
    out = {}
    for key, value in row.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flat(value, name))
        elif isinstance(value, (list, tuple)):
            out[name] = json.dumps(value, separators=(",", ":"), allow_nan=False)
        else:
            out[name] = value
    return out


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    rows = [flat(row) for row in rows]
    keys = sorted(set().union(*(row.keys() for row in rows))) if rows else ["no_records"]
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def paired_ratio(reference, candidate, reference_samples, candidate_samples, seed=20260925):
    """Strict same-case pairing; never pair by round across different groups."""
    if not eligible(reference) or not eligible(candidate):
        return None
    for key in ("group_id", "case_id", "scope", "shape_id", "seed"):
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"Unpaired comparison differs in {key}")
    by_round = []
    for samples in (reference_samples, candidate_samples):
        rounds = [sample["round"] for sample in samples]
        if len(rounds) != len(set(rounds)):
            raise ValueError("Duplicate timing rounds cannot be paired")
        if not samples or any(not finite(s.get("elapsed_ms"), True) for s in samples):
            raise ValueError("Pairing requires finite positive raw samples")
        by_round.append({s["round"]: s["elapsed_ms"] for s in samples})
    if set(by_round[0]) != set(by_round[1]):
        raise ValueError("Unequal raw round coverage")
    import numpy as np
    rounds = sorted(by_round[0])
    a, b = (np.asarray([d[r] for r in rounds], dtype=float) for d in by_round)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(rounds), size=(2000, len(rounds)))
    distribution = a[draws].mean(axis=1) / b[draws].mean(axis=1)
    ratio = float(a.mean() / b.mean())
    lo, hi = map(float, np.quantile(distribution, [.025, .975]))
    return {"speedup": ratio, "ci95_low": lo, "ci95_high": hi,
            "paired_round_count": len(rounds), "bootstrap_seed": seed,
            "method": "paired bootstrap ratio of arithmetic means; 2000 draws; pointwise 95%",
            "classification": "win" if lo > 1 else "loss" if hi < 1 else "inconclusive"}


class Evidence:
    def __init__(self, screen, confirmation, manifest):
        self.sources, self.warnings, self.rows, self.artifacts, self.metadata = {}, [], {}, {}, {}
        self.manifest_path = Path(manifest).resolve()
        self.manifest = self.read(self.manifest_path)
        shapes = self.manifest["shapes"]
        self.shapes = {s["id"]: s for s in shapes}
        if len(self.shapes) != len(shapes):
            raise ValueError("Duplicate manifest shape IDs")
        self.exploratory = list(self.manifest["exploratory_shape_ids"])
        self.holdout = set(self.manifest["holdout_shape_ids"])
        if set(self.exploratory) & self.holdout:
            raise ValueError("Exploratory and holdout overlap")
        self.samples = defaultdict(list)
        self.results, self.compilation, self.statuses = [], [], []
        self.audit = {"sample_checks": 0, "headline_rows": 0, "reserved_shapes_observed": []}
        for tag, directory in (("screen", screen), ("confirmation", confirmation)):
            directory = Path(directory).resolve()
            if (directory / "artifacts/results.jsonl").exists():
                directory = directory / "artifacts"
            self.artifacts[tag] = directory
            path = directory / "results.jsonl"
            self.track(path)
            rows = []
            with path.open(encoding="utf-8") as stream:
                for line_no, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row.update(_cohort=tag, _ref=f"{path}:{line_no}")
                    shape_id = row.get("shape_id")
                    if shape_id:
                        if shape_id not in self.shapes:
                            raise ValueError(f"Unknown shape {shape_id}")
                        if shape_id in self.holdout:
                            raise ValueError(f"Reserved holdout was observed: {shape_id}")
                        expected = [self.shapes[shape_id][axis] for axis in ("m", "k", "n")]
                        if row.get("shape_mkn") is not None and row["shape_mkn"] != expected:
                            raise ValueError(f"Shape order mismatch at {row['_ref']}")
                    rows.append(row)
                    event = row.get("event")
                    if event == "sample":
                        self.samples[(tag, row_key(row))].append(row)
                    elif event == "case_result":
                        self.results.append(row)
                    elif event == "compilation":
                        self.compilation.append(row)
                    if event == "error" or (event == "case_result" and row.get("status") != "ok"):
                        self.statuses.append(row)
            self.rows[tag] = rows
            self.metadata[tag] = {}
            for name in ("summary.json", "environment.json", "planned_cases.json", "source_manifest.json",
                         "artifact_manifest.json", "selection_input_provenance.json"):
                if (directory / name).exists():
                    self.metadata[tag][name] = self.read(directory / name)
        self.selection = self.read(self.artifacts["screen"] / "selections.json")
        self.confirmation = self.read(self.artifacts["confirmation"] / "confirmation.json")
        self.cross_input_guards()
        for directory in self.artifacts.values():
            self.verify_seal(directory)
        if self.selection.get("stage") != "screen":
            raise ValueError("Expected frozen screening selections")
        seen = set()
        for row in self.results:
            key = (row["_cohort"], row_key(row))
            if key in seen:
                raise ValueError(f"Duplicate scoped case_result: {key}")
            seen.add(key)
            raw = self.samples[key]
            rounds = [s["round"] for s in raw]
            if len(rounds) != len(set(rounds)):
                raise ValueError(f"Duplicate raw rounds: {key}")
            if any(not finite(s.get("elapsed_ms"), True) for s in raw):
                raise ValueError(f"Nonpositive or nonfinite sample: {key}")
            timing = row.get("timing") or {}
            if raw:
                if timing.get("sample_count") != len(raw):
                    raise ValueError(f"Raw sample count mismatch: {key}")
                actual = statistics.mean(s["elapsed_ms"] for s in raw)
                if not finite(timing.get("mean_ms"), True) or not math.isclose(actual, timing["mean_ms"], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f"Raw mean mismatch: {key}")
                self.audit["sample_checks"] += 1
            elif eligible(row):
                raise ValueError(f"Eligible result lacks raw samples: {key}")
            if eligible(row):
                phase = "GRID-screen" if row["_cohort"] == "screen" else "GRID-confirm"
                policy = self.campaign["experiments"][phase]["timing"]
                expected_count = self.campaign["timing"][policy]["repeats"]
                if sorted(rounds) != list(range(expected_count)):
                    raise ValueError(f"Eligible result does not cover all registered rounds: {key}")
        self.headline = {}
        for row in self.results:
            if row["_cohort"] != "confirmation" or not row.get("group_id", "").endswith("__headline_confirmation"):
                continue
            key = (row["shape_id"], row["scope"], row["candidate_id"])
            if key in self.headline:
                raise ValueError(f"Duplicate headline result {key}")
            self.headline[key] = row
        self.audit["headline_rows"] = len(self.headline)
        if not self.headline:
            raise ValueError("No __headline_confirmation groups; do not substitute shortlist controls")
        self.comparisons = []
        for shape_id in self.exploratory:
            for scope in SCOPES:
                for baseline, candidate in PAIRS:
                    left, right = self.selected(shape_id, scope, baseline), self.selected(shape_id, scope, candidate)
                    if not all(eligible(self.selected(shape_id, s, f)) for s in SCOPES for f in (baseline, candidate)):
                        continue
                    pair = paired_ratio(left, right, self.raw(left), self.raw(right), seed=self.campaign["seed"] + 811)
                    recorded = [p for p in self.confirmation.get("by_shape", {}).get(shape_id, {}).get("headline_comparisons", [])
                                if p.get("scope") == scope and p.get("candidate_id") == right["candidate_id"]
                                and p.get("reference_candidate_id") == left["candidate_id"]
                                and p.get("group_id") == right["group_id"]]
                    if len(recorded) > 1:
                        raise ValueError("Duplicate archived headline comparison")
                    if recorded:
                        published = recorded[0]
                        if not math.isclose(published["speedup_ratio_of_means"], pair["speedup"], rel_tol=1e-10):
                            raise ValueError("Archived headline ratio does not match raw samples")
                        lo, hi = published["speedup_ci95"]
                        if not (finite(lo, True) and finite(hi, True) and lo <= hi):
                            raise ValueError("Invalid archived headline interval")
                        if not (math.isclose(lo, pair["ci95_low"], rel_tol=1e-10, abs_tol=1e-12)
                                and math.isclose(hi, pair["ci95_high"], rel_tol=1e-10, abs_tol=1e-12)):
                            raise ValueError("Archived headline interval differs from exact-seed bootstrap replay")
                        pair.update(ci95_low=lo, ci95_high=hi, method=published["method"],
                                    uncertainty_source="archived headline comparison; raw means and interval replay verified",
                                    classification="win" if lo > 1 else "loss" if hi < 1 else "inconclusive")
                    else:
                        if left["candidate_id"] != right["candidate_id"]:
                            raise ValueError("Eligible headline contrast is missing from canonical confirmation report")
                        pair["uncertainty_source"] = "recomputed from exact headline paired rounds"
                    self.comparisons.append({"shape_id": shape_id, "scope": scope,
                        "reference": baseline, "candidate": candidate,
                        "reference_candidate_id": left["candidate_id"], "candidate_id": right["candidate_id"],
                        "reference_mean_ms": mean_ms(left), "candidate_mean_ms": mean_ms(right),
                        "same_executable_spec": left["candidate_id"] == right["candidate_id"],
                        "source_refs": [left["_ref"], right["_ref"]], **pair})

    def cross_input_guards(self):
        """Bind individually sealed artifacts into this exact study/cohort."""
        for tag, phase in (("screen", "GRID-screen"), ("confirmation", "GRID-confirm")):
            summary = self.metadata[tag].get("summary.json", {})
            if not (summary.get("completed") is True and summary.get("status") == "completed"
                    and summary.get("phase") == phase and not summary.get("not_completed_group_ids")):
                raise ValueError(f"A completed {phase} summary is required")
        identities = [self.metadata[tag].get("environment.json", {}).get("identity") for tag in ("screen", "confirmation")]
        if not identities[0] or identities[0] != identities[1] or self.selection.get("environment_identity") != identities[0]:
            raise ValueError("Screen/confirmation/selection environment identities differ")
        if self.selection.get("shape_manifest_sha256") != sha256(self.manifest_path):
            raise ValueError("Provided shape manifest is not the frozen screen manifest")
        if self.selection.get("screen_results_sha256") != sha256(self.artifacts["screen"] / "results.jsonl"):
            raise ValueError("Selection does not identify the supplied screen journal")
        if self.selection.get("source_manifest_sha256") != sha256(self.artifacts["screen"] / "source_manifest.json"):
            raise ValueError("Selection does not identify the supplied screen source manifest")
        selection_input = self.read(self.artifacts["confirmation"] / "selection_input.json")
        provenance = self.metadata["confirmation"].get("selection_input_provenance.json", {})
        if selection_input != self.selection or provenance.get("sha256") != sha256(self.artifacts["screen"] / "selections.json"):
            raise ValueError("Confirmation did not consume this exact screen selection")
        sources = [self.metadata[tag].get("source_manifest.json", {}).get("sha256", {}) for tag in ("screen", "confirmation")]
        code_sources = [{name: digest for name, digest in s.items() if name.startswith("source_snapshot/")} for s in sources]
        if not code_sources[0] or code_sources[0] != code_sources[1]:
            raise ValueError("Screen and confirmation frozen source snapshots differ")
        candidates = [name for name, digest in sources[1].items() if name.startswith("config_snapshot/")
                      and digest == self.selection.get("campaign_sha256")]
        if len(candidates) != 1:
            raise ValueError("Cannot identify one frozen campaign snapshot")
        campaign_path = self.artifacts["confirmation"] / candidates[0]
        self.campaign = self.read(campaign_path)
        if sha256(campaign_path) != self.selection.get("campaign_sha256"):
            raise ValueError("Frozen campaign snapshot hash mismatch")
        if set(self.campaign["experiments"]["GRID-screen"]["shape_ids"]) != set(self.exploratory):
            raise ValueError("Screen shape IDs differ from exploratory manifest")
        if set(self.campaign["experiments"]["GRID-confirm"]["shape_ids"]) != set(self.exploratory):
            raise ValueError("Confirmation shape IDs differ from exploratory manifest")
        self.audit["cross_input_identity_and_hash_checks"] = "passed"

    def track(self, path):
        path = Path(path).resolve()
        self.sources[str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    def read(self, path):
        self.track(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def verify_seal(self, directory):
        path = directory / "artifact_manifest.json"
        if not path.exists():
            self.warnings.append(f"No canonical seal available at {path}; hashes recorded independently")
            return
        seal = json.loads(path.read_text())
        hashes = seal.get("sha256", seal.get("artifacts_sha256", {}))
        for source, metadata in self.sources.items():
            source_path = Path(source)
            try:
                relative = str(source_path.relative_to(directory))
            except ValueError:
                continue
            if relative in hashes:
                entry = hashes[relative]
                expected = entry.get("sha256") if isinstance(entry, dict) else entry
                if metadata["sha256"] != expected:
                    raise ValueError(f"Canonical seal mismatch: {source_path}")

    def raw(self, row):
        return self.samples[(row["_cohort"], row_key(row))]

    def selected(self, shape_id, scope, family):
        if family == "native_default":
            items = [row for (sid, sc, _), row in self.headline.items()
                     if sid == shape_id and sc == scope and row.get("family") == "native"
                     and not row.get("compiler_options", {})]
            if len(items) > 1:
                raise ValueError("Multiple headline native defaults")
            return items[0] if items else None
        winner = self.selection.get("by_shape", {}).get(shape_id, {}).get(family, {}).get("winner")
        if not winner:
            return None
        return self.headline.get((shape_id, scope, winner["candidate_id"]))

    def pairs(self, scope="call", reference="native", candidate="strassen", ids=None):
        ids = set(ids) if ids is not None else set(self.exploratory)
        return [p for p in self.comparisons if p["scope"] == scope and p["reference"] == reference
                and p["candidate"] == candidate and p["shape_id"] in ids]


def shape_label(shape):
    return f"{shape['m']} × {shape['n']} × {shape['k']}"


def refs(rows):
    return sorted({ref for row in rows for ref in ([row["_ref"]] if "_ref" in row else row.get("source_refs", []))})


def wrap_canvas_text(fig, text, *, fontsize, weight="normal", width_fraction=.93):
    """Wrap at rendered glyph widths, not an assumed character count."""
    from matplotlib.font_manager import FontProperties
    renderer = fig.canvas.get_renderer()
    font = FontProperties(size=fontsize, weight=weight)
    maximum = fig.bbox.width * width_fraction
    # Presentation typography only: preserve mathematical signs and dimensions.
    text = text.translate(str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-",
                                        "\u2013": "-", "\u2014": "-"}))
    lines = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split():
            proposal = f"{current} {word}" if current else word
            width, _, _ = renderer.get_text_width_height_descent(proposal, font, ismath=False)
            if width > maximum and current:
                lines.append(current)
                current = word
            else:
                current = proposal
        lines.append(current)
    return "\n".join(lines)


def fit_page_layout(fig, title, note, category):
    """Reserve measured header/footer space and let grids fit their own labels.

    Constrained layout accounts for long shape/tile ticks and labels between
    panels. Colorbars must belong to their subplot grid rather than fixed
    figure coordinates. This changes presentation only, never plotted values.
    """
    title_artist = fig.suptitle(wrap_canvas_text(fig, title, fontsize=15, weight="bold"),
                               fontsize=15, fontweight="bold", x=.035, ha="left", y=.975, va="top")
    footer_artist = fig.text(.035, .025, wrap_canvas_text(fig, note, fontsize=8),
                             fontsize=8, va="bottom", color="#49525a")
    title_artist.set_in_layout(False)
    footer_artist.set_in_layout(False)
    renderer = fig.canvas.get_renderer()
    height = fig.bbox.height
    # Physical padding stays legible on both landscape and taller atlas pages.
    bottom = (footer_artist.get_window_extent(renderer).y1 + .25 * fig.dpi) / height
    top = (title_artist.get_window_extent(renderer).y0 - .22 * fig.dpi) / height
    horizontal_pad = .085 if category in ("native_tuning", "failures", "tiles") else .065
    fig.set_layout_engine("constrained", rect=(.02, bottom, .96, top - bottom),
                          w_pad=horizontal_pad, h_pad=.065, wspace=.045, hspace=.045)
    fig.canvas.draw()
    return {"engine": "constrained", "header_footer": "wrapped using rendered glyph widths",
            "body_rect": [.02, bottom, .96, top - bottom],
            "title_lines": len(title_artist.get_text().splitlines()),
            "footer_lines": len(footer_artist.get_text().splitlines()),
            "category": category}


class Book:
    def __init__(self, evidence, output):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                             "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42})
        self.plt, self.ev, self.output = plt, evidence, output
        self.figures = []
        (output / "figures").mkdir()
        self.pdf = PdfPages(output / "chartbook.pdf", metadata={"Title": "Power/midpoint TPU experiment chartbook", "Author": VERSION})

    def figure(self, rows=1, cols=1, **kwargs):
        kwargs.setdefault("layout", "constrained")
        return self.plt.subplots(rows, cols, figsize=(12, 8.2), squeeze=False, **kwargs)

    def empty(self, ax, message="No eligible measurements available"):
        ax.text(.5, .5, textwrap.fill(message, 46), transform=ax.transAxes, ha="center", va="center", color="#66717c")
        ax.set_xticks([])
        ax.set_yticks([])

    def save(self, fig, slug, title, rows=(), note="", category="overview", files=()):
        if any(item["id"] == slug for item in self.figures):
            raise ValueError(f"Duplicate figure ID {slug}")
        layout = fit_page_layout(fig, f"{len(self.figures) + 1}. {title}", note, category)
        png = self.output / "figures" / f"{slug}.png"
        fig.savefig(png, dpi=145)
        self.pdf.savefig(fig)
        self.plt.close(fig)
        self.figures.append({"id": slug, "title": title, "category": category, "note": note,
            "png": str(png.relative_to(self.output)), "pdf": f"chartbook.pdf#page={len(self.figures) + 1}",
            "source_row_refs": refs(rows), "source_files": sorted(set(map(str, files))),
            "page": len(self.figures) + 1, "layout": layout})

    def close(self):
        self.pdf.close()


def methods_sections(ev):
    """Presentation-only extraction from the already verified Evidence object."""
    campaign = ev.campaign
    missing = "not recorded"
    screen = campaign["experiments"]["GRID-screen"]
    confirm = campaign["experiments"]["GRID-confirm"]
    screen_timing = campaign["timing"][screen["timing"]]
    confirm_timing = campaign["timing"][confirm["timing"]]
    precision = campaign.get("precision", {})
    nice_dtype = lambda key: {"bfloat16": "BF16", "float32": "FP32"}.get(precision.get(key), precision.get(key, missing))
    off_lattice = sum(not ev.shapes[sid].get("on_lattice", True) for sid in ev.exploratory)
    distribution = screen.get("distribution", missing)
    counts = {family: len(screen.get("candidate_families", {}).get(family, [])) for family in FAMILIES}
    policy = ev.selection.get("selection_policy", {})
    top = policy.get("top_k", screen.get("shortlist_top_k", missing))
    cap = policy.get("max_confirm_per_family", screen.get("shortlist_max_per_family", missing))
    near = policy.get("near_fraction", screen.get("near_optimal_fraction"))
    near_label = f"{100 * near:g}%" if finite(near) else missing
    correctness = campaign.get("correctness", {})
    gate = correctness.get("gate_expression", missing)
    methods = [
        ("Study and algorithms",
         f"{len(ev.exploratory)} exploratory shapes; {len(ev.holdout)} reserved holdouts remain unmeasured. "
         f"{off_lattice} exploratory shapes are explicit off-lattice boundary probes. "
         f"The lattice has {len(dimensions())} power/midpoint values per axis; sampling is deliberately sparse. "
         f"Inputs: {distribution}. Compare native XLA, full-tile cubic, and one Strassen level per tile-panel. "
         "Public dimensions are M,N,K for A[M,K] @ B[K,N]; kernel records use M,K,N."),
        ("Precision and correctness",
         f"Inputs: {nice_dtype('input_dtype')}; Strassen pre-adds: {nice_dtype('pre_add_dtype')}; "
         f"accumulation: {nice_dtype('accumulator_dtype')}; output: {nice_dtype('output_dtype')}. "
         f"Native dot precision: {precision.get('native_precision', missing)}. "
         "References use the same quantized inputs. Full reference or sampled row/column cross-product coverage "
         "is identified in each result; sampled references retain all K. "
         f"Recorded eligibility gate: {gate}. Gaussian eligibility does not bound worst-case error."),
        ("Independent tuning and fresh confirmation",
         f"Screen: {screen_timing.get('repeats', missing)} measured rounds after {screen_timing.get('warmups', missing)} warmups. "
         f"Confirmation: {confirm_timing.get('repeats', missing)} rounds after {confirm_timing.get('warmups', missing)} warmups in a fresh execution. "
         f"Screen candidates per shape: native {counts['native']}, cubic {counts['cubic']}, Strassen {counts['strassen']}. "
         "Headline choices are the frozen screen winners, never a reselected confirmation minimum. "
         f"Finalists: top {top} plus the descriptive {near_label} screen set, capped at {cap} per family; "
         "the native default is retained. Native tiles are compiler-managed; native candidates vary per-compile options."),
        ("Timing scopes and uncertainty",
         "Complete device call includes required padding, matrix multiplication and cropping. Prepared kernel uses "
         "already prepared device inputs; report it separately. Compilation, host transfers, input generation and "
         "reference work are excluded from steady-state timing. Every measured output is synchronized. "
         "Ratios divide reference mean by candidate mean (>1 favors the candidate). Intervals are paired bootstrap "
         "95% intervals from rounds in the same group: pointwise, without multiplicity correction. Repeated finalist "
         "controls are not pooled into headline results. Tiny shapes can be dominated by host dispatch latency."),
    ]
    environment = ev.metadata["confirmation"].get("environment.json", {})
    identity = environment.get("identity", {})
    versions = identity.get("versions", {})
    kind = identity.get("device_kind") or next((d.get("kind") for d in environment.get("devices", []) if d.get("kind")), missing)
    hardware = (f"Device kind: {kind}\n"
                f"Devices / local devices / processes: {environment.get('device_count', missing)} / "
                f"{environment.get('local_device_count', missing)} / {environment.get('process_count', missing)}\n"
                f"Qualified single v5e: {environment.get('qualified_single_v5e', missing)}\n"
                f"Allocation: {identity.get('allocation_id', missing)}\n"
                f"Hostname: {identity.get('hostname', missing)}\n"
                f"Runtime image: {identity.get('runtime_image') or missing}\n"
                "Screen and confirmation identities match. Identity describes the logical allocation/runtime; "
                "no physical chip serial was verified.")
    software_lines = [f"Python: {str(environment.get('python', missing)).split()[0]}"]
    for package in ("jax", "jaxlib", "libtpu", "numpy", "ml_dtypes"):
        value = versions.get(package) or identity.get(f"{package}_version") or missing
        software_lines.append(f"{package}: {value}")
    software_lines += [f"Backend: {environment.get('backend', missing)}",
                       "Exact runtime flags, package versions and source hashes remain in environment.json and source_manifest.json."]
    source_lines = []
    for tag, label in (("screen", "Screen"), ("confirmation", "Confirmation")):
        directory = ev.artifacts[tag]
        run_id = directory.parent.name if directory.name == "artifacts" else directory.name
        source_lines.append(f"{label} run: {run_id}")
        for name in ("results.jsonl", "source_manifest.json"):
            record = ev.sources.get(str((directory / name).resolve()), {})
            digest = record.get("sha256", missing)
            source_lines.append(f"{label} {name} SHA256: {digest[:16]}")
    source_lines += [f"Campaign SHA256: {ev.selection.get('campaign_sha256', missing)[:16]}",
                     f"Shape manifest SHA256: {ev.selection.get('shape_manifest_sha256', missing)[:16]}"]
    provenance = [
        ("Captured hardware", hardware),
        ("Captured software", "\n".join(software_lines)),
        ("Run identifiers and hash prefixes", "\n".join(source_lines)),
        ("How to read this chartbook",
         "The native series is the frozen tuned native choice; native_default is the original library default. "
         "Near-optimal sets contain only tested tile tuples relative to the frozen family winner, not global tile optima "
         "or every tuple inside an axis range. Gray or crossed map cells lack an eligible comparison; do not interpolate "
         "a full performance surface. Reserved holdouts are excluded from timings and fitting. "
         "The 3D color scale is capped at ±2 in log₂ units; hover for exact ratios. "
         "Sealed input hashes and per-figure source references are available in audit.json and figures.json."),
    ]
    return [("Methods and reading guide", methods), ("Hardware, software and provenance", provenance)]


def frontmatter_pages(book):
    """Two concise presentation pages; no additional measurement or inference."""
    files = [book.ev.manifest_path]
    files += [book.ev.artifacts[tag] / name for tag in ("screen", "confirmation")
              for name in ("environment.json", "source_manifest.json", "results.jsonl")]
    for page_index, (title, sections) in enumerate(methods_sections(book.ev), 1):
        fig, axes = book.figure(2, 2)
        fig.set_size_inches(12, 10)
        for ax, (heading, body) in zip(axes.flat, sections):
            ax.set_axis_off()
            ax.set_title(heading, loc="left", fontsize=12, fontweight="bold", pad=14)
            width = ax.get_position().width * .98
            wrapped = wrap_canvas_text(fig, body, fontsize=10.5, width_fraction=width)
            ax.text(0, 1, wrapped, transform=ax.transAxes, ha="left", va="top",
                    fontsize=10.5, linespacing=1.35, color="#243442")
        book.save(fig, f"000_methods_{page_index:02d}", title,
                  note="Methods reflect the verified captured cohort and configuration. Hashes are shortened here; complete values are retained in the linked audit artifacts.",
                  category="methods", files=files)


def axes_log_lattice(ax, x="M", y="N"):
    ticks = list(range(0, 18, 2)) + [17]
    ax.set(xlim=(-.4, 17.4), ylim=(-.4, 17.4), xlabel=f"log₂ {x}", ylabel=f"log₂ {y}")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(alpha=.18)


def overview(book):
    ev, plt = book.ev, book.plt
    fig = plt.figure(figsize=(12, 8.2))
    for i, (stage, marker, color) in enumerate((("exploratory", "o", "#2472B4"), ("holdout", "^", "#888888"))):
        points = [s for s in ev.shapes.values() if s.get("stage") == stage]
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        for on_lattice, symbol in ((True, marker), (False, "x")):
            rows = [s for s in points if s.get("on_lattice", True) == on_lattice]
            if rows:
                ax.scatter(*[[math.log2(s[a]) for s in rows] for a in ("m", "n", "k")],
                           color=color, marker=symbol, s=36, alpha=.8)
        ax.set(xlabel="log₂ M", ylabel="log₂ N", zlabel="log₂ K", title=f"{stage.title()} ({len(points)} shapes)",
               xlim=(0, 17), ylim=(0, 17), zlim=(0, 17))
    book.save(fig, "001_design_3d", "Shape coverage in M, N, K", note="A[M,K] @ B[K,N]. Deliberately chosen shapes: win fractions are not prevalence over the 39,304-shape lattice. Crosses are off-lattice probes; holdouts stay unmeasured. This campaign contains Gaussian diagnostics only, without new stress, device-profile or fitted-model claims.", files=[ev.manifest_path])
    fig, axs = book.figure(1, 3)
    for ax, (x, y) in zip(axs.flat, (("m", "n"), ("m", "k"), ("n", "k"))):
        for stage, symbol in (("exploratory", "o"), ("holdout", "^")):
            points = [s for s in ev.shapes.values() if s.get("stage") == stage]
            ax.scatter([math.log2(s[x]) for s in points], [math.log2(s[y]) for s in points], marker=symbol, label=stage, alpha=.65)
        axes_log_lattice(ax, x.upper(), y.upper())
        ax.legend(fontsize=8)
    book.save(fig, "002_design_projections", "Coverage projections", note="Points with different hidden coordinates can overlap. Exact dimensions and strata are in shapes.csv.", files=[ev.manifest_path])
    fig, axs = book.figure(1, 2)
    counts = Counter(s.get("family", "unknown") for s in ev.shapes.values())
    axs[0, 0].barh(list(counts), list(counts.values()), color="#2472B4")
    axs[0, 0].tick_params(axis="y", labelsize=8)
    axs[0, 0].set_xlabel("Planned shapes, including reserved holdouts")
    for stage in ("exploratory", "holdout"):
        points = [s for s in ev.shapes.values() if s.get("stage") == stage]
        axs[0, 1].scatter([math.log2(s["m"] * s["n"] * s["k"]) for s in points],
                         [s.get("conservative_pair_buffer_bytes", 0) / 1024**3 for s in points], label=stage)
    axs[0, 1].set(xlabel="log₂(MNK)", ylabel="Planning pair-buffer estimate, GiB")
    axs[0, 1].legend()
    book.save(fig, "003_plan_accounting", "Design strata and analytical memory", note="Planning estimate is not measured peak memory and excludes executable/runtime peaks and VMEM. See campaign runtime preflight for admission.", files=[ev.manifest_path])


def forest_pages(book):
    ev = book.ev
    for scope in SCOPES:
        for baseline, candidate in PAIRS:
            pairs = sorted(ev.pairs(scope, baseline, candidate), key=lambda p: p["speedup"])
            for start in range(0, max(1, len(pairs)), 20):
                page = pairs[start:start + 20]
                fig, axs = book.figure()
                ax = axs[0, 0]
                for y, p in enumerate(page):
                    ax.plot([p["ci95_low"], p["ci95_high"]], [y, y], color=COLORS[candidate], lw=1.5)
                    ax.scatter(p["speedup"], y, color=COLORS[candidate], s=24)
                if page:
                    ax.set_yticks(range(len(page)), [shape_label(ev.shapes[p["shape_id"]]) + f"  n={p['paired_round_count']}" for p in page], fontsize=8)
                    ax.set_xscale("log", base=2)
                    ax.axvline(1, color="black", lw=1)
                    ax.axvline(1.05, color="#777", lw=.7, ls=":")
                    ax.set(xlabel=f"{baseline} mean / {candidate} mean", ylabel="Shape (M × N × K)")
                    ax.grid(axis="x", alpha=.2)
                else:
                    book.empty(ax)
                book.save(fig, f"forest_{scope}_{baseline}_{candidate}_{start//20+1:02d}",
                    f"{SCOPE_LABELS[scope]}: {baseline} / {candidate}", page,
                    CI_NOTE + " Frozen screening choices; dashed line=1.05×. Native_default is the library default; native is its frozen tuned choice.", "speedups")


def family_curves(book):
    ev = book.ev
    families = sorted({ev.shapes[s]["family"] for s in ev.exploratory})
    for design_family in families:
        ids = [s for s in ev.exploratory if ev.shapes[s]["family"] == design_family]
        axis = design_family[-1] if design_family.startswith("axis_") else None
        if axis:
            ids.sort(key=lambda s: ev.shapes[s][axis])
            xs = [ev.shapes[s][axis] for s in ids]
            xlabel = f"{axis.upper()} extent"
        elif "square" in design_family:
            ids.sort(key=lambda s: ev.shapes[s]["m"])
            xs = [ev.shapes[s]["m"] for s in ids]
            xlabel = "Square extent"
        else:
            ids.sort(key=lambda s: tuple(ev.shapes[s][a] for a in ("m", "n", "k")))
            xs = list(range(len(ids)))
            xlabel = "Shape (M × N × K)"
        for scope in SCOPES:
            fig, axs = book.figure(2, 2)
            used = []
            for family in FAMILIES:
                points = [(x, ev.selected(s, scope, family)) for x, s in zip(xs, ids)]
                available = [(x, r) for x, r in points if eligible(r)]
                used.extend(r for _, r in available)
                # NaNs preserve gaps and never bridge a failed/unmeasured point.
                ys = [mean_ms(r) if eligible(r) else math.nan for _, r in points]
                axs[0, 0].plot(xs, ys, marker="o", label=family, color=COLORS[family])
                throughput = [2 * ev.shapes[s]["m"] * ev.shapes[s]["n"] * ev.shapes[s]["k"] / mean_ms(r) / 1e9
                              if eligible(r) else math.nan for s, (_, r) in zip(ids, points)]
                axs[0, 1].plot(xs, throughput, marker="o", label=family, color=COLORS[family])
            for ax, baseline in ((axs[1, 0], "native"), (axs[1, 1], "cubic")):
                pairs = {p["shape_id"]: p for p in ev.pairs(scope, baseline, "strassen", ids)}
                for x, shape_id in zip(xs, ids):
                    p = pairs.get(shape_id)
                    if p:
                        ax.plot([x, x], [p["ci95_low"], p["ci95_high"]], color=COLORS["strassen"])
                        ax.scatter(x, p["speedup"], color=COLORS["strassen"])
                ax.axhline(1, color="black", lw=.7)
                ax.set_ylabel(f"{baseline} / Strassen")
            axs[0, 0].set_ylabel("Latency, ms")
            axs[0, 0].set_yscale("log")
            axs[0, 1].set_ylabel("Effective classical-equivalent TFLOP/s")
            axs[0, 0].legend(fontsize=8)
            for ax in axs.flat:
                ax.set_xlabel(xlabel)
                ax.grid(alpha=.2)
                if axis or "square" in design_family:
                    ax.set_xscale("log", base=2)
                else:
                    ax.set_xticks(xs, [shape_label(ev.shapes[s]) for s in ids], rotation=25, ha="right", fontsize=7)
            book.save(fig, f"path_{design_family}_{scope}", f"{design_family.replace('_', ' ')} · {SCOPE_LABELS[scope]}", used,
                      CI_NOTE + " Effective throughput uses 2MNK, not measured Strassen FLOP/s. Tiny shapes are host-latency-floor sensitive, not saturation evidence. Lines follow the registered family; missing points leave gaps.", "shape_paths")


def diagnostics(book):
    ev = book.ev
    for scope in SCOPES:
        fig, axs = book.figure(2, 3)
        used = []
        for family in FAMILIES:
            rows = [ev.selected(s, scope, family) for s in ev.exploratory]
            rows = [r for r in rows if r is not None]
            used.extend(rows)
            for row in rows:
                shape = ev.shapes[row["shape_id"]]
                volume = math.log2(shape["m"] * shape["n"] * shape["k"])
                metrics = row.get("correctness") or {}
                for ax, field in ((axs[0, 0], "relative_l2"), (axs[0, 1], "max_abs_error")):
                    value = metrics.get(field)
                    if finite(value):
                        ax.scatter(volume, value, c=COLORS[family], marker="o" if metrics.get("pass") else "x", s=22)
                if eligible(row):
                    timing = row["timing"]
                    raw = ev.raw(row)
                    cv = statistics.pstdev(s["elapsed_ms"] for s in raw) / mean_ms(row)
                    axs[0, 2].scatter(mean_ms(row), cv, color=COLORS[family], s=22)
                    axs[1, 0].scatter(volume, mean_ms(row), color=COLORS[family], s=22)
                    padding = (row.get("kernel_metadata") or {}).get("padded_volume_ratio")
                    if finite(padding):
                        axs[1, 1].scatter(volume, padding, color=COLORS[family], s=22)
                    axs[1, 2].scatter(mean_ms(row), metrics.get("relative_l2", 0), color=COLORS[family], s=22)
        for family in FAMILIES:
            axs[0, 0].scatter([], [], label=family, color=COLORS[family])
        axs[0, 0].legend(fontsize=8)
        axs[0, 0].set(xlabel="log₂ MNK", ylabel="Relative L2 error")
        axs[0, 0].set_yscale("symlog", linthresh=1e-8)
        axs[0, 1].set(xlabel="log₂ MNK", ylabel="Maximum absolute error")
        axs[0, 1].set_yscale("symlog", linthresh=1e-8)
        axs[0, 2].set(xlabel="Mean latency, ms", ylabel="Raw timing coefficient of variation", xscale="log")
        axs[1, 0].set(xlabel="log₂ MNK", ylabel="Latency, ms", yscale="log")
        axs[1, 1].set(xlabel="log₂ MNK", ylabel="Padded / useful volume", yscale="log")
        axs[1, 2].set(xlabel="Latency, ms", ylabel="Relative L2 error", xscale="log")
        axs[1, 2].set_yscale("symlog", linthresh=1e-8)
        book.save(fig, f"diagnostics_{scope}", f"Numerical accuracy, timing variation and padding · {SCOPE_LABELS[scope]}", used,
            "BF16 inputs and Strassen pre-adds; FP32 accumulation/output. Gaussian checks only. Error coverage follows each raw record. Coefficient of variation describes rounds within one allocation.", "diagnostics")
        fig, axs = book.figure(2, 2)
        pairrows = ev.pairs(scope)
        for p in pairrows:
            row = ev.selected(p["shape_id"], scope, "strassen")
            meta = row.get("kernel_metadata") or {}
            tile = row.get("tile") or meta.get("tile_bm_bn_bk")
            padded = meta.get("padded_shape_mkn")
            xs = [meta.get("padded_volume_ratio"), None, None, None]
            if tile and padded:
                bm, bn, bk = tile
                mp, kp, np_ = padded
                xs[1:] = [(mp / bm) * (np_ / bn), kp / bk, 5 / bm + 5 / bn + 8 / bk]
            for ax, x in zip(axs.flat, xs):
                if finite(x, True):
                    ax.scatter(x, p["speedup"], color=COLORS["strassen"], s=24)
        for ax, label in zip(axs.flat, ("Strassen padded/useful volume", "Parallel output tile count", "Sequential K panels", "Source vector-work / saved-dot-work feature")):
            ax.set(xlabel=label, ylabel="Native / Strassen", xscale="log")
            ax.axhline(1, color="black", lw=.8)
            ax.grid(alpha=.15)
        book.save(fig, f"mechanism_{scope}", f"Analytical features against measured speedup · {SCOPE_LABELS[scope]}", pairrows,
            "Features are source-level estimates, not measured occupancy, bandwidth or instruction counters. The vector feature 5/BM+5/BN+8/BK derives from equal-tile arithmetic; native uses compiler-managed tiles.", "mechanisms")
    fig, axs = book.figure(1, 2)
    for tag, ax in zip(("screen", "confirmation"), axs.flat):
        rows = [r for r in ev.compilation if r["_cohort"] == tag and finite(r.get("compile_ms"), True)]
        for family in FAMILIES:
            selected = [r for r in rows if r.get("family") == family]
            ax.scatter(range(len(selected)), [r["compile_ms"] / 1000 for r in selected], label=family, color=COLORS[family], s=9, alpha=.5)
        ax.set(title=tag.title(), xlabel="Compilation index within family", ylabel="Compilation time, seconds", yscale="log")
        ax.legend(fontsize=8)
    book.save(fig, "compiler_time", "Compilation costs are separate from steady-state latency", ev.compilation,
              "Successful compilations only; failures are in the failure maps and CSV. Compiler cost/temporary-memory estimates are not hardware measurements.", "compiler")
    fig, axs = book.figure(1, 2)
    book.empty(axs[0, 0], "No runtime HBM traffic or hardware utilization counters recorded by this runner")
    book.empty(axs[0, 1], "Compiler temporary-memory and cost-analysis dictionaries are unavailable unless explicitly recorded")
    book.save(fig, "unmeasured_mechanisms", "Unmeasured hardware mechanisms", note="Do not infer bandwidth by dividing original-operand bytes by warm-call latency. Do not label compiler estimates as hardware counters.", category="compiler")


def format_latency_ticks(ax):
    """Keep logarithmic latency axes while avoiding long adjacent tick labels."""
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter, NullLocator
    low, high = ax.get_xlim()
    if low > 0 and high / low < 5:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=2))
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1,), numticks=6))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.3f}".rstrip("0").rstrip(".")))
    ax.xaxis.set_minor_formatter(NullFormatter())


def detailed_shape_pages(book):
    ev = book.ev
    for index, shape_id in enumerate(ev.exploratory):
        shape = ev.shapes[shape_id]
        screen = [r for r in ev.results if r["_cohort"] == "screen" and r.get("shape_id") == shape_id]
        fig, axs = book.figure(2, 2, gridspec_kw={"height_ratios": [2, 1]})
        fig.set_size_inches(14, 13)
        for family, col in (("cubic", 0), ("strassen", 1)):
            rows = sorted([r for r in screen if r.get("family") == family and r.get("scope") == "call"],
                          key=lambda r: (tuple(r.get("tile") or []), r.get("candidate_id", "")))
            ax = axs[0, col]
            xs = list(range(len(rows)))
            screened_eligible = {c["candidate_id"] for c in ev.selection.get("by_shape", {}).get(shape_id, {}).get(family, {}).get("screen_eligible_candidates", [])}
            for x, row in zip(xs, rows):
                if row["candidate_id"] in screened_eligible:
                    ax.scatter(mean_ms(row), x, color=COLORS[family], s=25)
                else:
                    ax.scatter(.03, x, transform=ax.get_yaxis_transform(), marker="x", color="#b33030", s=25)
            sample_counts = sorted({(r.get("timing") or {}).get("sample_count") for r in rows if finite((r.get("timing") or {}).get("sample_count"))})
            ax.set(title=f"{family.title()} screen · complete call · n={sample_counts}", xlabel="Latency, ms", xscale="log")
            format_latency_ticks(ax)
            ax.set_yticks(xs, ["/".join(map(str, r.get("tile") or [])) for r in rows], fontsize=8)
            ax.set_ylabel("Actual tile BM/BN/BK; red × = not screen-eligible")
            ax.grid(axis="x", alpha=.2)
            if not rows:
                book.empty(ax)
            report = ev.confirmation.get("by_shape", {}).get(shape_id, {}).get(family, {})
            ax = axs[1, col]
            candidates = report.get("candidates", [])
            for i, candidate in enumerate(candidates):
                for offset, scope, color in ((-.1, "call", COLORS[family]), (.1, "prepared_kernel", "#555555")):
                    data = candidate.get("scopes", {}).get(scope, {})
                    value, ci = data.get("latency_ratio_to_frozen_winner"), data.get("latency_ratio_ci95")
                    if finite(value, True):
                        ax.scatter(value, i + offset, color=color, s=23)
                        if ci and all(finite(v, True) for v in ci):
                            ax.plot(ci, [i + offset] * 2, color=color)
            if candidates:
                ax.set_yticks(range(len(candidates)), ["/".join(map(str, c.get("tile") or [])) for c in candidates], fontsize=8)
                for value, linestyle in ((1, "-"), (1.01, ":"), (1.03, ":"), (1.05, "--")):
                    ax.axvline(value, color="#888888", lw=.6, ls=linestyle)
                ax.set(xlabel="Candidate latency / frozen family winner", title="Confirmation shortlist (colored=call; gray=prepared)")
            else:
                book.empty(ax, "No confirmation shortlist observations")
        confirmation_rows = [r for r in ev.results if r["_cohort"] == "confirmation" and r.get("shape_id") == shape_id]
        book.save(fig, f"tile_landscape_{index:03d}", f"Tile landscape · M,N,K = {shape_label(shape)}", screen + confirmation_rows,
            "Screen minima selected the frozen winners; confirmation does not reselect headline choices. Each BM/BN/BK label is one tested tuple. Marginal axis ranges do not establish Cartesian products. Bottom ratios >1 are slower.", "tiles", [ev.artifacts["confirmation"] / "confirmation.json"])
        fig, axs = book.figure(2, 2)
        used = []
        for i, scope in enumerate(SCOPES):
            for family in FAMILIES:
                row = ev.selected(shape_id, scope, family)
                if not row:
                    continue
                used.append(row)
                raw = sorted(ev.raw(row), key=lambda s: s["round"])
                used.extend(raw)
                if not raw:
                    continue
                axs[i, 0].plot([r["round"] for r in raw], [r["elapsed_ms"] for r in raw], marker=".", lw=.8, color=COLORS[family], label=f"{family} n={len(raw)}")
                ratios = sorted(r["elapsed_ms"] / mean_ms(row) for r in raw)
                axs[i, 1].step(ratios, [(k + 1) / len(ratios) for k in range(len(ratios))], where="post", color=COLORS[family], label=family)
            axs[i, 0].set(title=SCOPE_LABELS[scope], xlabel="Paired timing round", ylabel="Latency, ms")
            axs[i, 1].set(xlabel="Raw latency / its own mean", ylabel="Empirical cumulative fraction")
            if axs[i, 0].has_data():
                axs[i, 0].legend(fontsize=8)
            else:
                book.empty(axs[i, 0])
                book.empty(axs[i, 1])
        book.save(fig, f"raw_noise_{index:03d}", f"Raw confirmation rounds · M,N,K = {shape_label(shape)}", used,
            "Headline group only. Shortlist-group winner controls are separate and never pooled. ECDF normalizes each arm by its own mean; this shows spread, not speedup. All raw timings remain in samples.csv.", "raw_timing")


def failure_pages(book):
    ev = book.ev
    statuses = sorted({r.get("status", "unknown") for r in ev.results})
    import numpy as np
    from matplotlib.colors import ListedColormap, BoundaryNorm
    palette = ["#b5ddcf" if s == "ok" else "#e4ba89" if "skip" in s else "#d98484" for s in statuses]
    cmap, norm = ListedColormap(palette), BoundaryNorm(np.arange(len(statuses) + 1) - .5, len(statuses))
    for tag in ("screen", "confirmation"):
        for start in range(0, len(ev.exploratory), 20):
            ids = ev.exploratory[start:start + 20]
            fig, axs = book.figure(1, 2)
            used = []
            for ax, scope in zip(axs.flat, SCOPES):
                rows = [r for r in ev.results if r["_cohort"] == tag and r.get("shape_id") in ids and r.get("scope") == scope]
                used.extend(rows)
                counts = np.zeros((len(ids), len(statuses)), dtype=int)
                for row in rows:
                    counts[ids.index(row["shape_id"]), statuses.index(row.get("status", "unknown"))] += 1
                ax.imshow(np.tile(np.arange(len(statuses)), (len(ids), 1)), cmap=cmap, norm=norm, aspect="auto", alpha=.45)
                for y in range(len(ids)):
                    for x in range(len(statuses)):
                        ax.text(x, y, str(counts[y, x]), ha="center", va="center", fontsize=8)
                ax.set_xticks(range(len(statuses)), statuses, rotation=30, ha="right", fontsize=8)
                ax.set_yticks(range(len(ids)), [shape_label(ev.shapes[s]) for s in ids], fontsize=7)
                ax.set_title(SCOPE_LABELS[scope])
            book.save(fig, f"statuses_{tag}_{start//20:02d}", f"{tag.title()} outcome counts · all attempted configurations", used,
                "Counts are scoped result records, including repeated winner controls in confirmation. Zero means no such recorded outcome, not successful execution. Unattempted configurations have no fabricated timing.", "failures")


def slice_atlas(book):
    ev = book.ev
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    values = dimensions()
    lattice = set(values)
    logvalues = [math.log2(v) for v in values]
    norm = TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    slices = [(fixed, x, y, value) for fixed, x, y in
              (("k", "m", "n"), ("n", "m", "k"), ("m", "n", "k")) for value in values]
    for page in range(0, len(slices), 4):
        fig, axs = book.figure(4, 3)
        fig.set_size_inches(12, 13.5)
        used, labels = [], []
        for i, (fixed, x, y, value) in enumerate(slices[page:page + 4]):
            ids = [sid for sid in ev.exploratory if ev.shapes[sid][fixed] == value
                   and all(ev.shapes[sid][a] in lattice for a in ("m", "n", "k"))]
            labels.append(f"{fixed.upper()}={value}")
            for j, (baseline, candidate) in enumerate(PAIRS[:3]):
                ax = axs[i, j]
                gx, gy = np.meshgrid(logvalues, logvalues)
                ax.scatter(gx.ravel(), gy.ravel(), color="#e5e8eb", s=2, rasterized=True)
                pairs = ev.pairs("call", baseline, candidate, ids)
                used.extend(pairs)
                if pairs:
                    xs = [math.log2(ev.shapes[p["shape_id"]][x]) for p in pairs]
                    ys = [math.log2(ev.shapes[p["shape_id"]][y]) for p in pairs]
                    ax.scatter(xs, ys, c=[math.log2(p["speedup"]) for p in pairs], cmap="RdBu", norm=norm,
                               s=40, edgecolor="black", linewidth=.45)
                available = {p["shape_id"] for p in pairs}
                failed = [sid for sid in ids if sid not in available and
                          any(ev.selected(sid, "call", f) is not None for f in (baseline, candidate))]
                if failed:
                    ax.scatter([math.log2(ev.shapes[sid][x]) for sid in failed],
                               [math.log2(ev.shapes[sid][y]) for sid in failed], marker="x", color="#b33030", s=32)
                if not pairs and not failed:
                    ax.text(.5, .5, "No eligible measured points", transform=ax.transAxes, ha="center", fontsize=8, color="#66717c")
                axes_log_lattice(ax, x.upper(), y.upper())
                ax.tick_params(labelsize=7)
                ax.set_title(f"{fixed.upper()}={value:,} · {baseline}/{candidate}", fontsize=9)
        for ax in axs[len(slices[page:page + 4]):].flat:
            ax.set_visible(False)
        sm = book.plt.cm.ScalarMappable(norm=norm, cmap="RdBu")
        fig.colorbar(sm, ax=list(axs.flat), location="right", fraction=.022, pad=.025,
                     shrink=.82, label="log₂(reference/candidate), capped at ±2")
        book.save(fig, f"slice_atlas_{page//4:02d}", f"Complete-call lattice slices {page+1}–{min(page+4, len(slices))} of 102", used,
            "Gray cells have no eligible comparison; red × means an observed but ineligible comparison. No interpolation. Off-lattice probes excluded. Fixed scale across 102 slices. Prepared results have separate plots; native is the frozen tuned choice.", "slice_atlas", [ev.manifest_path])


def nearset_pages(book):
    ev = book.ev
    rows = []
    for shape_id, shape_report in ev.confirmation.get("by_shape", {}).items():
        for family in FAMILIES:
            report = shape_report.get(family, {})
            for scope, sets in report.get("near_sets", {}).items():
                for tolerance, projection in sets.items():
                    # The projection's scope is a semantic qualification, not a
                    # timing scope. Preserve both without overwriting call/prepared.
                    rows.append({**projection, "projection_scope": projection.get("scope"),
                                 "shape_id": shape_id, "family": family, "scope": scope,
                                 "definition": tolerance})
    write_csv(book.output / "tables/near_sets.csv", rows)
    for scope in SCOPES:
        fig, axs = book.figure(1, 2)
        for family, ax in zip(("cubic", "strassen"), axs.flat):
            for definition in ("descriptive_1_percent", "descriptive_3_percent", "descriptive_5_percent", "pointwise_ci95_certified_5_percent"):
                counts = []
                for sid in ev.exploratory:
                    found = next((r for r in rows if r["shape_id"] == sid and r["family"] == family and r["scope"] == scope and r["definition"] == definition), None)
                    counts.append(len(found.get("tuples_bm_bn_bk", [])) if found else math.nan)
                ax.plot(range(len(counts)), counts, marker=".", lw=.6, label=definition.replace("descriptive_", "").replace("pointwise_ci95_certified_", "CI95_"))
            ax.set(title=family.title(), xlabel="Exploratory shape index (manifest order)", ylabel="Number of actual confirmed tuples")
            ax.legend(fontsize=7)
        book.save(fig, f"nearset_sizes_{scope}", f"Near-tile tuple sets · {SCOPE_LABELS[scope]}", note="Thresholds are relative to the frozen screen winner, not a reselected confirmation minimum. CI95 qualification is pointwise, not simultaneous equivalence. The CSV contains the exact tuple members; missing data remain gaps.", category="tiles", files=[ev.artifacts["confirmation"] / "confirmation.json"])


def native_tuning_pages(book):
    ev = book.ev
    for start in range(0, len(ev.exploratory), 20):
        ids = ev.exploratory[start:start + 20]
        rows = [r for r in ev.results if r["_cohort"] == "screen" and r.get("family") == "native" and r.get("shape_id") in ids]
        options = sorted({r["candidate_id"] for r in rows})
        fig, axs = book.figure(1, 2)
        for ax, scope in zip(axs.flat, SCOPES):
            scoped = [r for r in rows if r.get("scope") == scope]
            for y, shape_id in enumerate(ids):
                candidates = [r for r in scoped if r["shape_id"] == shape_id]
                default = next((r for r in candidates if not r.get("compiler_options", {})), None)
                for x, candidate_id in enumerate(options):
                    row = next((r for r in candidates if r["candidate_id"] == candidate_id), None)
                    if eligible(row) and eligible(default):
                        value = mean_ms(default) / mean_ms(row)
                        ax.scatter(x, y, c=[math.log2(value)], cmap="RdBu", vmin=-1, vmax=1, s=130, marker="s")
                        ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7)
                    elif row:
                        ax.scatter(x, y, marker="x", color="#b33030")
            ax.set_xticks(range(len(options)), [o.replace("native_", "") for o in options], rotation=25, ha="right", fontsize=8)
            ax.set_yticks(range(len(ids)), [shape_label(ev.shapes[s]) for s in ids], fontsize=7)
            ax.set_title(SCOPE_LABELS[scope])
        book.save(fig, f"native_options_{start//20:02d}", "Native compiler-option screen · default / candidate", rows,
            "Tiles remain compiler-managed. These descriptive screening ratios may compare different candidate batches; no paired significance is claimed. Red × denotes missing/ineligible reference or candidate. Confirmation uses its frozen option winner.", "native_tuning")


def interactive(book):
    ev = book.ev
    try:
        import plotly.graph_objects as go
    except ImportError:
        return {"available": False, "reason": "Plotly unavailable; static 3D plus 102-slice atlas provided"}
    fig, panels = go.Figure(), []

    def hover(shape, scope, pairs, baseline, candidate):
        lines = [f"M,N,K={shape_label(shape)}", html.escape(shape["family"]),
                 "On lattice" if shape.get("on_lattice", True) else "Explicit off-lattice boundary probe"]
        for family in FAMILIES:
            row = ev.selected(shape["id"], scope, family)
            if row is None:
                lines.append(f"{family}: unmeasured")
                continue
            timing = f"{mean_ms(row):.6f} ms" if finite(mean_ms(row), True) else "no valid latency"
            if family == "native":
                options = row.get("compiler_options", {})
                preset = options.get("xla_tpu_scoped_vmem_limit_kib")
                config = f"VMEM option={preset} KiB" if preset else "library default options"
                config += "; tile compiler-managed"
            else:
                config = "tile BM,BN,BK=" + str(row.get("tile"))
            lines.append(f"{family}: {timing}; {config}; {html.escape(row.get('status', 'unknown'))}")
        pair = pairs.get(shape["id"])
        if pair:
            lines += [f"{baseline}/{candidate}={pair['speedup']:.4f}",
                      f"95% CI [{pair['ci95_low']:.4f}, {pair['ci95_high']:.4f}]; n={pair['paired_round_count']}"]
        else:
            lines.append("Reserved, unmeasured" if shape["id"] in ev.holdout else "No eligible comparison")
        return "<br>".join(lines)

    for scope in SCOPES:
        for baseline, candidate in PAIRS[:3]:
            index = len(panels)
            pairs = {p["shape_id"]: p for p in ev.pairs(scope, baseline, candidate)}
            groups = [
                ("Measured lattice", "circle", [s for s in ev.shapes.values() if s["id"] in pairs and s.get("on_lattice", True)], True),
                ("Off-lattice boundary probes", "x", [s for s in ev.shapes.values() if s["id"] in pairs and not s.get("on_lattice", True)], True),
                ("Missing/ineligible comparison", "x", [s for s in ev.shapes.values() if s.get("stage") == "exploratory" and s["id"] not in pairs], False),
                ("Reserved holdouts", "diamond", [s for s in ev.shapes.values() if s["id"] in ev.holdout], False),
            ]
            for name, symbol, shapes, measured in groups:
                marker = dict(size=5, symbol=symbol)
                if measured:
                    marker.update(color=[math.log2(pairs[s["id"]]["speedup"]) for s in shapes], coloraxis="coloraxis")
                else:
                    marker["color"] = "#999999"
                fig.add_trace(go.Scatter3d(x=[math.log2(s["m"]) for s in shapes], y=[math.log2(s["n"]) for s in shapes],
                    z=[math.log2(s["k"]) for s in shapes], mode="markers", name=name, visible=index == 0,
                    text=[hover(s, scope, pairs, baseline, candidate) for s in shapes],
                    hovertemplate="%{text}<extra></extra>", marker=marker))
            panels.append((scope, baseline, candidate))
    buttons = []
    for index, (scope, baseline, candidate) in enumerate(panels):
        title = f"{SCOPE_LABELS[scope]}: {baseline} / {candidate} · frozen choices"
        buttons.append(dict(label=f"{scope}: {baseline}/{candidate}", method="update",
                            args=[{"visible": [i // 4 == index for i in range(len(fig.data))]}, {"title.text": title}]))
    fig.update_layout(
        title=dict(text="Complete device call: native / strassen · frozen choices",
                   x=.055, y=.97, xanchor="left", yanchor="top"),
        scene=dict(xaxis_title="log₂ M", yaxis_title="log₂ N", zaxis_title="log₂ K"),
        coloraxis=dict(colorscale="RdBu", cmin=-2, cmax=2,
                       colorbar=dict(title="log₂ speedup", x=1.02, y=.5, len=.88)),
        updatemenus=[dict(buttons=buttons, direction="down", x=0, y=1.16,
                         xanchor="left", yanchor="top")],
        legend=dict(orientation="h", x=0, y=-.06, xanchor="left", yanchor="top"),
        margin=dict(l=70, r=130, t=180, b=120), height=900)
    fig.add_annotation(text="Color scale capped at ±2 in log₂ units; hover for exact ratios.",
                       xref="paper", yref="paper", x=0, y=1.075, showarrow=False,
                       xanchor="left", yanchor="top", font=dict(size=12, color="#49525a"))
    fig.write_html(book.output / "interactive_3d.html", include_plotlyjs=True, full_html=True, auto_open=False)
    return {"available": True, "path": "interactive_3d.html", "network_required": False,
            "comparison_scope_choices": len(panels), "shared_colorbars": 1}


def gallery(book, interactive_result):
    items = []
    for item in book.figures:
        title, category, note = (html.escape(item[k]) for k in ("title", "category", "note"))
        items.append(f'<article data-category="{category}"><h2>{item["page"]}. {title}</h2><a href="{item["pdf"]}">PDF</a><p>{note}</p><img loading="lazy" src="{item["png"]}" alt="{title}"></article>')
    link = '<a href="interactive_3d.html">Interactive offline 3D</a>' if interactive_result["available"] else html.escape(interactive_result["reason"])
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Power/midpoint TPU chartbook</title><style>body{{font:16px system-ui;margin:2rem auto;max-width:1250px;padding:0 1rem;background:#f4f6f8;color:#182632}}header,article{{background:white;padding:1.4rem;margin:1rem 0;border-radius:8px}}img{{width:100%;height:auto}}p{{line-height:1.5}}input{{font:inherit;padding:.6rem;width:90%}}a{{color:#176699}}</style><header><h1>Power/midpoint TPU experiment chartbook</h1><p>Measured observations, analytical features and unmeasured cells remain distinct. Native means its frozen screened choice; native_default is the original library default. Shapes display M,N,K; kernel arrays use M,K,N.</p><p><a href="chartbook.pdf">Complete PDF</a> · {link} · <a href="audit.json">Audit and hashes</a> · <a href="figures.json">Per-figure sources</a> · <a href="tables/comparisons.csv">Comparisons CSV</a></p><p>{html.escape(CI_NOTE)} Holdouts remain unmeasured. Search filters local figures only.</p><input id="search" placeholder="Filter by title, category or note" aria-label="Filter figures"></header>{''.join(items)}<script>document.getElementById('search').addEventListener('input',event=>{{const q=event.target.value.toLowerCase();document.querySelectorAll('article').forEach(el=>el.hidden=!el.textContent.toLowerCase().includes(q)&&!el.dataset.category.includes(q));}});</script></html>'''
    (book.output / "index.html").write_text(content, encoding="utf-8")


def generate(screen, confirmation, manifest, output_dir):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    ev = Evidence(screen, confirmation, manifest)
    output.mkdir(parents=True, exist_ok=False)
    (output / "tables").mkdir()
    write_csv(output / "tables/shapes.csv", ev.shapes.values())
    write_csv(output / "tables/case_results.csv", ev.results)
    write_csv(output / "tables/samples.csv", [r for rows in ev.rows.values() for r in rows if r.get("event") == "sample"])
    write_csv(output / "tables/compilations.csv", ev.compilation)
    write_csv(output / "tables/failures.csv", ev.statuses)
    write_csv(output / "tables/comparisons.csv", ev.comparisons)
    write_csv(output / "tables/headline_results.csv", ev.headline.values())
    book = Book(ev, output)
    try:
        frontmatter_pages(book)
        overview(book)
        forest_pages(book)
        family_curves(book)
        diagnostics(book)
        nearset_pages(book)
        native_tuning_pages(book)
        failure_pages(book)
        detailed_shape_pages(book)
        slice_atlas(book)
    finally:
        book.close()
    interactive_result = interactive(book)
    write_json(output / "figures.json", book.figures)
    gallery(book, interactive_result)
    audit = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
             "generator_sha256": sha256(__file__), "input_sources": ev.sources, "warnings": ev.warnings,
             "validation": ev.audit, "figure_count": len(book.figures), "slice_page_count": 26,
             "slice_count": 102,
             "case_result_counts": dict(Counter(r["_cohort"] for r in ev.results)),
             "raw_sample_count": sum(len(v) for v in ev.samples.values()),
             "confirmed_comparison_count": len(ev.comparisons), "interactive": interactive_result,
             "scope": "Exploratory cohort only; frozen screen choices. No prediction or heldout fitting performed.",
             "confidence_scope": CI_NOTE, "public_shape_order": ["m", "n", "k"],
             "kernel_shape_order": ["m", "k", "n"], "output_files": {}}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            audit["output_files"][str(path.relative_to(output))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    write_json(output / "audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.screen, args.confirmation, args.manifest, args.output_dir)
    print(json.dumps({k: result[k] for k in ("figure_count", "slice_page_count", "confirmed_comparison_count")}))


if __name__ == "__main__":
    main()
