"""Version 002: order N1-N9 and retain readable precision near equal performance.

Every plotted point retains its original journal references. Sources are checked
against both canonical and outer seals. All case-result rows must be represented.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import textwrap

RUNS = {
    "N1": "20260919T050233Z-N1-v5e-v001-93e39b",
    "N2": "20260919T050718Z-N2-v5e-v002-226e83",
    "N3": "20260919T051615Z-N3-v5e-v002-bc2214",
    "N4": "20260919T051854Z-N4-v5e-v002-3ef5fb",
    "N4-real": "20260919T052151Z-N4-real-weights-v5e-v003-cb86b9",
    "N5-screen": "20260919T055706Z-N5-screen-v5e-v004-25842c",
    "N5": "20260919T063644Z-N5-confirm-v5e-v004-155d9f",
    "N6-failed": "20260919T063959Z-N6-v5e-v004-d93a61",
    "N6": "20260919T064743Z-N6-v5e-v004-e7bc5e",
    "N6-supplement": "20260919T073356Z-N6-v5e-v004-dde5c0",
    "N7-screen": "20260919T065405Z-N7-screen-v5e-v004-6ef489",
    "N7": "20260919T073017Z-N7-evaluate-v5e-v004-8036b5",
    "N7-replica": "20260919T084449Z-N7-replicate-v5e-v004-955768",
    "N8": "20260919T073511Z-N8-v5e-v004-0a55e8",
    "N9": "20260919T081646Z-N9-v5e-v004-0fadc6",
}
EXPECTED = dict(zip(RUNS, [352,432,96,480,48,1312,96,18,18,12,1312,128,128,312,7]))
BLUE, ORANGE, GREEN, PURPLE, GRAY = "#3772C8", "#DD8734", "#24936D", "#8553B3", "#778492"
SCOPES = {"call": "Complete call", "prepared_kernel": "Prepared kernel",
          "accuracy": "Numerical accuracy", "device": "Profiled TPU module",
          "resident": "Resident layer", "streamed": "Streamed model",
          "quality": "Model quality", "preparation": "Separate preparation"}
ARM_LABELS = {"native_xla":"Native XLA", "native_joint_graph":"Native XLA",
    "native":"Native", "cubic_basic":"Cubic (full tile)",
    "cubic_quadrant":"Cubic (quadrant control)", "strassen_basic":"Strassen",
    "cubic_selected":"Cubic (selected)", "strassen_selected":"Strassen (selected)",
    "selector":"Frozen rule", "cubic_n8_selected":"Cubic MLP policy",
    "strassen_n8_selected":"Strassen MLP policy"}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def unique(items):
    return list(dict.fromkeys(items))

def color(name):
    if name == "selector": return PURPLE
    if name == "strassen_early": return "#125E47"
    if name.startswith("strassen_n5_"): return "#7EB9A4"
    if name.startswith("cubic_n5_"): return "#EAB987"
    if "strassen" in name: return GREEN
    if "cubic" in name: return ORANGE if "quadrant" not in name else "#AD641E"
    if "native" in name: return BLUE
    return GRAY

def short_arm(name):
    if name in ARM_LABELS: return ARM_LABELS[name]
    return (name.replace("cubic_full", "C-full").replace("cubic_quadrant", "C-quad")
            .replace("strassen", "Strassen").replace("unpacked_unfused", "unpacked")
            .replace("packed_unfused", "packed").replace("_", " "))

def dim(row):
    return row.get("shape_mkn") or (row.get("kernel_metadata") or {}).get("shape_mkn")

def tile(row):
    return row.get("tile") or (row.get("kernel_metadata") or {}).get("tile_bm_bn_bk")

def shape_label(key, shape):
    name = (key.replace("held_", "").replace("__frozen_rule", "")
            .replace("qwen32", "Qwen32").replace("qwen06", "Qwen0.6")
            .replace("mistral7b", "Mistral7").replace("mistral7", "Mistral7")
            .replace("gemma9", "Gemma9").replace("gemma_", "Gemma_")
            .replace("gate_up", "gate/up").replace("_", " "))
    name = textwrap.shorten(name, width=26, placeholder="...")
    return (name + "\n" + " x ".join(str(v) for v in shape)) if shape else name

class Evidence:
    def __init__(self, root):
        self.root, self.sources, self.rows, self.summaries = root, {}, {}, {}
        self.used = defaultdict(set)
        self.all_results = {}
        for tag, rid in RUNS.items():
            base = root / "runs" / rid
            completion_path = base / "completion.json"
            outer = json.loads((base / "artifact-manifest.json").read_text())
            assert outer["completion.json"] == {"bytes": completion_path.stat().st_size, "sha256": digest(completion_path)}, tag
            completion = json.loads(completion_path.read_text())
            assert completion["status"] == "completed", (tag, completion["status"])
            self.sources[tag] = {"run_id": rid, "files": {}, "completion_status": completion["status"]}
            rows = [json.loads(line) for line in self.read(tag, "results.jsonl").splitlines()]
            for row in rows: row["_tag"], row["_ref"] = tag, rid + ":" + str(row["sequence"])
            self.rows[tag] = rows
            self.all_results[tag] = [r for r in rows if r.get("event") == "case_result"]
            assert len(self.all_results[tag]) == EXPECTED[tag], tag
            self.summaries[tag] = self.load(tag, "summary.json")
            environment = self.load(tag, "environment.json")
            self.sources[tag]["identity"] = environment.get("identity", {})
            self.sources[tag]["case_status_counts"] = dict(Counter(r["status"] for r in self.all_results[tag]))

    def read(self, tag, rel):
        rid = RUNS[tag]
        base = self.root / "runs" / rid
        path = base / "artifacts" / rel
        outer = json.loads((base / "artifact-manifest.json").read_text())
        seal_path = base / "artifacts/artifact_manifest.json"
        assert outer["artifacts/artifact_manifest.json"] == {"bytes": seal_path.stat().st_size, "sha256": digest(seal_path)}, tag
        seal = json.loads(seal_path.read_text())
        actual = digest(path)
        assert seal["sha256"][rel] == actual, (tag, rel, "canonical")
        assert outer["artifacts/" + rel] == {"bytes": path.stat().st_size, "sha256": actual}, (tag, rel, "outer")
        self.sources[tag]["files"][rel] = {"sha256": actual, "bytes": path.stat().st_size}
        return path.read_text()

    def load(self, tag, rel):
        return json.loads(self.read(tag, rel))

    def use(self, rows):
        for r in rows: self.used[r["_tag"]].add(r["_ref"])
        return [r["_ref"] for r in rows]

    def coverage(self):
        result = {}
        for tag, rows in self.all_results.items():
            expected = {r["_ref"] for r in rows}
            missing = sorted(expected - self.used[tag])
            result[tag] = {"case_results":len(rows), "represented":len(expected & self.used[tag]),
                           "missing":missing, "status_counts":self.sources[tag]["case_status_counts"]}
            assert not missing, (tag, missing[:10])
        return result

class Book:
    def __init__(self, ev):
        self.ev, self.charts = ev, []

    def add(self, **kw):
        kw.setdefault("detail", False)
        kw.setdefault("takeaway", "")
        kw.setdefault("scope", "call")
        tags = kw.pop("tags")
        kw["source_run_ids"] = [RUNS[x] for x in tags]
        kw["cohort"] = ("Original N1-N4 allocation" if all(t.startswith(("N1","N2","N3","N4")) for t in tags)
                        else "Fresh v5e replication" if tags == ["N7-replica"]
                        else "Separate within-cohort comparisons" if "N7-replica" in tags
                        else "N5-N9 allocation")
        assert kw["id"] not in [x["id"] for x in self.charts]
        self.charts.append(kw)

    def point(self, row, category=None, series=None):
        timing = row.get("timing") or {}
        value = timing.get("mean_ms", row.get("mean_ms"))
        sd = timing.get("std_ms", row.get("std_ms", 0)) or 0
        return {"category": category or row.get("shape_id") or row["group_id"],
                "series":series or row["arm_id"], "value":value,
                "lo":max(0, value-sd) if value is not None else None,
                "hi":value+sd if value is not None else None,
                "status":row["status"], "row_refs":self.ev.use([row]),
                "comparisons":row.get("comparisons", []),
                "extra":{"arm_id":row.get("_original_arm_id",row["arm_id"]), "algorithm":row.get("algorithm"),
                         "variant":row.get("variant"), "tile_bm_bn_bk":tile(row),
                         "shape_mkn":dim(row), "sample_count":timing.get("sample_count"),
                         "seed":row.get("seed"), "distribution":row.get("distribution"),
                         "numerical_pass":(row.get("correctness") or {}).get("pass")}}

    def timing(self, ident, experiment, title, rows, tags, *, detail=False,
               baseline=None, takeaway="", subtitle="", category="shape_id", max_groups=10,
               series_key=None, label_overrides=None):
        cats = unique(r.get(category) or r["group_id"] for r in rows)
        arms = unique((series_key(r) if series_key else r["arm_id"]) for r in rows)
        # Keep the familiar algorithm order, with selector last.
        rank = lambda a: (0 if "native" in a else 1 if "cubic" in a else 2 if "strassen" in a else 3, arms.index(a))
        arms = sorted(arms, key=rank)
        scope = rows[0]["scope"]
        for page, start in enumerate(range(0, len(cats), max_groups), 1):
            subset = cats[start:start+max_groups]
            rr = [r for r in rows if (r.get(category) or r["group_id"]) in subset]
            labels = {cat:shape_label(cat, dim(next(r for r in rr if (r.get(category) or r["group_id"])==cat))) for cat in subset}
            points = [self.point(r, r.get(category) or r["group_id"], series_key(r) if series_key else None) for r in rr]
            assert len({(p["category"],p["series"]) for p in points}) == len(points), ident
            suffix = f" - shapes {start+1}-{start+len(subset)}" if len(cats)>max_groups else ""
            self.add(id=ident+f"-p{page}", experiment=experiment, kind="bar", title=title+suffix,
                     subtitle=subtitle, scope=scope, tags=tags, detail=detail, takeaway=takeaway,
                     categories=[{"id":c,"label":labels[c]} for c in subset],
                     series=[{"id":a,"label":(label_overrides or {}).get(a,short_arm(a)),
                              "color":([GRAY,BLUE,ORANGE,PURPLE][i] if label_overrides else color(a))} for i,a in enumerate(arms)],
                     points=points, ylabel="Mean time (ms) - lower is better", baseline=baseline,
                     error_note="Absolute whiskers: +/- 1 sample SD (spread, not a confidence interval).",
                     relative_note="Relative whiskers: archived paired 95% intervals where available; unadjusted for multiple comparisons.")

    def heat(self, ident, experiment, title, rows, tags, *, ykey, xkey, metric, ylabel,
             subtitle="", detail=True, center_one=False, higher_better=False, takeaway=""):
        xs, ys = unique(xkey(r) for r in rows), unique(ykey(r) for r in rows)
        points = []
        for row in rows:
            value = metric(row)
            points.append({"category":xkey(row),"series":ykey(row),"value":value,
                           "status":row.get("status","ok"),"lo":row.get("_lo"),"hi":row.get("_hi"),
                           "row_refs":self.ev.use([row]),"extra":{"arm_id":row.get("arm_id"),
                           "tile_bm_bn_bk":tile(row),"variant":row.get("variant")}})
        assert len({(p["category"],p["series"]) for p in points})==len(points), ident
        self.add(id=ident, experiment=experiment, kind="heat", title=title, subtitle=subtitle,
                 scope=rows[0].get("scope","call"), tags=tags, detail=detail,
                 categories=[{"id":x,"label":shape_label(x,dim(next(r for r in rows if xkey(r)==x)))} for x in xs],
                 series=[{"id":y,"label":y} for y in ys], points=points, ylabel=ylabel,
                 center_one=center_one, higher_better=higher_better, takeaway=takeaway)

def build(ev):
    book = Book(ev)
    for scope in ("call","prepared_kernel"):
        rows = [r for r in ev.all_results["N1"] if r["scope"]==scope]
        book.timing("n1-"+scope, "N1", "N1 | Fixed-tile baselines", rows, ["N1"],
                    baseline="native_xla", detail=scope!="call",
                    subtitle="All custom kernels use tile (BM,BN,BK)=(256,256,256). All 44 shapes are included.",
                    takeaway="Baseline choice matters: full-tile cubic and native are stronger than the matched quadrant control.")
        n2 = [r for r in ev.all_results["N2"] if r["scope"]==scope]
        book.heat("n2-sweep-"+scope,"N2","N2 | Every tile and algorithm",n2,["N2"],
                  ykey=lambda r:short_arm(r["arm_id"])+" | "+"/".join(map(str,tile(r))),
                  xkey=lambda r:r["shape_id"],metric=lambda r:r["timing"].get("mean_ms"),
                  ylabel="Mean time (ms; log color scale)",detail=False,
                  subtitle="Exploratory screen: 7 timings per passing cell. Tile order is BM/BN/BK. Native was not measured.",
                  takeaway="Tile choice changes performance. OOM cells are VMEM-limit failures, not zero-time results.")
        for t in unique(tuple(tile(r)) for r in n2):
            rows=[r for r in n2 if tuple(tile(r))==t]
            book.timing("n2-"+scope+"-"+"-".join(map(str,t)),"N2","N2 | Tile "+"/".join(map(str,t)),rows,["N2"],detail=True,
                        subtitle="Screening only; no native measurements in N2. Failed configurations remain visible.")
        n3=[r for r in ev.all_results["N3"] if r["scope"]==scope]
        variants=["plain","interleaved","output_accumulator","interleaved_output_accumulator"]
        for alg in ("cubic_quadrant","strassen"):
            rows=[r for r in n3 if r["algorithm"]==alg]
            book.timing("n3-"+alg+"-"+scope,"N3","N3 | "+short_arm(alg)+" optimization ablation",rows,["N3"],
                        baseline=alg+"__plain",subtitle="Fixed tile 1024/1024/512. Native and full-tile cubic were not measured.",
                        label_overrides={alg+"__"+v:{"plain":"Plain","interleaved":"Product order","output_accumulator":"Output accumulator","interleaved_output_accumulator":"Both changes"}[v] for v in variants},
                        takeaway="Compare each change with its own plain implementation; product ordering does not establish hardware overlap.")
        for v in variants:
            book.timing("n3-paired-"+v+"-"+scope,"N3","N3 | Matched "+v.replace("_"," "),
                        [r for r in n3 if r["variant"]==v],["N3"],detail=True,
                        subtitle="Two algorithms at the same tile and variant. All recorded cases are retained.")

    # Accuracy plots keep every seed, including failed Strassen outputs.
    for tag in ("N4","N4-real"):
        for distribution in unique(r["distribution"] for r in ev.all_results[tag]):
            rows=[r for r in ev.all_results[tag] if r["distribution"]==distribution]
            cats=unique(r["shape_id"] for r in rows)
            arms=unique(r["arm_id"] for r in rows)
            for metric in ("relative_l2","max_error_gate_ratio"):
                points=[]
                for cat in cats:
                    for arm in arms:
                        group=[r for r in rows if r["shape_id"]==cat and r["arm_id"]==arm]
                        vals=[r["correctness"]["relative_l2"] if metric=="relative_l2" else
                              r["correctness"]["max_abs_error"]/(0.001+0.05*r["correctness"]["max_abs_reference"]) for r in group]
                        points.append({"category":cat,"series":arm,"value":statistics.median(vals),
                            "lo":min(vals),"hi":max(vals),"raw_points":vals,
                            "status":"numerical_failure" if any(r["status"]!="ok" for r in group) else "ok",
                            "row_refs":ev.use(group),"extra":{"seeds":[r["seed"] for r in group],"values":vals}})
                book.add(id=tag.lower()+"-"+distribution+"-"+metric,experiment="N4",kind="bar",
                    title=("N4 | Actual Qwen weights" if tag=="N4-real" else "N4 | "+distribution.replace("_"," "))+
                          (" - relative error" if metric=="relative_l2" else " - maximum-error gate"),
                    subtitle="Accuracy only: median of 3 seeds; whiskers and dots show all seed values. No latency was measured.",
                    scope="accuracy",tags=[tag],detail=metric!="relative_l2",
                    categories=[{"id":c,"label":shape_label(c,dim(next(r for r in rows if r["shape_id"]==c)))} for c in cats],
                    series=[{"id":a,"label":short_arm(a),"color":color(a)} for a in arms],points=points,
                    ylabel="Relative L2 error (log scale)" if metric=="relative_l2" else "Maximum error / allowed maximum (log scale)",
                    log=True,hline=0.02 if metric=="relative_l2" else 1,
                    error_note="Seed range is descriptive, not a confidence interval. Hatched bars failed the unchanged numerical gate.",
                    takeaway="Actual weights use synthetic Gaussian activations; this is not a model-quality test." if tag=="N4-real" else
                        "Every cancellation Strassen case failed; successful cases must not hide this input sensitivity." if distribution=="cancellation" else
                        "All four implementations passed this distribution under the fixed gate.")

    # Exhaustive screen heatmaps, plus frozen selected-candidate screen summaries.
    for tag, exp in (("N5-screen","N5"),("N7-screen","N7")):
        choices=ev.load(tag,"selections.json")["by_shape"]
        allrows=ev.all_results[tag]
        for scope in ("call","prepared_kernel"):
            rr=[r for r in allrows if r["scope"]==scope]
            native={r["shape_id"]:r for r in rr if r["algorithm"]=="native"}
            selected=[]
            for shape, choices_for_shape in choices.items():
                selected.append(dict(native[shape],arm_id="native_xla"))
                for family in ("cubic","strassen"):
                    chosen=choices_for_shape[family]
                    if chosen is None: continue
                    target=next(r for r in rr if r["shape_id"]==shape and r.get("candidate_id")==chosen["candidate_id"])
                    selected.append(dict(target,arm_id=family+"_selected",_original_arm_id=target["arm_id"]))
            book.timing(tag+"-selected-"+scope,exp,exp+" | Screen selections (exploratory)",selected,[tag],detail=True,
                        baseline="native_xla",subtitle="Selected on these same 7-repeat measurements; use the separate fresh evaluation for claims.")
            for family in ("cubic_full","strassen"):
                rows=[r for r in rr if r["algorithm"]==family]
                for r in rows: ev.use([native[r["shape_id"]]])
                book.heat(tag+"-"+family+"-"+scope,exp,exp+" | All "+("cubic" if family=="cubic_full" else "Strassen")+" candidates",
                    rows,[tag],ykey=lambda r:r["variant"].replace("interleaved_output_accumulator","both").replace("output_accumulator","output_acc")+
                    " | "+"/".join(map(str,tile(r))),xkey=lambda r:r["shape_id"],
                    metric=lambda r:(r["timing"]["mean_ms"]/native[r["shape_id"]]["timing"]["mean_ms"]) if r["status"]=="ok" else None,
                    ylabel="Time / native in this screen (1 = equal; lower is better)",center_one=True,
                    subtitle="20 candidates per family and shape; all failed attempts count. Numbers are descriptive screening means.",
                    takeaway="This displays the whole bounded search, including VMEM failures; it is not independent confirmation.")
    for tag,exp in (("N5","N5"),("N7","N7"),("N7-replica","N7")):
        for scope in ("call","prepared_kernel"):
            rows=[r for r in ev.all_results[tag] if r["scope"]==scope]
            book.timing(tag.lower()+"-"+scope,exp,
                        ("N5 | Fresh confirmation" if tag=="N5" else "N7 | Frozen-rule evaluation" if tag=="N7" else "N7 | Fresh-v5e replication"),
                        rows,[tag],baseline="native_xla",
                        subtitle="30 paired rounds; independent family selections. Matrix dimensions are M x K x N.",
                        takeaway="Both algorithm families were tuned independently before these fresh confirmation measurements." if tag=="N5" else
                            "The rule is unchanged. Some rule bars duplicate an alternative; timing differences there are not algorithmic improvements.")
    # Present within-cohort selector speedups side by side, never raw timing pools.
    for scope in ("call","prepared_kernel"):
        points=[]
        cats=[]
        for tag in ("N7","N7-replica"):
            evaluation=ev.load(tag,"selector_evaluation.json")
            for case in evaluation["cases"]:
                if case["scope"]!=scope: continue
                cat=case["shape_id"]
                if cat not in cats: cats.append(cat)
                pair=case["paired_selector_speedups"]["native_xla"]
                row=next(r for r in ev.all_results[tag] if r["shape_id"]==cat and r["scope"]==scope and r["arm_id"]=="selector")
                points.append({"category":cat,"series":tag,"value":pair["speedup_ratio_of_means"],
                    "lo":pair["speedup_ci95"][0],"hi":pair["speedup_ci95"][1],"status":"ok","row_refs":ev.use([row]),
                    "extra":{"route":case["decision"]["label"],"shape_mkn":case["shape_mkn"]}})
        for page,start in enumerate(range(0,len(cats),8),1):
            subset=cats[start:start+8]
            book.add(id="n7-repeat-speedup-"+scope+f"-p{page}",experiment="N7",kind="bar",tags=["N7","N7-replica"],
                title="N7 | Does the frozen rule repeat its gain?",subtitle="Each bar is native time / rule time within its own allocation. No timing samples are pooled.",
                scope=scope,detail=True,categories=[{"id":c,"label":shape_label(c,next(p["extra"]["shape_mkn"] for p in points if p["category"]==c))} for c in subset],
                series=[{"id":"N7","label":"Original allocation","color":BLUE},{"id":"N7-replica","label":"Fresh allocation","color":PURPLE}],
                points=[p for p in points if p["category"] in subset],ylabel="Speedup vs native - higher is better",hline=1,
                error_note="Whiskers: archived paired 95% bootstrap intervals; per-comparison, not multiplicity-adjusted.",
                takeaway="Three custom routes win in both cohorts. Thirteen other routes use native; small duplicate-route differences reflect timing variability.")

    for tag in ("N6-failed","N6","N6-supplement"):
        for block in ("before","after"):
            rr=[r for r in ev.all_results[tag] if r["group_id"].endswith("__"+block)]
            book.timing(tag.lower()+"-"+block,"N6",
                ("N6 | Initial attempt" if tag=="N6-failed" else "N6 | Large-shape supplement" if tag=="N6-supplement" else "N6 | Registered representatives")+" - "+block+" profiling",
                rr,[tag],baseline="native_xla",detail=tag=="N6-failed",
                subtitle="Ordinary synchronized host-observed latency; profiled device duration is a separate chart.",
                takeaway="The initial attempt produced no valid TPU traces; these are ordinary timing measurements only." if tag=="N6-failed" else
                    "Supplement shapes were selected post hoc from N5 wins; preserve the original representatives and their reversals." if tag=="N6-supplement" else
                    "Host-observed and profiled-device rankings can differ; these measurements do not identify the cause.")
        if tag=="N6-failed": continue
        profiles=[r for r in ev.rows[tag] if r.get("event")=="profile_capture"]
        cats=unique(r["shape_id"] for r in profiles)
        arms=unique(r["arm_id"] for r in profiles)
        points=[]
        for cat in cats:
            for arm in arms:
                rr=[r for r in profiles if r["shape_id"]==cat and r["arm_id"]==arm]
                assert len(rr)==2 and all(r["status"]=="ok" for r in rr)
                vals=[r["device_analysis"]["module_mean_ms"] for r in rr]
                points.append({"category":cat,"series":arm,"value":statistics.mean(vals),"lo":min(vals),"hi":max(vals),
                               "status":"ok","raw_points":vals,"row_refs":ev.use(rr),
                               "extra":{"capture_means_ms":vals,"module_calls_per_capture":8,"arm_id":arm}})
        book.add(id=tag.lower()+"-device",experiment="N6",kind="bar",tags=[tag],scope="device",
            title="N6 | Profiled TPU module"+(" - large-shape supplement" if tag.endswith("supplement") else ""),
            subtitle="Mean of two 8-call captures; whiskers/dots show capture means. Profiler-instrumented, not ordinary call latency.",
            categories=[{"id":c,"label":shape_label(c,next(r["shape_mkn"] for r in profiles if r["shape_id"]==c))} for c in cats],
            series=[{"id":a,"label":short_arm(a),"color":color(a)} for a in arms],points=points,baseline="native_xla",
            ylabel="Device module time (ms) - lower is better",error_note="Capture-mean range is descriptive; no utilization, overlap or HBM-traffic claim.",
            takeaway="The large cases show device gains for Strassen." if tag.endswith("supplement") else
                     "Native has the shortest profiled module for these three representatives, despite some host-call ranking reversals.")

    n8=ev.all_results["N8"]
    planned={g["id"]:g for g in ev.load("N8","planned_cases.json")}
    choices=ev.load("N8","application_selections.json")["by_group"]
    for scope in ("call","prepared_kernel"):
        for kind in ("swiglu","residual"):
            rr=[r for r in n8 if r["scope"]==scope and planned[r["group_id"]]["kind"]==kind]
            selected=[]
            for group in unique(r["group_id"] for r in rr):
                for family in ("native","cubic","strassen"):
                    selected_arm=choices[group][family]["arm_id"]
                    row=next(r for r in rr if r["group_id"]==group and r["arm_id"]==selected_arm)
                    # Keep original comparison IDs in extra metadata when aliasing.
                    selected.append(dict(row,arm_id=family+"_selected" if family!="native" else "native_joint_graph",_original_arm_id=row["arm_id"]))
            book.timing("n8-selected-"+kind+"-"+scope,"N8","N8 | "+kind+" - selected family implementations",
                        selected,["N8"],baseline="native_joint_graph",category="group_id",
                        subtitle="N8-selected on these same measurements; not fresh confirmation. Prepared plots retain complete-call selections.",
                        takeaway="Selection includes layout/fusion and sometimes the quadrant-cubic control. These are synthetic inputs.")
            for stage in ("unpacked_unfused","packed_unfused","fused"):
                arms=["native_joint_graph","cubic_full_"+stage,"cubic_quadrant_"+stage,"strassen_"+stage]
                staged=[r for r in rr if r["arm_id"] in arms]
                if not any(r["algorithm"]!="native" for r in staged): continue
                book.timing("n8-"+kind+"-"+stage+"-"+scope,"N8","N8 | "+kind+" - "+stage.replace("_"," "),
                    staged,["N8"],baseline="native_joint_graph",category="group_id",detail=True,
                    subtitle="Matched custom tile 1024/1024/512. Native is a joint graph; each recorded layout/fusion path is distinct.")
            book.timing("n8-"+kind+"-early-"+scope,"N8","N8 | "+kind+" - early finalization",
                        [r for r in rr if r["arm_id"] in ("native_joint_graph","strassen_fused","strassen_early")],
                        ["N8"],baseline="native_joint_graph",category="group_id",detail=True,
                        subtitle="Early finalization compared with standard fused Strassen and native.",
                        takeaway="Early finalization frequently regresses; retained failures to improve are part of the result.")
            anchors=[r for r in rr if "_m512_swiglu" in r["group_id"] and r["arm_id"] in
                     ("native_joint_graph","cubic_full_unpacked_unfused","strassen_unpacked_unfused",
                      "cubic_n5_selected_unfused","strassen_n5_selected_unfused")]
            if anchors:
                book.timing("n8-anchors-"+scope,"N8","N8 | Stronger N5-selected anchors",anchors,["N8"],
                    baseline="native_joint_graph",category="group_id",detail=True,
                    subtitle="Fixed versus N5-selected implementations; tile and variant both change. Not an isolated tile effect.")
        # Every registered paired contrast is retained, not only favorable ones.
        contrasts=[]
        for row in n8:
            if row["scope"]!=scope: continue
            for c in row["comparisons"]:
                contrasts.append(dict(row,_contrast=short_arm(row["arm_id"])+" vs "+short_arm(c["reference_arm"]),
                    _ratio=c["speedup_ratio_of_means"],_lo=c["speedup_ci95"][0],_hi=c["speedup_ci95"][1]))
        book.heat("n8-all-contrasts-"+scope,"N8","N8 | All registered paired comparisons",contrasts,["N8"],
            ykey=lambda r:r["_contrast"],xkey=lambda r:r["group_id"],metric=lambda r:r["_ratio"],
            ylabel="Reference time / target time (>1 favors target)",center_one=True,higher_better=True,
            subtitle="Each row reads target vs reference. + or - marks a paired 95% interval wholly above or below 1. Blank = not registered.")
        book.heat("n8-accuracy-"+scope,"N8","N8 | Numerical error of every application arm",
            [r for r in n8 if r["scope"]==scope],["N8"],ykey=lambda r:short_arm(r["arm_id"]),
            xkey=lambda r:r["group_id"],metric=lambda r:r["correctness"]["relative_l2"],
            ylabel="Relative L2 error (log color scale)",subtitle="All application rows pass the unchanged 0.02 relative-L2 and maximum-error gates.")
    prep=[r for r in n8 if r["scope"]=="call"]
    prep=[dict(r,scope="preparation") for r in prep]
    book.heat("n8-preparation","N8","N8 | Separately measured preparation",prep,["N8"],
        ykey=lambda r:short_arm(r["arm_id"]),xkey=lambda r:r["group_id"],
        metric=lambda r:statistics.mean(r["preparation_raw_ms"]),
        ylabel="Preparation mean (ms; log color scale)",
        subtitle="Five samples per arm/group, deduplicated across scopes. Do not add or subtract this independent measurement from full-call timings.")

    models=[m for m in ev.summaries["N9"]["results"] if m["status"]=="completed"]
    arms=["native","cubic_n8_selected","strassen_n8_selected"]
    for scope,field,factor in (("resident","resident_layer_mean_ms",1),("streamed","streamed_forward_mean_ms",0.001)):
        for m in models:
            mid=m["model_id"]; model_short=mid.split("/")[-1]
            points=[]
            for arm in arms:
                case=next(r for r in ev.all_results["N9"] if r.get("model_id")==mid and r.get("arm_id")==arm)
                raw=[v*factor for v in m["streamed_forward_samples_ms"][arm]] if scope=="streamed" else []
                points.append({"category":mid,"series":arm,"value":m[field][arm]*factor,
                    "lo":min(raw) if raw else None,"hi":max(raw) if raw else None,
                    "raw_points":raw,"status":"ok","row_refs":ev.use([case]),"extra":{"arm_id":arm,"model_id":mid}})
            book.add(id="n9-"+scope+"-"+model_short.lower(),experiment="N9",kind="bar",tags=["N9"],scope=scope,
                title="N9 | "+model_short+" - "+("resident layer" if scope=="resident" else "streamed full forward"),
                subtitle="Complete transformer layer; weights resident. Mean across 28 or 32 different layers; layers are not independent replicates."
                    if scope=="resident" else "1,024-token full forward including reads, host layout and transfers. All three measured repeats are shown.",
                categories=[{"id":mid,"label":model_short}],series=[{"id":a,"label":short_arm(a),"color":color(a)} for a in arms],
                points=points,baseline="native",ylabel=("Layer time (ms)" if scope=="resident" else "Full-forward time (seconds)")+" - lower is better",
                error_note="No pooled layer confidence interval is inferred." if scope=="resident" else "Whiskers: minimum to maximum of 3 repeats; descriptive, not a confidence interval.",
                takeaway="Custom policies replace the MLP only. Attention and vocabulary projection remain native.")
            if scope=="resident":
                samples=[r for r in ev.rows["N9"] if r.get("event")=="resident_layer_sample" and r["model_id"]==mid]
                layers=sorted(set(r["layer"] for r in samples))
                rows=[]
                for layer in layers:
                    for arm in arms:
                        rr=[r for r in samples if r["layer"]==layer and r["arm_id"]==arm]
                        assert len(rr)==7
                        vals=[r["elapsed_ms"] for r in rr]
                        # Synthetic display row retains all actual raw sample refs below.
                        mean,sd=statistics.mean(vals),statistics.stdev(vals)
                        rows.append({"category":str(layer),"series":arm,"value":mean,"lo":max(0,mean-sd),"hi":mean+sd,
                            "status":"ok","row_refs":ev.use(rr),"extra":{"arm_id":arm,"samples":vals,"layer":layer}})
                for page,start in enumerate(range(0,len(layers),14),1):
                    subset=layers[start:start+14]
                    book.add(id="n9-layers-"+model_short.lower()+f"-p{page}",experiment="N9",kind="bar",tags=["N9"],scope="resident",detail=True,
                        title="N9 | "+model_short+" - layers "+str(subset[0])+" to "+str(subset[-1]),
                        subtitle="Seven paired rounds per layer; all policies receive the same native incoming first-window state.",
                        categories=[{"id":str(c),"label":"Layer "+str(c)} for c in subset],
                        series=[{"id":a,"label":short_arm(a),"color":color(a)} for a in arms],
                        points=[p for p in rows if int(p["category"]) in subset],baseline="native",ylabel="Layer time (ms) - lower is better",
                        error_note="Whiskers: +/- 1 sample SD within this layer; no across-layer significance claim.")
    for metric,label,threshold in (("nll_delta","NLL difference vs native (nats/token)",0.01),
                                   ("mean_kl","Mean KL vs native",0.02),
                                   ("top1_disagreement","Top-1 disagreement (%)",3)):
        points=[]
        for m in models:
            for arm in arms:
                q=m["quality"][arm]
                value=100*(1-q["top1_agreement"]) if metric=="top1_disagreement" else q[metric]
                case=next(r for r in ev.all_results["N9"] if r.get("model_id")==m["model_id"] and r.get("arm_id")==arm)
                points.append({"category":m["model_id"],"series":arm,"value":value,"status":"ok","row_refs":ev.use([case])})
        book.add(id="n9-quality-"+metric,experiment="N9",kind="bar",tags=["N9"],scope="quality",
            title="N9 | Corpus quality - "+metric.replace("_"," "),
            subtitle="32,736 scored positions per model and policy. Native zeros are self-comparisons; independent qualification is separate.",
            categories=[{"id":m["model_id"],"label":m["model_id"].split("/")[-1]} for m in models],
            series=[{"id":a,"label":short_arm(a),"color":color(a)} for a in arms],points=points,ylabel=label,
            hline=threshold,negative_hline=metric=="nll_delta",error_note="Dotted line: predeclared aggregate gate. No statistical interval is inferred.",
            takeaway="Both custom policies pass these fixed corpus gates; this does not establish arbitrary-input or downstream-task accuracy.")
    qpoints=[]; qcats=[]
    for m in models:
        q=m["qualification"]
        qualification_rows=[r for r in ev.rows["N9"] if r.get("event")=="native_qualification" and r["model_id"]==m["model_id"]]
        assert len(qualification_rows)==1
        for label,val,limit in [("Logit relative L2",q["relative_l2"],0.03),("Absolute NLL difference",abs(q["nll_delta"]),0.05),
                               ("Mean KL",q["mean_kl"],0.01),("Top-1 disagreement",1-q["top1_agreement"],0.10)]:
            cat=m["model_id"]+"|"+label
            qcats.append({"id":cat,"label":m["model_id"].split("/")[-1]+"\n"+label})
            qpoints.append({"category":cat,"series":"native","value":val/limit,"status":"ok","row_refs":ev.use(qualification_rows),
                            "extra":{"observed":val,"gate":limit,"scored_positions":63}})
    book.add(id="n9-native-qualification",experiment="N9",kind="bar",tags=["N9"],scope="quality",
        title="N9 | Independent native implementation qualification",
        subtitle="Native JAX vs official Transformers CPU BF16 on 64 input tokens / 63 scored positions.",
        categories=qcats,series=[{"id":"native","label":"Native JAX / allowed error","color":BLUE}],points=qpoints,
        ylabel="Fraction of permitted error (lower is better)",hline=1,
        error_note="The limits differ from the full-corpus custom-policy gates. Finite-output checks also passed.",
        takeaway="Both native implementations pass this short independent check; it is not full-corpus exact equivalence.")
    for model in models:
        mid=model["model_id"]
        windows=[r for r in ev.rows["N9"] if r.get("event")=="quality_window" and r["model_id"]==mid]
        for metric in ("nll_delta","mean_kl","top1_disagreement"):
            points=[]
            for r in windows:
                for arm in arms:
                    q=r["metrics"][arm]
                    value=100*(1-q["top1_agreement"]) if metric=="top1_disagreement" else q[metric]
                    points.append({"category":str(r["window"]),"series":arm,"value":value,"status":"ok","row_refs":ev.use([r])})
            book.add(id="n9-windows-"+mid.split("/")[-1].lower()+"-"+metric,experiment="N9",kind="line",tags=["N9"],scope="quality",detail=True,
                title="N9 | "+mid.split("/")[-1]+" - "+metric.replace("_"," ")+" by window",
                subtitle="32 contiguous 1,024-token windows, 1,023 scored positions each; context resets between windows.",
                categories=[{"id":str(r["window"]),"label":str(r["window"])} for r in windows],
                series=[{"id":a,"label":short_arm(a),"color":color(a)} for a in arms],points=points,
                ylabel=metric.replace("_"," ")+(" (%)" if metric=="top1_disagreement" else ""),
                error_note="Window variability is descriptive. Qualification gates apply to the aggregate, not separately to each window.")
    blocked=[r for r in ev.all_results["N9"] if r["status"]=="blocked_access"]
    ev.use(blocked)
    return book

def render(book, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm, TwoSlopeNorm
    from matplotlib.ticker import FuncFormatter
    import numpy as np
    matplotlib.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
        "axes.spines.right":False,"axes.labelcolor":"#344054","xtick.color":"#475467","ytick.color":"#475467",
        "axes.edgecolor":"#B7C0CC","savefig.facecolor":"white","pdf.fonttype":42,"svg.fonttype":"none"})
    (out/"figures").mkdir(); (out/"data").mkdir()
    metadata={"Title":"Strassen MM Focus - N1-N9 key comparisons","Author":"Strassen MM Focus",
              "Subject":"Sealed v5e experimental evidence; original scopes and cohorts preserved"}
    with PdfPages(out/"N1_N9_key_comparisons.pdf",metadata=metadata) as pdf:
        pages=0
        for index,c in enumerate(book.charts):
            c["data_file"]="data/"+c["id"]+".json"
            c["files"]={}
            modes=["absolute"] + (["relative"] if c.get("baseline") else [])
            for mode in modes:
                ncat,nseries=len(c["categories"]),len(c["series"])
                heat=c["kind"]=="heat"
                height=max(8.5,3.8+nseries*.34) if heat else 7.6
                width=16 if heat else 14
                fig,ax=plt.subplots(figsize=(width,height))
                fig.subplots_adjust(left=.28 if heat else .085,right=.965,top=.80,bottom=.22 if not heat else .27)
                fig.suptitle(textwrap.fill(c["title"],100),x=.045,y=.98,ha="left",fontsize=18,fontweight="bold",color="#162A43")
                fig.text(.045,.915,textwrap.fill(c["subtitle"],148),ha="left",va="top",fontsize=9.5,color="#475467")
                badge=SCOPES.get(c["scope"],c["scope"])+"  |  "+c["cohort"]+("  |  Detail" if c["detail"] else "")
                fig.text(.045,.845,badge,ha="left",fontsize=9,fontweight="bold",color="#344054")
                points={(p["category"],p["series"]):p for p in c["points"]}
                if heat:
                    a=np.full((nseries,ncat),np.nan)
                    for i,s in enumerate(c["series"]):
                        for j,cat in enumerate(c["categories"]):
                            p=points.get((cat["id"],s["id"]))
                            if p and p["value"] is not None and p["value"]>0: a[i,j]=p["value"]
                    assert np.isfinite(a).any(), c["id"]
                    if c.get("center_one"):
                        display=np.log2(a)
                        extent=max(.08,float(np.nanmax(abs(display))))
                        norm=TwoSlopeNorm(vmin=-extent,vcenter=0,vmax=extent)
                        cmap=plt.get_cmap("RdYlGn" if c.get("higher_better") else "RdYlGn_r").copy()
                    else:
                        display=a
                        lo,hi=float(np.nanmin(a)),float(np.nanmax(a))
                        norm=LogNorm(vmin=max(lo,1e-12),vmax=max(hi,lo*1.001))
                        cmap=plt.get_cmap("YlGnBu").copy()
                    cmap.set_bad("#EEF0F3")
                    im=ax.imshow(np.ma.masked_invalid(display),aspect="auto",cmap=cmap,norm=norm)
                    cb=fig.colorbar(im,ax=ax,pad=.015,fraction=.035)
                    if c.get("center_one"): cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{2**v:.2g}x"))
                    cb.set_label(c["ylabel"],fontsize=9)
                    ax.set_xticks(range(ncat),[x["label"] for x in c["categories"]],rotation=55,ha="right",fontsize=7.7)
                    ax.set_yticks(range(nseries),[s["label"] for s in c["series"]],fontsize=8)
                    for i,s in enumerate(c["series"]):
                        for j,cat in enumerate(c["categories"]):
                            p=points.get((cat["id"],s["id"]))
                            if p is None:
                                ax.text(j,i,"-",ha="center",va="center",fontsize=8,color="#8893A1"); continue
                            if p["value"] is None:
                                ax.text(j,i,"OOM" if p["status"]=="oom" else "FAIL",ha="center",va="center",fontsize=7,color="#B42318",fontweight="bold")
                            elif p["value"]==0:
                                ax.text(j,i,"0",ha="center",va="center",fontsize=7,color="#344054")
                            else:
                                suffix=""
                                if p.get("lo") is not None and c.get("higher_better"):
                                    suffix="+" if p["lo"]>1 else "-" if p["hi"]<1 else ""
                                rgba=cmap(norm(display[i,j])); lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                                value_label=f'{p["value"]:.3g}' if c.get("center_one") else f'{p["value"]:.2g}'
                                ax.text(j,i,value_label+suffix,ha="center",va="center",fontsize=7.2,color="white" if lum<.48 else "#162A43")
                else:
                    xs=np.arange(ncat); bw=.78/max(nseries,1); values=[]
                    for si,s in enumerate(c["series"]):
                        xpos=xs+(si-(nseries-1)/2)*bw
                        ys=[]; lows=[]; highs=[]; available=[]; failed=[]; ps=[]
                        for ci,cat in enumerate(c["categories"]):
                            p=points.get((cat["id"],s["id"])); ps.append(p)
                            value=p.get("value") if p else None
                            low=p.get("lo") if p else None; high=p.get("hi") if p else None
                            if mode=="relative" and value is not None:
                                base=points.get((cat["id"],c["baseline"]))
                                if not base or not base["value"]: value=None
                                else:
                                    value=value/base["value"]; low=high=None
                                    if s["id"]==c["baseline"]: low=high=1
                                    else:
                                        ref=(base.get("extra") or {}).get("arm_id",c["baseline"])
                                        pair=next((x for x in p.get("comparisons",[]) if x["reference_arm"]==ref and x.get("valid_numerical_comparison")),None)
                                        if pair:
                                            low,high=1/pair["speedup_ci95"][1],1/pair["speedup_ci95"][0]
                            ys.append(value if value is not None else np.nan)
                            lows.append(low); highs.append(high); available.append(value is not None)
                            failed.append(bool(p and p["status"]!="ok")); values.extend([v for v in [value,high] if v is not None])
                        if c["kind"]=="line":
                            ax.plot(xs,ys,color=s["color"],label=s["label"],marker="o",markersize=3,linewidth=1.5)
                        else:
                            bars=ax.bar(xpos,ys,width=bw*.9,color=s["color"],label=s["label"],zorder=3)
                            for i,bar in enumerate(bars):
                                p=ps[i]
                                if not available[i]:
                                    if p: ax.text(xpos[i],.045,"OOM" if p["status"]=="oom" else "N/A",transform=ax.get_xaxis_transform(),ha="center",rotation=90,fontsize=7,color="#B42318")
                                    continue
                                if failed[i]: bar.set_hatch("////"); bar.set_edgecolor("#B42318"); bar.set_linewidth(1.2)
                                if lows[i] is not None and highs[i] is not None:
                                    # Endpoints are drawn directly; bootstrap intervals need not contain the point estimate.
                                    ax.vlines(xpos[i],lows[i],highs[i],color="#20334A",linewidth=.8,zorder=5)
                                    ax.hlines([lows[i],highs[i]],xpos[i]-bw*.12,xpos[i]+bw*.12,color="#20334A",linewidth=.8,zorder=5)
                                if ys[i]==0: ax.plot(xpos[i],0,"o",color=s["color"],ms=4,zorder=6)
                                if ncat<=4 and not c.get("log"):
                                    ax.annotate(f"{ys[i]:.4g}",(xpos[i],max(ys[i],highs[i] or ys[i])),xytext=(0,6),textcoords="offset points",ha="center",fontsize=9)
                                if mode=="absolute" and p.get("raw_points"):
                                    raw=p["raw_points"]; offset=np.linspace(-bw*.16,bw*.16,len(raw))
                                    ax.scatter(xpos[i]+offset,raw,s=12,facecolor="white",edgecolor="#20334A",linewidth=.6,zorder=6)
                    threshold=1 if mode=="relative" else c.get("hline")
                    if threshold is not None:
                        ax.axhline(threshold,color="#B54739",ls="--",lw=1,zorder=2); values.append(threshold)
                        if c.get("negative_hline") and mode=="absolute": ax.axhline(-threshold,color="#B54739",ls="--",lw=1); values.append(-threshold)
                    if c.get("log") and mode=="absolute":
                        ax.set_yscale("log")
                        positive=[v for v in values if v>0]; ax.set_ylim(min(positive)*.4,max(positive)*4)
                    else:
                        lo=min([0]+values); hi=max([0]+values); span=max(hi-lo,1e-9)
                        ax.set_ylim(lo-.08*span if lo<0 else 0,hi+.22*span)
                    ax.set_ylabel(("Time / "+short_arm(c["baseline"])+" (1 = equal; lower is better)") if mode=="relative" else c["ylabel"])
                    ax.set_xticks(xs,[x["label"] for x in c["categories"]],fontsize=8.5,
                                  rotation=20 if ncat>8 and c["kind"]!="line" else 0,ha="right" if ncat>8 and c["kind"]!="line" else "center")
                    if c["kind"]=="line": ax.set_xlabel("Text window index")
                    ax.yaxis.grid(True,color="#E5E9EF",linewidth=.7); ax.set_axisbelow(True)
                    ax.legend(loc="lower left",bbox_to_anchor=(0,1.005),ncol=min(nseries,5),frameon=False,fontsize=9)
                note=(c.get("relative_note","Relative ratios are descriptive; no new confidence interval is inferred.") if mode=="relative" else c.get("error_note","Numbers are archived measurements; missing cells are not measured or failed."))
                fig.text(.045,.063,textwrap.fill(note,155),ha="left",fontsize=8,color="#475467")
                source="Sources: "+", ".join(c["source_run_ids"])
                fig.text(.045,.025,textwrap.fill(source,175),ha="left",fontsize=6.5,color="#667085")
                base="figures/"+c["id"]+"-"+mode
                fig.savefig(out/(base+".png"),dpi=150)
                fig.savefig(out/(base+".svg"))
                c["files"][mode]={"png":base+".png","svg":base+".svg"}
                if not c["detail"] and mode=="absolute":
                    pdf.savefig(fig); pages+=1
                plt.close(fig)
            (out/c["data_file"]).write_text(json.dumps(c,indent=2,allow_nan=False))
            print(f"Rendered {index+1}/{len(book.charts)}: {c['id']}",flush=True)
    return pages

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    evidence=Evidence(args.root.resolve())
    book=build(evidence)
    book.charts.sort(key=lambda c:int(c["experiment"][1:]))
    coverage=evidence.coverage()
    pages=render(book,out)
    manifest={"created_utc":datetime.now(timezone.utc).isoformat(),"charts":book.charts,
        "sources":evidence.sources,"coverage":coverage,"pdf_pages":pages,
        "notes":["No new TPU measurements were run.","N1-N4, N5-N9 and fresh N7 replication cohorts remain separate.",
                 "N2 and N3 did not measure native XLA. N4 measured accuracy only.",
                 "N9 Gemma was access-blocked: no latency or quality value exists.",
                 "N6 initial traces failed; retained ordinary timings are detail charts.",
                 "Zero quality error for native is a self-comparison, not independent qualification."]}
    for name,value in (("manifest.json",manifest),("sources.json",evidence.sources),("coverage.json",coverage)):
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False))
    from plot_gallery_v001 import write_gallery
    write_gallery(out,manifest)
    (out/"README.md").write_text("# N1-N9 experiment graphs\n\nOpen index.html, or serve this directory with a local HTTP server.\n\n"
        "Grouped latency bars show every measured shape. Full-call and prepared-kernel scopes, and all machine cohorts, remain separate.\n"
        "The gallery exposes overview/detail and absolute/relative views. Sources, plotted data, failures and numerical gates are retained.\n"
        f"\n{len(book.charts)} logical charts; {pages} key pages in N1_N9_key_comparisons.pdf.\n")
    files={str(p.relative_to(out)):{"bytes":p.stat().st_size,"sha256":digest(p)} for p in sorted(out.rglob("*")) if p.is_file()}
    (out/"plot_manifest.json").write_text(json.dumps({"files":files,"chart_count":len(book.charts),"pdf_pages":pages},indent=2))
    print(json.dumps({"charts":len(book.charts),"pdf_pages":pages,"output_dir":str(out),"coverage":coverage}))

if __name__=="__main__": main()
