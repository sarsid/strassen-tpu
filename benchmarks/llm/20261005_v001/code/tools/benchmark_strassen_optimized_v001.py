#!/usr/bin/env python3
"""Frozen, bounded Strassen optimization ablations on one TPU.

No autotuning or winner selection is performed. CPU interpretation is an
explicit correctness-only mode that emits no timing samples or speedup claims.
Public shapes use M,N,K; manual tiles use BM,BN,BK.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from strassen_mm import benchmark_v001 as base
from strassen_mm import strassen_schedule_v001 as schedules

DEFAULT_SHAPES = ((1024, 1024, 1024), (4096, 4096, 4096),
                  (8192, 8192, 8192), (16384, 16384, 16384),
                  (512, 4096, 16384), (4096, 512, 16384),
                  (4095, 4095, 4095), (4097, 4097, 4097))
INTERPRET_SHAPES = ((3, 5, 7), (17, 257, 259))
PRODUCT_ORDER = (4, 6, 5, 2, 7, 3, 1)
OPTIMIZED_VARIANTS = ("baseline", "masked_edges", "local_accumulators", "optimized", "optimized_nmk", "peeled_edges")
SCOPES = ("call", "prepared_kernel")


def triple(text):
    try:
        values = tuple(int(v.strip()) for v in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected three comma-separated integers") from error
    if len(values) != 3 or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected three positive comma-separated integers")
    return values


def make_plan(args):
    supplied = None
    if args.shapes_json:
        if args.shape:
            raise ValueError("Use either --shape or --shapes-json")
        supplied = json.loads(args.shapes_json.read_text())
        supplied = supplied["shapes"] if isinstance(supplied, dict) else supplied
        if not isinstance(supplied, list) or not supplied:
            raise ValueError("Shapes JSON must contain a nonempty shapes list")
        shapes = [tuple(s[d] for d in ("m", "n", "k")) for s in supplied]
    else:
        shapes = args.shape or (INTERPRET_SHAPES if args.interpret_correctness else DEFAULT_SHAPES)
    if any(len(s) != 3 or any(type(v) is not int or v <= 0 for v in s) for s in shapes):
        raise ValueError("Public shape dimensions must be positive integer M,N,K")
    tile = args.tile or ((16, 256, 256) if args.interpret_correctness else (1024, 1024, 512))
    variants = args.variant if args.variant is not None else OPTIMIZED_VARIANTS
    if len(shapes) > 32 or len(set(shapes)) != len(shapes):
        raise ValueError("A bounded ablation requires 1–32 unique shapes")
    if any(x % a for x, a in zip(tile, (16, 256, 256))):
        raise ValueError("Strassen tiles must align BM/BN/BK to 16/256/256")
    if len(set(variants)) != len(variants) or set(variants) - set(OPTIMIZED_VARIANTS):
        raise ValueError("Unknown or repeated optimized variant")
    candidates = [
        {"candidate_id": "native_default", "implementation": "native", "variant": "plain", "tile": None, "compiler_options": {}},
        {"candidate_id": "native_vmem_64m", "implementation": "native", "variant": "plain", "tile": None,
         "compiler_options": {"xla_tpu_scoped_vmem_limit_kib": 65536}},
        {"candidate_id": "old_strassen", "implementation": "kernels_v002", "variant": "interleaved_output_accumulator", "tile": list(tile), "compiler_options": {}},
    ]
    for variant in variants:
        ids = ["SOPT-BASE"]
        if variant in ("masked_edges", "optimized", "optimized_nmk"): ids.append("SOPT-001")
        if variant in ("local_accumulators", "optimized", "optimized_nmk", "peeled_edges"): ids.append("SOPT-002")
        if variant == "optimized_nmk": ids.append("SOPT-004")
        if variant == "peeled_edges": ids.append("SOPT-005")
        candidates.append({"candidate_id": f"new_{variant}", "implementation": "strassen_optimized",
                           "variant": "optimized" if variant == "optimized_nmk" else variant,
                           "tile": list(tile), "compiler_options": {}, "product_order": list(PRODUCT_ORDER),
                           "traversal": "nmk" if variant == "optimized_nmk" else "mnk",
                           "optimization_ids": ids, "schedule_id": schedules.schedule_id(PRODUCT_ORDER),
                           "algorithm_scope": "hybrid_strassen_core_and_native_tails" if variant == "peeled_edges" else "one_level_tile_strassen"})
    if args.include_schedule_candidates:
        for schedule in schedules.REGISTRY:
            if schedule.order == PRODUCT_ORDER: continue
            candidates.append({"candidate_id": f"new_optimized_schedule_{schedule.alias}", "implementation": "strassen_optimized",
                               "variant": "optimized", "tile": list(tile), "compiler_options": {},
                               "product_order": list(schedule.order), "traversal": "mnk", "schedule_id": schedule.schedule_id,
                               "optimization_ids": ["SOPT-BASE","SOPT-001","SOPT-002","SOPT-003"],
                               "algorithm_scope": "one_level_tile_strassen", "schedule_rationale": schedule.rationale,
                               "schedule_proxy_scope": "Source-order hypothesis; not a measured performance prediction"})
    shape_records = []
    for i, (m,n,k) in enumerate(shapes):
        fixed_tile = (supplied[i].get("tile_bm_bn_bk") if supplied else None) or args.tile
        if fixed_tile is None:
            fixed_tile = ((16,256,256) if args.interpret_correctness else
                          (2048,2048,512) if m == n == k and m >= 8192 else (1024,1024,512))
        if len(fixed_tile) != 3 or any(type(v) is not int or v <= 0 or v % a for v,a in zip(fixed_tile,(16,256,256))):
            raise ValueError("Each fixed per-shape BM/BN/BK tile must align to 16/256/256")
        shape_records.append({"id": f"ablation_m{m}_n{n}_k{k}", "m":m,"n":n,"k":k,"tile_bm_bn_bk":list(fixed_tile)})
    if args.interpret_correctness:
        for shape in shape_records:
            m,n,k = (shape[d] for d in ("m","n","k"));bm,bn,bk=shape["tile_bm_bn_bk"]
            padded = math.prod(((v + b - 1)//b)*b for v, b in zip((m, n, k), (bm, bn, bk)))
            if padded > 2**27:
                raise ValueError("CPU correctness interpretation is limited to small shapes/tiles; no large TPU workload is allowed")
    return {"schema_version": 1, "protocol": "fixed comparative ablation; no automatic selection or winner claim",
            "shape_order": "M,N,K", "tile_order": "BM,BN,BK", "shapes": shape_records,
            "candidates": candidates, "scopes": list(SCOPES), "interpret_correctness": args.interpret_correctness,
            "precision": {"input": "bfloat16", "output": "float32", "reference": "host_numpy_fp32 on quantized operands", "native_dot": "DEFAULT"},
            "correctness": {"near_zero_denominator": 1e-30, "sample_rows": 128, "sample_columns": 128,
                            "gate": {"require_finite": True, "relative_l2_max": 0.02, "max_abs_atol": 0.001, "max_abs_reference_rtol": 0.05}},
            "vmem_limit_bytes": args.vmem_limit_mib * 2**20,
            "memory_budget_bytes": int(args.memory_budget_gib * 2**30),
            "tile_binding": "Each shape.tile_bm_bn_bk overrides the catalog placeholder for every custom arm; native tiles are compiler-managed"}


def bound_candidates(plan, shape):
    return [dict(c, tile=shape["tile_bm_bn_bk"] if c["implementation"] != "native" else None)
            for c in plan["candidates"]]


def verify_artifact_hashes(directory):
    directory = Path(directory).resolve()
    declared = json.loads((directory/"artifact_manifest.json").read_text())["sha256"]
    for required in ("plan.json","summary.json","results.jsonl","source_manifest.json","environment.json"):
        if required not in declared:
            raise ValueError(f"Prior run does not seal required evidence: {required}")
    for name, expected in declared.items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or base.digest_file(path) != expected:
            raise ValueError(f"Prior-run artifact missing or hash mismatch: {name}")


def memory_estimate(shape, candidates):
    m, n, k = (shape[d] for d in ("m", "n", "k"))
    original = 2 * (m*k + k*n)
    extra, largest_inputs, largest_output = 0, original, 4*m*n
    for candidate in candidates:
        if candidate["implementation"] == "native":
            continue
        bm, bn, bk = candidate["tile"]
        mp, nn, kp = (((v+b-1)//b)*b for v,b in zip((m,n,k),(bm,bn,bk)))
        masked = candidate["implementation"] == "strassen_optimized" and candidate["variant"] in ("masked_edges", "optimized", "peeled_edges")
        if masked:
            mp, nn, kp = m, n, k
        size = 2 * (mp*kp + kp*nn)
        if (mp, nn, kp) != (m, n, k):
            extra += size  # Prepared operands are cached per arm, not merely by padded shape.
        largest_inputs = max(largest_inputs, size)
        largest_output = max(largest_output, 4*mp*nn)
    return {"estimated_live_device_bytes": original + extra + largest_inputs + 2*largest_output + 8*128*128,
            "original_input_bytes": original, "per_arm_prepared_padding_bytes": extra,
            "largest_padded_input_bytes": largest_inputs, "reserved_output_and_temporary_bytes": 2*largest_output,
            "estimate_is_not_measured_peak": True,
            "unmeasured": "compiler executable residency and runtime/device peak allocations"}


def error_status(error, during):
    message = str(error).lower()
    if any(term in message for term in ("unknown compiler", "unrecognized", "unsupported compiler", "no such option")):
        return "unsupported_compiler_option"
    return base.error_status(error, during)


def snapshot(out):
    target = out / "source_snapshot"; target.mkdir()
    paths = [Path(__file__).resolve(), *sorted((PROJECT / "src/strassen_mm").glob("*.py"))]
    hashes = {}
    for path in paths:
        dest = target / path.name; shutil.copyfile(path, dest)
        hashes[str(dest.relative_to(out))] = base.digest_file(dest)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    result = {"git_commit": commit, "sha256": hashes}
    base.exclusive_json(out / "source_manifest.json", result)
    return result


def device_memory():
    try:
        value = base.jax.devices()[0].memory_stats()
        return {"status": "available" if value is not None else "unavailable", "stats": base.json_safe(value),
                "scope": "Device allocator counters; any peak is runtime-global, not isolated to this arm"}
    except Exception as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}


def compiled_memory(executable):
    try:
        value = executable.memory_analysis()
        keys = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes",
                "generated_code_size_in_bytes", "host_argument_size_in_bytes", "host_output_size_in_bytes", "host_temp_size_in_bytes")
        return {"status": "available" if value is not None else "unavailable",
                "stats": {k: getattr(value, k) for k in keys if hasattr(value, k)},
                "scope": "Compiler memory analysis; not measured device peak"}
    except Exception as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}


class AblationJournal(base.Journal):
    def __init__(self, out, phase, plan):
        super().__init__(out, phase)
        self.shapes = {s["id"]: s for s in plan["shapes"]}
        self.candidates = {c["candidate_id"]: c for c in plan["candidates"]}
        self.rows, self.completed = [], set()

    def emit(self, event, **fields):
        if fields.get("shape_id") in self.shapes:
            shape = self.shapes[fields["shape_id"]]
            fields.setdefault("shape_mnk", [shape[d] for d in ("m", "n", "k")])
            fields.setdefault("shape_mkn", [shape[d] for d in ("m", "k", "n")])
        if fields.get("candidate_id") in self.candidates:
            candidate = dict(self.candidates[fields["candidate_id"]])
            if candidate["implementation"] != "native" and fields.get("shape_id") in self.shapes:
                candidate["tile"] = self.shapes[fields["shape_id"]]["tile_bm_bn_bk"]
            for k, v in candidate.items():
                fields.setdefault(k, v)
        if event == "case_result":
            key = tuple(fields.get(k) for k in ("shape_id", "candidate_id", "scope"))
            if key in self.completed:
                raise ValueError(f"Duplicate terminal case {key}")
            self.completed.add(key); self.rows.append(copy.deepcopy(fields))
        super().emit(event, **fields)


class Runner:
    def __init__(self, args, plan, journal, old, optimized):
        self.args, self.plan, self.journal, self.old, self.optimized = args, plan, journal, old, optimized
        self.started = time.monotonic()
        self.campaign = {"campaign_id": "strassen_optimized_ablation_v001", "correctness": plan["correctness"],
                         "memory": {"max_reference_output_elements": 2**20, "max_full_reference_flops": 2e8}}

    def check(self):
        if time.monotonic() - self.started > self.args.max_wall_seconds:
            raise TimeoutError("Ablation wall-time limit reached; resume only in a new output directory")

    def emit_error(self, context, error, during):
        self.journal.emit("error", **context, during=during, status=error_status(error, during),
                          error_type=type(error).__name__, message=str(error), traceback=traceback.format_exc())

    def make_fn(self, candidate, shape):
        mnk = tuple(shape[d] for d in ("m", "n", "k")); m,n,k = mnk
        if candidate["implementation"] == "strassen_optimized":
            return self.optimized.make_matmul(mnk, candidate["tile"], variant=candidate["variant"],
                interpret=self.args.interpret_correctness, vmem_limit_bytes=self.plan["vmem_limit_bytes"],
                product_order=tuple(candidate["product_order"]), traversal=candidate["traversal"])
        native = candidate["implementation"] == "native"
        return self.old.make_matmul("native" if native else "strassen", (m,k,n), candidate["tile"],
            variant=candidate["variant"], interpret=self.args.interpret_correctness,
            vmem_limit_bytes=None if native else self.plan["vmem_limit_bytes"])

    def compile(self, shape, a, b, context):
        entries = []
        for candidate in bound_candidates(self.plan,shape):
            self.check(); detail = {**context, "candidate_id": candidate["candidate_id"]}
            try:
                fn = self.make_fn(candidate, shape)
            except Exception as error:
                self.emit_error(detail, error, "compile")
                entries.extend({**candidate, "scope": scope, "error": error_status(error, "compile"),
                                "error_message": str(error), "samples": [], "correctness": None} for scope in SCOPES)
                continue
            for scope in SCOPES:
                self.check(); entry = {**candidate, "scope": scope, "fn": fn, "samples": [], "correctness": None,
                                      "metadata": fn.metadata}
                start = time.perf_counter_ns()
                try:
                    if scope == "call":
                        specs = (base.jax.ShapeDtypeStruct(a.shape, a.dtype), base.jax.ShapeDtypeStruct(b.shape, b.dtype))
                        lowered = base.jax.jit(fn).lower(*specs)
                    else:
                        mp, kp, nn = fn.metadata.get("prepared_shape_mkn") or fn.metadata["padded_shape_mkn"]
                        specs = (base.jax.ShapeDtypeStruct((mp,kp), a.dtype), base.jax.ShapeDtypeStruct((kp,nn), b.dtype))
                        lowered = base.jax.jit(fn.kernel).lower(*specs)
                    options = candidate["compiler_options"] if not self.args.interpret_correctness else {}
                    executable = lowered.compile(compiler_options=options) if options else lowered.compile()
                    entry["executable"] = executable
                    self.journal.emit("compilation", **detail, scope=scope, status="ok",
                        compile_ms=None if self.args.interpret_correctness else (time.perf_counter_ns()-start)/1e6,
                        kernel_metadata=fn.metadata, compiler_options_applied=options,
                        cpu_interpretation_has_no_timing_claim=self.args.interpret_correctness,
                        compiled_memory=compiled_memory(executable))
                except Exception as error:
                    unsupported_option = (not self.args.interpret_correctness and bool(candidate["compiler_options"])
                        and any(word in str(error).lower() for word in ("unknown", "unrecognized", "unsupported", "not supported", "no such")))
                    entry.update(error="unsupported_compiler_option" if unsupported_option else error_status(error, "compile"), error_message=str(error))
                    self.emit_error({**detail, "scope": scope}, error, "compile")
                entries.append(entry)
        return entries

    def group(self, shape, index):
        context = {"shape_id": shape["id"], "group_id": f"{shape['id']}__fixed_ablation", "distribution": "gaussian", "seed": self.args.seed}
        estimate = memory_estimate(shape, bound_candidates(self.plan,shape))
        self.journal.emit("group_start", **context, index=index, total=len(self.plan["shapes"]))
        self.journal.emit("memory_preflight", **context, **estimate)
        if estimate["estimated_live_device_bytes"] > self.plan["memory_budget_bytes"]:
            for candidate in self.plan["candidates"]:
                for scope in SCOPES:
                    self.journal.emit("case_result", **context, candidate_id=candidate["candidate_id"], scope=scope,
                                      status="skipped_memory_preflight", numerically_eligible=False, eligible_for_speedup_claim=False,
                                      timing={"sample_count": 0}, correctness=None)
            return
        self.check(); a_host,b_host = base.generate_inputs(tuple(shape[d] for d in ("m","k","n")), "gaussian", self.args.seed)
        self.journal.emit("case_start", **context, input_fingerprint=base.input_fingerprint(a_host,b_host), device_memory=device_memory())
        reference, rows, cols, info = base.make_reference(a_host,b_host,self.campaign,self.args.seed)
        entries = self.compile(shape,a_host,b_host,context)
        a,b = base.jax.device_put(a_host),base.jax.device_put(b_host); base.jax.block_until_ready((a,b))
        ri,ci = base.jnp.asarray(rows),base.jnp.asarray(cols)

        @base.jax.jit
        def sample_output(output):
            return base.jnp.all(base.jnp.isfinite(output)), output[ri[:,None],ci[None,:]]

        prepared = {}
        for entry in entries:
            self.check()
            if entry.get("error"): continue
            detail = {**context, "candidate_id": entry["candidate_id"], "scope": entry["scope"]}
            try:
                if entry["scope"] == "call":
                    entry["operands"] = (a,b)
                else:
                    key = entry["candidate_id"]
                    if key not in prepared:
                        prepared[key] = base.jax.jit(entry["fn"].prepare)(a,b)
                        base.jax.block_until_ready(prepared[key])
                        self.journal.emit("preparation", **detail, operand_shapes=[list(x.shape) for x in prepared[key]], cache_key=key,
                                          timing_role="excluded from prepared_kernel; included in complete-call executable")
                    entry["operands"] = prepared[key]
                output = entry["executable"](*entry["operands"]); output.block_until_ready()
                finite,sampled = base.jax.device_get(sample_output(output))
                metrics = base.numeric_metrics(sampled,finite,reference,info,self.campaign)
                entry["correctness"] = metrics
                self.journal.emit("correctness", **detail, metrics=metrics)
                output.delete(); del output
            except Exception as error:
                entry.update(error=error_status(error,"execute"),error_message=str(error)); self.emit_error(detail,error,"execute")
        if not self.args.interpret_correctness:
            rng = base.np.random.Generator(base.np.random.PCG64(self.args.seed+index*7919+17))
            for warmup in range(self.args.warmups):
                for i in rng.permutation(len(entries)):
                    entry = entries[int(i)]
                    if entry.get("error"): continue
                    self.check()
                    try:
                        output=entry["executable"](*entry["operands"]);output.block_until_ready();output.delete();del output
                    except Exception as error:
                        entry.update(error=error_status(error,"execute"),error_message=str(error))
                        self.emit_error({**context,"candidate_id":entry["candidate_id"],"scope":entry["scope"]},error,"warmup")
            for round_id in range(self.args.repeats):
                order=[entries[int(i)] for i in rng.permutation(len(entries))]
                self.journal.emit("round_order", **context, round=round_id,
                                  order=[{"candidate_id":e["candidate_id"],"scope":e["scope"],"active":not bool(e.get("error"))} for e in order])
                for position,entry in enumerate(order):
                    if entry.get("error"): continue
                    self.check()
                    try:
                        start=time.perf_counter_ns();output=entry["executable"](*entry["operands"]);output.block_until_ready()
                        elapsed=(time.perf_counter_ns()-start)/1e6
                        output.delete();del output
                        sample={"round":round_id,"position":position,"elapsed_ms":elapsed,"start_perf_counter_ns":start}
                        entry["samples"].append(sample)
                        self.journal.emit("sample",**context,candidate_id=entry["candidate_id"],scope=entry["scope"],**sample)
                    except Exception as error:
                        entry.update(error=error_status(error,"execute"),error_message=str(error))
                        self.emit_error({**context,"candidate_id":entry["candidate_id"],"scope":entry["scope"]},error,"timing")
        for entry in entries:
            metrics=entry["correctness"];status=entry.get("error") or ("ok" if metrics and metrics["pass"] else "numerical_failure")
            eligible=(not self.args.interpret_correctness and status=="ok" and len(entry["samples"])==self.args.repeats)
            comparisons=[]
            if eligible:
                for other in entries:
                    if other is entry or other["scope"]!=entry["scope"] or other["candidate_id"] not in ("native_default","native_vmem_64m","old_strassen"):continue
                    if other.get("error") or not (other["correctness"] or {}).get("pass") or len(other["samples"])!=self.args.repeats:continue
                    pair=base.paired_comparison(other["samples"],entry["samples"],self.args.seed+811)
                    pair.update(reference_candidate_id=other["candidate_id"],valid_numerical_comparison=True,
                                screen_results_are_not_confirmation=self.args.phase!="confirm",
                                scope=entry["scope"],group_id=context["group_id"])
                    comparisons.append(pair)
            self.journal.emit("case_result",**context,candidate_id=entry["candidate_id"],scope=entry["scope"],status=status,
                numerically_eligible=bool(status=="ok"),eligible_for_speedup_claim=eligible,
                error_message=entry.get("error_message"),correctness=metrics,timing=base.basic_statistics(entry["samples"]),
                comparisons=comparisons,kernel_metadata=entry.get("metadata"),
                fidelity_scope="Finite Gaussian-input gate only; not a universal accuracy guarantee",
                interpretation_only=self.args.interpret_correctness)
        self.journal.emit("group_complete",**context,device_memory=device_memory())
        del prepared,entries,a,b,a_host,b_host,reference,ri,ci,sample_output
        base.jax.clear_caches();gc.collect()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--phase",choices=("screen","confirm"),default="screen")
    p.add_argument("--prior-run",type=Path,help="Frozen screen artifact directory; required for independent confirmation")
    p.add_argument("--shape",type=triple,action="append",help="Public M,N,K; repeat for multiple shapes")
    p.add_argument("--shapes-json",type=Path,help="JSON shapes [{m,n,k,tile_bm_bn_bk?}], frozen before execution")
    p.add_argument("--tile",type=triple,help="Shared BM,BN,BK override; otherwise fixed per-shape defaults")
    p.add_argument("--variant",choices=OPTIMIZED_VARIANTS,action="append",help="Filter new variants; native and old-Strassen controls remain")
    p.add_argument("--include-schedule-candidates",action="store_true",help="Add registered non-current product orders, optimized/mnk only")
    p.add_argument("--repeats",type=int,choices=(7,30),help="Default 7 for screen, 30 for confirmation")
    p.add_argument("--warmups",type=int,default=5)
    p.add_argument("--seed",type=int)
    p.add_argument("--vmem-limit-mib",type=int,default=48)
    p.add_argument("--memory-budget-gib",type=float,default=8)
    p.add_argument("--max-wall-seconds",type=float,default=3600)
    p.add_argument("--allocation-id")
    p.add_argument("--expected-identity",type=Path)
    p.add_argument("--interpret-correctness",action="store_true")
    p.add_argument("--plan-only",action="store_true",help="Write and seal the frozen plan without importing JAX")
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    supplied_flags = {value.partition("=")[0] for value in (argv if argv is not None else sys.argv[1:]) if value.startswith("--")}
    if args.warmups<1 or args.vmem_limit_mib<=0 or any(not math.isfinite(v) or v<=0 for v in (args.memory_budget_gib,args.max_wall_seconds)):
        p.error("Warmups, VMEM, memory budget and wall-time limit must be positive and finite")
    args.repeats=args.repeats or (30 if args.phase=="confirm" else 7)
    args.seed=args.seed if args.seed is not None else (20260924 if args.phase=="confirm" else 20260923)
    if args.seed<0:p.error("Seed must be nonnegative")
    if args.phase=="confirm" and (not args.prior_run or args.repeats!=30 or args.interpret_correctness):
        p.error("Confirmation requires --prior-run, 30 rounds and TPU timing mode")
    if not args.plan_only and not args.interpret_correctness and not args.allocation_id:p.error("TPU timing requires --allocation-id")
    prior=None
    if args.prior_run:
        prior=args.prior_run.resolve()
        if (prior/"artifacts/plan.json").is_file():prior=prior/"artifacts"
    plan=make_plan(args)
    if args.phase=="confirm":
        verify_artifact_hashes(prior)
        oldplan=json.loads((prior/"plan.json").read_text())
        for flag, actual, expected in (("--vmem-limit-mib",args.vmem_limit_mib*2**20,oldplan["vmem_limit_bytes"]),
                                       ("--memory-budget-gib",int(args.memory_budget_gib*2**30),oldplan["memory_budget_bytes"])):
            if flag in supplied_flags and actual != expected:
                p.error(f"Confirmation cannot change frozen {flag}")
        if args.shape is None and args.shapes_json is None and args.tile is None and args.variant is None and not args.include_schedule_candidates:
            plan=oldplan
        elif plan!=oldplan:
            p.error("Confirmation cannot change the frozen screen plan")
        prior_summary=json.loads((prior/"summary.json").read_text())
        if not prior_summary.get("completed") or prior_summary.get("phase")!="screen" or prior_summary.get("plan_only") or prior_summary.get("interpretation_only"):
            p.error("Confirmation requires a completed TPU screening execution")
        if args.seed==prior_summary["seed"]:p.error("Confirmation must use a fresh input seed")
    args.output_dir.mkdir(parents=True,exist_ok=False);out=args.output_dir.resolve()
    base.exclusive_json(out/"plan.json",plan)
    execution_plan = [{"group_id":f"{s['id']}__fixed_ablation","shape":s,"candidates":bound_candidates(plan,s),
                       "scopes":list(SCOPES),"input":{"distribution":"gaussian","seed":args.seed},
                       "timing":{"repeats":0 if args.interpret_correctness else args.repeats,
                                 "warmups":0 if args.interpret_correctness else args.warmups,
                                 "order":"Fresh deterministic PCG64 permutation of candidate/scope entries each round"},
                       "memory_preflight":memory_estimate(s,bound_candidates(plan,s))} for s in plan["shapes"]]
    base.exclusive_json(out/"planned_cases.json",execution_plan)
    source=snapshot(out)
    journal=AblationJournal(out,args.phase,plan)
    started=time.monotonic();completed=False;error=None;done=[]
    try:
        journal.emit("run_start",argv=list(argv) if argv is not None else sys.argv[1:],seed=args.seed,repeats=args.repeats,
                     warmups=args.warmups,interpretation_only=args.interpret_correctness,source_manifest=source,
                     plan_sha256=base.digest_file(out/"plan.json"),execution_plan=execution_plan)
        if args.plan_only:
            completed=True
        else:
            if "jax" in sys.modules:raise RuntimeError("Launch a fresh process before importing JAX")
            os.environ.update(base.FIXED_ENVIRONMENT)
            if args.interpret_correctness:os.environ["JAX_PLATFORMS"]="cpu"
            import jax
            import jax.numpy as jnp
            import numpy as np
            import ml_dtypes
            from strassen_mm import kernels_v002 as old
            from strassen_mm import strassen_optimized as optimized
            base.jax,base.jnp,base.np,base.ml_dtypes=jax,jnp,np,ml_dtypes
            compat=old.enable_qualified_mosaic_v7_compat()
            environment=base.capture_environment(args,{"campaign_id":"strassen_optimized_ablation_v001"},compat)
            environment["interpretation_only"]=args.interpret_correctness
            base.exclusive_json(out/"environment.json",environment)
            if not args.interpret_correctness and (len(jax.devices())!=1 or jax.local_device_count()!=1 or jax.process_count()!=1 or jax.devices()[0].platform!="tpu"):
                raise RuntimeError("Performance ablations require exactly one local/global TPU and one JAX process")
            identity_path=args.expected_identity or (prior/"environment.json" if args.phase=="confirm" else None)
            if identity_path:
                expected=json.loads(identity_path.read_text());expected=expected.get("identity",expected)
                if expected!=environment["identity"]:raise RuntimeError("Environment identity differs from frozen expected cohort")
            if args.phase=="confirm":
                prior_source=json.loads((prior/"source_manifest.json").read_text())
                if source["sha256"]!=prior_source["sha256"]:raise RuntimeError("Source changed between screening and confirmation")
                base.exclusive_json(out/"screen_provenance.json",{"path":str(prior),"plan_sha256":base.digest_file(prior/"plan.json"),
                    "results_sha256":base.digest_file(prior/"results.jsonl"),"source_manifest_sha256":base.digest_file(prior/"source_manifest.json")})
            runner=Runner(args,plan,journal,old,optimized)
            for index,shape in enumerate(plan["shapes"]):
                print(f"[{index+1}/{len(plan['shapes'])}] {shape['id']} · {len(plan['candidates'])} frozen arms",flush=True)
                runner.check();runner.group(shape,index);done.append(shape["id"])
            completed=True
    except BaseException as exc:
        error={"type":type(exc).__name__,"message":str(exc),"status":error_status(exc,"execute")}
        journal.emit("run_error",**error,traceback=traceback.format_exc());print(f"Ablation failed: {exc}",file=sys.stderr,flush=True)
    finally:
        if not args.plan_only:
            for shape in plan["shapes"]:
                for candidate in plan["candidates"]:
                    for scope in SCOPES:
                        key=(shape["id"],candidate["candidate_id"],scope)
                        if key not in journal.completed:
                            journal.emit("case_result",shape_id=shape["id"],group_id=f"{shape['id']}__fixed_ablation",
                                candidate_id=candidate["candidate_id"],scope=scope,status="not_run",seed=args.seed,distribution="gaussian",
                                numerically_eligible=False,eligible_for_speedup_claim=False,correctness=None,timing={"sample_count":0},
                                error_message="Execution stopped before terminal case reporting; any partial samples remain in the journal")
        summary={"schema_version":1,"phase":args.phase,"completed":completed,"plan_only":args.plan_only,
                 "interpretation_only":args.interpret_correctness,"seed":args.seed,"repeats":0 if args.interpret_correctness else args.repeats,
                 "warmups":0 if args.interpret_correctness else args.warmups,"case_status_counts":dict(journal.status_counts),
                 "planned_shapes":len(plan["shapes"]),"planned_candidates":len(plan["candidates"]),"completed_shape_ids":done,
                 "wall_seconds":time.monotonic()-started,"error":error,"automatic_selection_performed":False,
                 "scope":"Fixed-candidate Gaussian ablation; screen results are exploratory; interpret mode is correctness only"}
        journal.emit("run_complete",**summary);journal.close()
        base.exclusive_json(out/"summary.json",summary)
        base.exclusive_json(out/"case_results.json",journal.rows)
        hashes={str(x.relative_to(out)):base.digest_file(x) for x in sorted(out.rglob("*")) if x.is_file()}
        base.exclusive_json(out/"artifact_manifest.json",{"sha256":hashes,"sealed_utc":base.utc_now()})
    return 0 if completed else 1


if __name__=="__main__":
    raise SystemExit(main())
