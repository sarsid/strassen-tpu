"""Immutable, single-device v5e runner for the preregistered N1--N4 study.

Run as ``python -m strassen_mm.benchmark_v001 --campaign CONFIG --phase smoke
--output-dir NEW_DIRECTORY --allocation-id ALLOCATION``. JAX is deliberately
imported only after fixed runtime settings are applied. This runner has no
CPU timing mode. CPU NumPy is used solely as an independent numerical reference.

Every results.jsonl line is flushed and fsynced; existing output directories
are rejected. A phase can be retried only in a new directory. Source/config
copies, hashes, environment, planned cases, errors and raw chronological
samples remain reviewable even when compilation, correctness or timing fails.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata as package_metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback


FIXED_ENVIRONMENT = {
    "JAX_PLATFORMS": "tpu",
    "JAX_ENABLE_X64": "false",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
FLAG_NAMES = tuple(FIXED_ENVIRONMENT) + (
    "XLA_FLAGS", "LIBTPU_INIT_ARGS", "TPU_NUM_CHIPS", "TPU_VISIBLE_CHIPS",
    "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS", "PJRT_DEVICE",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def exclusive_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def digest_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def installed_version(name):
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return None


class Journal:
    def __init__(self, directory, phase):
        self.directory = directory
        self.phase = phase
        self.sequence = 0
        self.handle = (directory / "results.jsonl").open("x", encoding="utf-8")
        self.status_counts = Counter()

    def emit(self, event, **fields):
        self.sequence += 1
        row = {"event": event, "sequence": self.sequence, "utc": utc_now(),
               "monotonic_ns": time.monotonic_ns(), "phase": self.phase, **fields}
        self.handle.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        if event == "case_result":
            self.status_counts[row["status"]] += 1

    def close(self):
        self.handle.close()


def snapshot_sources(out, campaign_path, shape_path, distribution_path):
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    package_root = Path(__file__).resolve().parent
    entries = {}
    for path in sorted(package_root.glob("*.py")):
        target = source_dir / path.name
        shutil.copyfile(path, target)
        entries[str(target.relative_to(out))] = digest_file(target)
    config_dir = out / "config_snapshot"
    config_dir.mkdir()
    for path in (campaign_path, shape_path, distribution_path):
        target = config_dir / path.name
        shutil.copyfile(path, target)
        entries[str(target.relative_to(out))] = digest_file(target)
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=campaign_path.parent,
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        git_commit = None
    manifest = {"git_commit": git_commit, "sha256": entries,
                "note": "Source/config hashes are authoritative; transport may omit .git."}
    exclusive_json(out / "source_manifest.json", manifest)
    return manifest


def capture_environment(args, campaign, compatibility):
    devices = jax.devices()
    device = devices[0] if len(devices) == 1 else None
    kind = device.device_kind if device else None
    versions = {name: installed_version(name) for name in (
        "jax", "jaxlib", "libtpu", "numpy", "ml_dtypes", "scipy", "requests"
    )}
    device_records = []
    for item in devices:
        record = {"kind": item.device_kind, "id": int(item.id),
                  "platform": item.platform, "process_index": int(item.process_index)}
        for field in ("coords", "core_on_chip", "slice_index", "local_hardware_id"):
            if hasattr(item, field) and getattr(item, field) is not None:
                record[field] = json_safe(getattr(item, field))
        device_records.append(record)
    identity = {
        "allocation_id": args.allocation_id,
        "colab_endpoint": args.allocation_id,
        "hostname": socket.gethostname(),
        "host_id": socket.gethostname(),
        "boot_id": read_optional("/proc/sys/kernel/random/boot_id"),
        "device_kind": kind,
        "device_id": int(device.id) if device else None,
        "process_index": int(device.process_index) if device else None,
        "jax_version": jax.__version__, "jaxlib_version": installed_version("jaxlib"),
        "libtpu_version": installed_version("libtpu"),
        "numpy_version": np.__version__,
        "runtime_image": os.environ.get("TPU_RUNTIME_VERSION") or os.environ.get("COLAB_RELEASE_TAG"),
        "runtime_flags": {name: os.environ.get(name) for name in FLAG_NAMES},
        "mosaic_compatibility": compatibility,
        "versions": versions, "devices": device_records,
    }
    normalized = (kind or "").lower().replace(" ", "").replace("-", "")
    qualified = (len(devices) == 1 and jax.local_device_count() == 1
                 and jax.process_count() == 1 and device.platform == "tpu"
                 and ("v5e" in normalized or "v5lite" in normalized))
    environment = {
        "created_utc": utc_now(), "identity": identity,
        "device_count": len(devices), "local_device_count": jax.local_device_count(),
        "process_count": jax.process_count(), "devices": device_records,
        "backend": jax.default_backend(), "platform": platform.platform(),
        "python": sys.version, "pid": os.getpid(), "qualified_single_v5e": qualified,
        "fixed_environment_policy": FIXED_ENVIRONMENT,
        "numpy_blas": str(np.__config__.CONFIG) if hasattr(np.__config__, "CONFIG") else "unknown",
        "identity_scope": "One logical TPU allocation and runtime; physical chip serial unavailable.",
        "identity_unknown_fields": [key for key, value in identity.items() if value is None],
        "campaign_id": campaign["campaign_id"],
    }
    return environment


def verify_identity(environment, expected_path, campaign):
    if not environment["qualified_single_v5e"]:
        raise RuntimeError("requires exactly one local/global TPU v5e device and one JAX process")
    if expected_path is None:
        return {"status": "new_cohort", "expected_path": None}
    prior = json.loads(expected_path.read_text())
    prior = prior.get("identity", prior)
    current = environment["identity"]
    if "versions" in prior and "colab_endpoint" in prior and "runtime_flags" not in prior:
        # First smoke compares to immutable provisioning identity. Later runs
        # should use smoke's environment.json, also sealing runtime flags.
        fields = {"colab_endpoint", "hostname", "boot_id", "versions", "devices"}
    else:
        fields = set(campaign["device"]["identity_fields"]) | {
            "hostname", "boot_id", "numpy_version", "mosaic_compatibility", "versions", "devices"
        }
    mismatches = {key: {"expected": prior.get(key), "actual": current.get(key)}
                  for key in sorted(fields) if prior.get(key) != current.get(key)}
    if mismatches:
        raise RuntimeError("environment identity mismatch: " + json.dumps(mismatches, sort_keys=True))
    return {"status": "matched", "fields": sorted(fields), "expected_path": str(expected_path),
            "unknown_fields": environment["identity_unknown_fields"]}


def planned_groups(campaign, manifest, phase):
    experiment = campaign["experiments"][phase]
    shapes = {shape["id"]: shape for shape in manifest["shapes"]}
    selected = [shapes[key] for key in experiment["shape_ids"]]
    selected.sort(key=lambda s: (s.get("stage") == "anchor",
                                 2 * (s["m"] * s["k"] + s["k"] * s["n"])
                                 + 8 * s["m"] * s["n"], s["id"]))
    rng = np.random.Generator(np.random.PCG64(campaign["seed"]))
    result = []
    for shape in selected:
        tiles = list(experiment.get("tiles", [experiment.get("tile")]))
        if phase == "N2":
            tiles = [tiles[int(i)] for i in rng.permutation(len(tiles))]
        for tile in tiles:
            if phase == "N3":
                arms = [{"arm_id": f"{algorithm}__{variant}", "algorithm": algorithm,
                         "variant": variant} for algorithm in experiment["algorithms"]
                        for variant in experiment["variants"]]
            else:
                arms = [{"arm_id": key, **campaign["arms"][key]}
                        for key in experiment["arm_ids"]]
            case_inputs = ([{"distribution": dist, "seed": seed}
                            for dist in experiment["distributions"]
                            for seed in experiment["seeds"]] if phase == "N4" else
                           [{"distribution": experiment.get("distribution", "gaussian"),
                             "seed": campaign["seed"]}])
            result.append({"group_id": f'{shape["id"]}__{"x".join(map(str, tile))}',
                           "shape": shape, "tile": tile, "arms": arms,
                           "inputs": case_inputs, "timing": experiment.get("timing"),
                           "scopes": ["call"] if phase == "N4" else
                           campaign["timing"]["measure_scopes"]})
    return result


def memory_estimate(shape, tile, arms, scopes):
    m, k, n = shape
    bm, bn, bk = tile
    mp, kp, nn = [((x + b - 1) // b) * b for x, b in zip(shape, (bm, bk, bn))]
    original = 2 * (m * k + k * n)
    padded = 2 * (mp * kp + kp * nn)
    # Prepared inputs are shared by equal shapes across arms. Reserve two
    # output-sized buffers for result + runtime temporary and a host-reference
    # sample transfer. Executable memory is unmeasured and separately reported.
    extra_prepared = padded if (mp, kp, nn) != shape and "prepared_kernel" in scopes else 0
    live = original + padded + extra_prepared + 8 * mp * nn + 8 * 128 * 128
    return {"estimated_live_device_bytes": live, "original_input_bytes": original,
            "padded_input_bytes": padded, "reserved_output_and_temporary_bytes": 8 * mp * nn,
            "padded_shape_mkn": [mp, kp, nn], "unmeasured": "executable/runtime peak memory",
            "estimate_is_not_measured_peak": True}


def generate_inputs(shape, distribution, seed):
    """Exact host PCG64 construction specified in distributions_v1.json."""
    m, k, n = shape
    rng = np.random.Generator(np.random.PCG64(seed))
    scale = np.float32(math.sqrt(k))
    if distribution == "cancellation":
        h = k // 2
        left = rng.standard_normal((m, h), dtype=np.float32) / scale
        top = rng.standard_normal((h, n), dtype=np.float32)
        delta = rng.standard_normal((m, h), dtype=np.float32) / np.float32(128 * math.sqrt(k))
        a = np.zeros((m, k), np.float32)
        b = np.zeros((k, n), np.float32)
        a[:, :h], a[:, h:2 * h] = left, left + delta
        b[:h, :], b[h:2 * h, :] = top, -top
        del left, top, delta
    elif distribution == "uniform":
        bound = math.sqrt(3)
        a = rng.uniform(-bound, bound, (m, k)).astype(np.float32) / scale
        b = rng.uniform(-bound, bound, (k, n)).astype(np.float32)
    else:
        a = rng.standard_normal((m, k), dtype=np.float32) / scale
        b = rng.standard_normal((k, n), dtype=np.float32)
        if distribution == "log_scale":
            a *= np.exp2(rng.integers(-8, 9, size=k)).astype(np.float32)[None, :]
            b *= np.exp2(rng.integers(-8, 9, size=k)).astype(np.float32)[:, None]
        elif distribution == "outliers":
            a[rng.random(a.shape) < 0.001] *= np.float32(128)
            b[rng.random(b.shape) < 0.001] *= np.float32(128)
        elif distribution != "gaussian":
            raise ValueError(f"unknown distribution: {distribution}")
    # Quantize once on the host. All algorithms and the reference then use
    # identical operands; pre-quantization FP32 arrays are not the reference.
    a = a.astype(ml_dtypes.bfloat16)
    b = b.astype(ml_dtypes.bfloat16)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("input_error: nonfinite generated BF16 input")
    return a, b


def input_fingerprint(a, b):
    return {"a_sha256": hashlib.sha256(memoryview(a.view(np.uint8))).hexdigest(),
            "b_sha256": hashlib.sha256(memoryview(b.view(np.uint8))).hexdigest(),
            "quantized_dtype": "bfloat16"}


def sample_indices(size, count, rng):
    if size <= count:
        return np.arange(size, dtype=np.int32)
    if count < 2:
        raise ValueError("reference sampling requires at least two rows/columns")
    middle = rng.choice(np.arange(1, size - 1), count - 2, replace=False)
    return np.sort(np.concatenate(([0, size - 1], middle))).astype(np.int32)


def make_reference(a, b, campaign, seed):
    m, k = a.shape
    n = b.shape[1]
    memory = campaign["memory"]
    full = (m * n <= memory["max_reference_output_elements"]
            and 2 * m * k * n <= memory["max_full_reference_flops"])
    rng = np.random.Generator(np.random.PCG64(seed + 911))
    rows = (np.arange(m, dtype=np.int32) if full else
            sample_indices(m, campaign["correctness"]["sample_rows"], rng))
    cols = (np.arange(n, dtype=np.int32) if full else
            sample_indices(n, campaign["correctness"]["sample_columns"], rng))
    ar = np.asarray(a[rows, :], dtype=np.float32)
    bc = np.asarray(b[:, cols], dtype=np.float32)
    ref = ar @ bc
    metadata = {"reference_scope": "full" if full else "sampled_cross_product",
                "reference_backend": "host_numpy_fp32", "reference_input_dtype": "quantized_bf16",
                "reference_accumulation_dtype": "float32", "all_k_used": k,
                "sample_rows": rows.tolist(), "sample_columns": cols.tolist(),
                "sample_count": int(ref.size),
                "normwise_denominator": float(np.linalg.norm(ar.astype(np.float64))
                                               * np.linalg.norm(bc.astype(np.float64)))}
    return ref, rows, cols, metadata


def numeric_metrics(output_sample, finite_output, reference, reference_info, campaign):
    got = np.asarray(output_sample, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    finite = bool(finite_output) and bool(np.isfinite(got).all()) and bool(np.isfinite(ref).all())
    result = {**reference_info, "finite": finite, "pass": False,
              "invalid_metrics": not finite}
    metric_names = ("relative_l2", "max_abs_error", "max_abs_reference", "rmse",
                    "mean_abs_error", "p50_abs_error", "p99_abs_error", "normwise_error")
    if not finite:
        result.update({name: None for name in metric_names})
        return result
    delta = got - ref
    absolute = np.abs(delta)
    floor = campaign["correctness"]["near_zero_denominator"]
    l2 = float(np.linalg.norm(delta))
    result.update(
        relative_l2=l2 / max(float(np.linalg.norm(ref)), floor),
        max_abs_error=float(absolute.max()), max_abs_reference=float(np.abs(ref).max()),
        rmse=float(np.sqrt(np.mean(delta * delta))), mean_abs_error=float(absolute.mean()),
        p50_abs_error=float(np.quantile(absolute, .5)), p99_abs_error=float(np.quantile(absolute, .99)),
        normwise_error=l2 / max(reference_info["normwise_denominator"], floor),
    )
    gate = campaign["correctness"]["gate"]
    result["pass"] = (result["relative_l2"] <= gate["relative_l2_max"]
                      and result["max_abs_error"] <= gate["max_abs_atol"]
                      + gate["max_abs_reference_rtol"] * result["max_abs_reference"])
    return result


def error_status(error, during="compile"):
    message = str(error).lower()
    if any(term in message for term in ("out of memory", "resource_exhausted", "vmem", "allocation failed")):
        return "oom"
    if isinstance(error, (KeyboardInterrupt, TimeoutError)):
        return "interrupted"
    return "compile_error" if during == "compile" else "execution_error"


def basic_statistics(samples):
    if not samples:
        return {"sample_count": 0}
    values = np.asarray([s["elapsed_ms"] for s in samples], np.float64)
    return {"sample_count": len(values), "mean_ms": float(values.mean()),
            "median_ms": float(np.median(values)),
            "std_ms": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min_ms": float(values.min()), "max_ms": float(values.max())}


def paired_comparison(reference, candidate, seed):
    left = {s["round"]: s["elapsed_ms"] for s in reference}
    right = {s["round"]: s["elapsed_ms"] for s in candidate}
    rounds = sorted(left.keys() & right.keys())
    if not rounds:
        return None
    a = np.asarray([left[r] for r in rounds], np.float64)
    b = np.asarray([right[r] for r in rounds], np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(rounds), size=(2000, len(rounds)))
    ratios = a[indices].mean(axis=1) / b[indices].mean(axis=1)
    return {"paired_rounds": rounds, "speedup_ratio_of_means": float(a.mean() / b.mean()),
            "time_reduction_fraction": float(1 - b.mean() / a.mean()),
            "speedup_ci95": np.quantile(ratios, [.025, .975]).tolist(),
            "method": "paired bootstrap of rounds, ratio of arithmetic means, 2000 resamples",
            "screen_results_are_not_confirmation": len(rounds) < 30}


class Runner:
    def __init__(self, args, campaign, journal):
        self.args, self.campaign, self.journal = args, campaign, journal
        self.started = time.monotonic()
        self.finished_groups = 0

    def check_deadline(self):
        if time.monotonic() - self.started > self.args.max_wall_seconds:
            raise TimeoutError("phase wall-time budget exhausted; restart only in a new run")

    def emit_error(self, context, error, during):
        self.journal.emit("error", **context, during=during,
                          status=error_status(error, during), error_type=type(error).__name__,
                          message=str(error), traceback=traceback.format_exc())

    def compile_entries(self, group, a, b):
        shape = tuple(group["shape"][d] for d in ("m", "k", "n"))
        entries = []
        for arm in group["arms"]:
            self.check_deadline()
            context = {"group_id": group["group_id"], "arm_id": arm["arm_id"]}
            try:
                fn = make_matmul(arm["algorithm"], shape, tuple(group["tile"]),
                                 variant=arm["variant"], interpret=False,
                                 vmem_limit_bytes=(None if arm["algorithm"] == "native" else
                                                   self.campaign["memory"]["kernel_vmem_limit_mib"] * 1024**2))
            except Exception as error:
                self.emit_error(context, error, "compile")
                for scope in group["scopes"]:
                    entries.append({**context, **arm, "scope": scope,
                                    "error": error_status(error), "error_message": str(error)})
                continue
            for scope in group["scopes"]:
                self.check_deadline()
                entry = {**context, **arm, "scope": scope, "fn": fn, "metadata": fn.metadata}
                start = time.perf_counter_ns()
                try:
                    if scope == "call":
                        specs = (jax.ShapeDtypeStruct(a.shape, a.dtype),
                                 jax.ShapeDtypeStruct(b.shape, b.dtype))
                        executable = jax.jit(fn).lower(*specs).compile()
                    else:
                        mp, kp, nn = fn.metadata["padded_shape_mkn"]
                        specs = (jax.ShapeDtypeStruct((mp, kp), a.dtype),
                                 jax.ShapeDtypeStruct((kp, nn), b.dtype))
                        executable = jax.jit(fn.kernel).lower(*specs).compile()
                    entry["executable"] = executable
                    self.journal.emit("compilation", **context, scope=scope, status="ok",
                                      compile_ms=(time.perf_counter_ns() - start) / 1e6,
                                      kernel_metadata=fn.metadata)
                except Exception as error:
                    entry.update(error=error_status(error), error_message=str(error))
                    self.emit_error({**context, "scope": scope}, error, "compile")
                entries.append(entry)
        return entries

    def execute_case(self, group, input_case, entries, a_host, b_host):
        self.check_deadline()
        case_id = f'{group["group_id"]}__{input_case["distribution"]}__{input_case["seed"]}'
        context = {"group_id": group["group_id"], "case_id": case_id,
                   "shape_id": group["shape"]["id"], **input_case}
        self.journal.emit("case_start", **context, input_fingerprint=input_fingerprint(a_host, b_host))
        reference, rows, cols, ref_info = make_reference(a_host, b_host, self.campaign, input_case["seed"])
        a, b = jax.device_put(a_host), jax.device_put(b_host)
        jax.block_until_ready((a, b))
        row_indices, col_indices = jnp.asarray(rows), jnp.asarray(cols)

        @jax.jit
        def check_output(output):
            return (jnp.all(jnp.isfinite(output)),
                    output[row_indices[:, None], col_indices[None, :]])

        prepared = {}
        live_entries = []
        for original in entries:
            self.check_deadline()
            entry = {**original, "samples": [], "correctness": None}
            live_entries.append(entry)
            if "error" in entry:
                continue
            detail = {**context, "arm_id": entry["arm_id"], "scope": entry["scope"]}
            try:
                if entry["scope"] == "call":
                    operands = (a, b)
                else:
                    key = tuple(entry["metadata"]["padded_shape_mkn"])
                    if key not in prepared:
                        prep_start = time.perf_counter_ns()
                        prepared[key] = jax.jit(entry["fn"].prepare)(a, b)
                        jax.block_until_ready(prepared[key])
                        self.journal.emit("preparation", **context, padded_shape_mkn=list(key),
                                          compile_and_prepare_ms=(time.perf_counter_ns() - prep_start) / 1e6,
                                          timing_role="excluded from prepared_kernel; included in call executable")
                    operands = prepared[key]
                entry["operands"] = operands
                output = entry["executable"](*operands)
                output.block_until_ready()
                finite, sampled = jax.device_get(check_output(output))
                metrics = numeric_metrics(sampled, finite, reference, ref_info, self.campaign)
                entry["correctness"] = metrics
                self.journal.emit("correctness", **detail, metrics=metrics)
                output.delete()
                del output
            except Exception as error:
                entry.update(error=error_status(error, "execute"), error_message=str(error))
                self.emit_error(detail, error, "execute")
        timing = group["timing"]
        if timing:
            policy = self.campaign["timing"][timing]
            active = [entry for entry in live_entries if "error" not in entry]
            rng = np.random.Generator(np.random.PCG64(input_case["seed"] + 17))
            base = [active[int(i)] for i in rng.permutation(len(active))]
            for warmup in range(policy["warmups"]):
                for entry in base[warmup % max(1, len(base)):] + base[:warmup % max(1, len(base))]:
                    if "error" in entry:
                        continue
                    self.check_deadline()
                    try:
                        output = entry["executable"](*entry["operands"])
                        output.block_until_ready()
                        output.delete()
                        del output
                    except Exception as error:
                        entry.update(error=error_status(error, "execute"), error_message=str(error))
                        self.emit_error({**context, "arm_id": entry["arm_id"], "scope": entry["scope"]},
                                        error, "execute")
            for round_id in range(policy["repeats"]):
                shift = round_id % max(1, len(base))
                for position, entry in enumerate(base[shift:] + base[:shift]):
                    if "error" in entry:
                        continue
                    self.check_deadline()
                    try:
                        start = time.perf_counter_ns()
                        output = entry["executable"](*entry["operands"])
                        output.block_until_ready()
                        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                        sample = {"round": round_id, "position": position, "elapsed_ms": elapsed_ms,
                                  "start_perf_counter_ns": start}
                        entry["samples"].append(sample)
                        output.delete()
                        del output
                        self.journal.emit("sample", **context, arm_id=entry["arm_id"],
                                          scope=entry["scope"], **sample)
                    except Exception as error:
                        entry.update(error=error_status(error, "execute"), error_message=str(error))
                        self.emit_error({**context, "arm_id": entry["arm_id"], "scope": entry["scope"]},
                                        error, "execute")
        for entry in live_entries:
            metrics = entry["correctness"]
            status = entry.get("error") or ("ok" if metrics and metrics["pass"] else "numerical_failure")
            eligible = status == "ok"
            comparisons = []
            for other in live_entries:
                is_baseline = (other["arm_id"] in ("native_xla", "cubic_basic")
                               or (other["algorithm"] == "cubic_quadrant"
                                   and other["variant"] == entry["variant"])
                               or (other["algorithm"] == entry["algorithm"] and other["variant"] == "plain"))
                if is_baseline and other is not entry and other["scope"] == entry["scope"]:
                    pair = paired_comparison(other["samples"], entry["samples"], input_case["seed"])
                    if pair:
                        pair.update(reference_arm=other["arm_id"],
                                    valid_numerical_comparison=bool(eligible and not other.get("error")
                                                                  and other["correctness"]
                                                                  and other["correctness"]["pass"]))
                        comparisons.append(pair)
            self.journal.emit("case_result", **context, arm_id=entry["arm_id"], scope=entry["scope"],
                              algorithm=entry["algorithm"], variant=entry["variant"],
                              status=status, eligible_for_speedup_claim=eligible,
                              error_message=entry.get("error_message"), kernel_metadata=entry.get("metadata"),
                              correctness=metrics, timing=basic_statistics(entry["samples"]),
                              comparisons=comparisons)
            entry.pop("operands", None)
        del prepared, live_entries, a, b, reference, row_indices, col_indices, check_output
        gc.collect()

    def run_group(self, group):
        shape = tuple(group["shape"][d] for d in ("m", "k", "n"))
        estimate = memory_estimate(shape, group["tile"], group["arms"], group["scopes"])
        self.journal.emit("memory_preflight", group_id=group["group_id"], **estimate)
        if estimate["estimated_live_device_bytes"] > self.campaign["memory"]["estimated_live_device_budget_gib"] * 1024**3:
            for input_case in group["inputs"]:
                for arm in group["arms"]:
                    for scope in group["scopes"]:
                        self.journal.emit("case_result", group_id=group["group_id"], **input_case,
                                          arm_id=arm["arm_id"], scope=scope, status="skipped_memory_preflight",
                                          eligible_for_speedup_claim=False, memory_estimate=estimate)
            return
        entries = None
        try:
            for input_case in group["inputs"]:
                self.check_deadline()
                try:
                    a_host, b_host = generate_inputs(shape, **input_case)
                except Exception as error:
                    self.emit_error({"group_id": group["group_id"], **input_case}, error, "input_generation")
                    for arm in group["arms"]:
                        for scope in group["scopes"]:
                            self.journal.emit("case_result", group_id=group["group_id"], **input_case,
                                              arm_id=arm["arm_id"], scope=scope, status="input_error",
                                              eligible_for_speedup_claim=False, error_message=str(error))
                    continue
                if entries is None:
                    entries = self.compile_entries(group, a_host, b_host)
                self.execute_case(group, input_case, entries, a_host, b_host)
                del a_host, b_host
                gc.collect()
        finally:
            del entries
            jax.clear_caches()
            gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "N1", "N2", "N3", "N4"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-identity", type=Path)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=21600,
                        help="Checked before every compile/call; use an outer process timeout for hung drivers.")
    args = parser.parse_args(argv)
    if args.max_wall_seconds <= 0:
        parser.error("--max-wall-seconds must be positive")
    if args.phase != "smoke" and args.expected_identity is None:
        parser.error("N1--N4 require --expected-identity from the qualified smoke run")
    args.campaign = args.campaign.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir
    journal = Journal(out, args.phase)
    started = time.monotonic()
    completed = False
    planned = []
    done = []
    error_summary = None
    try:
        campaign = json.loads(args.campaign.read_text())
        shape_path = args.campaign.parent / campaign["shape_manifest"]
        distribution_path = args.campaign.parent / campaign.get("distribution_manifest", "distributions_v1.json")
        manifest = json.loads(shape_path.read_text())
        distributions = json.loads(distribution_path.read_text())
        if distributions["generator_id"] != "host_numpy_pcg64_v1":
            raise ValueError("unsupported input generator version")
        if campaign["precision"]["native_precision"] != "DEFAULT":
            raise ValueError("kernels_v001 uses DEFAULT BF16 dots; campaign precision does not match")
        if campaign["device"]["target"] != "v5e" or campaign["device"].get("allow_v6e"):
            raise ValueError("this runner is frozen for v5e only")
        source_manifest = snapshot_sources(out, args.campaign, shape_path, distribution_path)
        journal.emit("run_start", allocation_id=args.allocation_id, source_manifest=source_manifest,
                     argv=sys.argv, max_wall_seconds=args.max_wall_seconds)
        if "jax" in sys.modules:
            raise RuntimeError("run in a fresh Python process: JAX was imported before fixed environment setup")
        os.environ.update(FIXED_ENVIRONMENT)
        global jax, jnp, np, ml_dtypes, make_matmul
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        from strassen_mm.kernels_v001 import make_matmul, enable_qualified_mosaic_v7_compat
        compatibility = enable_qualified_mosaic_v7_compat()
        environment = capture_environment(args, campaign, compatibility)
        exclusive_json(out / "environment.json", environment)
        identity_check = verify_identity(environment, args.expected_identity, campaign)
        journal.emit("identity_check", **identity_check)
        planned = planned_groups(campaign, manifest, args.phase)
        exclusive_json(out / "planned_cases.json", planned)
        runner = Runner(args, campaign, journal)
        print(f"{args.phase}: {len(planned)} groups; {environment['identity']['device_kind']}; allocation {args.allocation_id}", flush=True)
        for index, group in enumerate(planned):
            runner.check_deadline()
            print(f"[{index + 1}/{len(planned)}] {group['group_id']} starting", flush=True)
            journal.emit("group_start", group_id=group["group_id"], index=index + 1, total=len(planned))
            runner.run_group(group)
            done.append(group["group_id"])
            journal.emit("group_complete", group_id=group["group_id"], index=index + 1, total=len(planned))
            print(f"[{index + 1}/{len(planned)}] complete; {dict(journal.status_counts)}", flush=True)
        completed = True
    except BaseException as error:
        error_summary = {"type": type(error).__name__, "message": str(error),
                         "status": error_status(error, "execute")}
        journal.emit("run_error", **error_summary, traceback=traceback.format_exc())
        print(f"{args.phase} stopped: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    finally:
        summary = {"phase": args.phase, "completed": completed,
                   "status": "completed" if completed else "failed_or_interrupted",
                   "finished_utc": utc_now(), "wall_seconds": time.monotonic() - started,
                   "completed_groups": done, "planned_group_count": len(planned),
                   "not_completed_group_ids": [g["group_id"] for g in planned if g["group_id"] not in done],
                   "case_status_counts": dict(journal.status_counts), "error": error_summary,
                   "scientific_success": "Not inferred from phase completion; inspect numerical eligibility and timings.",
                   "n4_scope": "call correctness only; no timing" if args.phase == "N4" else None}
        journal.emit("run_complete", **summary)
        journal.close()
        exclusive_json(out / "summary.json", summary)
        files = {str(p.relative_to(out)): digest_file(p) for p in sorted(out.rglob("*")) if p.is_file()}
        exclusive_json(out / "artifact_manifest.json", {"sha256": files, "sealed_utc": utc_now()})
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
