"""Measure whether equivalent layer errors can cancel across depth."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import time


os.environ["STRASSEN_VMEM_PROFILE"] = "max48"

import benchmark_checkpoint_layer as layer_bench
import benchmark_checkpoint_permutations as permutation_bench
import benchmark_common as bench

import jax
import jax.numpy as jnp


OUTPUT = Path("/content/results/strassen_checkpoint_error_correlation.jsonl")
VARIANTS = {
    "classical_p0": ("classical", 0, 0),
    "classical_p1": ("classical", 1, 1),
    "classical_p2": ("classical", 2, 2),
    "classical_p3": ("classical", 3, 3),
    "classical_selected": ("classical", 4, 7),
    "dual_best": ("dual", 1, 0),
}


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("CHECKPOINT_CORRELATION_JSON " + line, flush=True)


@jax.jit
def error_relationship(first, second, reference):
    first_error = first.astype(jnp.float32) - reference.astype(jnp.float32)
    second_error = second.astype(jnp.float32) - reference.astype(jnp.float32)
    first_norm = jnp.linalg.norm(first_error)
    second_norm = jnp.linalg.norm(second_error)
    denominator = jnp.maximum(
        first_norm * second_norm, jnp.finfo(jnp.float32).tiny
    )
    cosine = jnp.vdot(first_error, second_error) / denominator
    sum_ratio = jnp.linalg.norm(first_error + second_error) / jnp.maximum(
        first_norm + second_norm, jnp.finfo(jnp.float32).tiny
    )
    return cosine, sum_ratio


def compile_variant(name, formula, up_permutation, down_permutation, x, params):
    permutation_bench.FORMULA_VARIANT = formula
    function = permutation_bench.make_layer(up_permutation, down_permutation)
    started = time.perf_counter()
    lowered = jax.jit(function).lower(x, params)
    executable = lowered.compile()
    emit({
        "kind": "compile",
        "name": name,
        "formula": formula,
        "up_permutation": up_permutation,
        "down_permutation": down_permutation,
        "compile_s": time.perf_counter() - started,
        "lowered_audit": layer_bench.lowered_audit(lowered),
    })
    return executable


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata",
        "repository": layer_bench.REPOSITORY,
        "revision": layer_bench.REVISION,
        "device": jax.devices()[0].device_kind,
        "variants": VARIANTS,
    })
    x, params = layer_bench.load_inputs()
    native = jax.jit(permutation_bench.make_layer(None, None)).lower(
        x, params
    ).compile()
    reference = native(x, params)
    outputs = {}
    for name, (formula, up_permutation, down_permutation) in VARIANTS.items():
        executable = compile_variant(
            name, formula, up_permutation, down_permutation, x, params
        )
        outputs[name] = executable(x, params)
        jax.block_until_ready(outputs[name])
        emit({
            "kind": "accuracy",
            "name": name,
            "vs_native": bench.device_error(outputs[name], reference),
        })

    relationships = []
    for first, second in itertools.combinations(VARIANTS, 2):
        cosine, sum_ratio = jax.device_get(
            error_relationship(outputs[first], outputs[second], reference)
        )
        record = {
            "kind": "relationship",
            "first": first,
            "second": second,
            "error_cosine": float(cosine),
            "error_sum_norm_ratio": float(sum_ratio),
        }
        relationships.append(record)
        emit(record)
    best = min(relationships, key=lambda record: record["error_cosine"])
    emit({
        "kind": "final",
        "lowest_cosine_pair": [best["first"], best["second"]],
        "lowest_cosine": best["error_cosine"],
        "useful_cancellation": best["error_cosine"] < 0.0,
    })


if __name__ == "__main__":
    main()
