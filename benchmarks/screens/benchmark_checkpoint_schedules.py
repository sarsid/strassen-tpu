"""Full-checkpoint accuracy frontier for sparse Strassen layer schedules.

Latency values are projections from the separately measured resident layer-0
executables.  The streamed 32-layer pass is used only for hidden-state, logit,
top-1, and next-token-loss accuracy.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import time


os.environ["STRASSEN_VMEM_PROFILE"] = "max48"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import benchmark_checkpoint_layer as layer_bench
import benchmark_checkpoint_model as model_bench
import benchmark_common as bench

import jax
import jax.numpy as jnp


OUTPUT = Path("/content/results/strassen_checkpoint_schedules.jsonl")
CALIBRATION_SOURCE_SHA256 = None
LAYER_RESULT_SHA256 = (
    "4ef55cec132662d060c16d3fbc85b0602e97b3528741aa2c930f2ddf8da5c9b6"
)
MEASURED_LAYER_MS = {
    "native": 31.806857,
    "both": 31.2128416,
    "up": 31.65455005,
}

# Uniform schedules place each selected layer at the midpoint of an equal
# interval.  This avoids choosing layers after inspecting their errors.
SCHEDULES = {
    "both_1": {"both": frozenset({16}), "up": frozenset()},
    "both_2": {"both": frozenset({8, 24}), "up": frozenset()},
    "both_4": {"both": frozenset({4, 12, 20, 28}), "up": frozenset()},
    "both_8": {
        "both": frozenset({2, 6, 10, 14, 18, 22, 26, 30}),
        "up": frozenset(),
    },
    "both_16": {
        "both": frozenset(range(1, model_bench.NUM_LAYERS, 2)),
        "up": frozenset(),
    },
    "up_all": {
        "both": frozenset(),
        "up": frozenset(range(model_bench.NUM_LAYERS)),
    },
}


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("CHECKPOINT_SCHEDULE_JSON " + line, flush=True)


def policy_for(schedule, layer):
    if layer in schedule["both"]:
        return "both"
    if layer in schedule["up"]:
        return "up"
    return "native"


def projected_latency(schedule):
    counts = {"native": 0, "both": 0, "up": 0}
    for layer in range(model_bench.NUM_LAYERS):
        counts[policy_for(schedule, layer)] += 1
    native_total = model_bench.NUM_LAYERS * MEASURED_LAYER_MS["native"]
    candidate_total = sum(
        counts[name] * MEASURED_LAYER_MS[name] for name in counts
    )
    return {
        "policy_counts": counts,
        "projected_resident_ms": candidate_total,
        "projected_speedup": native_total / candidate_total,
        "projected_saving_ms": native_total - candidate_total,
        "source_layer_result_sha256": LAYER_RESULT_SHA256,
    }


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
        "quality_only": True,
        "calibration_source_sha256": CALIBRATION_SOURCE_SHA256,
        "schedules": {
            name: {
                "both_layers": sorted(value["both"]),
                "up_layers": sorted(value["up"]),
            }
            for name, value in SCHEDULES.items()
        },
        "quality_gate": {
            "post_norm_hidden_l2_relative_max": 0.02,
            "post_norm_hidden_maxnorm_relative_max": 0.02,
            "logit_l2_relative_max": 0.02,
            "logit_maxnorm_relative_max": 0.02,
            "top1_agreement_min": 0.95,
            "absolute_loss_delta_max": 0.01,
        },
    })

    # Reuse the cached pinned shards. Redirect helper progress records here.
    model_bench.OUTPUT = OUTPUT
    token_ids = model_bench.make_tokens()
    checkpoint = model_bench.Checkpoint(model_bench.download_checkpoint())
    embeddings = checkpoint.tensor("model.embed_tokens.weight")
    initial = embeddings[token_ids].reshape(
        layer_bench.TOKENS, layer_bench.MODEL_DIM
    )
    states = {"native": jnp.asarray(initial)}
    states.update({name: jnp.asarray(initial) for name in SCHEDULES})
    rope_cos, rope_sin = layer_bench.rope_values()
    first_params = model_bench.load_layer(
        checkpoint, 0, rope_cos, rope_sin
    )

    started = time.perf_counter()
    executables = {
        "native": jax.jit(layer_bench.make_layer("matched_native"))
        .lower(states["native"], first_params)
        .compile(),
        "both": jax.jit(layer_bench.make_layer("both"))
        .lower(states["native"], first_params)
        .compile(),
        "up": jax.jit(layer_bench.make_layer("mlp_up"))
        .lower(states["native"], first_params)
        .compile(),
    }
    emit({"kind": "compile", "compile_s": time.perf_counter() - started})

    all_finite = {name: True for name in SCHEDULES}
    for layer in range(model_bench.NUM_LAYERS):
        params = (
            first_params
            if layer == 0
            else model_bench.load_layer(
                checkpoint, layer, rope_cos, rope_sin
            )
        )
        states["native"] = executables["native"](states["native"], params)
        for name, schedule in SCHEDULES.items():
            policy = policy_for(schedule, layer)
            states[name] = executables[policy](states[name], params)
        jax.block_until_ready(states)
        for name in SCHEDULES:
            error = bench.device_error(states[name], states["native"])
            finite = bool(
                jax.device_get(jnp.all(jnp.isfinite(states[name])))
            )
            all_finite[name] = all_finite[name] and finite
            emit({
                "kind": "layer_accuracy",
                "schedule": name,
                "layer": layer,
                "finite": finite,
                "vs_native": error,
            })
        del params
        if layer == 0:
            del first_params
        gc.collect()

    final_norm = jnp.asarray(checkpoint.tensor("model.norm.weight"))
    for name in states:
        states[name] = layer_bench.rms_norm(states[name], final_norm)
    jax.block_until_ready(states)
    hidden_errors = {
        name: bench.device_error(states[name], states["native"])
        for name in SCHEDULES
    }

    lm_head = model_bench.device_matrix(checkpoint, "lm_head.weight")
    targets = jnp.asarray(token_ids[:, 1:].reshape(-1))
    logits_executable = jax.jit(model_bench.logits_and_loss).lower(
        states["native"], lm_head, targets
    ).compile()
    native_logits, native_loss = logits_executable(
        states["native"], lm_head, targets
    )
    jax.block_until_ready((native_logits, native_loss))
    native_loss = float(jax.device_get(native_loss))

    passing = []
    for name, schedule in SCHEDULES.items():
        logits, loss = logits_executable(states[name], lm_head, targets)
        jax.block_until_ready((logits, loss))
        logits_error = bench.device_error(logits, native_logits)
        top1_agreement = float(
            jax.device_get(
                jnp.mean(
                    jnp.argmax(logits, axis=-1)
                    == jnp.argmax(native_logits, axis=-1)
                )
            )
        )
        loss = float(jax.device_get(loss))
        loss_delta = abs(loss - native_loss)
        hidden_error = hidden_errors[name]
        passes = (
            all_finite[name]
            and hidden_error["l2_relative"] <= 0.02
            and hidden_error["maxnorm_relative"] <= 0.02
            and logits_error["l2_relative"] <= 0.02
            and logits_error["maxnorm_relative"] <= 0.02
            and top1_agreement >= 0.95
            and loss_delta <= 0.01
        )
        performance = projected_latency(schedule)
        if passes and performance["projected_speedup"] > 1.0:
            passing.append(name)
        emit({
            "kind": "schedule_accuracy",
            "schedule": name,
            "final_hidden_post_norm": hidden_error,
            "logits": logits_error,
            "top1_agreement": top1_agreement,
            "native_next_token_loss": native_loss,
            "candidate_next_token_loss": loss,
            "absolute_loss_delta": loss_delta,
            "passes": passes,
            "projected_performance": performance,
        })
        del logits
        gc.collect()

    success = bool(passing)
    emit({"kind": "final", "success": success, "passing_schedules": passing})
    if not success:
        raise RuntimeError("no sparse checkpoint schedule passed")


if __name__ == "__main__":
    main()
