"""Full-model gate for all-layer sequences of equivalent Strassen formulas.

Every candidate uses Strassen at both MLP GEMM sites in all 32 layers.  The
only change is which algebraically equivalent seven-product formula is assigned
to each layer.  This tests whether rotating weakly correlated rounding patterns
prevents the depth amplification seen with a single formula.
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
import benchmark_checkpoint_permutations as permutation_bench
import benchmark_common as bench

import jax
import jax.numpy as jnp


OUTPUT = Path("/content/results/strassen_checkpoint_variant_sequences.jsonl")
EXECUTABLE_CONFIGS = {
    "c00": ("classical", 0, 0),
    "c11": ("classical", 1, 1),
    "c22": ("classical", 2, 2),
    "c33": ("classical", 3, 3),
    "c47": ("classical", 4, 7),
    "d10": ("dual", 1, 0),
}
SEQUENCES = {
    "classical_cycle": tuple(
        ("c00", "c11", "c22", "c33")[layer % 4]
        for layer in range(model_bench.NUM_LAYERS)
    ),
    "classical_selected_cycle": tuple(
        ("c00", "c11", "c22", "c47")[layer % 4]
        for layer in range(model_bench.NUM_LAYERS)
    ),
    "classical_dual_alternating": tuple(
        ("c33", "d10")[layer % 2]
        for layer in range(model_bench.NUM_LAYERS)
    ),
}


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("CHECKPOINT_SEQUENCE_JSON " + line, flush=True)


def compile_variant(name, config, state, params):
    formula, up_permutation, down_permutation = config
    permutation_bench.FORMULA_VARIANT = formula
    function = permutation_bench.make_layer(up_permutation, down_permutation)
    lowered = jax.jit(function).lower(state, params)
    executable = lowered.compile()
    emit({
        "kind": "compile_variant",
        "name": name,
        "formula": formula,
        "up_permutation": up_permutation,
        "down_permutation": down_permutation,
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
        "jax": jax.__version__,
        "quality_only": True,
        "all_layers_use_both_strassen_sites": True,
        "executable_configs": EXECUTABLE_CONFIGS,
        "sequences": SEQUENCES,
        "quality_gate": {
            "post_norm_hidden_l2_relative_max": 0.02,
            "post_norm_hidden_maxnorm_relative_max": 0.02,
            "logit_l2_relative_max": 0.02,
            "logit_maxnorm_relative_max": 0.02,
            "top1_agreement_min": 0.95,
            "absolute_loss_delta_max": 0.01,
        },
    })

    model_bench.OUTPUT = OUTPUT
    token_ids = model_bench.make_tokens()
    checkpoint = model_bench.Checkpoint(model_bench.download_checkpoint())
    embeddings = checkpoint.tensor("model.embed_tokens.weight")
    initial = embeddings[token_ids].reshape(
        layer_bench.TOKENS, layer_bench.MODEL_DIM
    )
    states = {"native": jnp.asarray(initial)}
    states.update({name: jnp.asarray(initial) for name in SEQUENCES})
    rope_cos, rope_sin = layer_bench.rope_values()
    first_params = model_bench.load_layer(checkpoint, 0, rope_cos, rope_sin)

    started = time.perf_counter()
    native_lowered = jax.jit(layer_bench.make_layer("matched_native")).lower(
        states["native"], first_params
    )
    native_executable = native_lowered.compile()
    emit({
        "kind": "compile_native",
        "lowered_audit": layer_bench.lowered_audit(native_lowered),
    })
    candidate_executables = {
        name: compile_variant(name, config, states["native"], first_params)
        for name, config in EXECUTABLE_CONFIGS.items()
    }
    emit({"kind": "compile", "compile_s": time.perf_counter() - started})

    all_finite = {name: True for name in SEQUENCES}
    for layer in range(model_bench.NUM_LAYERS):
        params = (
            first_params
            if layer == 0
            else model_bench.load_layer(checkpoint, layer, rope_cos, rope_sin)
        )
        states["native"] = native_executable(states["native"], params)
        for name, sequence in SEQUENCES.items():
            states[name] = candidate_executables[sequence[layer]](
                states[name], params
            )
        jax.block_until_ready(states)
        for name in SEQUENCES:
            error = bench.device_error(states[name], states["native"])
            finite = bool(jax.device_get(jnp.all(jnp.isfinite(states[name]))))
            all_finite[name] = all_finite[name] and finite
            emit({
                "kind": "layer_accuracy",
                "sequence": name,
                "layer": layer,
                "variant": SEQUENCES[name][layer],
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
        for name in SEQUENCES
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
    for name in SEQUENCES:
        logits, loss = logits_executable(states[name], lm_head, targets)
        jax.block_until_ready((logits, loss))
        logits_error = bench.device_error(logits, native_logits)
        top1_agreement = float(jax.device_get(jnp.mean(
            jnp.argmax(logits, axis=-1) == jnp.argmax(native_logits, axis=-1)
        )))
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
        if passes:
            passing.append(name)
        emit({
            "kind": "sequence_accuracy",
            "sequence": name,
            "final_hidden_post_norm": hidden_error,
            "logits": logits_error,
            "top1_agreement": top1_agreement,
            "native_next_token_loss": native_loss,
            "candidate_next_token_loss": loss,
            "absolute_loss_delta": loss_delta,
            "passes": passes,
        })
        del logits
        gc.collect()

    success = bool(passing)
    emit({"kind": "final", "success": success, "passing_sequences": passing})
    if not success:
        raise RuntimeError("no all-layer formula sequence passed the quality gate")


if __name__ == "__main__":
    main()
