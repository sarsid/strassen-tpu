"""Predeclared task-level gate on a broad public corpus (WikiText-2 test).

Registered BEFORE evaluation (see the git commit adding this file). The
tensor-error gate measured worst-case hidden/logit perturbation; this gate
measures what deployment observes. Candidates are evaluated with completely
frozen configurations — no calibration on this corpus.

Corpus: WikiText-2 raw test split, concatenated, tokenized with the pinned
tokenizer, first 8x1024 contiguous tokens (no padding).

Streams (all 32 real layers + final norm + LM head, quality only):
- matched native (reference)
- classical Strassen, both MLP sites (the promoted kernel, no hybrid)
- outlier-panel hybrid with permutations and budgets FROZEN from the
  calibration-text run (results/hybrid_iperms_graded.npy).

Predeclared pass thresholds, per candidate, chosen before any run:
- absolute next-token loss delta <= 0.01
- greedy top-1 agreement >= 99.0%
- mean next-token KL(native || candidate) <= 0.005 nats
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import time

import benchmark_checkpoint_layer as layer_bench
import benchmark_checkpoint_model as model_bench
import benchmark_checkpoint_model_hybrid as hybrid_bench
import benchmark_common as bench
import strassen_pallas as sp

import jax
import jax.numpy as jnp
import numpy as np


OUTPUT = Path("/content/results/strassen_checkpoint_task_gate.jsonl")
IPERM_FILE = "/content/hybrid_iperms_frozen.npy"
BUDGET_FILE = "/content/hybrid_budgets_frozen.npy"
LOSS_DELTA_MAX = 0.01
TOP1_MIN = 0.99
KL_MAX = 0.005
NUM_LAYERS = model_bench.NUM_LAYERS


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("TASK_GATE_JSON " + line, flush=True)


model_bench.emit = emit
layer_bench.emit = emit
hybrid_bench.emit = emit


def wikitext_tokens():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
    )
    text = "\n".join(row["text"] for row in dataset)
    tokenizer = AutoTokenizer.from_pretrained(
        layer_bench.REPOSITORY, revision=layer_bench.REVISION,
        use_fast=True, token=False,
    )
    ids = tokenizer(text, add_special_tokens=False, return_tensors="np")[
        "input_ids"
    ][0]
    needed = layer_bench.BATCH * layer_bench.SEQUENCE
    if ids.shape[0] < needed:
        raise RuntimeError(f"corpus too small: {ids.shape[0]} < {needed}")
    token_ids = ids[:needed].reshape(
        layer_bench.BATCH, layer_bench.SEQUENCE
    ).astype(np.int32)
    emit({
        "kind": "tokens",
        "corpus": "wikitext-2-raw-v1/test first contiguous tokens",
        "shape": list(token_ids.shape),
        "sha256": hashlib.sha256(token_ids.tobytes()).hexdigest(),
        "unique_tokens": int(np.unique(token_ids).size),
    })
    return token_ids


@jax.jit
def kl_chunk(native_logits, candidate_logits):
    lp = jax.nn.log_softmax(native_logits.astype(jnp.float32), axis=-1)
    lq = jax.nn.log_softmax(candidate_logits.astype(jnp.float32), axis=-1)
    return jnp.sum(jnp.exp(lp) * (lp - lq), axis=-1)


def mean_kl(native_logits, candidate_logits, chunks=8):
    n = native_logits.shape[0]
    step = (n + chunks - 1) // chunks
    values = []
    for start in range(0, n, step):
        values.append(
            kl_chunk(
                native_logits[start:start + step],
                candidate_logits[start:start + step],
            )
        )
    return float(jnp.mean(jnp.concatenate(values)))


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
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "predeclared_gate": {
            "abs_loss_delta_max": LOSS_DELTA_MAX,
            "top1_agreement_min": TOP1_MIN,
            "mean_kl_max_nats": KL_MAX,
        },
        "candidates": ["strassen_control", "hybrid_frozen"],
        "calibration": "frozen from calibration-text run; none on this corpus",
    })

    token_ids = wikitext_tokens()
    paths = model_bench.download_checkpoint()
    checkpoint = model_bench.Checkpoint(paths)
    embeddings = checkpoint.tensor("model.embed_tokens.weight")
    initial = embeddings[token_ids].reshape(
        layer_bench.TOKENS, layer_bench.MODEL_DIM
    )
    rope_cos, rope_sin = layer_bench.rope_values()

    frozen_iperms = np.load(IPERM_FILE)
    frozen_budgets = np.load(BUDGET_FILE)

    native_fn = hybrid_bench.make_layer(strassen=False)
    control_fn = hybrid_bench.make_layer(strassen=True)
    hidden = {
        "native": jnp.asarray(initial),
        "control": jnp.asarray(initial),
        "hybrid": jnp.asarray(initial),
    }
    native_exec = control_exec = None
    hybrid_execs = {}

    for index in range(NUM_LAYERS):
        params = model_bench.load_layer(checkpoint, index, rope_cos, rope_sin)
        iperm = jnp.asarray(frozen_iperms[index].astype(np.int32))
        budget = int(frozen_budgets[index])
        params_hybrid = hybrid_bench.permute_intermediate(params, iperm)
        jax.block_until_ready(params_hybrid.gate_up)

        if native_exec is None:
            started = time.perf_counter()
            native_exec = jax.jit(native_fn).lower(
                hidden["native"], params
            ).compile()
            control_exec = jax.jit(control_fn).lower(
                hidden["control"], params
            ).compile()
            emit({"kind": "compile", "compile_s": time.perf_counter() - started})
        if budget not in hybrid_execs:
            fn = hybrid_bench.make_layer(
                strassen=True, down_panels=tuple(range(budget))
            )
            hybrid_execs[budget] = jax.jit(fn).lower(
                hidden["hybrid"], params_hybrid
            ).compile()

        hidden["native"] = native_exec(hidden["native"], params)
        hidden["control"] = control_exec(hidden["control"], params)
        hidden["hybrid"] = hybrid_execs[budget](hidden["hybrid"], params_hybrid)
        jax.block_until_ready(tuple(hidden.values()))
        if index % 8 == 7:
            emit({"kind": "progress", "layer": index})
        del params, params_hybrid
        gc.collect()

    final_norm = jnp.asarray(checkpoint.tensor("model.norm.weight"))
    for key in hidden:
        hidden[key] = layer_bench.rms_norm(hidden[key], final_norm)
    lm_head = model_bench.device_matrix(checkpoint, "lm_head.weight")
    targets = jnp.asarray(token_ids[:, 1:].reshape(-1))
    logits_exec = jax.jit(model_bench.logits_and_loss).lower(
        hidden["native"], lm_head, targets
    ).compile()

    logits, loss = {}, {}
    for key in hidden:
        logits[key], loss[key] = logits_exec(hidden[key], lm_head, targets)
    jax.block_until_ready(tuple(logits.values()))

    verdicts = {}
    for name, key in (("strassen_control", "control"), ("hybrid_frozen", "hybrid")):
        top1 = float(jax.device_get(jnp.mean(
            jnp.argmax(logits["native"], axis=-1)
            == jnp.argmax(logits[key], axis=-1)
        )))
        loss_delta = abs(float(loss[key]) - float(loss["native"]))
        kl = mean_kl(logits["native"], logits[key])
        passes = (
            loss_delta <= LOSS_DELTA_MAX
            and top1 >= TOP1_MIN
            and kl <= KL_MAX
        )
        verdicts[name] = passes
        emit({
            "kind": "task_metrics",
            "candidate": name,
            "native_loss": float(loss["native"]),
            "candidate_loss": float(loss[key]),
            "abs_loss_delta": loss_delta,
            "top1_agreement": top1,
            "mean_kl_nats": kl,
            "passes": passes,
        })

    emit({"kind": "final", "success": all(verdicts.values()), "verdicts": verdicts})
    if not all(verdicts.values()):
        raise RuntimeError(f"task gate: {verdicts}")


if __name__ == "__main__":
    main()
