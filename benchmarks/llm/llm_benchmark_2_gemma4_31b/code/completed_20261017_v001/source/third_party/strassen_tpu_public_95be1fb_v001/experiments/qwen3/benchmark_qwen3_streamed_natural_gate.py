"""Streamed natural-corpus quality gate for the promoted product-aware policy.

The existing streamed task gate repeats eight hand-written sentences (97
unique token IDs, native perplexity about 1.09).  Its saturated next-token
distributions make loss-delta and KL insensitive almost by construction, so
it demonstrates agreement on one frozen repetitive corpus, not language
quality.  This gate re-runs the identical frozen policy over WikiText-2 test
text -- the standard held-out corpus in the quantization literature -- and
gates on distributional metrics computed against the same-run XLA reference.

Harness design:

* The first 32,768 contiguous corpus tokens are split into 32 non-overlapping
  1024-token windows and processed as four batches of eight windows. These are
  evaluation windows, not statistically independent samples. There are 32,736
  scored positions after dropping one boundary target per window.
* No timing is recorded.  Performance is already registered elsewhere; this
  is a quality gate only, and the arms share one layer download.
* Declared thresholds for the promoted arm (gated product-aware Strassen):
  absolute next-token loss delta <= 0.01 nats (retained from the easy-corpus
  gate; about 1% perplexity), mean per-token KL <= 0.02 nats, and top-1
  agreement >= 0.97.  KL and top-1 are deliberately looser than the
  easy-corpus gate because natural text leaves the softmax unsaturated;
  the thresholds are declared in this harness and emitted in each artifact's
  opening metadata record before evaluation begins.
* Logit L2/maxnorm drift is reported as a diagnostic, never gated on.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ["QWEN3_STREAM_PRODUCT_AWARE"] = "1"
os.environ.setdefault("QWEN3_PRODUCT_TILE", "2048,2048,512")
os.environ.setdefault("QWEN3_CUBIC_TILE", "2048,2048,512")

import jax
import jax.numpy as jnp
import numpy as np

import benchmark_common as common
import benchmark_qwen3_32b_layer as layer
import benchmark_qwen3_32b_streamed_inference as stream
import mosaic_compat
import strassen_pallas as sp


SEGMENTS = int(os.environ.get("QWEN3_NATURAL_SEGMENTS", "4"))
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
ARMS = stream.ARMS
LOGIT_CHUNK = stream.LOGIT_CHUNK
THRESHOLDS = {
    "absolute_loss_delta_max": 0.01,
    "mean_kl_nats_max": 0.02,
    "top1_agreement_min": 0.97,
}
SUFFIX = os.environ.get("QWEN3_OUTPUT_SUFFIX", "")
# run.py exports STRASSEN_OUTPUT_DIR so --output-dir actually takes effect;
# the Colab default is kept for direct invocation.
RESULTS_DIR = os.environ.get("STRASSEN_OUTPUT_DIR", "/content/results")
OUTPUT = Path(
    f"{RESULTS_DIR}/strassen_qwen3_{layer.MODEL_NAME}"
    f"_natural_task_gate{SUFFIX}.jsonl")


def emit(record):
    line = json.dumps(record, sort_keys=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as stream_out:
        stream_out.write(line + "\n")
    print("QWEN3_NATURAL_GATE_JSON " + line, flush=True)


def wikitext_segments():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test",
        revision=WIKITEXT_REVISION)
    text = "\n".join(row["text"] for row in dataset)
    tokenizer = AutoTokenizer.from_pretrained(
        layer.REPOSITORY, revision=layer.REVISION, use_fast=True, token=False)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="np")[
        "input_ids"][0]
    needed = SEGMENTS * layer.BATCH * layer.SEQUENCE
    if ids.shape[0] < needed:
        raise RuntimeError(f"corpus too small: {ids.shape[0]} < {needed}")
    token_ids = ids[:needed].reshape(
        SEGMENTS, layer.BATCH, layer.SEQUENCE).astype(np.int32)
    emit({
        "kind": "tokens",
        "corpus": "wikitext-2-raw-v1/test first contiguous tokens",
        "dataset_revision": WIKITEXT_REVISION,
        "shape": list(token_ids.shape),
        "sha256": hashlib.sha256(token_ids.tobytes()).hexdigest(),
        "unique_tokens": int(np.unique(token_ids).size),
    })
    return token_ids


def aggregate_task_metrics(hidden, checkpoint, token_ids):
    """Chunked next-token metrics over every segment, one lm_head load."""
    final_norm = jnp.asarray(checkpoint.tensor("model.norm.weight"))
    lm_head = stream.device_matrix(checkpoint, "lm_head.weight")
    metric_exec = None
    totals = {
        "native_loss": 0.0, "native_sq": 0.0, "native_max": 0.0, "count": 0,
        "gated_cubic": [0.0, 0.0, 0.0, 0.0, 0.0],
        "gated_strassen": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    for segment in range(SEGMENTS):
        normalized = {
            arm: layer.rms_norm(hidden[arm][segment], final_norm)
            for arm in ARMS
        }
        jax.block_until_ready(tuple(normalized.values()))
        rows = {
            arm: value.reshape(
                layer.BATCH, layer.SEQUENCE, layer.MODEL_DIM)[:, :-1]
            .reshape(-1, layer.MODEL_DIM)
            for arm, value in normalized.items()
        }
        targets = token_ids[segment][:, 1:].reshape(-1)
        if metric_exec is None:
            metric_exec = jax.jit(stream.logit_chunk_metrics).lower(
                *[rows[arm][:LOGIT_CHUNK] for arm in ARMS],
                lm_head,
                jnp.asarray(targets[:LOGIT_CHUNK]),
                jnp.ones((LOGIT_CHUNK,), jnp.int32),
            ).compile()
        for start in range(0, len(targets), LOGIT_CHUNK):
            stop = min(start + LOGIT_CHUNK, len(targets))
            valid = stop - start
            padded_rows = []
            for arm in ARMS:
                part = rows[arm][start:stop]
                padded_rows.append(
                    jnp.pad(part, ((0, LOGIT_CHUNK - valid), (0, 0))))
            target = np.zeros((LOGIT_CHUNK,), np.int32)
            target[:valid] = targets[start:stop]
            mask = np.zeros((LOGIT_CHUNK,), np.int32)
            mask[:valid] = 1
            result = jax.device_get(metric_exec(
                *padded_rows, lm_head, jnp.asarray(target), jnp.asarray(mask)))
            native_loss, native_sq, native_max, cubic_v, strassen_v, count = (
                result)
            totals["native_loss"] += float(native_loss)
            totals["native_sq"] += float(native_sq)
            totals["native_max"] = max(totals["native_max"], float(native_max))
            totals["count"] += int(count)
            for name, values in (("gated_cubic", cubic_v),
                                 ("gated_strassen", strassen_v)):
                for index, value in enumerate(values):
                    if index == 4:
                        totals[name][index] = max(
                            totals[name][index], float(value))
                    else:
                        totals[name][index] += float(value)
        del normalized, rows
        gc.collect()
    count = totals["count"]
    native_loss = totals["native_loss"] / count
    records = {}
    for name in ("gated_cubic", "gated_strassen"):
        loss, kl, top1, delta_sq, max_delta = totals[name]
        candidate_loss = loss / count
        records[name] = {
            "native_loss": native_loss,
            "native_perplexity": math.exp(native_loss),
            "candidate_loss": candidate_loss,
            "candidate_perplexity": math.exp(candidate_loss),
            "absolute_loss_delta": abs(candidate_loss - native_loss),
            "relative_perplexity_delta": (
                math.exp(candidate_loss - native_loss) - 1.0),
            "mean_kl_nats": kl / count,
            "top1_agreement": top1 / count,
            "logit_l2_relative": math.sqrt(
                delta_sq / max(totals["native_sq"], 1e-30)),
            "logit_maxnorm_relative": (
                max_delta / max(totals["native_max"], 1e-30)),
            "positions": count,
        }
        records[name]["passes_task_gate"] = (
            records[name]["absolute_loss_delta"]
            <= THRESHOLDS["absolute_loss_delta_max"]
            and records[name]["top1_agreement"]
            >= THRESHOLDS["top1_agreement_min"]
            and records[name]["mean_kl_nats"]
            <= THRESHOLDS["mean_kl_nats_max"]
        )
    return records


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata", "model": layer.MODEL_NAME,
        "repository": layer.REPOSITORY, "revision": layer.REVISION,
        "layers": stream.NUM_LAYERS, "segments": SEGMENTS,
        "device": jax.devices()[0].device_kind, "jax": jax.__version__,
        "mosaic_compat": mosaic_compat.compatibility_info(
            sp.MOSAIC_IR_V7_COMPAT),
        "arms": list(ARMS),
        "policy": "product-aware fused gate/up+SwiGLU; XLA down",
        "product_tile": list(stream.PRODUCT_TILE),
        "cubic_tile": list(stream.CUBIC_TILE),
        "thresholds": THRESHOLDS,
        "scope": (
            "quality only; no timing; thresholds declared in the harness "
            "and emitted before evaluation; logit drift diagnostic, never gated"),
    })
    token_ids = wikitext_segments()
    checkpoint = layer.ShardedSafetensors()
    embeddings = checkpoint.tensor("model.embed_tokens.weight")
    initial = [
        jnp.asarray(embeddings[token_ids[segment]].reshape(
            layer.TOKENS, layer.MODEL_DIM).copy())
        for segment in range(SEGMENTS)
    ]
    del embeddings
    hidden = {arm: [jnp.array(value, copy=True) for value in initial]
              for arm in ARMS}
    jax.block_until_ready(hidden)
    del initial
    rope_cos, rope_sin = layer.rope_values()
    first = stream.load_layer(checkpoint, 0, rope_cos, rope_sin)
    executables = {}
    for arm in ARMS:
        started = time.perf_counter()
        lowered = jax.jit(stream.make_product_layer(arm)).lower(
            hidden[arm][0], first)
        executables[arm] = lowered.compile()
        emit({"kind": "compile", "arm": arm,
              "seconds": time.perf_counter() - started})

    all_finite = True
    for index in range(stream.NUM_LAYERS):
        transfer_started = time.perf_counter()
        params = (first if index == 0
                  else stream.load_layer(checkpoint, index, rope_cos, rope_sin))
        transfer_s = time.perf_counter() - transfer_started
        for arm in ARMS:
            hidden[arm] = [
                executables[arm](value, params) for value in hidden[arm]]
        jax.block_until_ready(hidden)
        finite = bool(jax.device_get(jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(value))
            for values in hidden.values() for value in values
        ]))))
        all_finite = all_finite and finite
        errors = {
            arm: float(np.mean([
                common.device_error(value, reference)["l2_relative"]
                for value, reference in zip(
                    hidden[arm], hidden["regular_xla"], strict=True)
            ]))
            for arm in ARMS if arm != "regular_xla"
        }
        emit({"kind": "layer", "layer": index, "transfer_s": transfer_s,
              "mean_l2_vs_xla": errors, "finite": finite})
        del params
        if index == 0:
            del first
        gc.collect()

    tasks = aggregate_task_metrics(hidden, checkpoint, token_ids)
    emit({"kind": "task", "results": tasks})
    passes = all_finite and tasks["gated_strassen"]["passes_task_gate"]
    emit({
        "kind": "verdict", "passes": passes,
        "task_passes": tasks["gated_strassen"]["passes_task_gate"],
        "cubic_task_passes": tasks["gated_cubic"]["passes_task_gate"],
        "all_finite": all_finite,
        "scope": (
            "natural-corpus streamed quality gate for the frozen "
            "product-aware policy"),
    })


if __name__ == "__main__":
    main()
