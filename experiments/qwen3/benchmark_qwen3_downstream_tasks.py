"""Downstream-task agreement gate for the frozen product-aware policy.

The natural-corpus gate measured teacher-forced next-token metrics; this
gate measures the two standard zero-shot harness patterns on real tasks:

* HellaSwag (validation slice): four-way loglikelihood choice.  A candidate
  ending is scored by the summed log-probability of its tokens given the
  context; the selected ending is the argmax (per-token-mean selection is
  also recorded).
* LAMBADA (openai test slice): greedy final-word prediction.  An example is
  correct when every target-word token is the argmax at its position.

All sequences are scored teacher-forced through the streamed layer stack,
one sequence per row (causal attention, so right padding is inert), with the
layer compiled at a short sequence length so every scored sequence rides one
checkpoint download.  No timing is recorded.

Registered thresholds, fixed before the first run, for the promoted arm:
choice/greedy agreement with the same-run XLA arm >= 0.97 on each task, and
absolute accuracy delta <= 0.02 on each task.  The matched cubic arm is
reported under the same metrics as the substrate control.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["QWEN3_STREAM_PRODUCT_AWARE"] = "1"
os.environ.setdefault("QWEN3_SEQUENCE", "256")
os.environ.setdefault("QWEN3_PRODUCT_TILE", "2048,2048,512")
os.environ.setdefault("QWEN3_CUBIC_TILE", "2048,2048,512")

import jax
import jax.numpy as jnp
import numpy as np

import benchmark_qwen3_32b_layer as layer
import benchmark_qwen3_32b_streamed_inference as stream
import mosaic_compat
import strassen_pallas as sp


HELLASWAG_N = 160
LAMBADA_N = 320
SEQ = layer.SEQUENCE
ARMS = stream.ARMS
CHUNK = 256
THRESHOLDS = {
    "choice_agreement_min": 0.97,
    "greedy_agreement_min": 0.97,
    "absolute_accuracy_delta_max": 0.02,
}
SUFFIX = os.environ.get("QWEN3_OUTPUT_SUFFIX", "")
# run.py exports STRASSEN_OUTPUT_DIR so --output-dir actually takes effect;
# the Colab default is kept for direct invocation.
RESULTS_DIR = os.environ.get("STRASSEN_OUTPUT_DIR", "/content/results")
OUTPUT = Path(
    f"{RESULTS_DIR}/strassen_qwen3_{layer.MODEL_NAME}"
    f"_downstream_tasks{SUFFIX}.jsonl")


def emit(record):
    line = json.dumps(record, sort_keys=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as stream_out:
        stream_out.write(line + "\n")
    print("QWEN3_DOWNSTREAM_JSON " + line, flush=True)


def build_rows():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        layer.REPOSITORY, revision=layer.REVISION, use_fast=True, token=False)

    def encode(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    rows, spans, meta = [], [], []

    def add_sequence(context_ids, target_ids, record):
        ids = list(context_ids) + list(target_ids)
        if len(ids) > SEQ:
            ids = ids[-SEQ:]
        start = len(ids) - len(target_ids)
        if start < 1:
            raise RuntimeError(f"target fills the whole row: {record}")
        row = np.zeros((SEQ,), np.int32)
        row[: len(ids)] = ids
        rows.append(row)
        spans.append((len(rows) - 1, start, len(ids)))
        meta.append(record)

    hellaswag = load_dataset("Rowan/hellaswag", split="validation")
    for index in range(HELLASWAG_N):
        example = hellaswag[index]
        context_ids = encode(example["ctx"])
        for candidate, ending in enumerate(example["endings"]):
            add_sequence(context_ids, encode(" " + ending), {
                "task": "hellaswag", "example": index,
                "candidate": candidate, "gold": int(example["label"]),
            })

    lambada = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    for index in range(LAMBADA_N):
        text = lambada[index]["text"]
        context, _, last_word = text.rpartition(" ")
        add_sequence(encode(context), encode(" " + last_word), {
            "task": "lambada", "example": index, "candidate": 0, "gold": None,
        })

    while len(rows) % layer.BATCH:
        rows.append(np.zeros((SEQ,), np.int32))
    matrix = np.stack(rows)
    emit({
        "kind": "tokens", "shape": list(matrix.shape),
        "sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        "hellaswag_examples": HELLASWAG_N, "lambada_examples": LAMBADA_N,
        "scored_sequences": len(spans),
        "scored_positions": int(sum(e - s for _, s, e in spans)),
    })
    return matrix, spans, meta


def position_metrics(hidden_rows, lm_head, gold_tokens):
    logits = jnp.matmul(
        hidden_rows.astype(jnp.float32), lm_head.astype(jnp.float32),
        precision=jax.lax.Precision.DEFAULT)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    gold_logprob = jnp.take_along_axis(
        log_probs, gold_tokens[:, None], axis=1)[:, 0]
    return gold_logprob, jnp.argmax(logits, axis=-1)


def main():
    if jax.default_backend() != "tpu" or len(jax.devices()) != 1:
        raise RuntimeError(f"expected one TPU device, got {jax.devices()}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("", encoding="utf-8")
    emit({
        "kind": "metadata", "model": layer.MODEL_NAME,
        "repository": layer.REPOSITORY, "revision": layer.REVISION,
        "layers": stream.NUM_LAYERS, "sequence": SEQ,
        "device": jax.devices()[0].device_kind, "jax": jax.__version__,
        "mosaic_compat": mosaic_compat.compatibility_info(
            sp.MOSAIC_IR_V7_COMPAT),
        "arms": list(ARMS),
        "policy": "product-aware fused gate/up+SwiGLU; XLA down",
        "product_tile": list(stream.PRODUCT_TILE),
        "cubic_tile": list(stream.CUBIC_TILE),
        "product_panels": list(stream.PRODUCT_PANELS),
        "tasks": {"hellaswag": HELLASWAG_N, "lambada": LAMBADA_N},
        "thresholds": THRESHOLDS,
        "scope": (
            "zero-shot task agreement; teacher-forced scoring; no timing; "
            "thresholds registered before the first run"),
    })
    matrix, spans, meta = build_rows()
    batches = matrix.reshape(-1, layer.BATCH, SEQ)
    checkpoint = layer.ShardedSafetensors()
    embeddings = checkpoint.tensor("model.embed_tokens.weight")
    initial = [
        jnp.asarray(embeddings[batch].reshape(
            layer.TOKENS, layer.MODEL_DIM).copy())
        for batch in batches
    ]
    del embeddings
    hidden = {arm: [jnp.array(v, copy=True) for v in initial] for arm in ARMS}
    jax.block_until_ready(hidden)
    del initial
    gc.collect()
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

    for index in range(stream.NUM_LAYERS):
        transfer_started = time.perf_counter()
        params = (first if index == 0
                  else stream.load_layer(checkpoint, index, rope_cos, rope_sin))
        transfer_s = time.perf_counter() - transfer_started
        for arm in ARMS:
            hidden[arm] = [executables[arm](v, params) for v in hidden[arm]]
        jax.block_until_ready(hidden)
        emit({"kind": "layer", "layer": index, "transfer_s": transfer_s})
        del params
        if index == 0:
            del first
        gc.collect()

    finite = bool(jax.device_get(jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(v)) for values in hidden.values() for v in values
    ]))))
    emit({"kind": "finite", "all_finite": finite})

    # Gather the hidden vectors that predict each target token: position p
    # is predicted by hidden index p - 1.
    gather_rows, gather_positions, gold_tokens, owners = [], [], [], []
    for span_index, (row, start, end) in enumerate(spans):
        for position in range(start, end):
            gather_rows.append(row)
            gather_positions.append(position - 1)
            gold_tokens.append(int(matrix[row, position]))
            owners.append(span_index)
    final_norm = jnp.asarray(checkpoint.tensor("model.norm.weight"))
    lm_head = stream.device_matrix(checkpoint, "lm_head.weight")
    gather_rows = np.asarray(gather_rows)
    gather_positions = np.asarray(gather_positions)
    gold_array = np.asarray(gold_tokens, np.int32)
    total = len(gold_array)

    metric_exec = None
    results = {}
    for arm in ARMS:
        stacked = jnp.stack(hidden[arm]).reshape(-1, SEQ, layer.MODEL_DIM)
        needed = stacked[jnp.asarray(gather_rows), jnp.asarray(gather_positions)]
        needed = layer.rms_norm(needed, final_norm)
        jax.block_until_ready(needed)
        del stacked
        logprobs = np.zeros((total,), np.float64)
        argmaxes = np.zeros((total,), np.int64)
        for start_index in range(0, total, CHUNK):
            stop = min(start_index + CHUNK, total)
            valid = stop - start_index
            block = jnp.pad(needed[start_index:stop],
                            ((0, CHUNK - valid), (0, 0)))
            gold_block = np.zeros((CHUNK,), np.int32)
            gold_block[:valid] = gold_array[start_index:stop]
            if metric_exec is None:
                metric_exec = jax.jit(position_metrics).lower(
                    block, lm_head, jnp.asarray(gold_block)).compile()
            gold_logprob, argmax = jax.device_get(
                metric_exec(block, lm_head, jnp.asarray(gold_block)))
            logprobs[start_index:stop] = gold_logprob[:valid]
            argmaxes[start_index:stop] = argmax[:valid]
        results[arm] = (logprobs, argmaxes)
        del needed
        gc.collect()

    verdict = {}
    task_records = {}
    for arm in ARMS:
        logprobs, argmaxes = results[arm]
        span_sum = {}
        span_count = {}
        span_greedy = {}
        span_argmax = {}
        for value_index, span_index in enumerate(owners):
            span_sum[span_index] = span_sum.get(span_index, 0.0) + logprobs[
                value_index]
            span_count[span_index] = span_count.get(span_index, 0) + 1
            ok = argmaxes[value_index] == gold_array[value_index]
            span_greedy[span_index] = span_greedy.get(span_index, True) and ok
            span_argmax.setdefault(span_index, []).append(
                int(argmaxes[value_index]))
        hs_choice, hs_choice_mean, hs_gold = {}, {}, {}
        lam_correct, lam_argmax = {}, {}
        for span_index, record in enumerate(meta):
            if record["task"] == "hellaswag":
                example = record["example"]
                hs_gold[example] = record["gold"]
                best = hs_choice.setdefault(example, (None, -np.inf))
                score = span_sum[span_index]
                if score > best[1]:
                    hs_choice[example] = (record["candidate"], score)
                best_mean = hs_choice_mean.setdefault(example, (None, -np.inf))
                mean_score = score / span_count[span_index]
                if mean_score > best_mean[1]:
                    hs_choice_mean[example] = (record["candidate"], mean_score)
            else:
                example = record["example"]
                lam_correct[example] = bool(span_greedy[span_index])
                lam_argmax[example] = tuple(span_argmax[span_index])
        task_records[arm] = {
            "hellaswag_choices": {k: v[0] for k, v in hs_choice.items()},
            "hellaswag_choices_mean": {
                k: v[0] for k, v in hs_choice_mean.items()},
            "hellaswag_accuracy": float(np.mean([
                hs_choice[k][0] == hs_gold[k] for k in hs_choice])),
            "lambada_correct": lam_correct,
            "lambada_argmax": lam_argmax,
            "lambada_accuracy": float(np.mean(list(lam_correct.values()))),
            "mean_gold_logprob": float(np.mean(logprobs)),
        }
    reference = task_records["regular_xla"]
    for arm in ARMS:
        record = task_records[arm]
        summary = {
            "kind": "task", "arm": arm,
            "hellaswag_accuracy": record["hellaswag_accuracy"],
            "lambada_accuracy": record["lambada_accuracy"],
            "mean_gold_logprob": record["mean_gold_logprob"],
        }
        if arm != "regular_xla":
            choice_agreement = float(np.mean([
                record["hellaswag_choices"][k]
                == reference["hellaswag_choices"][k]
                for k in record["hellaswag_choices"]]))
            greedy_agreement = float(np.mean([
                record["lambada_argmax"][k] == reference["lambada_argmax"][k]
                for k in record["lambada_argmax"]]))
            summary.update({
                "hellaswag_choice_agreement": choice_agreement,
                "hellaswag_accuracy_delta": abs(
                    record["hellaswag_accuracy"]
                    - reference["hellaswag_accuracy"]),
                "lambada_greedy_agreement": greedy_agreement,
                "lambada_accuracy_delta": abs(
                    record["lambada_accuracy"]
                    - reference["lambada_accuracy"]),
            })
            summary["passes_task_gate"] = bool(
                choice_agreement >= THRESHOLDS["choice_agreement_min"]
                and greedy_agreement >= THRESHOLDS["greedy_agreement_min"]
                and summary["hellaswag_accuracy_delta"]
                <= THRESHOLDS["absolute_accuracy_delta_max"]
                and summary["lambada_accuracy_delta"]
                <= THRESHOLDS["absolute_accuracy_delta_max"])
            verdict[arm] = summary["passes_task_gate"]
        emit(summary)
    emit({
        "kind": "verdict", "all_finite": finite,
        "passes": bool(finite and verdict.get("gated_strassen")),
        "strassen_passes": verdict.get("gated_strassen"),
        "cubic_passes": verdict.get("gated_cubic"),
        "scope": "zero-shot downstream-task agreement gate",
    })


if __name__ == "__main__":
    main()
