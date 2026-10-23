"""Zero-shot downstream-task agreement for Gemma 3 27B.

Logit-agreement gates answer "did the distribution move".  This answers the
question that actually matters for deployment: does the routed model pick
the same answers?  HellaSwag is scored by summed log-likelihood over the
four endings, LAMBADA by whether the final word is produced greedily, and
both are compared against the native path rather than only against the
label -- a routed model that is wrong in exactly the same places as the
native one has not been damaged by routing.

Gemma specifics carried over from the streamed harness: four norms per
block, ``(1 + weight)`` RMSNorm, GeGLU, per-layer RoPE selection, the
sqrt(hidden_size) embedding scale, and a head tied to ``embed_tokens``
because the checkpoint has no ``lm_head``.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("QWEN3_MODEL", "gemma3-27b")
os.environ.setdefault("QWEN3_SEQUENCE", "256")
SCOPED_VMEM_KIB = int(os.environ.get("QWEN3_SCOPED_VMEM_KIB", "49152"))
os.environ["LIBTPU_INIT_ARGS"] = (
    "--xla_tpu_use_enhanced_launch_barrier=true "
    f"--xla_tpu_scoped_vmem_limit_kib={SCOPED_VMEM_KIB}")

import jax
import jax.numpy as jnp
import ml_dtypes  # noqa: F401  (registers bfloat16 with numpy)
import numpy as np

import benchmark_gemma3_block as block
import benchmark_gemma3_streamed as stream
import benchmark_qwen3_32b_layer as layer
import mosaic_compat
import strassen_pallas as sp

SEQ = layer.SEQUENCE
HELLASWAG_N = int(os.environ.get("GEMMA_HELLASWAG_N", "160"))
LAMBADA_N = int(os.environ.get("GEMMA_LAMBADA_N", "320"))
NUM_LAYERS = int(os.environ.get("GEMMA_LAYERS", layer._MODEL["num_layers"]))
# Streaming all 62 layers in one exec takes about 2.5 hours, and a free-tier
# runtime has died three times at that length, losing everything.  The layer
# loop is therefore resumable: each chunk loads the hidden state left by the
# previous one and saves its own, so a death costs one chunk.  Scoring runs
# only in the chunk that reaches the last layer.
LAYER_START = int(os.environ.get("GEMMA_LAYER_START", "0"))
LAYER_STOP = int(os.environ.get("GEMMA_LAYER_STOP", NUM_LAYERS))
CHECKPOINT = os.environ.get("GEMMA_STATE_PATH", "/content/gemma_state.npz")
CHUNK = 256
ARMS = ("regular_xla", "gated_cubic", "gated_strassen")
SUFFIX = os.environ.get("QWEN3_OUTPUT_SUFFIX", "")
# run.py exports STRASSEN_OUTPUT_DIR so --output-dir actually takes effect;
# the Colab default is kept for direct invocation.
RESULTS_DIR = os.environ.get("STRASSEN_OUTPUT_DIR", "/content/results")
OUTPUT = Path(
    f"{RESULTS_DIR}/strassen_gemma3_27b_downstream{SUFFIX}.jsonl")


def emit(record):
    line = json.dumps(record, sort_keys=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("a", encoding="utf-8") as stream_out:
        stream_out.write(line + "\n")
    print("GEMMA3_DOWNSTREAM_JSON " + line, flush=True)


def build_rows():
    from datasets import load_dataset

    tokenizer = stream._tokenizer()

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
                # lm-evaluation-harness normalises HellaSwag by the
                # continuation's character length; without it the metric
                # favours short endings and is not the published acc_norm.
                "characters": len(" " + ending)})

    lambada = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    for index in range(LAMBADA_N):
        text = lambada[index]["text"]
        context, _, last_word = text.rpartition(" ")
        add_sequence(encode(context), encode(" " + last_word), {
            "task": "lambada", "example": index, "candidate": 0,
            "gold": None, "characters": len(" " + last_word)})

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


def position_metrics(hidden_rows, head, gold_tokens):
    logits = jnp.matmul(
        hidden_rows.astype(jnp.float32), head.astype(jnp.float32),
        precision=jax.lax.Precision.DEFAULT)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    gold = jnp.take_along_axis(log_probs, gold_tokens[:, None], axis=1)[:, 0]
    return gold, jnp.argmax(logits, axis=-1)


def score(results, spans, meta):
    """HellaSwag choice by summed log-likelihood; LAMBADA by greedy match."""
    records = {}
    native_choice, native_choice_norm, native_greedy = None, None, None
    for arm in ARMS:
        logprobs, argmaxes, owners, gold = results[arm]
        totals, counts, greedy = {}, {}, {}
        for value_index, span in enumerate(owners):
            totals[span] = totals.get(span, 0.0) + logprobs[value_index]
            counts[span] = counts.get(span, 0) + 1
            greedy[span] = greedy.get(span, True) and bool(
                argmaxes[value_index] == gold[value_index])
        hella, hella_norm, lam_correct, lam_greedy = {}, {}, 0, {}
        for span_index, record in enumerate(meta):
            if record["task"] == "hellaswag":
                hella.setdefault(record["example"], {})[
                    record["candidate"]] = totals[span_index]
                # acc_norm: log-likelihood per character of continuation.
                hella_norm.setdefault(record["example"], {})[
                    record["candidate"]] = (
                        totals[span_index] / max(record["characters"], 1))
            else:
                lam_greedy[record["example"]] = greedy[span_index]
                lam_correct += int(greedy[span_index])
        choice = {k: max(v, key=v.get) for k, v in hella.items()}
        choice_norm = {k: max(v, key=v.get) for k, v in hella_norm.items()}
        gold_label = {r["example"]: r["gold"] for r in meta
                      if r["task"] == "hellaswag"}
        entry = {
            "hellaswag_accuracy": sum(
                int(choice[k] == gold_label[k]) for k in choice) / len(choice),
            # The published HellaSwag figure is acc_norm, not acc; reporting
            # only raw accuracy invited a comparison against a different
            # metric and made a faithful model look broken.
            "hellaswag_accuracy_norm": sum(
                int(choice_norm[k] == gold_label[k])
                for k in choice_norm) / len(choice_norm),
            "lambada_accuracy": lam_correct / max(len(lam_greedy), 1),
        }
        if arm == "regular_xla":
            native_choice, native_greedy = choice, lam_greedy
            entry["hellaswag_choice_agreement"] = 1.0
            entry["hellaswag_choice_agreement_norm"] = 1.0
            entry["lambada_greedy_agreement"] = 1.0
            native_choice_norm = choice_norm
        else:
            entry["hellaswag_choice_agreement"] = sum(
                int(choice[k] == native_choice[k]) for k in choice
            ) / len(choice)
            entry["hellaswag_choice_agreement_norm"] = sum(
                int(choice_norm[k] == native_choice_norm[k])
                for k in choice_norm) / len(choice_norm)
            entry["lambada_greedy_agreement"] = sum(
                int(lam_greedy[k] == native_greedy[k]) for k in lam_greedy
            ) / max(len(lam_greedy), 1)
            entry["passes_task_gate"] = (
                entry["hellaswag_choice_agreement"] >= 0.97
                and entry["lambada_greedy_agreement"] >= 0.97)
        records[arm] = entry
        emit({"kind": "task", "arm": arm, **entry})
    return records


def main():
    if SEQ > block.SLIDING_WINDOW:
        raise ValueError("sequence exceeds the sliding window; add a mask")
    layer.emit = emit
    stream.emit = emit
    emit({
        "kind": "metadata", "model": layer.MODEL_NAME,
        "repository": layer.REPOSITORY, "revision": layer.REVISION,
        "device": jax.devices()[0].device_kind, "jax": jax.__version__,
        "mosaic_compat": mosaic_compat.compatibility_info(
            sp.MOSAIC_IR_V7_COMPAT),
        "sequence": SEQ, "layers": NUM_LAYERS,
        "tile": [block.BM, block.BN, block.BK], "arms": list(ARMS),
        "thresholds": {"choice_agreement": 0.97, "greedy_agreement": 0.97},
        "scope": "zero-shot downstream-task agreement, Gemma 3 27B",
    })
    matrix, spans, meta = build_rows()
    checkpoint = layer.ShardedSafetensors()
    embeddings = checkpoint.tensor(
        f"{layer.TENSOR_PREFIX}.embed_tokens.weight")
    scale = math.sqrt(layer.MODEL_DIM)
    blocks = matrix.reshape(-1, layer.BATCH, SEQ)
    if LAYER_START:
        saved = np.load(CHECKPOINT)
        if int(saved["next_layer"]) != LAYER_START:
            raise ValueError(
                f"checkpoint resumes at {int(saved['next_layer'])}, "
                f"not {LAYER_START}")
        if saved["tokens_sha"].item() != hashlib.sha256(
                matrix.tobytes()).hexdigest():
            raise ValueError("checkpoint was built from different tokens")
        hidden = {arm: [jnp.asarray(v.view(ml_dtypes.bfloat16))
                        for v in saved[f"hidden_{arm}"]] for arm in ARMS}
        emit({"kind": "resumed", "from_layer": LAYER_START,
              "path": CHECKPOINT})
    else:
        hidden = {arm: [
            (jnp.asarray(np.asarray(embeddings)[chunk.reshape(-1)],
                         jnp.float32) * scale).astype(jnp.bfloat16)
            for chunk in blocks] for arm in ARMS}

    # Build the jitted layer functions once.  make_layer returns a fresh
    # closure each call, so jitting inside the loop would defeat JAX's cache
    # and recompile every arm at every layer -- 186 compiles for 62 layers.
    runners = {arm: jax.jit(stream.make_layer(arm)) for arm in ARMS}

    for index in range(LAYER_START, LAYER_STOP):
        params = stream.load_layer(checkpoint, index)
        for arm in ARMS:
            run = runners[arm]
            hidden[arm] = [jax.block_until_ready(run(h, params))
                           for h in hidden[arm]]
        del params
        gc.collect()
        if index % 10 == 0 or index == NUM_LAYERS - 1:
            emit({"kind": "progress", "layer": index})

    finite = all(bool(jnp.isfinite(v).all())
                 for values in hidden.values() for v in values)
    emit({"kind": "finite", "all_finite": finite})

    if LAYER_STOP < NUM_LAYERS:
        np.savez(
            CHECKPOINT, next_layer=np.asarray(LAYER_STOP),
            tokens_sha=np.asarray(hashlib.sha256(matrix.tobytes()).hexdigest()),
            # The state is BF16.  npz does not preserve the bfloat16 dtype
            # -- it round-trips as an opaque 2-byte void -- so store an
            # explicit uint16 view and restore it on load.  Exact, and half
            # the size of promoting to FP32.
            **{f"hidden_{arm}": np.stack(
                [np.asarray(v).view(np.uint16) for v in hidden[arm]])
               for arm in ARMS})
        emit({"kind": "checkpoint", "next_layer": LAYER_STOP,
              "path": CHECKPOINT, "note": "scoring deferred to the last chunk"})
        return

    final_norm = jnp.asarray(
        checkpoint.tensor(f"{layer.TENSOR_PREFIX}.norm.weight"))
    head = jnp.asarray(embeddings).T          # tied to the embedding table
    gather_rows, gather_positions, gold_tokens, owners = [], [], [], []
    for span_index, (row, start, end) in enumerate(spans):
        for position in range(start, end):
            gather_rows.append(row)
            gather_positions.append(position - 1)
            gold_tokens.append(int(matrix[row, position]))
            owners.append(span_index)
    gold_array = np.asarray(gold_tokens, np.int32)
    total = len(gold_array)

    metric_exec, results = None, {}
    for arm in ARMS:
        stacked = jnp.concatenate(hidden[arm]).reshape(-1, SEQ, layer.MODEL_DIM)
        needed = stacked[jnp.asarray(np.asarray(gather_rows)),
                         jnp.asarray(np.asarray(gather_positions))]
        needed = block.gemma_rms_norm(needed, final_norm)
        jax.block_until_ready(needed)
        del stacked
        logprobs = np.zeros((total,), np.float64)
        argmaxes = np.zeros((total,), np.int64)
        for start_index in range(0, total, CHUNK):
            stop = min(start_index + CHUNK, total)
            valid = stop - start_index
            rows_block = jnp.pad(needed[start_index:stop],
                                 ((0, CHUNK - valid), (0, 0)))
            gold_block = np.zeros((CHUNK,), np.int32)
            gold_block[:valid] = gold_array[start_index:stop]
            if metric_exec is None:
                metric_exec = jax.jit(position_metrics).lower(
                    rows_block, head, jnp.asarray(gold_block)).compile()
            gold_logprob, argmax = jax.device_get(
                metric_exec(rows_block, head, jnp.asarray(gold_block)))
            logprobs[start_index:stop] = gold_logprob[:valid]
            argmaxes[start_index:stop] = argmax[:valid]
        results[arm] = (logprobs, argmaxes, owners, gold_array)
        del needed
        gc.collect()

    records = score(results, spans, meta)
    emit({"kind": "verdict", "all_finite": finite,
          "passes": all(v.get("passes_task_gate", True)
                        for v in records.values()),
          "scope": "zero-shot downstream-task agreement, Gemma 3 27B"})


if __name__ == "__main__":
    main()
