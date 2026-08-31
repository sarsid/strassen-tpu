"""Streamed full-checkpoint quality gate for Mistral-7B-v0.1.

This is an accuracy experiment, not a latency benchmark.  The two pinned
checkpoint shards stay on runtime disk while one layer of weights at a time is
copied to the TPU.  Matched-native and two-MLP Strassen states then traverse
all 32 layers side by side before real LM-head logits and next-token loss are
compared.
"""

from __future__ import annotations

import gc
import hashlib
import json
import mmap
import os
from pathlib import Path
import time


os.environ["STRASSEN_VMEM_PROFILE"] = "max48"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import benchmark_checkpoint_layer as layer_bench
import benchmark_common as bench
import strassen_pallas as sp

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np


OUTPUT = Path("/content/results/strassen_checkpoint_model.jsonl")
CHECKPOINT_DIR = Path("/content/mistral-7b-v0.1")
SHARDS = (
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)
NUM_LAYERS = 32


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with OUTPUT.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    print("CHECKPOINT_MODEL_JSON " + line, flush=True)


def download_checkpoint():
    from huggingface_hub import hf_hub_download

    paths = []
    for filename in SHARDS:
        started = time.perf_counter()
        path = hf_hub_download(
            repo_id=layer_bench.REPOSITORY,
            filename=filename,
            revision=layer_bench.REVISION,
            local_dir=CHECKPOINT_DIR,
            token=False,
        )
        size = Path(path).stat().st_size
        emit({
            "kind": "shard",
            "filename": filename,
            "bytes": size,
            "download_s": time.perf_counter() - started,
        })
        paths.append(Path(path))
    return paths


class LocalSafetensors:
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open("rb")
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        header_length = int.from_bytes(self.mapping[:8], "little")
        self.metadata = json.loads(self.mapping[8:8 + header_length])
        self.data_start = 8 + header_length

    def tensor(self, name):
        metadata = self.metadata[name]
        if metadata["dtype"] != "BF16":
            raise ValueError(f"{name} has unsupported dtype {metadata['dtype']}")
        first, last = metadata["data_offsets"]
        return np.frombuffer(
            self.mapping,
            dtype=ml_dtypes.bfloat16,
            count=(last - first) // 2,
            offset=self.data_start + first,
        ).reshape(metadata["shape"])


class Checkpoint:
    def __init__(self, paths):
        self.shards = [LocalSafetensors(path) for path in paths]
        self.locations = {}
        for shard in self.shards:
            for name in shard.metadata:
                if name != "__metadata__":
                    self.locations[name] = shard

    def tensor(self, name):
        return self.locations[name].tensor(name)


def make_tokens():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        layer_bench.REPOSITORY,
        revision=layer_bench.REVISION,
        use_fast=True,
        token=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    expanded = [(text + "\n") * 80 for text in layer_bench.TEXTS]
    encoded = tokenizer(
        expanded,
        add_special_tokens=True,
        max_length=layer_bench.SEQUENCE,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )
    if not np.all(encoded["attention_mask"] == 1):
        raise RuntimeError("text batch did not fill every sequence position")
    token_ids = encoded["input_ids"].astype(np.int32)
    emit({
        "kind": "tokens",
        "shape": list(token_ids.shape),
        "sha256": hashlib.sha256(token_ids.tobytes()).hexdigest(),
        "unique_tokens": int(np.unique(token_ids).size),
    })
    return token_ids


def device_matrix(checkpoint, name):
    value = jnp.asarray(checkpoint.tensor(name).T)
    jax.block_until_ready(value)
    return value


def load_layer(checkpoint, index, rope_cos, rope_sin):
    prefix = f"model.layers.{index}"
    gate = device_matrix(checkpoint, f"{prefix}.mlp.gate_proj.weight")
    up = device_matrix(checkpoint, f"{prefix}.mlp.up_proj.weight")
    gate_up = jnp.concatenate((gate, up), axis=1)
    params = layer_bench.Parameters(
        attention_norm=jnp.asarray(
            checkpoint.tensor(f"{prefix}.input_layernorm.weight")
        ),
        query=device_matrix(checkpoint, f"{prefix}.self_attn.q_proj.weight"),
        key=device_matrix(checkpoint, f"{prefix}.self_attn.k_proj.weight"),
        value=device_matrix(checkpoint, f"{prefix}.self_attn.v_proj.weight"),
        attention_output=device_matrix(
            checkpoint, f"{prefix}.self_attn.o_proj.weight"
        ),
        mlp_norm=jnp.asarray(
            checkpoint.tensor(f"{prefix}.post_attention_layernorm.weight")
        ),
        gate_up=gate_up,
        mlp_down=device_matrix(checkpoint, f"{prefix}.mlp.down_proj.weight"),
        rope_cos=rope_cos,
        rope_sin=rope_sin,
    )
    jax.block_until_ready(params)
    return params


def logits_and_loss(hidden, lm_head, targets):
    hidden = hidden.reshape(
        layer_bench.BATCH, layer_bench.SEQUENCE, layer_bench.MODEL_DIM
    )[:, :-1, :]
    hidden = hidden.reshape(-1, layer_bench.MODEL_DIM)
    logits = sp.native_matmul(hidden, lm_head)
    logits32 = logits.astype(jnp.float32)
    selected = jnp.take_along_axis(logits32, targets[:, None], axis=1)[:, 0]
    loss = jnp.mean(jax.nn.logsumexp(logits32, axis=1) - selected)
    return logits, loss


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
        "profile": sp.VMEM_PROFILE,
        "layers": NUM_LAYERS,
        "quality_only": True,
        "candidate": "combined gate/up and down use tuned_matmul",
        "reference": "combined gate/up and down use native_matmul",
    })

    token_ids = make_tokens()
    paths = download_checkpoint()
    checkpoint = Checkpoint(paths)
    embeddings = checkpoint.tensor("model.embed_tokens.weight")
    initial = embeddings[token_ids].reshape(
        layer_bench.TOKENS, layer_bench.MODEL_DIM
    )
    native_hidden = jnp.asarray(initial)
    strassen_hidden = jnp.asarray(initial)
    rope_cos, rope_sin = layer_bench.rope_values()
    first_params = load_layer(checkpoint, 0, rope_cos, rope_sin)

    started = time.perf_counter()
    native_executable = jax.jit(
        layer_bench.make_layer("matched_native")
    ).lower(native_hidden, first_params).compile()
    strassen_executable = jax.jit(
        layer_bench.make_layer("both")
    ).lower(strassen_hidden, first_params).compile()
    emit({"kind": "compile", "compile_s": time.perf_counter() - started})

    all_finite = True
    final_hidden_error = None
    for index in range(NUM_LAYERS):
        params = (
            first_params
            if index == 0
            else load_layer(checkpoint, index, rope_cos, rope_sin)
        )
        native_hidden = native_executable(native_hidden, params)
        strassen_hidden = strassen_executable(strassen_hidden, params)
        jax.block_until_ready((native_hidden, strassen_hidden))
        error = bench.device_error(strassen_hidden, native_hidden)
        finite = bool(
            jax.device_get(
                jnp.all(jnp.isfinite(native_hidden))
                & jnp.all(jnp.isfinite(strassen_hidden))
            )
        )
        all_finite = all_finite and finite
        final_hidden_error = error
        emit({
            "kind": "layer_accuracy",
            "layer": index,
            "finite": finite,
            "strassen_vs_native": error,
        })
        del params
        if index == 0:
            del first_params
        gc.collect()

    final_norm = jnp.asarray(checkpoint.tensor("model.norm.weight"))
    native_hidden = layer_bench.rms_norm(native_hidden, final_norm)
    strassen_hidden = layer_bench.rms_norm(strassen_hidden, final_norm)
    jax.block_until_ready((native_hidden, strassen_hidden))
    normalized_error = bench.device_error(strassen_hidden, native_hidden)

    lm_head = device_matrix(checkpoint, "lm_head.weight")
    targets = jnp.asarray(token_ids[:, 1:].reshape(-1))
    logits_executable = jax.jit(logits_and_loss).lower(
        native_hidden, lm_head, targets
    ).compile()
    native_logits, native_loss = logits_executable(
        native_hidden, lm_head, targets
    )
    strassen_logits, strassen_loss = logits_executable(
        strassen_hidden, lm_head, targets
    )
    jax.block_until_ready(
        (native_logits, native_loss, strassen_logits, strassen_loss)
    )
    logits_error = bench.device_error(strassen_logits, native_logits)
    top1_agreement = float(
        jax.device_get(
            jnp.mean(
                jnp.argmax(native_logits, axis=-1)
                == jnp.argmax(strassen_logits, axis=-1)
            )
        )
    )
    native_loss, strassen_loss = map(
        float, jax.device_get((native_loss, strassen_loss))
    )
    absolute_loss_delta = abs(strassen_loss - native_loss)
    passes = (
        all_finite
        and normalized_error["l2_relative"] <= 0.02
        and normalized_error["maxnorm_relative"] <= 0.02
        and logits_error["l2_relative"] <= 0.02
        and logits_error["maxnorm_relative"] <= 0.02
        and top1_agreement >= 0.95
        and absolute_loss_delta <= 0.01
    )
    emit({
        "kind": "model_accuracy",
        "final_hidden_pre_norm": final_hidden_error,
        "final_hidden_post_norm": normalized_error,
        "logits": logits_error,
        "top1_agreement": top1_agreement,
        "native_next_token_loss": native_loss,
        "strassen_next_token_loss": strassen_loss,
        "absolute_loss_delta": absolute_loss_delta,
        "passes": passes,
    })
    emit({"kind": "final", "success": passes})
    if not passes:
        raise RuntimeError("full-checkpoint quality gate failed")


if __name__ == "__main__":
    main()
