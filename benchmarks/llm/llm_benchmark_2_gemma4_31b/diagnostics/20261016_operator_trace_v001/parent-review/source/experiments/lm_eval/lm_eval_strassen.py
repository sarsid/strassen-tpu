"""Expose the streamed Strassen model to lm-evaluation-harness.

Our own task harness reimplements scoring, and that reimplementation is
where the published comparison broke down: we reported raw HellaSwag
accuracy against tables that publish acc_norm, and no amount of care in our
code makes our protocol identical to theirs.  The only way to claim a
like-for-like number is to let the reference harness drive.

The awkward part is that this model streams 54 GB of weights layer by layer,
so a forward pass is enormously expensive and must be amortised over as many
sequences as possible.  lm-eval hands us every request at once, which is
exactly what that needs: all requests are packed into (BATCH, SEQUENCE)
blocks, every block is carried through each layer together, and the whole
model is streamed once per call rather than once per request.

Memory is the binding constraint.  Each block holds
BATCH*SEQUENCE*model_dim BF16 values, about 22 MB at 8x256 for Gemma, so a
few hundred blocks is the practical ceiling on one chip.  Use lm-eval's
--limit and report it: a subsample scored under the real protocol is worth
more than the full set scored under ours.

    python3 run_lm_eval_strassen.py --arm gated_strassen \\
        --tasks hellaswag,lambada_openai --limit 200
"""

from __future__ import annotations

import math
import os
from typing import List, Tuple

import numpy as np

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@register_model("strassen")
class StrassenLM(LM):
    """A streamed Gemma/Qwen block stack, scored one arm at a time."""

    def __init__(self, arm="gated_strassen", batch_size=None, **kwargs):
        super().__init__()
        import jax
        import jax.numpy as jnp

        import benchmark_qwen3_32b_layer as layer

        self.jax, self.jnp = jax, jnp
        self.layer = layer
        self.arm = arm
        self.gemma = layer.MODEL_NAME.startswith("gemma")
        if self.gemma:
            import benchmark_gemma3_block as block
            import benchmark_gemma3_streamed as stream
        else:
            import benchmark_qwen3_32b_streamed_inference as stream
            block = None
        self.block, self.stream = block, stream
        self.seq = layer.SEQUENCE
        self.batch = int(batch_size or layer.BATCH)
        self.num_layers = int(
            os.environ.get("GEMMA_LAYERS", layer._MODEL["num_layers"]))
        self._tokenizer = None
        self._checkpoint = None

    # -- plumbing -------------------------------------------------------
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.layer.REPOSITORY, revision=self.layer.REVISION,
                use_fast=True, token=self.layer._hub_token())
        return self._tokenizer

    @property
    def checkpoint(self):
        if self._checkpoint is None:
            self._checkpoint = self.layer.ShardedSafetensors()
        return self._checkpoint

    def _encode(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    # -- the LM interface ------------------------------------------------
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        pairs = [request.args for request in requests]
        # Gemma is trained with a leading <bos> and degrades measurably
        # without one; lm-eval's own HF model prepends it, so a wrapper that
        # omits it is not running the published protocol.  Only the context
        # gets it, never the continuation.
        bos = getattr(self.tokenizer, "bos_token_id", None)
        rows, spans = [], []
        for context, continuation in pairs:
            context_ids = self._encode(context) if context else []
            if bos is not None:
                context_ids = [bos] + list(context_ids)
            target_ids = self._encode(continuation)
            ids = list(context_ids) + list(target_ids)
            if len(ids) > self.seq:
                ids = ids[-self.seq:]
            start = len(ids) - len(target_ids)
            if start < 1:
                # lm-eval permits an empty context; keep one token of room so
                # every scored position has a predecessor.
                ids = ids[-(self.seq - 1):]
                start = max(1, len(ids) - len(target_ids))
            row = np.zeros((self.seq,), np.int32)
            row[: len(ids)] = ids
            spans.append((len(rows), start, len(ids)))
            rows.append(row)

        while len(rows) % self.batch:
            rows.append(np.zeros((self.seq,), np.int32))
        matrix = np.stack(rows)
        hidden = self._forward(matrix)
        return self._score(hidden, matrix, spans)

    def loglikelihood_rolling(self, requests) -> List[float]:
        raise NotImplementedError(
            "rolling loglikelihood needs windowed scoring over documents "
            "longer than the pinned sequence; not implemented rather than "
            "silently wrong")

    def generate_until(self, requests) -> List[str]:
        raise NotImplementedError(
            "this wrapper scores likelihoods only; generation would need a "
            "decode path the streamed harness does not have")

    # -- model execution -------------------------------------------------
    def _forward(self, matrix):
        """Stream every layer once, carrying all blocks together."""
        jnp = self.jnp
        layer = self.layer
        embeddings = self.checkpoint.tensor(
            f"{layer.TENSOR_PREFIX}.embed_tokens.weight")
        blocks = matrix.reshape(-1, self.batch, self.seq)
        state = [jnp.asarray(np.asarray(embeddings)[chunk.reshape(-1)],
                             jnp.bfloat16) for chunk in blocks]
        if self.gemma:
            scale = math.sqrt(layer.MODEL_DIM)
            state = [(v.astype(jnp.float32) * scale).astype(jnp.bfloat16)
                     for v in state]
        run = self.jax.jit(self.stream.make_layer(self.arm))
        for index in range(self.num_layers):
            params = self.stream.load_layer(self.checkpoint, index)
            state = [self.jax.block_until_ready(run(v, params)) for v in state]
            del params
        return state

    def _score(self, state, matrix, spans):
        jnp, jax = self.jnp, self.jax
        layer = self.layer
        final = jnp.asarray(
            self.checkpoint.tensor(f"{layer.TENSOR_PREFIX}.norm.weight"))
        if self.gemma:
            head = jnp.asarray(self.checkpoint.tensor(
                f"{layer.TENSOR_PREFIX}.embed_tokens.weight")).T
            normalise = self.block.gemma_rms_norm
        else:
            head = jnp.asarray(
                self.checkpoint.tensor("lm_head.weight")).T
            normalise = layer.rms_norm

        stacked = jnp.concatenate(state).reshape(-1, self.seq, layer.MODEL_DIM)
        out = []
        for row, start, end in spans:
            hidden = normalise(stacked[row, start - 1:end - 1], final)
            logits = jnp.matmul(hidden.astype(jnp.float32),
                                head.astype(jnp.float32))
            logp = jax.nn.log_softmax(logits, axis=-1)
            targets = jnp.asarray(matrix[row, start:end], jnp.int32)
            gold = jnp.take_along_axis(logp, targets[:, None], axis=1)[:, 0]
            greedy = bool(jnp.all(logits.argmax(-1) == targets))
            out.append((float(gold.sum()), greedy))
        return out
