"""Check our block against the published reference implementation.

Every quality gate in this project compares a Pallas arm to a ``regular_xla``
arm, but both are *our* block with different matmuls, so any error in the
block itself is shared and cancels.  The gates therefore measure
self-consistency, not fidelity.  The task accuracies say that matters:
HellaSwag reads 0.506 for Qwen and 0.338 for Gemma against published figures
near 0.83 and 0.85, and 0.25 is chance.

This runs one real decoder layer through transformers and through our own
formulation, on identical weights and identical input, on CPU.  It needs no
TPU and no Pallas: the question is whether the surrounding block arithmetic
is right, which is upstream of anything the kernel does.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def load_layer_tensors(repository, revision, prefix, names, token):
    """Fetch a few named tensors by ranged read, as the harnesses do."""
    import requests

    base = f"https://huggingface.co/{repository}/resolve/{revision}"
    session = requests.Session()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    index = session.get(f"{base}/model.safetensors.index.json", timeout=120)
    index.raise_for_status()
    weight_map = index.json()["weight_map"]

    headers, out = {}, {}
    for name in names:
        full = f"{prefix}.{name}"
        shard = weight_map[full]
        if shard not in headers:
            url = f"{base}/{shard}"
            size = int.from_bytes(
                session.get(url, headers={"Range": "bytes=0-7"},
                            timeout=120).content, "little")
            blob = session.get(
                url, headers={"Range": f"bytes=8-{7 + size}"},
                timeout=300).content
            headers[shard] = (json.loads(blob), 8 + size)
        meta, start = headers[shard]
        info = meta[full]
        lo, hi = info["data_offsets"]
        raw = session.get(
            f"{base}/{shard}",
            headers={"Range": f"bytes={start + lo}-{start + hi - 1}"},
            timeout=600).content
        array = np.frombuffer(raw, dtype=np.uint16).reshape(info["shape"])
        out[name] = torch.from_numpy(array.copy()).view(torch.bfloat16)
    return out


def gemma_reference(config, weights, hidden):
    from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer

    layer = Gemma3DecoderLayer(config, layer_idx=0).to(torch.float32).eval()
    state = {
        "self_attn.q_proj.weight": "self_attn.q_proj.weight",
        "self_attn.k_proj.weight": "self_attn.k_proj.weight",
        "self_attn.v_proj.weight": "self_attn.v_proj.weight",
        "self_attn.o_proj.weight": "self_attn.o_proj.weight",
        "self_attn.q_norm.weight": "self_attn.q_norm.weight",
        "self_attn.k_norm.weight": "self_attn.k_norm.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight": "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight":
            "post_feedforward_layernorm.weight",
    }
    loaded = layer.state_dict()
    for dst, src in state.items():
        loaded[dst] = weights[src].to(torch.float32)
    layer.load_state_dict(loaded)

    seq = hidden.shape[1]
    position_ids = torch.arange(seq)[None, :]
    rotary = layer.self_attn.rotary_emb if hasattr(
        layer.self_attn, "rotary_emb") else None
    with torch.no_grad():
        # Gemma 3 keeps a separate inv_freq per attention kind, so the
        # rotary module needs to be told which one layer 0 is: sliding.
        from transformers.models.gemma3.modeling_gemma3 import (
            Gemma3RotaryEmbedding)
        rotary = rotary or Gemma3RotaryEmbedding(config)
        embeddings = rotary(hidden, position_ids,
                            layer_type="sliding_attention")
        # attention_mask=None means *no* mask in the eager path, i.e.
        # bidirectional attention.  Our block is causal, so comparing
        # against an unmasked reference measures the mask, not the block.
        seq_len = hidden.shape[1]
        causal = torch.full((seq_len, seq_len), float("-inf")).triu(1)
        out = layer(hidden, position_embeddings=embeddings,
                    position_ids=position_ids,
                    attention_mask=causal[None, None])
    return out[0] if isinstance(out, tuple) else out


def our_block(config, w, hidden):
    """Our formulation, in torch, exactly as benchmark_gemma3_block states it."""
    eps = 1e-6
    heads, kv_heads, head_dim = 32, 16, 128
    dim = hidden.shape[-1]

    def norm(x, weight):
        v = x.to(torch.float32)
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        return v * (1.0 + weight.to(torch.float32))

    def rope_tables(seq, base=10000.0):
        pos = torch.arange(seq, dtype=torch.float32)[:, None]
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        ang = torch.cat([pos * inv[None, :]] * 2, dim=-1)
        return ang.cos(), ang.sin()

    def apply_rope(x, cos, sin):
        first, second = x.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return x * cos[None, :, None, :] + rotated * sin[None, :, None, :]

    seq = hidden.shape[1]
    residual = hidden
    normed = norm(hidden, w["input_layernorm.weight"])
    q = (normed @ w["self_attn.q_proj.weight"].to(torch.float32).T
         ).view(1, seq, heads, head_dim)
    k = (normed @ w["self_attn.k_proj.weight"].to(torch.float32).T
         ).view(1, seq, kv_heads, head_dim)
    v = (normed @ w["self_attn.v_proj.weight"].to(torch.float32).T
         ).view(1, seq, kv_heads, head_dim)
    q = norm(q, w["self_attn.q_norm.weight"])
    k = norm(k, w["self_attn.k_norm.weight"])
    cos, sin = rope_tables(seq)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    repeat = heads // kv_heads
    kk = k.repeat_interleave(repeat, dim=2)
    vv = v.repeat_interleave(repeat, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, kk) * (168 ** -0.5)
    mask = torch.full((seq, seq), float("-inf")).triu(1)
    attended = torch.einsum(
        "bhqk,bkhd->bqhd", (scores + mask).softmax(-1), vv
    ).reshape(1, seq, heads * head_dim)

    out = attended @ w["self_attn.o_proj.weight"].to(torch.float32).T
    residual = residual + norm(out, w["post_attention_layernorm.weight"])

    h = norm(residual, w["pre_feedforward_layernorm.weight"])
    gate = h @ w["mlp.gate_proj.weight"].to(torch.float32).T
    up = h @ w["mlp.up_proj.weight"].to(torch.float32).T
    act = torch.nn.functional.gelu(gate, approximate="tanh") * up
    down = act @ w["mlp.down_proj.weight"].to(torch.float32).T
    return residual + norm(down, w["post_feedforward_layernorm.weight"])


def qwen_reference(config, weights, hidden):
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3DecoderLayer, Qwen3RotaryEmbedding)

    layer = Qwen3DecoderLayer(config, layer_idx=0).to(torch.float32).eval()
    loaded = layer.state_dict()
    for key in loaded:
        if key in weights:
            loaded[key] = weights[key].to(torch.float32)
    layer.load_state_dict(loaded)
    seq = hidden.shape[1]
    position_ids = torch.arange(seq)[None, :]
    with torch.no_grad():
        embeddings = Qwen3RotaryEmbedding(config)(hidden, position_ids)
        causal = torch.full((seq, seq), float("-inf")).triu(1)
        out = layer(hidden, position_embeddings=embeddings,
                    position_ids=position_ids,
                    attention_mask=causal[None, None])
    return out[0] if isinstance(out, tuple) else out


def our_qwen_block(config, w, hidden):
    """Qwen3 is pre-norm and gates with SiLU; scale is head_dim ** -0.5."""
    eps = config.rms_norm_eps
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    def norm(x, weight):
        v = x.to(torch.float32)
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        return v * weight.to(torch.float32)          # Qwen: w, not (1 + w)

    def rope_tables(seq, base):
        pos = torch.arange(seq, dtype=torch.float32)[:, None]
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        ang = torch.cat([pos * inv[None, :]] * 2, dim=-1)
        return ang.cos(), ang.sin()

    def apply_rope(x, cos, sin):
        first, second = x.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return x * cos[None, :, None, :] + rotated * sin[None, :, None, :]

    seq = hidden.shape[1]
    residual = hidden
    normed = norm(hidden, w["input_layernorm.weight"])
    q = (normed @ w["self_attn.q_proj.weight"].to(torch.float32).T
         ).view(1, seq, heads, head_dim)
    k = (normed @ w["self_attn.k_proj.weight"].to(torch.float32).T
         ).view(1, seq, kv_heads, head_dim)
    v = (normed @ w["self_attn.v_proj.weight"].to(torch.float32).T
         ).view(1, seq, kv_heads, head_dim)
    q = norm(q, w["self_attn.q_norm.weight"])
    k = norm(k, w["self_attn.k_norm.weight"])
    params = getattr(config, "rope_parameters", None) or {}
    base = params.get("rope_theta", getattr(config, "rope_theta", 1e6))
    cos, sin = rope_tables(seq, base)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    repeat = heads // kv_heads
    kk = k.repeat_interleave(repeat, dim=2)
    vv = v.repeat_interleave(repeat, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, kk) * (head_dim ** -0.5)
    mask = torch.full((seq, seq), float("-inf")).triu(1)
    attended = torch.einsum(
        "bhqk,bkhd->bqhd", (scores + mask).softmax(-1), vv
    ).reshape(1, seq, heads * head_dim)
    residual = residual + attended @ w["self_attn.o_proj.weight"].to(
        torch.float32).T

    h = norm(residual, w["post_attention_layernorm.weight"])
    gate = h @ w["mlp.gate_proj.weight"].to(torch.float32).T
    up = h @ w["mlp.up_proj.weight"].to(torch.float32).T
    down = (torch.nn.functional.silu(gate) * up) @ w[
        "mlp.down_proj.weight"].to(torch.float32).T
    return residual + down


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma3-27b")
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()

    import huggingface_hub as hub
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import benchmark_qwen3_32b_layer as layer  # noqa: E402

    qwen = layer.MODEL_NAME != "gemma3-27b"
    names = [
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        "input_layernorm.weight", "post_attention_layernorm.weight",
    ] + ([] if qwen else [
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight"])
    weights = load_layer_tensors(
        layer.REPOSITORY, layer.REVISION, f"{layer.TENSOR_PREFIX}.layers.0",
        names, hub.get_token())
    torch.manual_seed(0)
    dim = weights["input_layernorm.weight"].shape[0]
    hidden = (torch.randn(1, args.tokens, dim) * 0.05).to(torch.float32)

    if qwen:
        from transformers import Qwen3Config
        config = Qwen3Config(
            hidden_size=dim,
            intermediate_size=weights["mlp.up_proj.weight"].shape[0],
            num_hidden_layers=1, num_attention_heads=layer.HEADS,
            num_key_value_heads=layer.KV_HEADS, head_dim=layer.HEAD_DIM,
            rms_norm_eps=layer.RMS_EPS,
            rope_parameters={"rope_type": "default",
                             "rope_theta": layer.ROPE_THETA},
            attn_implementation="eager")
        reference = qwen_reference(config, weights, hidden)
        ours = our_qwen_block(config, weights, hidden)
        report(layer, args, ours, reference)
        return

    from transformers import Gemma3TextConfig
    config = Gemma3TextConfig(
        hidden_size=dim, intermediate_size=weights["mlp.up_proj.weight"].shape[0],
        num_hidden_layers=1, num_attention_heads=32, num_key_value_heads=16,
        head_dim=128, query_pre_attn_scalar=168, sliding_window=1024,
        rope_scaling={"factor": 8.0, "rope_type": "linear"},
        attn_implementation="eager")
    reference = gemma_reference(config, weights, hidden)
    ours = our_block(config, weights, hidden)
    report(layer, args, ours, reference)


def report(layer, args, ours, reference):
    diff = (ours - reference).abs()
    scale = reference.abs().mean().item()
    print(json.dumps({
        "repository": layer.REPOSITORY, "revision": layer.REVISION,
        "tokens": args.tokens,
        "l2_relative": (torch.linalg.vector_norm(ours - reference)
                        / torch.linalg.vector_norm(reference)).item(),
        "max_abs": diff.max().item(),
        "mean_abs_reference": scale,
        "verdict": "FAITHFUL" if (
            torch.linalg.vector_norm(ours - reference)
            / torch.linalg.vector_norm(reference)).item() < 1e-3 else
        "DIVERGENT -- our block does not reproduce the reference",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
