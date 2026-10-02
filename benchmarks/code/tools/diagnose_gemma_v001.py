#!/usr/bin/env python3
"""Offline common-input Gemma diagnostics; no qualification gates or timings.

Run the pinned official 64-token model once, hook layer-0 operations and final
RMS, then compare JAX v001 eager/JIT operations on those exact official inputs.
No second complete adapter forward is performed. Old evidence is never edited.
Output is a new STRASSEN_EXECUTION_DIR/artifacts directory owned by the archive.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from qualify_gemma_v001 import (environment, manifest_identity, official_dependencies,
                               save_array, sha256, verify_manifest, write_json)


def metric(reference, candidate):
    """Descriptive whole-array metrics only; no new or relaxed acceptance gate."""
    import numpy as np
    r, c = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if r.shape != c.shape:
        return {"status": "shape_mismatch", "reference_shape": list(r.shape), "candidate_shape": list(c.shape)}
    finite = bool(np.isfinite(r).all() and np.isfinite(c).all())
    result = {"status": "finite" if finite else "nonfinite", "shape": list(r.shape),
              "exact_equal_fraction": float(np.mean(r == c)), "element_count": int(r.size)}
    if finite:
        delta = c - r
        result.update(relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(r), 1e-30)),
                      max_abs_error=float(np.max(np.abs(delta))),
                      reference_l2=float(np.linalg.norm(r)), candidate_l2=float(np.linalg.norm(c)),
                      unequal_elements=int(np.count_nonzero(delta)))
    return result


def capture(model, tokens):
    """Capture actual pinned module/function outputs, never reimplement oracle math."""
    import numpy as np
    import torch
    from transformers.models.gemma3 import modeling_gemma3 as official
    layer = model.model.layers[0]
    modules = {
        "input_norm": layer.input_layernorm,
        "q_linear": layer.self_attn.q_proj, "k_linear": layer.self_attn.k_proj,
        "v_linear": layer.self_attn.v_proj, "q_norm": layer.self_attn.q_norm,
        "k_norm": layer.self_attn.k_norm, "o_linear": layer.self_attn.o_proj,
        "post_attention_norm": layer.post_attention_layernorm,
        "pre_ffn_norm": layer.pre_feedforward_layernorm,
        "gate_linear": layer.mlp.gate_proj, "up_linear": layer.mlp.up_proj,
        "gelu": layer.mlp.act_fn, "down_linear": layer.mlp.down_proj,
        "post_ffn_norm": layer.post_feedforward_layernorm,
        "layer0": layer, "final_norm": model.model.norm,
    }
    records, handles, metadata = {}, [], {}

    def host(value):
        # Every capture is batch one; remove only that dimension.
        if value.shape[0] != 1:
            raise ValueError("Diagnostic oracle unexpectedly changed batch size")
        return value[0].detach().float().cpu().numpy().copy()

    for name, module in modules.items():
        def pre(mod, args, name=name):
            records[name + "_input"] = host(args[0])
        def post(mod, args, output, name=name):
            output = output[0] if isinstance(output, tuple) else output
            records[name + "_output"] = host(output)
        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))

    original_rope, original_attention = official.apply_rotary_pos_emb, official.eager_attention_forward
    calls = {"rope": 0, "attention": 0}

    def rotary(*args, **kwargs):
        output = original_rope(*args, **kwargs)
        if calls["rope"] == 0:
            q, k, cos, sin = args[:4]
            for name, value in (("q_input", q), ("k_input", k), ("cos", cos), ("sin", sin),
                                ("q_output", output[0]), ("k_output", output[1])):
                records["rope_" + name] = host(value)
        calls["rope"] += 1
        return output

    def attention(module, query, key, value, attention_mask, *args, **kwargs):
        output = original_attention(module, query, key, value, attention_mask, *args, **kwargs)
        if calls["attention"] == 0:
            for name, tensor in (("q", query), ("k", key), ("v", value),
                                 ("output", output[0]), ("probability", output[1])):
                records["attention_" + name] = host(tensor)
            if attention_mask is None:
                raise ValueError("Pinned eager prefill unexpectedly omitted causal mask")
            records["attention_mask"] = host(attention_mask)
            metadata["attention_scaling"] = kwargs.get("scaling", module.head_dim ** -0.5)
            metadata["attention_softcap_argument"] = kwargs.get("softcap")
        calls["attention"] += 1
        return output

    official.apply_rotary_pos_emb, official.eager_attention_forward = rotary, attention
    try:
        with torch.no_grad():
            result = model(torch.as_tensor(tokens, dtype=torch.long).unsqueeze(0),
                           use_cache=False, output_hidden_states=False, return_dict=True,
                           logits_to_keep=1)
            records["last_position_logits"] = host(result.logits)
    finally:
        official.apply_rotary_pos_emb, official.eager_attention_forward = original_rope, original_attention
        for handle in handles:
            handle.remove()
    if calls != {"rope": model.config.num_hidden_layers, "attention": model.config.num_hidden_layers}:
        raise ValueError("Did not capture the expected official forward call counts")
    required = {name + suffix for name in modules for suffix in ("_input", "_output")}
    required |= {"rope_q_input", "rope_q_output", "rope_k_input", "rope_k_output", "rope_cos", "rope_sin",
                 "attention_q", "attention_k", "attention_v", "attention_probability", "attention_output", "attention_mask"}
    if not required.issubset(records) or any(not np.isfinite(value).all() for value in records.values()):
        raise ValueError("Incomplete or nonfinite official intermediate capture")
    metadata["official_function_call_counts"] = calls
    metadata["saved_batch_dimension_removed"] = True
    metadata["qk_layout"] = "heads,sequence,head_dim; transpose(1,0,2) for adapter RoPE"
    return records, metadata


def run(args, output):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from strassen_mm import model_gemma_v001 as adapter
    manifest = verify_manifest(args.manifest)
    tokens = np.load(args.tokens, allow_pickle=False)
    if (tokens.ndim != 1 or tokens.dtype.kind not in "iu" or len(tokens) < 64 or
            tokens.min() < 0 or tokens.max() >= manifest["config"]["vocab_size"]):
        raise ValueError("Expected at least 64 valid frozen one-dimensional integer tokens")
    tokens = np.asarray(tokens[:64], dtype=np.int64)
    torch, Model, _ = official_dependencies()
    model = Model.from_pretrained(manifest["cache_dir"], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager", use_safetensors=True).to(device="cpu").eval()
    if any(p.dtype != torch.bfloat16 or p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Official parameters must be CPU BF16")
    records, oracle_metadata = capture(model, tokens)
    print("Captured official 64-token layer-0 and final-normalization intermediates", flush=True)
    for name, array in records.items():
        save_array(output / ("official_" + name + ".npy"), array)
    save_array(output / "tokens_first64.npy", tokens)
    del model
    gc.collect()

    checkpoint = adapter.Checkpoint(args.manifest)
    config = checkpoint.normalized_config
    weights = jax.tree_util.tree_map(jnp.asarray, checkpoint.layer(0))
    final_norm = jnp.asarray(checkpoint.tensor("model.norm.weight"))
    comparisons, candidates = {}, {}

    def array(value):
        return jnp.asarray(value, dtype=jnp.bfloat16)

    def compare(name, reference, fn, *inputs):
        # Explicit contexts make both variants real even if an earlier caller
        # happened to set JAX_DISABLE_JIT in its environment.
        with jax.disable_jit(True):
            eager = np.asarray(jax.block_until_ready(fn(*inputs)), dtype=np.float32)
        with jax.disable_jit(False):
            compiled = np.asarray(jax.block_until_ready(jax.jit(fn)(*inputs)), dtype=np.float32)
        comparisons[name] = {"eager_vs_official": metric(reference, eager),
                             "jit_vs_official": metric(reference, compiled),
                             "jit_vs_eager": metric(eager, compiled)}
        candidates[name] = {"eager": eager, "jit": compiled}
        save_array(output / ("jax_" + name + "_eager.npy"), eager)
        save_array(output / ("jax_" + name + "_jit.npy"), compiled)
        print(name + ": " + json.dumps(comparisons[name]), flush=True)

    norms = {"input_norm": "norm1", "post_attention_norm": "norm2",
             "pre_ffn_norm": "norm3", "post_ffn_norm": "norm4", "q_norm": "qnorm", "k_norm": "knorm"}
    for name, key in norms.items():
        # RMS is over the final dimension, so the official Q/K layout is valid.
        compare(name, records[name + "_output"],
                lambda x, w: adapter.rms(x, w, config["rms_norm_eps"]),
                array(records[name + "_input"]), weights[key])
    compare("final_norm_official_input", records["final_norm_output"],
            lambda x, w: adapter.rms(x, w, config["rms_norm_eps"]),
            array(records["final_norm_input"]), final_norm)

    projections = {"q_linear": weights["q"], "k_linear": weights["k"], "v_linear": weights["v"],
                   "o_linear": weights["o"], "down_linear": weights["down"]}
    gate, up = jnp.split(weights["gateup"], 2, axis=1)
    projections.update(gate_linear=gate, up_linear=up)
    for name, weight in projections.items():
        compare(name, records[name + "_output"], adapter.dot, array(records[name + "_input"]), weight)
    if not np.array_equal(records["gate_linear_input"], records["up_linear_input"]):
        raise ValueError("Official gate/up do not share the expected normalized input")
    joined_reference = np.concatenate((records["gate_linear_output"], records["up_linear_output"]), axis=-1)
    supplied = array(records["gate_linear_input"])
    compare("gateup_concatenated", joined_reference, adapter.dot, supplied, weights["gateup"])
    compare("gateup_split", joined_reference,
            lambda x, a, b: jnp.concatenate((adapter.dot(x, a), adapter.dot(x, b)), axis=-1), supplied, gate, up)
    comparisons["gateup_concatenated_vs_split"] = {
        mode: metric(candidates["gateup_split"][mode], candidates["gateup_concatenated"][mode])
        for mode in ("eager", "jit")}
    compare("gelu", records["gelu_output"], adapter.gelu, array(records["gelu_input"]))
    compare("gelu_times_up", records["down_linear_input"],
            lambda a, b: (a * b).astype(jnp.bfloat16),
            array(records["gelu_output"]), array(records["up_linear_output"]))
    compare("gate_gelu_times_up", records["down_linear_input"],
            lambda a, b: (adapter.gelu(a) * b).astype(jnp.bfloat16),
            array(records["gate_linear_output"]), array(records["up_linear_output"]))

    theta = config["rope_local_base_freq"] if config["layer_types"][0] == "sliding_attention" else config["rope_theta"]
    cosine, sine = array(records["rope_cos"])[:, None, :], array(records["rope_sin"])[:, None, :]
    for key in ("q", "k"):
        supplied = array(records["rope_" + key + "_input"].transpose(1, 0, 2))
        reference = records["rope_" + key + "_output"].transpose(1, 0, 2)
        compare("rope_" + key, reference, lambda x: adapter.rope(x, theta), supplied)
        # Common official cos/sin separates trigonometric generation differences
        # from multiply/add BF16 rounding within apply_rotary_pos_emb.
        def supplied_trig(x, cos, sin):
            left = (x * cos).astype(jnp.bfloat16)
            right = (adapter.rotate_half(x) * sin).astype(jnp.bfloat16)
            return (left + right).astype(jnp.bfloat16)
        compare("rope_" + key + "_official_trig", reference, supplied_trig, supplied, cosine, sine)
        # Independent official primitive products, not a reconstructed oracle.
        with torch.no_grad():
            tx = torch.as_tensor(np.asarray(supplied, dtype=np.float32), dtype=torch.bfloat16)
            tc = torch.as_tensor(records["rope_cos"], dtype=torch.bfloat16)[:, None, :]
            ts = torch.as_tensor(records["rope_sin"], dtype=torch.bfloat16)[:, None, :]
            from transformers.models.gemma3.modeling_gemma3 import rotate_half
            left = (tx * tc).float().numpy().copy()
            right = (rotate_half(tx) * ts).float().numpy().copy()
        save_array(output / ("official_rope_" + key + "_left.npy"), left)
        save_array(output / ("official_rope_" + key + "_right.npy"), right)
        compare("rope_" + key + "_left", left, lambda x, y: (x * y).astype(jnp.bfloat16), supplied, cosine)
        compare("rope_" + key + "_right", right,
                lambda x, y: (adapter.rotate_half(x) * y).astype(jnp.bfloat16), supplied, sine)

    # Attention replay uses captured official Q/K/V and its additive causal
    # mask. Targets are returned by official eager_attention_forward itself.
    q, k, v = (array(records["attention_" + name].transpose(1, 0, 2)) for name in ("q", "k", "v"))
    mask = array(records["attention_mask"])
    scaling = float(oracle_metadata["attention_scaling"])
    if oracle_metadata["attention_softcap_argument"] is not None:
        raise ValueError("Pinned diagnostic expects the actual null attention softcap")
    def probabilities(q, k, mask):
        k = jnp.repeat(k, config["num_attention_heads"] // config["num_key_value_heads"], axis=1)
        scores = jnp.einsum("thd,shd->hts", q, k, precision=jax.lax.Precision.DEFAULT,
                            preferred_element_type=jnp.float32).astype(jnp.bfloat16)
        scores = (scores.astype(jnp.float32) * scaling).astype(jnp.bfloat16)
        scores = (scores + mask).astype(jnp.bfloat16)
        return jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)
    compare("attention_probability", records["attention_probability"], probabilities, q, k, mask)
    def attended(probability, v):
        v = jnp.repeat(v, config["num_attention_heads"] // config["num_key_value_heads"], axis=1)
        return jnp.einsum("hts,shd->thd", probability, v, precision=jax.lax.Precision.DEFAULT,
                          preferred_element_type=jnp.float32).astype(jnp.bfloat16)
    compare("attention_output_official_probability", records["attention_output"], attended,
            array(records["attention_probability"]), v)
    compare("attention_combined", records["attention_output"],
            lambda q, k, v, mask: attended(probabilities(q, k, mask), v), q, k, v, mask)
    compare("attention_residual", records["pre_ffn_norm_input"],
            lambda a, b: (a + b).astype(jnp.bfloat16),
            array(records["layer0_input"]), array(records["post_attention_norm_output"]))
    compare("ffn_residual", records["layer0_output"],
            lambda a, b: (a + b).astype(jnp.bfloat16),
            array(records["pre_ffn_norm_input"]), array(records["post_ffn_norm_output"]))
    layer = adapter.build_layer(checkpoint.config, 64, adapter.native_policy(), layer_index=0)
    compare("layer0_common_input", records["layer0_output"], layer,
            array(records["layer0_input"]), weights)

    return {"status": "completed", "diagnostic_only": True, "performance_claim": False,
            "qualification_gate_changes": False, "sequence_length": 64,
            "scope": "common official inputs; one official full forward; adapter layer0 only",
            "model_identity": manifest_identity(manifest), "manifest_sha256": sha256(args.manifest),
            "source_tokens_sha256": sha256(args.tokens), "diagnostic_source_sha256": sha256(__file__),
            "environment": environment(adapter), "ambient_jax_disable_jit": bool(jax.config.jax_disable_jit),
            "oracle_metadata": oracle_metadata, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    args = parser.parse_args()
    if not args.manifest.is_absolute() or not args.tokens.is_absolute():
        parser.error("Input paths must be absolute so archived source execution is unambiguous")
    output = Path(os.environ["STRASSEN_EXECUTION_DIR"]) / "artifacts"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "arguments.json", {"manifest": str(args.manifest), "tokens": str(args.tokens)})
    try:
        result = run(args, output)
    except Exception as error:
        result = {"status": "failed", "diagnostic_only": True, "performance_claim": False,
                  "error": str(error), "traceback": traceback.format_exc()}
    write_json(output / "summary.json", result)
    write_json(output / "artifact_manifest.json", {"schema_version": 1, "sha256": {
        p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(json.dumps({"status": result["status"], "output_dir": str(output)}), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
