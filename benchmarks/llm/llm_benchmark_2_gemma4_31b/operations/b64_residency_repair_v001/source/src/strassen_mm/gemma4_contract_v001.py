"""Gemma4 dense text metadata contract; not yet an executable model adapter.

Targets the pinned 31B architecture and smaller instances of the same design.
Reject unsupported model features before checkpoint acquisition or TPU allocation.
No checkpoint bytes, device work, numerical qualification, or queue changes occur
here. Reference: Transformers c1c34249fa27deefbd4a377dfbf883a39baf5c6d.
"""
from copy import deepcopy
import math

VERSION = "gemma4_contract_v001"
LAYER_TYPES = ("sliding_attention", "full_attention")


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def validate_config(raw):
    """Require explicit serialized geometry; preserve the original config."""
    if not isinstance(raw, dict):
        raise ValueError("Expected a serialized Gemma4 configuration")
    if raw.get("model_type") == "gemma4":
        if not isinstance(raw.get("text_config"), dict):
            raise ValueError("Gemma4 requires text_config")
        c = deepcopy(raw["text_config"])
    elif raw.get("model_type") == "gemma4_text":
        c = deepcopy(raw)
    else:
        raise ValueError("Only Gemma4 text configuration is supported")
    if c.get("model_type") != "gemma4_text":
        raise ValueError("Expected Gemma4 text decoder")
    for source in (raw, c):
        if source.get("quantization_config") is not None:
            raise ValueError("Quantized checkpoints are unsupported")
        for key in ("dtype", "torch_dtype"):
            if source.get(key) not in (None, "bfloat16"):
                raise ValueError("BF16 checkpoint parameters are required")
    for key in ("hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "head_dim",
                "global_head_dim", "num_global_key_value_heads", "vocab_size",
                "max_position_embeddings", "sliding_window"):
        _positive_int(c.get(key), key)
    for key in ("head_dim", "global_head_dim"):
        if c[key] % 2:
            raise ValueError("Rotary head widths must be even")
    for key in ("num_key_value_heads", "num_global_key_value_heads"):
        if c["num_attention_heads"] % c[key]:
            raise ValueError("Grouped-query head counts must divide query heads")
    for key in ("attention_bias", "enable_moe_block", "use_double_wide_mlp"):
        if c.get(key) is not False:
            raise ValueError(f"{key} must explicitly be false")
    for key in ("hidden_size_per_layer_input", "num_kv_shared_layers"):
        if type(c.get(key)) is not int or c[key] != 0:
            raise ValueError(f"{key} is unsupported and must be zero")
    if c.get("attention_k_eq_v") is not True:
        raise ValueError("This contract requires shared global K/V projection")
    if c.get("tie_word_embeddings") is not True:
        raise ValueError("This contract requires tied word embeddings")
    if "tie_word_embeddings" in raw and raw["tie_word_embeddings"] is not True:
        raise ValueError("Outer and text configurations must agree on tied embeddings")
    if c.get("attention_dropout") != 0 or c.get("mlp_bias", False):
        raise ValueError("Only dropout-free bias-free text inference is supported")
    if c.get("hidden_activation") != "gelu_pytorch_tanh":
        raise ValueError("Only gelu_pytorch_tanh is supported")
    if c.get("use_bidirectional_attention") not in (None, "vision"):
        raise ValueError("Only causal text attention is supported")
    if c.get("is_encoder_decoder", False):
        raise ValueError("Only decoder inference is supported")
    c["rms_norm_eps"] = _positive_float(c.get("rms_norm_eps"), "rms_norm_eps")
    if c.get("final_logit_softcapping") is not None:
        c["final_logit_softcapping"] = _positive_float(c["final_logit_softcapping"], "final_logit_softcapping")
    layer_types = c.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != c["num_hidden_layers"]:
        raise ValueError("Explicit layer_types must name every decoder layer")
    if any(t not in LAYER_TYPES for t in layer_types) or layer_types[-1] != "full_attention":
        raise ValueError("Unsupported layer_types; the final layer must be global")
    rope = c.get("rope_parameters")
    if not isinstance(rope, dict) or set(rope) != set(LAYER_TYPES):
        raise ValueError("Both local and global rope_parameters are required")
    if c.get("rope_scaling") not in (None, {}):
        raise ValueError("Legacy rope_scaling is unsupported")
    for kind, expected in zip(LAYER_TYPES, ("default", "proportional")):
        rp = rope[kind]
        allowed = {"rope_type", "rope_theta"}
        if kind == "full_attention":
            allowed |= {"partial_rotary_factor", "factor"}
        if not isinstance(rp, dict) or rp.get("rope_type") != expected or set(rp) - allowed:
            raise ValueError(f"Unsupported {kind} RoPE configuration")
        rp["rope_theta"] = _positive_float(rp.get("rope_theta"), "rope_theta")
    rp = rope["full_attention"]
    proportion = _positive_float(rp.get("partial_rotary_factor"), "partial_rotary_factor")
    rotated = proportion * c["global_head_dim"]
    if proportion > 1 or not rotated.is_integer() or int(rotated) % 2:
        raise ValueError("Proportional RoPE requires an integral even rotated width")
    if rp.get("factor", 1.0) != 1.0:
        raise ValueError("Scaled proportional RoPE is outside this contract")
    return c


def layer_geometry(raw, layer_index, batch=1, sequence=1):
    """Return actual MM sites in (M,K,N) order, never a fictitious global V."""
    c = validate_config(raw)
    if type(layer_index) is not int or not 0 <= layer_index < c["num_hidden_layers"]:
        raise ValueError("Layer index out of range")
    _positive_int(batch, "batch")
    _positive_int(sequence, "sequence")
    if sequence > c["max_position_embeddings"]:
        raise ValueError("Sequence exceeds the configured position limit")
    kind = c["layer_types"][layer_index]
    shared = kind == "full_attention"
    d = c["global_head_dim"] if shared else c["head_dim"]
    kv_heads = c["num_global_key_value_heads"] if shared else c["num_key_value_heads"]
    m, h, inner = batch * sequence, c["hidden_size"], c["intermediate_size"]
    q, kv = c["num_attention_heads"] * d, kv_heads * d
    shapes = {"q": [m, h, q], "k": [m, h, kv]}
    if not shared:
        shapes["v"] = [m, h, kv]
    shapes.update(o=[m, q, h], gateup=[m, h, 2 * inner], down=[m, inner, h])
    return dict(layer_type=kind, head_dim=d, query_heads=c["num_attention_heads"],
                kv_heads=kv_heads, shared_raw_kv=shared, attention_scale=1.0,
                value_rms_norm=True, sliding_window=None if shared else c["sliding_window"],
                projection_shapes_mkn=shapes)


def required_tensor_shapes(raw):
    """Canonical text state, including persistent buffers and the tied head.

    Shapes describe checkpoint layout, before projection transposes/packing.
    A serialized lm_head.weight alias is optional; any reader must verify its
    equality to embeddings if present. Do not allocate a second tied parameter.
    """
    c = validate_config(raw)
    h, inner = c["hidden_size"], c["intermediate_size"]
    result = {"model.embed_tokens.weight": [c["vocab_size"], h], "model.norm.weight": [h]}
    for i in range(c["num_hidden_layers"]):
        g = layer_geometry(c, i)
        shapes = g["projection_shapes_mkn"]
        prefix = f"model.layers.{i}."
        for site in ("q", "k", "v", "o"):
            if site in shapes:
                _, k, n = shapes[site]
                result[prefix + f"self_attn.{site}_proj.weight"] = [n, k]
        for site in ("q", "k"):
            result[prefix + f"self_attn.{site}_norm.weight"] = [g["head_dim"]]
        for name in ("input_layernorm", "post_attention_layernorm",
                     "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            result[prefix + name + ".weight"] = [h]
        for site in ("gate", "up"):
            result[prefix + f"mlp.{site}_proj.weight"] = [inner, h]
        result[prefix + "mlp.down_proj.weight"] = [h, inner]
        result[prefix + "layer_scalar"] = [1]
    return result
