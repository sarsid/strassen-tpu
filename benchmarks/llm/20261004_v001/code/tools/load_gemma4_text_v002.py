"""Independent official Gemma4 text loader using safetensors/PyTorch.

Accepts a caller-verified local checkpoint directory. It does not use the JAX
reader, parser or tensor geometry helper. Preserve persistent scalar dtypes and
official FP32 rotary buffers; require exact text state and tied aliases.
"""
import json
from pathlib import Path


def load_text(cache):
    import torch
    import transformers
    from safetensors import safe_open
    from transformers import Gemma4TextConfig, Gemma4ForCausalLM
    from transformers.initialization import no_init_weights
    if transformers.__version__ != "5.5.0":
        raise ValueError("Official Gemma4 oracle requires Transformers 5.5.0")
    cache = Path(cache)
    raw = json.loads((cache / "config.json").read_text())
    if raw.get("model_type") not in ("gemma4", "gemma4_text"):
        raise ValueError("Expected official Gemma4 configuration")
    config = Gemma4TextConfig(**raw.get("text_config", raw))
    config._attn_implementation = "eager"
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with no_init_weights():
            model = Gemma4ForCausalLM(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.tie_weights()
    expected = dict(model.state_dict())
    if not config.tie_word_embeddings:
        raise ValueError("Oracle loader requires tied embeddings")
    expected.pop("lm_head.weight")
    loaded, aliases = set(), []
    with torch.no_grad():
        for shard in sorted(cache.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as reader:
                for original in reader.keys():
                    if original.startswith("model.language_model."):
                        name = "model." + original[len("model.language_model."):]
                    elif original.startswith("language_model."):
                        name = original[len("language_model."):]
                        if not name.startswith("model.") and name != "lm_head.weight":
                            name = "model." + name
                    elif original.startswith(("model.layers.", "model.embed_tokens.", "model.norm.")) or original == "lm_head.weight":
                        name = original
                    else:
                        continue
                    if name == "lm_head.weight":
                        if aliases:
                            raise ValueError("Duplicate tied head alias")
                        aliases.append((shard, original))
                        continue
                    if name not in expected or name in loaded:
                        raise ValueError("Unexpected or duplicate official text state: " + name)
                    value = reader.get_tensor(original)
                    target = expected[name]
                    scalar = name.endswith(".layer_scalar")
                    if value.shape != target.shape or value.dtype not in ((torch.bfloat16, torch.float32) if scalar else (torch.bfloat16,)):
                        raise ValueError("Official text tensor shape/dtype mismatch: " + name)
                    if scalar:
                        owner = model.get_submodule(name.rsplit(".", 1)[0])
                        owner.layer_scalar = value.clone()
                    else:
                        target.copy_(value)
                    loaded.add(name)
        if loaded != set(expected):
            raise ValueError("Missing official text state: " + str(sorted(set(expected)-loaded)))
        for shard, name in aliases:
            with safe_open(shard, framework="pt", device="cpu") as reader:
                value = reader.get_tensor(name)
                embedding = model.model.embed_tokens.weight
                if value.shape != embedding.shape or value.dtype != embedding.dtype:
                    raise ValueError("Tied head shape/dtype mismatch")
                for start in range(0, len(value), 1024):
                    if not torch.equal(value[start:start+1024].view(torch.int16), embedding[start:start+1024].view(torch.int16)):
                        raise ValueError("Tied head differs from embeddings")
    for _, value in model.model.rotary_emb.named_buffers():
        if value.dtype != torch.float32:
            raise ValueError("Official rotary buffers must remain FP32")
    return model.eval()
