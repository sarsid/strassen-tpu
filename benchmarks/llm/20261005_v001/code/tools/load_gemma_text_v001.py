"""Load the official Gemma text oracle independently of our JAX checkpoint reader.

Allocate only text parameters, copy shard tensors with strict name/shape checks,
and retain Transformers' CPU RoPE buffers. No vision parameters are allocated.
"""
import json
from pathlib import Path


def load_text(cache):
    import torch
    from safetensors import safe_open
    from transformers import Gemma3TextConfig, Gemma3ForCausalLM
    from transformers.modeling_utils import no_init_weights
    cache=Path(cache);raw=json.loads((cache/'config.json').read_text())
    if raw['model_type']!='gemma3':raise ValueError('Expected multimodal Gemma3 checkpoint')
    config=Gemma3TextConfig(**raw['text_config']);config._attn_implementation='eager'
    old=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with no_init_weights():model=Gemma3ForCausalLM(config)
    finally:torch.set_default_dtype(old)
    model.tie_weights()
    expected=dict(model.named_parameters());loaded=set()
    with torch.no_grad():
        for shard in sorted(cache.glob('*.safetensors')):
            with safe_open(shard,framework='pt',device='cpu') as reader:
                for original in reader.keys():
                    if original.startswith('language_model.'):
                        name=original[len('language_model.'):]
                    elif original.startswith('model.language_model.'):
                        name='model.'+original[len('model.language_model.'):]
                    elif original=='lm_head.weight':name=original
                    else:continue
                    if name=='lm_head.weight' and config.tie_word_embeddings and name not in expected:
                        continue  # tied parameter is strictly loaded through embed_tokens
                    if name not in expected or name in loaded:raise ValueError('Unexpected or duplicate text tensor: '+name)
                    value=reader.get_tensor(original);target=expected[name]
                    if value.shape!=target.shape or value.dtype!=torch.bfloat16:raise ValueError('Text tensor shape/dtype mismatch: '+name)
                    target.copy_(value);loaded.add(name)
    if loaded!=set(expected):raise ValueError('Missing text parameters: '+str(sorted(set(expected)-loaded)))
    if any(b.device.type!='cpu' for b in model.buffers()):raise ValueError('Non-CPU oracle buffer')
    return model.eval()
