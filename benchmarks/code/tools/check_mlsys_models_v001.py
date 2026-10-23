"""CPU numerical regressions for the newly composed model execution path."""
import json
import os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from strassen_mm import model_n9_v001 as qm,model_gemma_v001 as gm,model_composed_v001 as composed
from strassen_mm.kernels_v002 import make_matmul


def main():
    rng=np.random.default_rng(2171);checks=[]
    def tensor(shape):return jnp.asarray(rng.standard_normal(shape).astype(np.float32)*.1,dtype=jnp.bfloat16)
    for kind in ('qwen3','mistral','gemma3_text'):
        cfg=dict(model_type=kind,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,rms_norm_eps=1e-6,rope_theta=10000.,
            hidden_act='silu',rope_scaling=None,sliding_window=None)
        if kind=='gemma3_text':
            cfg.update(vocab_size=97,max_position_embeddings=1024,sliding_window=8,
                layer_types=['sliding_attention','full_attention'],hidden_activation='gelu_pytorch_tanh',
                rope_local_base_freq=10000.,query_pre_attn_scalar=8,tie_word_embeddings=True)
        mod=gm if kind=='gemma3_text' else qm
        w={k:tensor(shape) for k,shape in dict(q=(32,32),k=(32,16),v=(32,16),o=(32,32),gateup=(32,128),down=(64,32),norm1=(32,),norm2=(32,)).items()}
        if kind in ('qwen3','gemma3_text'):w.update(qnorm=tensor((8,)),knorm=tensor((8,)))
        if kind=='gemma3_text':w.update(norm3=tensor((32,)),norm4=tensor((32,)))
        x=tensor((17,32))
        for index in range(2):
            full=(mod.build_layer(cfg,17,mod.native_policy(),layer_index=index) if mod is gm else mod.build_layer(cfg,17,mod.native_policy()))
            fn=composed.Layer(composed.prefix(cfg,17,index),jax.jit(make_matmul('native',(17,32,128))),
                composed.activation(cfg),jax.jit(make_matmul('native',(17,64,32))),composed.suffix(cfg))
            a=np.asarray(full(x,w),np.float32);b=np.asarray(fn(x,w),np.float32)
            l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30))
            assert np.isfinite(b).all() and l2<.02,(kind,index,l2)
            checks.append(dict(model_type=kind,layer=index,relative_l2=l2,passed=True))
    from strassen_mm import benchmark_mlsys_shapes_v001 as grid
    source=Path(__file__).resolve().parents[1]
    cfg=json.loads((source/'configs/mlsys_real_models_v001/campaign.json').read_text())
    registered=grid.candidates(cfg)
    assert all(sum(x['family']==f for x in registered)==16 for f in ('cubic','one_level','two_level'))
    assert sum(x['family']=='native' for x in registered)==4
    assert cfg['operand_windows']==[0,1,2,3] and len(set(cfg['confirm_seeds']))==3
    assert cfg['quality_windows']==32
    # Exact checkpoint dimensions from preserved immutable official manifests.
    manifest_paths=[source/'configs/generated_n9_v001/model-00-model_manifest.json',
                    source/'configs/generated_n9_v001/model-01-model_manifest.json']
    shapes=json.loads((source/'configs/mlsys_real_models_v001/shapes.json').read_text())['shapes']
    for key,path in zip(('qwen','mistral'),manifest_paths):
        c=json.loads(path.read_text())['config']
        assert next(s for s in shapes if s['id']==key+'_gateup_m1024')['k']==c['hidden_size']
        assert next(s for s in shapes if s['id']==key+'_gateup_m1024')['n']==2*c['intermediate_size']
    out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    result=dict(passed=True,checks=checks,scope='CPU composed Native parity with preserved full-JIT adapters; actual checkpoint official qualification separately required on TPU',registered_candidates=len(registered))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
