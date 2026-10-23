"""CPU Native model equivalence and shape/menu checks before TPU execution."""
import ast, importlib, json, os
from pathlib import Path
import parent_stage_v001 as setup

def main():
    setup.configure(Path('/tmp/unused-parent-adapter-output'))
    layer=importlib.import_module('benchmark_qwen3_32b_layer')
    layer.BATCH=1;layer.SEQUENCE=8;layer.TOKENS=8;layer.MODEL_DIM=256
    layer.INTERMEDIATE_DIM=512;layer.HEADS=2;layer.KV_HEADS=1
    import jax
    import jax.numpy as jnp
    import numpy as np
    from strassen_mm import parent_fused_adapter_v001 as adapter
    from strassen_mm.tuner_joint_v001 import arm
    import tune_parent_fused_v001 as tune
    rng=np.random.default_rng(20261002)
    def rand(shape):return jnp.asarray(rng.normal(size=shape)*.03,jnp.bfloat16)
    cos,sin=layer.rope_values()
    p=layer.Parameters(jnp.ones((256,),jnp.bfloat16),rand((256,256)),jnp.ones((128,),jnp.bfloat16),rand((256,128)),jnp.ones((128,),jnp.bfloat16),
                       rand((256,128)),rand((256,256)),jnp.ones((256,),jnp.bfloat16),rand((256,1024)),rand((512,256)),cos,sin)
    policy={site:dict(arm(dtype='bfloat16'),architecture='v6e') for site in adapter.SITES}
    builder=adapter.build(layer,policy)
    x=rand((8,256))
    expected=jax.jit(layer.make_layer('regular_xla'))(x,p)
    actual=jax.jit(builder)(x,builder.prepare_weights(p))
    np.testing.assert_array_equal(np.asarray(actual),np.asarray(expected))
    operands=jax.jit(lambda x,p:adapter.tuning_operands(layer,p,x))(x,p)
    assert all(v[0].shape==adapter.shapes(layer)[name][:2] for name,v in operands.items())
    counts={}
    for site,shape in dict(q=(8192,5120,8192),k=(8192,5120,1024),o=(8192,8192,5120),gateup=(8192,5120,51200),down=(8192,25600,5120)).items():
        offered=list(tune.menu(site,shape,'bfloat16'))
        assert all(a['depth'] in (0,1,2) for a,_,_ in offered)
        assert all(not a['early'] or (site=='gateup' and a['depth']==1 and a['accumulator']=='outputs') for a,_,_ in offered)
        counts[site]=dict(offered=sum(reason is None for _,_,reason in offered),pruned=sum(reason is not None for _,_,reason in offered))
    for path in [Path(__file__).with_name('tune_parent_fused_v001.py'),Path(__file__).with_name('compare_parent_fused_v001.py')]:ast.parse(path.read_text())
    out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    result=dict(status='passed',native_adapter_bitwise_equal=True,tuning_operand_shapes_match=True,candidates_per_dtype=counts,
                scope='CPU model adapter and search plan checks; no TPU timing claim')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
