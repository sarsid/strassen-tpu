"""Run frozen tuned projections through the parent's exact stream/quality harness.

Only layer construction, pre-timing weight packing and compiler options change.
The parent's timing loop, tokenization, attention, scoring and gates are reused.
"""
import argparse, hashlib, importlib, importlib.metadata as md, json, math, os
from pathlib import Path
import parent_stage_v001 as setup


class JaxOptionsProxy:
    """Apply explicit whole-function options without changing global jax.jit."""
    def __init__(self,jax):self.original=jax
    def __getattr__(self,name):return getattr(self.original,name)
    def jit(self,fn,*args,**kwargs):
        if hasattr(fn,'replication_compiler_options'):
            kwargs['compiler_options']=fn.replication_compiler_options
        return self.original.jit(fn,*args,**kwargs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--profile',type=Path,required=True);p.add_argument('--dtype',choices=['bfloat16','float32'],required=True)
    p.add_argument('--family',choices=['s1','s2'],required=True);p.add_argument('--stage',choices=['streamed','quality'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    profile=json.loads(a.profile.read_text())
    def emit(value):
        if value.get('kind')=='tokens':
            expected=('4aa8e3628d16c9a26876aedb8c99525c6a15ee601c88d93deadae15855a84d74' if a.stage=='quality' else
                      '70b86198317db1dee0c351568cf420d78242aa74e0b9a6cbb14925eaa55ca2a5')
            if value['sha256']!=expected:raise RuntimeError('Frozen evaluation tokens differ')
        if value.get('kind')=='task':
            for result in value['results'].values():
                result['native_perplexity']=math.exp(result['native_loss'])
                result['candidate_perplexity']=math.exp(result['candidate_loss'])
        if value.get('kind')=='metadata':
            value=dict(kind='metadata',parent_harness_metadata=value,actual_profiles=profile['profiles'][a.dtype],
                       actual_family=a.family,custom_store_dtype=a.dtype,consumer_dtype='bfloat16',
                       compiler_options=profile['native_compiler_options'],
                       field_mapping=dict(regular_xla='parent_native_tuned',gated_cubic='our_tuned_cubic',gated_strassen='our_tuned_'+a.family),
                       profile_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),
                       changes='projection kernels/epilogues, offline packing and selected whole-layer options; parent timing/scoring unchanged')
        line=json.dumps(value,sort_keys=True)
        with (a.output/'events.jsonl').open('a') as f:f.write(line+'\n')
        print(line,flush=True)
    setup.configure(a.output);os.environ['STRASSEN_MANAGE_CEILING']='0'
    stream=importlib.import_module('benchmark_qwen3_32b_streamed_inference')
    module=stream if a.stage=='streamed' else importlib.import_module('benchmark_qwen3_streamed_natural_gate')
    layer=importlib.import_module('benchmark_qwen3_32b_layer')
    os.environ['LIBTPU_INIT_ARGS']='--xla_tpu_use_enhanced_launch_barrier=true'
    setup.install_transport(layer,a.cache,emit)
    import jax
    from strassen_mm.parent_fused_adapter_v001 import build
    versions={n:md.version(n) for n in ('jax','jaxlib','libtpu')}
    if versions!=dict(jax='0.11.2',jaxlib='0.11.2',libtpu='0.0.48'):raise RuntimeError('Unexpected current stack')
    if len(jax.devices())!=1 or 'v6' not in jax.devices()[0].device_kind.lower():raise RuntimeError('Expected one v6e')
    emit(dict(kind='environment',versions=versions,device=jax.devices()[0].device_kind,libtpu_init_args=os.environ['LIBTPU_INIT_ARGS']))
    builders={key:build(layer,profile['profiles'][a.dtype][family])
              for key,family in [('gated_cubic','cubic'),('gated_strassen',a.family)]}
    original_load=stream.load_layer
    original_native=stream.make_product_layer('regular_xla')
    def load(checkpoint,index,cos,sin):
        params=original_load(checkpoint,index,cos,sin)
        cache={}
        packed={name:builder.prepare_weights(params,cache) for name,builder in builders.items()}
        result=dict(parent=params,prepared=packed)
        jax.block_until_ready(result)
        return result
    def make(name):
        if name=='regular_xla':
            def fn(x,p):return original_native(x,p['parent'])
        else:
            builder=builders[name]
            def fn(x,p):return builder(x,p['prepared'][name])
        fn.replication_compiler_options=profile['native_compiler_options']
        return fn
    stream.load_layer=load;stream.make_product_layer=make
    stream.jax=JaxOptionsProxy(jax)
    if module is not stream:module.jax=JaxOptionsProxy(jax)
    module.emit=stream.emit=layer.emit=emit
    module.main()
    (a.output/'summary.json').write_text(json.dumps(dict(status='completed',stage=a.stage,family=a.family,dtype=a.dtype),indent=2)+'\n')


if __name__=='__main__':main()
