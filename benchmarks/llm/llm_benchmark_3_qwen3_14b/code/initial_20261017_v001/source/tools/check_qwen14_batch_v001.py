"""CPU-only regression checks for segmented capture and the authorized B64 plan."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from collections import namedtuple

os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.numpy as jnp
import numpy as np
import qwen14_batch_stage_v001 as study

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'third_party/strassen_tpu_public_95be1fb_v001'


def functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def errors(got, ref):
    a = np.asarray(got, dtype=np.float32); b = np.asarray(ref, dtype=np.float32)
    return dict(max_abs=float(np.max(np.abs(a-b))),
                l2_relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-20)))


def main():
    assert jax.default_backend() == 'cpu'
    study.verify_parent()
    old = ast.parse((ROOT/'tools/qwen32_batch_stage_v001.py').read_text())
    new = ast.parse((ROOT/'tools/qwen14_batch_stage_v001.py').read_text())
    old_funcs = {x.name:ast.dump(x) for x in old.body if isinstance(x,ast.FunctionDef)}
    new_funcs = {x.name:ast.dump(x) for x in new.body if isinstance(x,ast.FunctionDef)}
    unchanged = ['options','arm_options','timed','round_robin','block_exes','projection',
                 'error_set','metric_chunk','task_metrics','confirm','quality','prefill','set_profile']
    for name in unchanged: assert old_funcs[name] == new_funcs[name], name
    plan = json.loads((ROOT/'configs/qwen14_parent_batch_v001.json').read_text())
    assert plan['batches']==[8,16,32,64,4,1] and plan['sequence']==1024
    assert plan['model']=='Qwen/Qwen3-14B'
    assert plan['software']=={'jax':'0.11.2','jaxlib':'0.11.2','libtpu':'0.0.48'}
    baseline=json.loads((ROOT/'configs/qwen32_b64_current_stack_v004.json').read_text())
    for key in ['arms','output_dtype','strassen_depth','block_warmups','block_runs','layer_warmups','layer_runs','quality_segments','natural_quality_gates','repetitive_quality_gates','offloaded_prefill_warmups','offloaded_prefill_runs']:
        assert plan[key]==baseline[key], key
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'profile.json'
        fixture=dict(profile=study.default_profile(64,1024),model=plan['model'],revision=plan['revision'],batch=64,sequence=1024)
        p.write_text(json.dumps(fixture));sha=hashlib.sha256(p.read_bytes()).hexdigest()
        assert study.verify_retry_profile(p,64,plan,sha)==fixture
        fixture['model']='Qwen/Qwen3-32B';p.write_text(json.dumps(fixture))
        try:study.verify_retry_profile(p,64,plan,hashlib.sha256(p.read_bytes()).hexdigest())
        except ValueError:pass
        else:raise AssertionError('Wrong model profile accepted')
    # Execute the exact pinned parent math at reduced CPU dimensions, including
    # grouped-query attention, RoPE, BF16 boundaries and the whole native block.
    spns=dict(jax=jax,jnp=jnp)
    functions(PARENT/'strassen_pallas.py',['native_matmul'],spns)
    sp=types.SimpleNamespace(native_matmul=spns['native_matmul'])
    ns=dict(jax=jax,jnp=jnp,sp=sp,RMS_EPS=1e-6,ROPE_THETA=1e6,
            BATCH=1,SEQUENCE=8,TOKENS=8,HEADS=4,KV_HEADS=2,HEAD_DIM=4,MODEL_DIM=16)
    functions(PARENT/'experiments/qwen3/benchmark_qwen3_32b_layer.py',
              ['rms_norm','apply_rope','native_projection','rope_values','attention_prefix'],ns)
    layer=types.SimpleNamespace(**ns)
    study.jax=jax;study.jnp=jnp;study.np=np;study.layer=layer;study.sp=sp
    # Use the original eager collection body as the independent regression reference.
    original=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name=='projection_confirm')
    body=original.body[:next(i for i,x in enumerate(original.body) if isinstance(x,ast.For))]
    legacy=ast.FunctionDef(name='legacy',args=original.args,body=body+[ast.Return(value=ast.Tuple(elts=[ast.Name(id='traces',ctx=ast.Load()),ast.Name(id='residual',ctx=ast.Load())],ctx=ast.Load()))],decorator_list=[])
    tree=ast.fix_missing_locations(ast.Module(body=[legacy],type_ignores=[]))
    legacy_ns=dict(jax=jax,jnp=jnp,layer=layer,sp=sp)
    exec(compile(tree,'original-eager-capture','exec'),legacy_ns)
    native_ns=dict(jax=jax,jnp=jnp,layer=layer,sp=sp,PRODUCT_TILE=(1,1,1),
                   FUSED_QK=True,EXTENDED_SITES=('o','down'),
                   make_site_gemm=lambda arm:lambda site,a,w:sp.native_matmul(a,w))
    functions(PARENT/'experiments/qwen3/benchmark_qwen3_32b_streamed_inference.py',['make_product_layer'],native_ns)
    Params=namedtuple('Params','attention_norm query key value query_norm key_norm rope_cos rope_sin attention_output mlp_norm gate_up mlp_down')
    reports=[]
    for b in [1,8,32,64]:
        ns.update(BATCH=b,TOKENS=b*8);layer.BATCH=b;layer.TOKENS=b*8
        rng=np.random.default_rng(103+b)
        def mat(m,n): return jnp.asarray(rng.normal(0,.15,(m,n)),dtype=jnp.bfloat16)
        cos,sin=ns['rope_values']()
        p=Params(jnp.ones(16,jnp.bfloat16),mat(16,16),mat(16,8),mat(16,8),
                 jnp.ones(4,jnp.bfloat16),jnp.ones(4,jnp.bfloat16),cos,sin,
                 mat(16,16),jnp.ones(16,jnp.bfloat16),mat(16,64),mat(32,16))
        x=mat(b*8,16)
        reference=legacy_ns['legacy'](x,p,None,None)
        events=[]
        captured=study.capture_to_host(x,p,events.append)
        assert [e['part'] for e in events if e['kind']=='capture_part_complete'] == ['qkv','attention','postattention','swiglu']
        assert events[-1]['kind'] == 'activation_capture'
        assert events[-1]['mode'] == 'segmented_full_batch'
        assert all(isinstance(v,np.ndarray) for v in (*captured[0].values(),captured[1]))
        report={}
        for site in ['q','k','v','o','gateup','down']:
            assert captured[0][site].shape == reference[0][site].shape
            assert captured[0][site].dtype == reference[0][site].dtype
            report[site]=errors(captured[0][site],reference[0][site])
            assert report[site]['l2_relative'] < .01, (b,site,report[site])
        native=jax.jit(native_ns['make_product_layer']('regular_xla'))(x,p)
        rebuilt=jax.jit(study.projection('down','native',{}))(
            jnp.asarray(captured[0]['down']),p,jnp.asarray(captured[1]))
        report['native_reconstruction']=errors(rebuilt,native)
        assert report['native_reconstruction']['l2_relative'] < .01, report
        reports.append(dict(batch=b,sequence=8,model_dim=16,errors=report))
        del reference,captured,native,rebuilt,p,x
        jax.clear_caches()
    # Abstract lowering checks the true B64 geometry without allocating model
    # weights, activations or a TPU. It does not test TPU backend compilation.
    ns.update(BATCH=64,SEQUENCE=1024,TOKENS=65536,HEADS=40,KV_HEADS=8,HEAD_DIM=128,MODEL_DIM=5120)
    for key in ('BATCH','SEQUENCE','TOKENS','HEADS','KV_HEADS','HEAD_DIM','MODEL_DIM'):
        setattr(layer,key,ns[key])
    def abstract(*dims): return jax.ShapeDtypeStruct(dims,jnp.bfloat16)
    p=Params(abstract(5120),abstract(5120,5120),abstract(5120,1024),abstract(5120,1024),
             abstract(128),abstract(128),abstract(1024,128),abstract(1024,128),
             abstract(5120,5120),abstract(5120),abstract(5120,34816),abstract(17408,5120))
    abstract_checks={}
    for name,fn,args in (
        ('qkv',study.capture_qkv,(abstract(65536,5120),p)),
        ('attention',study.capture_attention,(abstract(64,1024,40,128),abstract(64,1024,8,128),abstract(64,1024,8,128))),
        ('postattention',study.capture_postattention,(abstract(65536,5120),abstract(65536,5120),p)),
        ('swiglu',study.capture_swiglu,(abstract(65536,5120),abstract(5120,34816)))):
        lowered=jax.jit(fn).lower(*args)
        text=lowered.as_text()
        abstract_checks[name]=dict(stablehlo_bytes=len(text.encode()),sha256=hashlib.sha256(text.encode()).hexdigest())
        if os.environ.get('STRASSEN_EXECUTION_DIR'):
            path=Path(os.environ['STRASSEN_EXECUTION_DIR'])/('b64-'+name+'.stablehlo.txt')
            path.write_text(text)
    # The retry starter must reject calls lacking explicit resumption before
    # opening a lock, reading a run, contacting Colab or spawning a process.
    attempt=subprocess.run([sys.executable,str(ROOT/'tools/start_qwen14_batch_v001.py'),
                            '--run','/does-not-exist','--controller-python',sys.executable],
                           capture_output=True,text=True)
    assert attempt.returncode == 2 and 'explicit user resumption' in attempt.stderr
    for name in ['runtime/qwen14_batch_remote_v001.py','tools/run_qwen14_batch_v001.py','tools/start_qwen14_batch_v001.py']:
        ast.parse((ROOT/name).read_text())
    print(json.dumps(dict(status='passed',scope='CPU correctness only; no full-size TPU allocation or validation',
                         unchanged_functions=unchanged,profile_integrity='passed',explicit_start_guard='passed',
                         capture_reports=reports,full_size_abstract_lowering=abstract_checks)))


if __name__=='__main__': main()
