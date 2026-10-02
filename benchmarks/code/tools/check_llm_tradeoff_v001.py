"""CPU-only independent metrics oracle and tiny official-model integration."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
import ml_dtypes
from scipy.special import logsumexp
from strassen_mm import llm_error_metrics_v001 as metrics
from strassen_mm import benchmark_llm_tradeoff_v001 as runner
from strassen_mm import benchmark_large_models_v001 as prior
from strassen_mm import model_n9_v001 as qm, model_gemma_v002 as gm, model_composed_v002 as composed
from strassen_mm.kernels_v002 import make_matmul


def main():
    checks=[];rng=np.random.default_rng(99);compare=metrics.make_comparator(jax,jnp)
    r=rng.normal(size=(12,11)).astype(np.float32);c=(r+.2*rng.normal(size=r.shape)).astype(np.float32)
    t=np.arange(12)%11
    def run(candidate):return {k:np.asarray(v) for k,v in jax.device_get(compare(jnp.asarray(r),jnp.asarray(candidate),jnp.asarray(t))).items()}
    raw=run(c);lp=r.astype(float)-logsumexp(r.astype(float),axis=-1,keepdims=True)
    lq=c.astype(float)-logsumexp(c.astype(float),axis=-1,keepdims=True);p,q=np.exp(lp),np.exp(lq)
    oracle=dict(candidate_nll=-lq[np.arange(12),t],kl_forward=np.sum(p*(lp-lq),axis=-1),
        kl_reverse=np.sum(q*(lq-lp),axis=-1),total_variation=.5*np.sum(abs(p-q),axis=-1),
        candidate_brier=np.sum(q*q,axis=-1)-2*q[np.arange(12),t]+1,
        logit_squared_error=np.sum((c.astype(float)-r)**2,axis=-1),
        candidate_top1_correct=np.argmax(c,axis=-1)==t,
        candidate_top5_correct=np.any(np.argsort(c,axis=-1)[:,-5:]==t[:,None],axis=-1))
    for key,wanted in oracle.items():
        np.testing.assert_allclose(raw[key],wanted,rtol=2e-5,atol=3e-7);checks.append(key+'_float64_oracle')
    for candidate,label in [(r,'identical'),(r+4,'constant_shift')]:
        d=run(candidate);d['window']=np.repeat(np.arange(3),4)
        s=metrics.summarize(d,11)
        assert s['top1_agreement']==1 and abs(s['mean_kl'])<1e-6
        assert s['correct_to_wrong']==0 and s['wrong_to_correct']==0
        if label=='constant_shift':assert s['relative_l2']>1 and s['mean_total_variation']<1e-6
        checks.append(label)
    d=run(c);d['window']=np.repeat(np.arange(3),4);d['finite']=d['finite'].copy();d['finite'][0]=False
    assert metrics.summarize(d,11)['passed'] is False;checks.append('nonfinite_blocks_quality_claim')
    from transformers import Qwen3Config,Qwen3ForCausalLM
    import torch
    torch.set_num_threads(1);torch.manual_seed(17)
    config=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=512,
        attention_bias=False,hidden_act='silu',tie_word_embeddings=False)
    config._attn_implementation='eager';official=Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
    ids=rng.integers(0,97,size=(4,33),dtype=np.int32)
    with torch.no_grad():reference=official(torch.as_tensor(ids[1:2],dtype=torch.long),use_cache=False).logits[0].float().numpy()
    for module in (runner,prior,prior.base):
        module.jax,module.jnp,module.np,module.ml_dtypes=jax,jnp,np,ml_dtypes
    for module in (runner,prior):module.qm,module.gm,module.composed=qm,gm,composed
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);checkpoint=root/'checkpoint';official.save_pretrained(checkpoint,safe_serialization=True)
        manifest=dict(model_id='local/official-tiny-qwen3',revision='0'*40,cache_dir=str(checkpoint),config=json.loads((checkpoint/'config.json').read_text()),
            files=[dict(path=p.name,bytes=p.stat().st_size,sha256=qm.sha256(p)) for p in checkpoint.iterdir() if p.is_file()])
        manifest_path=root/'manifest.json';manifest_path.write_text(json.dumps(manifest));cp=qm.Checkpoint(manifest_path)
        cfg=dict(heldout_windows=[1,2,3],timing_windows=[1,2,3],order_seed=922,resident_timing={'repeats':1},full_model_warmups=1,full_model_repeats=2)
        class Log:
            def __init__(self):self.rows=[]
            def emit(self,event,**data):self.rows.append(dict(event=event,**data))
        class CPUStudy(runner.Study):
            def mm(self,arm,shape):return jax.jit(make_matmul('native',shape))
            def load_resident(self,cp):
                return ([jax.device_put(cp.layer(i)) for i in range(2)],jax.device_put(np.array(cp.tensor('model.embed_tokens.weight'))),
                        jax.device_put(np.array(cp.tensor('model.norm.weight'))),jax.device_put(cp.head()))
        policy={k:{'family':'native'} for k in ('gateup','down')};policies={k:policy for k in ('native_default','native','cubic','one_level','two_level')}
        log=Log();out=root/'quality';out.mkdir()
        study=CPUStudy(SimpleNamespace(output_dir=out,max_wall_seconds=300),cfg,log)
        got,_=study.forward(cp,ids[1],policy)
        relative=float(np.linalg.norm(np.asarray(got,dtype=float)-reference)/np.linalg.norm(reference))
        assert relative<.03,relative;checks.append('official_qwen_forward_relative_l2_'+str(relative))
        quality=study.quality(cp,ids,policies)
        for arm in policies:
            assert quality['quality'][arm]['positions']==96
            assert quality['quality'][arm]['passed'] and quality['quality'][arm]['top1_agreement']==1
        checks.append('heldout_quality_full_stack_and_token_archives')
        out=root/'resident';out.mkdir();study.args.output_dir=out
        original_layer=cp.layer;resident_values=study.load_resident(cp)
        study.load_resident=lambda _:resident_values
        def forbidden(*a,**kw):raise AssertionError('Checkpoint read inside resident operation')
        cp.layer=cp.tensor=cp.head=cp.embeddings=forbidden
        result=study.resident(cp,ids,policies)
        assert result['fully_resident_model_timed'] and len([r for r in log.rows if r['event']=='resident_model_sample'])==72
        checks.append('resident_operations_do_not_read_checkpoint_after_loading')
        assert len(result['resident_timings'])==2;checks.append('both_prompt_forward_and_scoring_scopes_execute')
    out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    report=dict(passed=True,checks=checks,scope='CPU correctness/integration only; synthetic tiny official Qwen weights, never TPU performance')
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
