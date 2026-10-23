"""Verify parent registry, real geometry, candidate legality and unchanged math."""
import ast
import importlib
import json
import os
from pathlib import Path
import tempfile
import parent_stage_v001 as transport
from parent_exact_stage_v001 import GATE_ENV, verify_parent
import qwen14_batch_stage_v001 as study

ROOT=Path(__file__).resolve().parents[1]

def main():
    verify_parent()
    plan=json.loads((ROOT/'configs/qwen14_parent_batch_v001.json').read_text())
    with tempfile.TemporaryDirectory() as tmp:
        transport.configure(Path(tmp))
        os.environ.update(QWEN3_MODEL='14b',QWEN3_SEQUENCE='1024',**GATE_ENV)
        layer=importlib.import_module('benchmark_qwen3_32b_layer')
        gate=importlib.import_module('benchmark_qwen3_scaling_tile_tune')
        site=importlib.import_module('benchmark_qwen3_site_tile_tune')
        assert (layer.REPOSITORY,layer.REVISION)==(plan['model'],plan['revision'])
        geometry=dict(hidden_size=layer.MODEL_DIM,intermediate_size=layer.INTERMEDIATE_DIM,num_hidden_layers=layer.NUM_LAYERS,num_attention_heads=layer.HEADS,num_key_value_heads=layer.KV_HEADS,head_dim=layer.HEAD_DIM)
        assert geometry==plan['model_geometry']
        official=json.loads((ROOT/'plans/qwen14_parent_batch_20261017_v001/official-config.json').read_text())
        assert all(official[k]==v for k,v in geometry.items())
        site.STRASSEN_LIMIT=120*1024**2
        reports=[]
        for b in plan['batches']:
            gate.TOKENS=site.TOKENS=b*1024
            kept,filtered=gate.enumerate_candidates();assert kept
            for row in kept:
                bm,bn,bk=row['tile'];assert b*1024%bm==0 and 34816%bn==0 and 17408%(bn//2)==0 and 5120%bk==0
            counts={}
            for s in ('o','down'):
                k,n,_=site.SITES[s];rows,_=site.enumerate_site(k,n);assert rows
                for row in rows:
                    bm,bn,bk=row['tile'];assert b*1024%bm==0 and n%bn==0 and k%bk==0
                counts[s]=len(rows)
            p=study.default_profile(b,1024)
            assert p['qk_tile']==[min(2048,b*1024),1024,5120]
            assert p['strassen_sites']['o'][2]==5120
            reports.append(dict(batch=b,gate_candidates=len(kept),sites=counts))
    def funcs(path):return {n.name:ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    old=funcs(ROOT/'tools/qwen32_batch_stage_v004.py');new=funcs(ROOT/'tools/qwen14_batch_stage_v001.py')
    changed={'setup','default_profile','tune','verify_retry_profile','main'}
    for name in old.keys()-changed:assert old[name]==new[name],name
    print(json.dumps(dict(status='passed',model=plan['model'],revision=plan['revision'],geometry=geometry,candidate_counts=reports,unchanged_functions=sorted(old.keys()-changed),scope='CPU contracts and candidate enumeration; not TPU performance')))

if __name__=='__main__':main()
