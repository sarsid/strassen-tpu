"""Offline import/config validation before actual TPU application executions."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

root=Path(__file__).resolve().parents[1]
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
paths=['src/strassen_mm/benchmark_n8_v001.py','src/strassen_mm/model_n9_v001.py',
       'src/strassen_mm/benchmark_n9_v001.py','src/strassen_mm/support_n9_v001.py',
       'tools/build_application_campaign_v001.py','tools/prepare_n9_inputs_v001.py']
for name in paths:ast.parse((root/name).read_text(),filename=name)
from strassen_mm import benchmark_n8_v001 as n8,benchmark_n9_v001 as n9,model_n9_v001
spec=importlib.util.spec_from_file_location('application_generator',root/'tools/build_application_campaign_v001.py')
generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
config=generator.n8(SimpleNamespace(base_campaign='../campaign_n5_n9_v1.json',n5_selections=None),out)
assert len(config['groups'])==16
assert len({tuple(g['shape_mkn'])+(g['kind'],) for g in config['groups']})==16
assert sum(g['shape_mkn'][0]==512 for g in config['groups'])==4
train=json.loads((root/'configs/shapes_v1.json').read_text())['shapes']
ids=set(json.loads((root/'configs/campaign_n5_n9_v1.json').read_text())['training_shape_ids'])
seen={tuple(s[x] for x in ('m','k','n')) for s in train if s['id'] in ids}
matching=[g['id'] for g in config['groups'] if tuple(g['shape_mkn']) in seen]
assert set(matching)=={'mistral7b_m512_swiglu','qwen32b_m512_swiglu'}
base={'all_finite':True,'nll_delta':0.0,'mean_kl':0.0,'top1_agreement':1.0,'relative_l2':0.0}
assert n9.eligible(base,n9.QUALITY)
for key,value in [('all_finite',False),('nll_delta',.1),('mean_kl',.1),('top1_agreement',.1),('nll_delta',float('nan'))]:
 assert not n9.eligible({**base,key:value},n9.QUALITY)
report={'status':'passed','syntax_files':paths,'imports':'N8/N9/model/support succeeded',
        'n8_group_count':16,'n5_exact_matching_application_groups':matching,
        'numerical_failure_guards':'nonfinite, NLL, KL and top1 failures rejected',
        'scope':'Offline validation only; generated no-selection config is a test fixture, not the TPU campaign'}
with (out/'validation.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(report))
