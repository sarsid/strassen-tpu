"""Check actual MXU leaf metadata and model-specific candidate constraints."""
import json,os
from pathlib import Path
from strassen_mm.llm_campaign_v002 import geometry,candidates
from strassen_mm.fusion_v003 import Epilogue
results=[]
for family in ('cubic_tuned','s1','s2'):
    offered=[r for r in candidates((4096,4096,8192),Epilogue('swiglu',rounding='accumulator'),'bfloat16',family) if r['disposition']=='offered']
    for r in offered:
        a=r['arm'];bm,bn,bk=a['tile'];divisor=1 if a['implementation']=='cubic_full' else 2**a['depth'] if a['depth'] else 2
        assert r['leaf_mkn']==[bm//divisor,bk//divisor,bn//divisor]
    results.append(dict(family=family,leaf_records_checked=len(offered)))
models=json.loads(Path('results/v6e/llm_campaign_20261003_v001/inputs_v002/models.json').read_text())
for model in models:
    c,shapes,specs=geometry(model['config'],4,1024)
    if c['model_type']=='mistral':assert specs['q'].kind=='rope' and shapes['q'][2]==c['num_attention_heads']*c['head_dim']
    if c['model_type']=='gemma3_text':
        for site in ('o','down'):
            for family in ('cubic_tuned','s1','s2'):
                rows=[r for r in candidates(shapes[site],specs[site],'bfloat16',family) if r['disposition']=='offered']
                assert rows and all(r['arm']['tile'][1]>=shapes[site][2] for r in rows)
    results.append(dict(model=model['id'],geometry_checked=True))
out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir();(out/'summary.json').write_text(json.dumps(dict(passed=True,checks=results,scope='Tuner metadata and geometry only; no performance claim'),indent=2)+'\n');print('PASS: full cubic and recursive leaf metadata; Mistral head width; Gemma full-row normalization')
