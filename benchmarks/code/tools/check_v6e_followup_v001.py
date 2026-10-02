"""Check per-shape incumbent isolation and exact follow-up candidate coverage."""
import json
import os
from pathlib import Path
from strassen_mm.benchmark_v6e_opt_v002 import groups


def main():
    folder=Path('configs/v6e_opt_v002')
    cfg=json.loads((folder/'campaign.json').read_text());shapes=json.loads((folder/'shapes.json').read_text())['shapes']
    planned=groups(cfg,shapes,'screen',None)
    ids=[]
    for group in planned:
        for arm in group['arms']:
            assert arm.get('only_shape',group['shape']['id'])==group['shape']['id']
            assert arm.get('depth',0) in (0,1,2)
            ids.append((group['shape']['id'],arm['arm_id']))
    assert len(ids)==30 and len(set(ids))==30
    for shape in shapes:
        arms=[a for g in planned if g['shape']['id']==shape['id'] for a in g['arms']]
        assert len(arms)==15 and sum(a['family']=='hybrid2' for a in arms)==6
        assert sum(a['family']=='native' for a in arms)==5
        for family in ('original1','original2','optimized1','optimized2'):
            assert sum(a['family']==family for a in arms)==1
    smoke=groups(cfg,[dict(id='aligned_integer_probe',m=128,k=2048,n=1024)],'smoke',None)
    assert any(a.get('implementation')=='hybrid' for g in smoke for a in g['arms'])
    out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    summary=dict(passed=True,screen_attempts=len(ids),shapes=[s['id'] for s in shapes],
                 smoke_arms=sum(len(g['arms']) for g in smoke),scope='Offline plan coverage; no TPU timing.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
