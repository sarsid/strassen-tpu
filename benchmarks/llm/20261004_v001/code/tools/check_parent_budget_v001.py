"""Validate that the cap audit includes previously excluded arms and preserves controls."""
import ast,json
from pathlib import Path
from parent_budget_menu_v001 import shortlist


def main():
    root=Path(__file__).resolve().parents[1];inputs=root/'runs/parent-comparison-20261002-v004/stage-backups'
    events=[json.loads(l) for l in (inputs/'tune/events.jsonl').read_text().splitlines()]
    profile=json.loads((inputs/'refine/profile.json').read_text());before=json.dumps(profile,sort_keys=True)
    for dtype in ('bfloat16','float32'):
        arms=shortlist(events,'down',dtype,profile)
        new=[a for a in arms if a['vmem_limit_bytes']!=112*1024**2]
        assert len(new)==6,(dtype,len(new))
        assert {a['family'] for a in new}=={'cubic','s1','s2'}
        assert all(a['tile']==[2048,2560,1024] and a['buffers']==2 for a in arms)
        assert all(a['vmem_limit_bytes']==(124 if a['family']=='cubic' else 120)*1024**2 for a in new)
        for family in ('cubic','s1','s2'):
            old=profile['profiles'][dtype][family]['down']
            if old['implementation']!='native':assert old in arms
    assert json.dumps(profile,sort_keys=True)==before
    for name in ('tools/tune_parent_budget_v001.py','tools/run_parent_budget_v001.py','runtime/parent_budget_remote_v001.py'):
        ast.parse((root/name).read_text())
    source=(root/'tools/tune_parent_budget_v001.py').read_text()
    assert "choices=json.loads(json.dumps(screen_profile['profiles']))" in source
    assert "for site in ('down',):" in source
    assert 'scores[winner]>=0.99*scores[previous_name]' in source
    print('PASS: twelve higher-cap down arms, prior controls retained, other profiles preserved, orchestration syntax')


if __name__=='__main__':main()
