"""Write the planned grid and menu summaries without allocating a TPU."""
import argparse,json,os
from pathlib import Path
from strassen_mm.llm_campaign_v001 import entries,geometry,candidates

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path('configs/llm_v6e_v002.json'));p.add_argument('--output',type=Path);a=p.parse_args()
    out=a.output or Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir(parents=True,exist_ok=False)
    config=json.loads(a.config.read_text());models=json.loads(Path(config['resolved_metadata']).read_text())
    rows=list(entries(config,models));assert len(rows)==700 and len({r['entry_id'] for r in rows})==700
    (out/'planned_entries.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summaries=[]
    # Menu audit at representative M=4096; this is a plan, not tuned choices.
    for item in models:
        c,shapes,specs=geometry(item['config'],4,1024)
        for dtype in config['output_dtypes']:
            for family in config['algorithms']:
                for site in shapes:
                    rows=list(candidates(shapes[site],specs[site],dtype,family))
                    ids=[r['candidate_id'] for r in rows];assert len(ids)==len(set(ids))
                    offered=[r for r in rows if r['disposition']=='offered']
                    assert offered,(item['id'],dtype,family,site)
                    summaries.append(dict(model=item['id'],output_dtype=dtype,family=family,site=site,shape_mkn=shapes[site],offered=len(offered),pruned=len(rows)-len(offered),full_contraction_offered=sum(bool(r.get('metadata',{}).get('full_contraction')) for r in offered)))
    (out/'menu_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    summary=dict(passed=True,planned_main_entries=700,measured_main_entries=0,menu_groups=len(summaries),scope='Pinned model geometry and candidate coverage audit; no tuning result')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
