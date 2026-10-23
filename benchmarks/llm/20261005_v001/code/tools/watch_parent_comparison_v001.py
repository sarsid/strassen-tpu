"""Mirror changing local remote snapshots into the existing live journal."""
import argparse,json,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];run=a.run.resolve();previous=None
    if not run.is_relative_to(root/'runs'):raise ValueError('Expected this project run directory')
    deadline=time.monotonic()+3*3600
    while time.monotonic()<deadline:
        latest=run/'latest.json'
        if latest.exists():
            snapshot=json.loads(latest.read_text());state=snapshot.get('status.json',{})
            records=[]
            for path,tail in snapshot.get('tails',{}).items():
                if path.endswith('events.jsonl'):
                    for line in tail.splitlines():
                        try:records.append(json.loads(line))
                        except ValueError:pass
            progress=[r for r in records if r.get('kind') in ('candidate','candidate_failure','selected','layer','confirmation','task','verdict')]
            last=progress[-1] if progress else {}
            key=(state.get('active'),last.get('kind'),last.get('site'),last.get('layer'),
                 last.get('arm',{}).get('arm_id') if isinstance(last.get('arm'),dict) else None,
                 last.get('selected',{}).get('arm_id'))
            if key!=previous:
                previous=key
                title='Fused Qwen3-32B comparison: '+state.get('active',state.get('status','starting'))
                detail='The latest remote snapshot is saved; source and previous outcomes remain preserved.'
                if last.get('kind')=='candidate':
                    arm=last['arm'];title=f'Fused tuning: {last["site"]}, {arm["output_dtype"]}'
                    detail=f'Latest completed candidate {arm["arm_id"]}; mean {sum(last["samples_ms"])/len(last["samples_ms"]):.4f} ms; relative L2 {last["errors_vs_parent_native"]["l2_relative"]:.6g}. This is a site screen, not a complete-model result.'
                elif last.get('kind')=='layer':
                    detail=f'Completed layer {last["layer"]+1}/64 for {state.get("active")}; finite={last.get("finite")}. Final prediction-quality gate is still evaluated after all layers.'
                elif last.get('kind')=='selected':
                    detail=f'Frozen site selection for {last["site"]}, {last["dtype"]}, {last["family"]}: {last["selected"]["arm_id"]}. Whole-block and model confirmation follow.'
                elif last.get('kind')=='candidate_failure':
                    detail=f'Candidate {last["arm"]["arm_id"]} failed with {last["error_type"]}; its evidence is saved and independent candidates continue.'
                subprocess.run(['python3',str(root/'tools/record_progress_v001.py'),'--kind','activity','--id','current','--state','working',
                                '--title',title,'--detail',detail,'--evidence',str(latest.relative_to(root))],check=False,stdout=subprocess.DEVNULL)
        if (run/'completion.json').exists():return
        time.sleep(45)

if __name__=='__main__':main()
