"""Ten frozen v5e winners, with all screen arms timed in paired rounds."""
from . import benchmark_v6e_diagnostic_v001 as routing
from . import benchmark_v6e_opt_v001 as benchmark
previous_groups=benchmark.groups

def groups(cfg,shapes,stage,selection):
    result=previous_groups(cfg,shapes,stage,selection)
    if stage!='screen':return result
    merged={}
    for group in result:
        key=group['shape']['id']
        if key not in merged:merged[key]={**group,'arms':[],'group_id':key+'__screen_paired'}
        merged[key]['arms'].extend(group['arms'])
    return list(merged.values())

benchmark.groups=groups
def main():
    code=routing.main()
    import json,sys
    from pathlib import Path
    out=Path(sys.argv[sys.argv.index('--output-dir')+1])
    path=out/'summary.json';summary=json.loads(path.read_text())
    summary['interpretation']='Ten frozen historical v5e winners, synthetic BF16 inputs/preadds, FP32 output, modern compiler. Paired screen, frozen finalists, three fresh confirmation inputs. Not a full LLM test.'
    path.write_text(json.dumps(summary,indent=2)+'\n')
    return code

if __name__=='__main__':raise SystemExit(main())
