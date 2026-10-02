"""Use the original v5e statistics/report with two explicit output contracts."""
import argparse,json
from pathlib import Path
import report_mlsys_campaign_v001 as original

def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    for suffix,dtype in [('fp32','float32'),('bf16','bfloat16')]:
        original.SCOPES['arch_v5e_168_'+suffix+'_v001']='Historical v5e kernels, synthetic Gaussian inputs, '+dtype+' output'
    result=original.report(a.cohort,a.output_dir)
    complete=len(result['coverage'])==2 and all(v['confirmed_shapes']==v['expected_shapes']==168 for v in result['coverage'].values()) and result['confirmed_method_rows']==1680
    result['completed']=complete
    result['output_contracts']=['float32','bfloat16']
    result['v5e_policy']='Historical code/menu/timing unchanged; BF16 uses one final cast inside the complete call.'
    (a.output_dir/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    return 0 if complete else 1

if __name__=='__main__':raise SystemExit(main())
