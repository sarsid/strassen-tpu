"""Check that a slow Native outlier cannot suppress the raw-time shortlist."""
import ast
from pathlib import Path
from parent_shortlist_v001 import robust_score,shortlist


def main():
    profile={'profiles':{'float32':{f:{'k':{'implementation':'native'}} for f in ('cubic','s1','s2')}}}
    events=[]
    for i,(candidate,control) in enumerate([(0.46,0.50),(0.48,0.50),(0.51,0.81)]):
        a=dict(arm_id=str(i),implementation='current',output_dtype='float32',family='s2')
        events.append(dict(kind='candidate',site='k',eligible=True,arm=a,samples_ms=[candidate]*8,
                           native_control_samples_ms=[control]*8))
    profile['profiles']['float32']['s2']['k']=events[-1]['arm']
    chosen={a['arm_id'] for a in shortlist(events,'k','float32',profile)}
    assert chosen=={'0','1','2'},chosen
    values=[0.5]*32;values[7]=3.0
    assert robust_score(values)==0.5
    assert robust_score([4.0,6.0]*16)==5.0
    for invalid in ([1.0]*31,[float('nan')]*32,[0.0]*32):
        try:robust_score(invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid measurements accepted')
    root=Path(__file__).resolve().parents[1]
    ast.parse((root/'tools/tune_parent_fused_v005.py').read_text())
    print('PASS: outlier regression, alternating latency, input validation, refinement syntax')


if __name__=='__main__':main()
