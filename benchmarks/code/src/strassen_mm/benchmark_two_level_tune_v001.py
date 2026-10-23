"""Small equal-budget tile search, followed by frozen-choice fresh-seed confirmation."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback
from strassen_mm import benchmark_v001 as base
from strassen_mm.benchmark_two_level_poc_v001 import Runner as PriorRunner


def select_tiles(cases, campaign, shapes):
    selected = {}
    for shape in shapes:
        selected[shape['id']] = {}
        for arm in campaign['arms']:
            valid = [r for r in cases if r['shape_id'] == shape['id'] and r['arm_id'] == arm
                     and r['status'] == 'ok' and r['eligible_for_speedup_claim']
                     and r['timing']['sample_count'] == campaign['timing']['screen']['repeats']]
            if not valid: raise ValueError('No eligible tile for ' + shape['id'] + '/' + arm)
            best = min(valid, key=lambda r: (r['timing']['mean_ms'], tuple(r['kernel_metadata']['tile_bm_bn_bk'])))
            selected[shape['id']][arm] = dict(tile=best['kernel_metadata']['tile_bm_bn_bk'],
                screen_group_id=best['group_id'], screen_mean_ms=best['timing']['mean_ms'],
                screen_relative_l2=best['correctness']['relative_l2'])
    return selected


def make_groups(campaign, shapes, stage, selected=None):
    groups = []
    for i, shape in enumerate(shapes):
        if stage == 'screen':
            tiles = list(campaign['tiles'])
            random.Random(campaign['order_seed'] + i).shuffle(tiles)
            for tile in tiles:
                groups.append(dict(group_id=shape['id']+'__screen__tile_'+'_'.join(map(str,tile)),
                    shape=shape,tile=tile,arms=[dict(arm_id=a,algorithm='strassen',variant=a,tile=tile) for a in campaign['arms']],
                    scopes=['call'],timing='screen',inputs=[dict(distribution='gaussian',seed=campaign['screen_seed']+i)]))
        else:
            arms = [dict(arm_id=a,algorithm='strassen',variant=a,tile=selected[shape['id']][a]['tile']) for a in campaign['arms']]
            for arm in arms:
                if arm['tile'] not in campaign['tiles']: raise ValueError('Selected tile outside frozen grid')
            groups.append(dict(group_id=shape['id']+'__confirm',shape=shape,tile=None,arms=arms,scopes=['call'],timing='confirm',
                inputs=[dict(distribution='gaussian',seed=s+i) for s in campaign['confirm_seeds']]))
    return groups


class Journal(base.Journal):
    def __init__(self,*args):
        super().__init__(*args)
        self.cases = []

    def emit(self,event,**fields):
        super().emit(event,**fields)
        if event == 'case_result': self.cases.append(base.json_safe(fields))


class Runner(PriorRunner):
    def compile_entries(self, group, a, b):
        entries = []
        for arm in group['arms']:
            entries.extend(super().compile_entries({**group,'arms':[arm],'tile':arm['tile']}, a, b))
        return entries

    def run_group(self, group):
        import jax
        shape = tuple(group['shape'][d] for d in ('m','k','n'))
        entries = None
        for case in group['inputs']:
            self.check_deadline()
            a,b = base.generate_inputs(shape,**case)
            if entries is None: entries = self.compile_entries(group,a,b)
            self.execute_case(group,case,entries,a,b)
            del a,b
            gc.collect()
        del entries
        jax.clear_caches()
        gc.collect()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--phase',required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--expected-identity',type=Path,required=True)
    parser.add_argument('--allocation-id',required=True);parser.add_argument('--max-wall-seconds',type=float,default=1800)
    parser.add_argument('--selection',type=Path)
    args=parser.parse_args();stage=args.phase.rsplit('-',1)[-1]
    if stage not in ('screen','confirm'):raise ValueError('Expected screen or confirm phase')
    args.output_dir.mkdir(parents=True,exist_ok=False);out=args.output_dir;journal=Journal(out,args.phase)
    campaign=json.loads(args.campaign.read_text());groups=[];done=[];error=None
    try:
        shapes=json.loads((args.campaign.parent/campaign['shape_manifest']).read_text())['shapes']
        base.snapshot_sources(out,args.campaign,args.campaign.parent/campaign['shape_manifest'],args.campaign.parent/campaign['distribution_manifest'])
        selected=None
        if stage=='confirm':
            selection=json.loads(args.selection.read_text())
            if selection['allocation_id']!=args.allocation_id or selection['campaign_sha256']!=hashlib.sha256(args.campaign.read_bytes()).hexdigest():
                raise ValueError('Selection allocation/configuration mismatch')
            selected=selection['selected'];base.exclusive_json(out/'selection_used.json',selection)
        groups=make_groups(campaign,shapes,stage,selected)
        base.exclusive_json(out/'planned_cases.json',groups)
        if 'jax' in sys.modules:raise RuntimeError('Expected fresh process')
        os.environ.update(base.FIXED_ENVIRONMENT)
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        base.jax,base.jnp,base.np,base.ml_dtypes=jax,jnp,np,ml_dtypes
        from strassen_mm.kernels_v002 import enable_qualified_mosaic_v7_compat
        env=base.capture_environment(args,campaign,enable_qualified_mosaic_v7_compat())
        base.exclusive_json(out/'environment.json',env)
        journal.emit('identity_check',**base.verify_identity(env,args.expected_identity,campaign))
        runner=Runner(args,campaign,journal)
        for i,g in enumerate(groups):
            print(f'[{i+1}/{len(groups)}] {g["group_id"]} starting',flush=True)
            journal.emit('group_start',group_id=g['group_id'],index=i+1,total=len(groups))
            runner.check_deadline();runner.run_group(g);done.append(g['group_id'])
            journal.emit('group_complete',group_id=g['group_id'],index=i+1,total=len(groups))
            print(f'[{i+1}/{len(groups)}] complete; {dict(journal.status_counts)}',flush=True)
        if stage=='screen':
            base.exclusive_json(out/'selections.json',dict(allocation_id=args.allocation_id,
                campaign_sha256=hashlib.sha256(args.campaign.read_bytes()).hexdigest(),
                rule='Fastest eligible arithmetic mean per shape and depth; exact ties broken by tile tuple.',
                selected=select_tiles(journal.cases,campaign,shapes)))
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc))
        journal.emit('run_error',**error,traceback=traceback.format_exc());print(error,flush=True)
    summary=dict(completed=error is None and len(done)==len(groups),completed_groups=done,planned_group_count=len(groups),
        case_status_counts=dict(journal.status_counts),error=error,stage=stage,
        interpretation='Two shapes, eight tiles per depth, independent selection then three fresh Gaussian seeds. Same kernels and one allocation. A bounded tuning probe, not a global optimum or LLM quality study.')
    base.exclusive_json(out/'summary.json',summary);journal.close()
    return 0 if summary['completed'] else 1


if __name__=='__main__':raise SystemExit(main())
