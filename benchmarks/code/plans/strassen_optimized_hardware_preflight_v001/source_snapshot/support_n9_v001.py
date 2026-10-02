"""Append-only common application-study recording; no network access."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from . import benchmark_v001 as base


def parser(phase):
    p=argparse.ArgumentParser()
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--phase',choices=(phase,),required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--expected-identity',type=Path,required=True)
    p.add_argument('--allocation-id',required=True)
    p.add_argument('--max-wall-seconds',type=float,default=21600)
    return p


def begin(args):
    if args.max_wall_seconds<=0: raise ValueError('Positive wall budget required')
    args.campaign=args.campaign.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    campaign=json.loads(args.campaign.read_text())
    parent_path=(args.campaign.parent/campaign.get('base_campaign','campaign_v1.json')).resolve()
    parent=json.loads(parent_path.read_text())
    effective={**parent,**campaign,'device':parent['device']}
    if parent['device']['target']!='v5e' or parent['device'].get('allow_v6e'):
        raise ValueError('Application study frozen for single v5e cohort')
    source=args.output_dir/'source_snapshot';source.mkdir()
    for path in sorted(Path(__file__).parent.glob('*.py')): shutil.copyfile(path,source/path.name)
    configs=args.output_dir/'config_snapshot';configs.mkdir()
    shutil.copyfile(args.campaign,configs/args.campaign.name)
    if parent_path.name!=args.campaign.name: shutil.copyfile(parent_path,configs/parent_path.name)
    base.exclusive_json(args.output_dir/'effective_campaign.json',effective)
    base.exclusive_json(args.output_dir/'source_manifest.json',{'sha256':{
        str(p.relative_to(args.output_dir)):base.digest_file(p) for d in (source,configs)
        for p in sorted(d.rglob('*')) if p.is_file()}})
    return effective


def initialize_device(args,campaign,journal):
    if 'jax' in sys.modules: raise RuntimeError('Fresh process required before runtime policy setup')
    os.environ.update(base.FIXED_ENVIRONMENT)
    import jax
    import jax.numpy as jnp
    import numpy as np
    import ml_dtypes
    base.jax=jax;base.jnp=jnp;base.np=np;base.ml_dtypes=ml_dtypes
    from .kernels_v001 import enable_qualified_mosaic_v7_compat
    compatibility=enable_qualified_mosaic_v7_compat()
    environment=base.capture_environment(args,campaign,compatibility)
    base.exclusive_json(args.output_dir/'environment.json',environment)
    journal.emit('identity_check',**base.verify_identity(environment,args.expected_identity,campaign))
    return environment


def resolve(config,value):
    path=Path(value)
    return path.resolve() if path.is_absolute() else (config.parent/path).resolve()


def seal(args,journal,started,completed,results,error=None):
    summary={'phase':args.phase,'completed':completed,'status':'completed' if completed else 'failed_or_interrupted',
             'finished_utc':base.utc_now(),'wall_seconds':time.monotonic()-started,
             'case_status_counts':dict(journal.status_counts),'results':results,'error':error,
             'scientific_success':'Execution completion does not imply numerical eligibility or a speedup.'}
    journal.emit('run_complete',**summary);journal.close()
    base.exclusive_json(args.output_dir/'summary.json',summary)
    base.exclusive_json(args.output_dir/'artifact_manifest.json',{'sha256':{
        str(p.relative_to(args.output_dir)):base.digest_file(p) for p in sorted(args.output_dir.rglob('*'))
        if p.is_file()},'sealed_utc':base.utc_now()})


def checked_file(directory,relative,expected):
    path=(Path(directory)/relative).resolve()
    if not path.is_relative_to(Path(directory).resolve()) or base.digest_file(path)!=expected:
        raise ValueError('Application input hash/path mismatch')
    return path


class Budget:
    def __init__(self,seconds): self.deadline=time.monotonic()+seconds
    def check(self):
        if time.monotonic()>=self.deadline: raise TimeoutError('Application wall-time budget exhausted')
