"""Freeze and transport a small separate cohort, with scoped commits only."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import run_region_cohort_v001 as legacy

ROOT=Path(os.environ.get('STRASSEN_PROJECT_ROOT',Path(__file__).resolve().parents[1])).resolve()
CAMPAIGN='configs/alphatensor_poc_v001/campaign.json'
SOURCES=['src','runtime','tools','configs/alphatensor_poc_v001','plans/alphatensor_poc_v001','third_party/alphatensor_1949163_v001']


def commit(paths,message):
    relative=[str(Path(p).resolve().relative_to(ROOT)) for p in paths]
    subprocess.run(['git','add','--',*relative],cwd=ROOT,check=True)
    subprocess.run(['git','commit','--only','-m',message,'--',*relative],cwd=ROOT,check=True)
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def freeze(cohort):
    meta=legacy.read(cohort/'cohort.json')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','diff','--exit-code','HEAD','--',*SOURCES],cwd=ROOT,check=True)
    with (cohort/'source.tar').open('xb') as stream:
        subprocess.run(['git','archive',revision,'--',*SOURCES],cwd=ROOT,stdout=stream,check=True)
    source=cohort/'source';source.mkdir()
    with tarfile.open(cohort/'source.tar') as archive:archive.extractall(source,filter='data')
    legacy.write(cohort/'frozen.json',{'baseline_commit':revision,'campaign_relative':CAMPAIGN,
        'archive_sha256':legacy.sha(cohort/'source.tar'),
        'source_sha256':{str(p.relative_to(source)):legacy.sha(p) for p in sorted(source.rglob('*')) if p.is_file()},
        **meta})
    legacy.verify_frozen(cohort)
    print(commit([cohort],'Freeze AlphaTensor proof of concept source'))


class Archive(legacy.CohortArchive):
    def finish(self,path,status,**details):
        super().finish(path,status,**details)
        return commit([path,self.cohort/(self.phase+'-started.json'),self.cohort/(self.phase+'-finished.json')],
                      'Archive AlphaTensor proof of concept '+status)


def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['freeze','run'])
    p.add_argument('--cohort',type=Path,required=True);p.add_argument('--expected-identity')
    p.add_argument('--controller-python',default=sys.executable)
    args=p.parse_args();cohort=args.cohort.resolve()
    assert cohort.is_relative_to(ROOT/'runs') and cohort!=ROOT/'runs'
    if args.operation=='freeze':freeze(cohort);return 0
    if not args.expected_identity:raise ValueError('Setup identity is required')
    meta=legacy.read(cohort/'cohort.json');legacy.verify_frozen(cohort)
    spec=importlib.util.spec_from_file_location('poc_transport',cohort/'source/tools/run_phase_v004.py')
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    transport.archive=Archive(cohort,'ALPHA-poc')
    transport.emit=lambda *a,**kw: print(json.dumps(kw),flush=True)
    sys.argv=['run_phase_v004.py','--phase','ALPHA-poc','--session',meta['session'],'--endpoint',meta['allocation_id'],
        '--expected-identity',args.expected_identity,'--controller-python',args.controller_python,
        '--campaign-relative',CAMPAIGN,'--benchmark-module','strassen_mm.benchmark_alphatensor_poc_v001',
        '--timeout-seconds','2400']
    return transport.main()


if __name__=='__main__':raise SystemExit(main())
