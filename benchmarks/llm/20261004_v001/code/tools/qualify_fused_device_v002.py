"""One allocation, frozen v6e setup/check, download, then release in finally."""
import argparse, hashlib, json, os, subprocess, tarfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--controller-python',required=True);a=p.parse_args()
    run=Path(os.environ['STRASSEN_EXECUTION_DIR']).resolve();source=run/'source'
    out=run/'artifacts';out.mkdir()
    controller=[a.controller_python,str(source/'runtime/colab_control_v003.py')]
    session='fusion-check-'+run.name.split('-')[0].lower()
    def command(name,argv,timeout=240):
        with (out/(name+'.log')).open('x') as log:
            result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        return result.returncode
    allocation=[a.controller_python,str(source/'runtime/allocate_tradeoff_v002.py'),'v6e','create','--session',session]
    code=command('allocate',allocation)
    records=[]
    for line in (out/'allocate.log').read_text().splitlines():
        try:records.append(json.loads(line))
        except ValueError:pass
    found=[r for r in records if r.get('kind')=='allocation' and r.get('created')]
    if code or len(found)!=1:
        raise RuntimeError('No confirmed new allocation; inspect sanitized allocation log before retrying')
    endpoint=found[0]['endpoint'];common=['--session',session,'--expect-endpoint',endpoint]
    receipt=dict(session=session,endpoint=endpoint,source_archive_sha256=hashlib.sha256((run/'source.tar').read_bytes()).hexdigest())
    (out/'allocation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    ok=False
    try:
        base='/content/Strassen_MM_Focus/'+run.name
        code=command('setup',controller+['exec-file',*common,'--file',str(source/'runtime/setup_v004.py'),'--timeout','1500',
            '--script-arg=--output-dir','--script-arg='+base+'-setup','--script-arg=--expected-endpoint','--script-arg='+endpoint],1620)
        if code:raise RuntimeError('Runtime setup did not complete')
        code=command('upload',controller+['upload',*common,'--local',str(run/'source.tar'),'--remote',base+'-source.tar'])
        if code:raise RuntimeError('Frozen source upload failed')
        code=command('execute',controller+['exec-file',*common,'--file',str(source/'runtime/check_fusion_remote_v002.py'),'--timeout','1020',
            '--script-arg=--archive','--script-arg='+base+'-source.tar','--script-arg=--sha256','--script-arg='+receipt['source_archive_sha256'],
            '--script-arg=--output','--script-arg='+base+'-check'],1140)
        # Even a lost execution response may have a complete result archive.
        package=out/'remote-check.tar.gz'
        downloaded=command('download',controller+['download',*common,'--remote',base+'-check.tar.gz','--local',str(package)])
        if downloaded:raise RuntimeError('Qualification archive was not downloaded')
        receipt['result_sha256']=hashlib.sha256(package.read_bytes()).hexdigest()
        with tarfile.open(package) as t:t.extractall(out/'remote',filter='data')
        summary=json.loads((out/'remote'/(run.name+'-check')/'artifacts/summary.json').read_text())
        receipt['qualification']=summary;ok=summary['failed']==0
    finally:
        released=command('release',[a.controller_python,str(source/'runtime/release_allocation_v002.py'),*common])
        receipt.update(release_exit_code=released,all_checks_passed=ok)
        (out/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if released:raise RuntimeError('Release was not confirmed; inspect release.log')
    return 0 if ok else 1

if __name__=='__main__':raise SystemExit(main())
