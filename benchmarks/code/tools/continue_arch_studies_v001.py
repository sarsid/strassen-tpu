"""One-shot handoff: wait for successful/released v6e, then launch historical v5e.

This is not a retry loop. Any incomplete v6e outcome, allocation ambiguity,
setup failure or launch failure stops the handoff with a saved diagnostic.
Credentials stay in the existing Colab SDK store.
"""
import argparse,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from run_large_real_v002 import commit

def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--prior-cohort',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    source=Path(__file__).resolve().parents[1];root=Path(os.environ['STRASSEN_PROJECT_ROOT']);out=a.output_dir;out.mkdir(parents=True,exist_ok=False)
    prior=json.loads((a.prior_cohort/'plan.json').read_text());controller=prior['controller_python'];analysis=prior['analysis_python']
    save(out/'started.json',dict(pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),prior_cohort=str(a.prior_cohort),source=str(source),action='Wait for completed, released v6e; then start v5e once'))
    while not (a.prior_cohort/'controller-completion.json').exists():time.sleep(30)
    done=json.loads((a.prior_cohort/'controller-completion.json').read_text())
    if done['status']!='completed' or not done['allocation_released'] or done['remote_execution_uncertain']:
        save(out/'completion.json',dict(status='blocked',reason='v6e did not complete and release cleanly',prior=done));commit([out],'Record blocked architecture-study handoff');return 1
    session='strassen-arch-v5e-20260927';cohort=root/'runs/20260927-arch-v5e-168-v001';endpoint=None
    def run(label,cmd,timeout):
        save(out/(label+'-command.json'),dict(command=cmd))
        with (out/(label+'.log')).open('x') as f:
            result=subprocess.run(cmd,cwd=source,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        if result.returncode:raise RuntimeError(label+' failed; inspect saved sanitized log, do not blindly retry')
        return (out/(label+'.log')).read_text()
    try:
        # The general archive runner locates its repository from its own file.
        # Use that root copy only after proving it matches our frozen copy.
        for name in ('tools/archive_scoped_v001.py','tools/setup_allocation_v004.py'):
            if (root/name).read_bytes()!=(source/name).read_bytes():raise ValueError('Setup tool changed since handoff was frozen: '+name)
        for name in ('kernels_v001.py','kernels_v002.py','kernels_two_level_v001.py','benchmark_mlsys_shapes_v001.py'):
            if (root/'src/strassen_mm'/name).read_bytes()!=(source/'src/strassen_mm'/name).read_bytes():raise ValueError('Historical v5e implementation changed: '+name)
        allocation=run('allocate',[controller,str(source/'runtime/allocate_tradeoff_v002.py'),'v5e','create','--session',session],600)
        records=[json.loads(l) for l in allocation.splitlines()];record=next(r for r in records if r.get('kind')=='allocation');endpoint=record['endpoint']
        # Create and commit a frozen setup execution using the established archiver.
        setup=run('setup',[sys.executable,str(root/'tools/archive_scoped_v001.py'),'--label','arch-v5e-setup-v001','--source','runtime','--source','tools/setup_allocation_v004.py','--timeout-seconds','1800','--',controller,'tools/setup_allocation_v004.py','--session',session,'--endpoint',endpoint,'--hardware','v5e','--controller-python',controller],1900)
        setupdir=Path(next(l.split('=',1)[1] for l in setup.splitlines() if l.startswith('EXECUTION_DIR=')))
        identity=json.loads((setupdir/'artifacts/summary.json').read_text())['remote_identity']
        prepare=[sys.executable,str(source/'tools/prepare_arch_v5e_v001.py')]
        run('freeze',prepare+['freeze','--cohort',str(cohort),'--hardware','v5e','--session',session,'--endpoint',endpoint,'--identity',identity,'--controller-python',controller,'--analysis-python',analysis],300)
        launch=run('launch',prepare+['launch','--cohort',str(cohort),'--port','8789'],120)
        save(out/'completion.json',dict(status='launched',v5e_cohort=str(cohort),dashboard='http://127.0.0.1:8789/',launch=launch))
        commit([out],'Launch historical v5e after completed v6e study');return 0
    except Exception as exc:
        # If setup/freeze failed before launch, release only the known allocation.
        if endpoint and not (cohort/'controller-process.json').exists():
            try:run('release-failed-handoff',[controller,str(source/'runtime/release_allocation_v001.py'),'--session',session,'--expect-endpoint',endpoint],180)
            except Exception:pass
        save(out/'completion.json',dict(status='attention',error_type=type(exc).__name__,message=str(exc),endpoint=endpoint))
        commit([out],'Archive architecture-study handoff problem');return 1

if __name__=='__main__':raise SystemExit(main())
