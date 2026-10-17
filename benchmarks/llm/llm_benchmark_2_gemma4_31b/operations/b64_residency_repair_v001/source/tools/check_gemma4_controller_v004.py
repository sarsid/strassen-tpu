"""Exercise timeout recovery through the controller without a TPU or network."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import copy
from unittest.mock import patch

import run_gemma4_batch_v004 as controller


def main():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
    import poll_gemma4_batch_v001 as poll
    with tempfile.TemporaryDirectory() as tmp:
        remote=Path(tmp)/'remote';local=Path(tmp)/'local';remote.mkdir();local.mkdir()
        profile=remote/'artifacts/b64-tune-a0/profile.json';profile.parent.mkdir(parents=True)
        profile.write_text('{"profile":{"native_options":{}},"batch":64,"sequence":1024}\n')
        (remote/'status.json').write_text(json.dumps(dict(status='running',active='b64-qualify-a0',stages=[dict(name='b64-tune-a0',returncode=0)])))
        snapshot=poll.make_snapshot(remote)
        files=controller.save_stage_exports(local,snapshot)
        assert len(files)==1 and files[0].read_bytes()==profile.read_bytes()
        assert controller.save_stage_exports(local,snapshot)==[]
        bad=copy.deepcopy(snapshot);bad['stage_exports'][0]['files']['profile.json']['content_utf8']='{}'
        try:controller.save_stage_exports(local,bad)
        except ValueError:pass
        else:raise AssertionError('Corrupt checkpoint accepted')
        bad=copy.deepcopy(snapshot);bad['stage_exports'][0]['name']='../escape'
        try:controller.save_stage_exports(local,bad)
        except ValueError:pass
        else:raise AssertionError('Unsafe path accepted')
        original=json.loads((remote/'status.json').read_text());original['stages'][0]['returncode']=1
        (remote/'status.json').write_text(json.dumps(original));assert poll.make_snapshot(remote)['stage_exports']==[]
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); logs=root/'logs'; logs.mkdir()
        code,rows=controller.control_command(logs,'real-timeout',
            [sys.executable,'-c',"import time; print('{\"kind\":\"before_timeout\"}',flush=True); time.sleep(2)"],.3)
        assert code==124
        assert any(r.get('kind')=='control_command_timeout' for r in rows)
        assert any(r.get('kind')=='before_timeout' for r in rows)
        code,rows=controller.control_command(logs,'after-timeout',
            [sys.executable,'-c',"print('{\"kind\":\"recovered\"}')"],2)
        assert code==0 and rows==[dict(kind='recovered')]

        for scenario in ('recovered','lost','empty','worker_failed','bad_export','quality_failed'):
            run=root/'runs'/('simulated-b64-'+scenario);run.mkdir(parents=True)
            source=run/'source';(source/'configs').mkdir(parents=True)
            (source/'configs/gemma4_parent_batch_v003.json').write_text(json.dumps(
                dict(batches=[64],result_root='results/simulated')))
            (run/'source.tar').write_bytes(b'fake-source')
            endpoint='owned-test-endpoint'
            (run/'allocation-events.jsonl').write_text(json.dumps(dict(kind='allocation',
                created=True,session=run.name,endpoint=endpoint))+'\n')
            result_file=root/'result.json';result_file.write_text('{"status":"completed"}\n')
            package=root/'export.tar.gz'
            with tarfile.open(package,'w:gz') as t:t.add(result_file,arcname='test/artifacts/summary.json')
            ready=dict(path='/remote/results.tar.gz',sha256=hashlib.sha256(package.read_bytes()).hexdigest())
            calls=[];polls=0
    
            def fake_run(argv,**kw):
                nonlocal polls
                argv=[str(x) for x in argv];calls.append(argv)
                def emit(value):kw['stdout'].write(json.dumps(value)+'\n')
                if argv[1].endswith('release_allocation_v002.py'):
                    assert scenario=='recovered', 'Release forbidden on failure'
                    assert (run/'remote/test/artifacts/summary.json').exists(), 'Release before verified export'
                    emit(dict(kind='allocation_released',verified_absent=True))
                elif argv[1].endswith('colab_control_v003.py'):
                    op=argv[2]
                    if op=='list':emit(dict(kind='allocations',assignments=[] if scenario=='lost' and polls>=2 else [dict(endpoint=endpoint,name=run.name,accelerator='V6E1')]))
                    elif op=='download':Path(argv[argv.index('--local')+1]).write_bytes(b'corrupt' if scenario=='bad_export' else package.read_bytes())
                    elif op=='exec-file' and argv[argv.index('--file')+1].endswith('poll_gemma4_batch_v001.py'):
                        polls+=1
                        if scenario=='empty':
                            emit(dict(kind='remote_stream',text=json.dumps({'kind':'gemma4_batch_snapshot','tails':{},'stage_exports':[]})));return subprocess.CompletedProcess(argv,0)
                        if polls==1:raise subprocess.TimeoutExpired(argv,120)
                        if scenario=='lost':
                            emit(dict(kind='controller_error',message='The saved endpoint is not currently allocated'));return subprocess.CompletedProcess(argv,1)
                        snapshot={'kind':'gemma4_batch_snapshot','status.json':dict(status='failed' if scenario=='worker_failed' else 'completed',batches=[dict(batch=64,status='completed',quality_pass=scenario!='quality_failed')],batch_archives=[]),'archive-ready.json':ready,'tails':{}}
                        emit(dict(kind='remote_stream',text=json.dumps(snapshot)))
                return subprocess.CompletedProcess(argv,0)
    
            args=['run_gemma4_batch_v004.py','--run',str(run),'--controller-python',sys.executable,'--allocated-endpoint',endpoint]
            with patch.object(controller,'__file__',str(root/'tools/run_gemma4_batch_v004.py')), \
                 patch.object(controller.subprocess,'run',side_effect=fake_run), \
                 patch.object(controller.time,'sleep'),patch.object(sys,'argv',args):
                assert controller.main()==(0 if scenario=='recovered' else 1)
            completion=json.loads((run/'completion.json').read_text())
            assert completion['status']==('runtime_lost' if scenario=='lost' else 'controller_failed' if scenario in ('empty','bad_export') else 'failed' if scenario=='worker_failed' else 'completed')
            assert completion['release_attempted']==(scenario=='recovered')
            assert completion['allocation_retained']==(scenario not in ('lost','recovered'))
            assert completion['release_exit_code']==(0 if scenario in ('lost','recovered') else None)
            assert (run/'retained-allocation.json').exists()==(scenario not in ('lost','recovered'))
            if scenario=='lost':assert completion['allocation_verified_absent'] and completion['release_not_needed']
            if scenario!='empty':assert 'control_command_timeout' in (run/'control/poll-000.log').read_text()
            else:assert 'no worker status' in completion['error']
            assert len([c for c in calls if c[1].endswith('release_allocation_v002.py')])==(1 if scenario=='recovered' else 0)
            assert len([c for c in calls if 'exec-file' in c and any('launch_gemma4_bundle_v004.py' in v for v in c)])==1
            assert not any(Path(c[1]).name.startswith('allocate_') or (Path(c[1]).name=='colab_control_v003.py' and c[2]=='create') for c in calls)
    print(json.dumps(dict(status='passed',actual_subprocess_timeout='recovered',
        controller_poll_timeout='retried_without_release_or_relaunch',
        archive_download='verified_or_failure_preserved',retained_on_failure_and_released_only_after_full_success=True,checkpoint_integrity='passed',verified_absent_runtime='stopped_without_unassigning',scope='local orchestration regression; no TPU/network')))


if __name__=='__main__':main()
