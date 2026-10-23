"""One slow-provisioning attempt with a persistent request identity and status.

No credential/header/response-body logging. A fixed notebook UUID lets the same
request be reconciled after a lost response. Waiting events do not imply a TPU
allocation or computation. Refuse to create alongside any existing assignment.
"""
import argparse,json,subprocess,threading,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import colab_control_v003 as control

def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True,type=control.session_name)
    p.add_argument('--journal-root',type=Path,required=True);p.add_argument('--events',type=Path,required=True)
    p.add_argument('--config',type=Path,default=control.DEFAULT_CONFIG);p.add_argument('--token-file',type=Path,default=control.DEFAULT_TOKEN)
    a=p.parse_args();from colab_cli.client import Accelerator,Variant,Assignment
    from colab_cli.state import SessionState
    nbh=uuid.uuid5(uuid.NAMESPACE_URL,'strassen-llm-provisioning/'+a.session)
    started=time.monotonic();stop=threading.Event();state={'phase':'preflight'};lock=threading.Lock()
    def emit(row):
        record=control.safe_value(dict(utc=datetime.now(timezone.utc).isoformat(),**row))
        with lock:
            with a.events.open('a') as f:f.write(json.dumps(record)+'\n')
        control.emit(record)
    def progress():
        while not stop.wait(30):
            elapsed=round(time.monotonic()-started)
            emit(dict(kind='provisioning_wait',phase=state['phase'],elapsed_seconds=elapsed,request_notebook_uuid=str(nbh),measurement_started=False))
            subprocess.run(['python3',str(a.journal_root/'tools/record_progress_v001.py'),'--kind','activity','--state','working',
                '--title','Waiting for Colab v6e provisioning','--detail','Single extended request: '+str(elapsed)+' seconds elapsed; phase '+state['phase']+'. No TPU endpoint or measurements yet.',
                '--next-step','Await response, reconcile the same request identity, and archive any service failure.',
                '--evidence',str(a.events.relative_to(a.journal_root))],stdout=subprocess.DEVNULL,check=False)
    thread=threading.Thread(target=progress,daemon=True);thread.start()
    try:
        with control.api(a.token_file) as client:
            store=control.store_at(a.config);saved=store.get(a.session)
            if saved is not None:raise RuntimeError('Session already exists; explicit reconciliation required')
            assignments=client.list_assignments()
            emit(dict(kind='preflight_assignments',count=len(assignments)))
            if assignments:raise RuntimeError('Existing assignment prevents new allocation')
            request=client.session.request
            def observed(method,url,*args,**kwargs):
                assignment=url.split('?',1)[0].endswith('/assign')
                if assignment:
                    kwargs['timeout']=(20,600);state['phase']='assign_'+method.lower()
                    emit(dict(kind='provisioning_request',method=method,timeout_seconds=600,request_notebook_uuid=str(nbh)))
                t=time.monotonic()
                response=request(method,url,*args,**kwargs)
                if assignment:emit(dict(kind='provisioning_response',method=method,http_status=response.status_code,elapsed_seconds=round(time.monotonic()-t,3)))
                return response
            client.session.request=observed
            assigned=client.assign(nbh,variant=Variant.TPU,accelerator=Accelerator.V6E1)
            control.remember_secret(assigned.runtime_proxy_info.token)
            saved=SessionState(name=a.session,token=assigned.runtime_proxy_info.token,url=assigned.runtime_proxy_info.url,endpoint=assigned.endpoint,variant='TPU',accelerator='V6E1')
            store.add(saved);pid=control.launch_heartbeat(a,saved,store)
            stop.set()
            emit(dict(kind='allocation',created=True,session=a.session,endpoint=saved.endpoint,accelerator='V6E1',heartbeat_pid=pid,elapsed_seconds=round(time.monotonic()-started,3)))
            return 0
    except Exception as error:
        stop.set();emit(dict(kind='allocation_error',phase=state['phase'],**control.safe_error(error),elapsed_seconds=round(time.monotonic()-started,3)))
        return 1
    finally:stop.set();thread.join(timeout=2)

if __name__=='__main__':raise SystemExit(main())
