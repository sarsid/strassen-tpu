"""Recover the sole late v6e assignment after an archived POST timeout."""
import argparse
import hashlib
import json
from pathlib import Path
import requests
import colab_control_v003 as control


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',required=True,type=control.session_name)
    p.add_argument('--expect-endpoint',required=True)
    p.add_argument('--failed-request',type=Path,required=True)
    p.add_argument('--empty-check',type=Path,required=True)
    p.add_argument('--config',type=Path,default=control.DEFAULT_CONFIG)
    p.add_argument('--token-file',type=Path,default=control.DEFAULT_TOKEN)
    a=p.parse_args();failure=json.loads(a.failed_request.read_text());empty=json.loads(a.empty_check.read_text())
    if failure.get('error_type')!='ReadTimeout' or failure.get('method')!='POST' or failure.get('hardware')!='v6e':
        raise RuntimeError('Expected the recorded v6e allocation POST timeout')
    if empty.get('kind')!='allocations' or empty.get('assignments')!=[]:
        raise RuntimeError('Expected the empty post-timeout allocation check')
    from colab_cli.state import SessionState
    with control.api(a.token_file) as client:
        assignments=client.list_assignments()
        if len(assignments)!=1 or assignments[0].endpoint!=a.expect_endpoint or assignments[0].accelerator.value!='V6E1':
            raise RuntimeError('Expected the sole late v6e endpoint')
        assigned=assignments[0];store=control.store_at(a.config)
        if store.get(a.session) is not None or any(s.endpoint==a.expect_endpoint for s in store.list().values()):
            raise RuntimeError('Session or endpoint already registered')
        token=assigned.runtime_proxy_info.token;control.remember_secret(token)
        response=requests.get(assigned.runtime_proxy_info.url.rstrip('/')+'/api/kernels',
            params={'authuser':'0','colab-runtime-proxy-token':token},timeout=(20,45))
        response.raise_for_status()
        if response.json()!=[]:
            raise RuntimeError('Late runtime already has kernels; refusing to adopt a potentially active notebook')
        saved=SessionState(name=a.session,token=token,url=assigned.runtime_proxy_info.url,
            endpoint=a.expect_endpoint,variant='TPU',accelerator='V6E1')
        store.add(saved);pid=control.launch_heartbeat(a,saved,store)
        control.emit(dict(kind='late_allocation_recovered',session=saved.name,endpoint=saved.endpoint,
            created=False,accelerator='V6E1',kernel_count_before_adoption=0,heartbeat_pid=pid,
            evidence_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.failed_request,a.empty_check)}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        control.emit(dict(kind='late_adoption_error',**control.safe_error(exc)))
        raise SystemExit(1)
