"""Adopt a sole idle assignment after an owned allocation request lost its response."""
import argparse
import json
from pathlib import Path
import requests
import colab_control_v003 as v6
import colab_control_v002 as v5

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--expect-endpoint', required=True)
    a = p.parse_args()
    request = json.loads(a.request.read_text())
    if request['before_assignments'] != [] or request['hardware'] not in ('v5e', 'v6e'):
        raise ValueError('Adoption requires a recorded empty allocation list before our request')
    control = v6 if request['hardware'] == 'v6e' else v5
    a.config = control.DEFAULT_CONFIG
    a.token_file = control.DEFAULT_TOKEN
    a.session = request['session']
    wanted = 'V6E1' if request['hardware'] == 'v6e' else 'V5E1'
    from colab_cli.state import SessionState
    with control.api(a.token_file) as client:
        assigned = client.list_assignments()
        if len(assigned) != 1 or assigned[0].endpoint != a.expect_endpoint or assigned[0].accelerator.value != wanted:
            raise ValueError('Ambiguous late allocation; do not adopt')
        assigned = assigned[0]
        store = control.store_at(a.config)
        saved = store.get(a.session)
        if saved:
            if saved.endpoint != a.expect_endpoint:
                raise ValueError('Session endpoint changed')
            control.emit(dict(kind='allocation', endpoint=saved.endpoint, session=saved.name, created=False))
            return
        if any(s.endpoint == a.expect_endpoint for s in store.list().values()):
            raise ValueError('Assignment belongs to another saved session')
        token = assigned.runtime_proxy_info.token
        control.remember_secret(token)
        response = requests.get(assigned.runtime_proxy_info.url.rstrip('/') + '/api/kernels',
            params={'authuser': '0', 'colab-runtime-proxy-token': token}, timeout=(20, 45))
        response.raise_for_status()
        if response.json() != []:
            raise ValueError('Late assignment already has kernels; do not adopt')
        saved = SessionState(name=a.session, token=token, url=assigned.runtime_proxy_info.url,
            endpoint=a.expect_endpoint, variant='TPU', accelerator=wanted)
        store.add(saved)
        control.launch_heartbeat(a, saved, store)
        control.emit(dict(kind='allocation', endpoint=saved.endpoint, session=saved.name, created=False))

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        v6.emit(dict(kind='adoption_error', **v6.safe_error(exc)))
        raise SystemExit(1)
