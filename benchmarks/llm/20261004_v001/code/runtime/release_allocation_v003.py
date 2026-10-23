"""Idempotently release exactly the owned v6e; accept verified prior expiry."""
import argparse
from pathlib import Path
import colab_control_v003 as control


def release(client,store,session,endpoint):
    saved=store.get(session)
    if saved is not None and saved.endpoint!=endpoint:raise RuntimeError('Saved endpoint changed; refusing release')
    matches=[a for a in client.list_assignments() if a.endpoint==endpoint]
    if matches:
        if saved is None or len(matches)!=1 or matches[0].accelerator.value!='V6E1':raise RuntimeError('Cannot establish ownership of the expected v6e')
        client.unassign(endpoint)
        if any(a.endpoint==endpoint for a in client.list_assignments()):raise RuntimeError('Endpoint still allocated after release')
    if saved is not None:store.remove(session)
    return dict(kind='allocation_released',session=session,endpoint=endpoint,verified_absent=True,already_absent=not matches)


def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--expect-endpoint',required=True)
    p.add_argument('--config',type=Path,default=control.DEFAULT_CONFIG);p.add_argument('--token-file',type=Path,default=control.DEFAULT_TOKEN);a=p.parse_args()
    with control.api(a.token_file) as client:control.emit(release(client,control.store_at(a.config),a.session,a.expect_endpoint))

if __name__=='__main__':
    try:main()
    except Exception as error:control.emit(dict(kind='release_error',**control.safe_error(error)));raise SystemExit(1)
