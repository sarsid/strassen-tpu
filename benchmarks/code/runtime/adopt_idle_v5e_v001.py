"""Register an explicitly user-authorized idle v5e without creating a machine."""
import argparse
from pathlib import Path
import colab_control_v001 as control


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',required=True,type=control.session_name)
    parser.add_argument('--expect-endpoint',required=True)
    parser.add_argument('--config',type=Path,default=control.DEFAULT_CONFIG)
    parser.add_argument('--token-file',type=Path,default=control.DEFAULT_TOKEN)
    args=parser.parse_args()
    from colab_cli.state import SessionState
    with control.api(args.token_file) as client:
        assignments=client.list_assignments()
        matches=[a for a in assignments if a.endpoint==args.expect_endpoint]
        if len(assignments)!=1 or len(matches)!=1:
            raise RuntimeError('Expected the one explicitly authorized idle allocation')
        assigned=matches[0]
        if assigned.accelerator.value!='V5E1':raise RuntimeError('Expected V5E1')
        store=control.store_at(args.config)
        if store.get(args.session) is not None:raise RuntimeError('Session name already exists')
        if any(s.endpoint==assigned.endpoint for s in store.list().values()):
            raise RuntimeError('Allocation already registered under another session')
        control.remember_secret(assigned.runtime_proxy_info.token)
        saved=SessionState(name=args.session,token=assigned.runtime_proxy_info.token,
            url=assigned.runtime_proxy_info.url,endpoint=assigned.endpoint,variant='TPU',accelerator='V5E1')
        store.add(saved)
        heartbeat=control.launch_heartbeat(args,saved,store)
        control.emit(dict(kind='idle_allocation_adopted',session=saved.name,endpoint=saved.endpoint,
            accelerator='V5E1',created=False,heartbeat_pid=heartbeat,
            authorization='User confirmed this existing TPU notebook is not running a job.'))


if __name__=='__main__':
    try:main()
    except Exception as error:
        control.emit(dict(kind='adoption_error',**control.safe_error(error)))
        raise SystemExit(1)
