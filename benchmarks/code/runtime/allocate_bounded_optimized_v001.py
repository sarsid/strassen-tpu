"""Allocate one explicitly authorized extra v5e with a local release watchdog.

The existing controller rejects concurrent allocations. This separate entry
point requires naming the one allocation that may coexist, never touches it,
and records an exclusive receipt. Use only after user approval for extra compute.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import colab_control_v001 as control


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def watchdog(args):
    # API release includes several requests. Reserve five minutes for cleanup.
    while time.time() < args.deadline - 300:
        saved = control.store_at(args.config).get(args.session)
        if saved is None or saved.endpoint != args.expect_endpoint:
            return 0
        time.sleep(min(30, max(0, args.deadline - 300 - time.time())))
    for attempt in range(3):
        try:
            with control.api(args.token_file) as client:
                matches = [a for a in client.list_assignments() if a.endpoint == args.expect_endpoint]
                if matches:
                    client.unassign(args.expect_endpoint)
                remaining = any(a.endpoint == args.expect_endpoint for a in client.list_assignments())
                if remaining:
                    raise RuntimeError('Endpoint still allocated after release request')
                store = control.store_at(args.config)
                saved = store.get(args.session)
                if saved is not None and saved.endpoint == args.expect_endpoint:
                    store.remove(saved.name)
                write(args.receipt_dir / 'watchdog-release.json',
                      {'endpoint': args.expect_endpoint, 'verified_absent': True,
                       'released_unix': time.time(), 'deadline_unix': args.deadline})
                return 0
        except Exception as error:
            write(args.receipt_dir / f'watchdog-error-{attempt}.json', control.safe_error(error))
            if attempt < 2:
                time.sleep(10)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=control.session_name)
    parser.add_argument('--coexist-endpoint')
    parser.add_argument('--receipt-dir', required=True, type=Path)
    parser.add_argument('--minutes', type=int, default=30)
    parser.add_argument('--config', type=Path, default=control.DEFAULT_CONFIG)
    parser.add_argument('--token-file', type=Path, default=control.DEFAULT_TOKEN)
    parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--expect-endpoint')
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if args.watchdog:
        if not args.expect_endpoint or args.deadline is None:
            parser.error('Watchdog requires endpoint and deadline')
        return watchdog(args)
    if not 2 <= args.minutes <= 30 or not args.coexist_endpoint:
        parser.error('Explicit coexist endpoint and a 2–30 minute budget are required')
    args.receipt_dir.mkdir(parents=True, exist_ok=False)
    from colab_cli.client import Accelerator, Variant
    from colab_cli.state import SessionState
    with control.api(args.token_file) as client:
        assignments = client.list_assignments()
        if any(a.endpoint != args.coexist_endpoint for a in assignments):
            raise RuntimeError('Unexpected live allocation; refusing another allocation')
        store = control.store_at(args.config)
        if store.get(args.session) is not None:
            raise RuntimeError('Session name already exists; refusing ambiguous reuse')
        assigned = client.assign(uuid.uuid4(), variant=Variant.TPU, accelerator=Accelerator.V5E1)
        if assigned.endpoint == args.coexist_endpoint:
            raise RuntimeError('Service returned the other task endpoint; no ownership or release action will be taken')
        started = time.time()
        deadline = started + args.minutes * 60
        control.remember_secret(assigned.runtime_proxy_info.token)
        saved = SessionState(name=args.session, token=assigned.runtime_proxy_info.token,
                  url=assigned.runtime_proxy_info.url, endpoint=assigned.endpoint,
                  variant='TPU', accelerator='V5E1')
        try:
            store.add(saved)
            command = [sys.executable, str(Path(__file__).resolve()), '--watchdog', '--session', args.session,
                       '--receipt-dir', str(args.receipt_dir.resolve()), '--expect-endpoint', saved.endpoint,
                       '--deadline', str(deadline), '--config', str(args.config), '--token-file', str(args.token_file)]
            with (args.receipt_dir / 'watchdog.log').open('x') as log:
                guard = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            heartbeat = control.launch_heartbeat(args, saved, store)
            receipt = {'kind': 'allocation', 'created': True, 'session': saved.name,
                       'endpoint': saved.endpoint, 'accelerator': 'V5E1', 'started_unix': started,
                       'deadline_unix': deadline, 'watchdog_pid': guard.pid, 'heartbeat_pid': heartbeat,
                       'coexisting_endpoint_untouched': args.coexist_endpoint}
            write(args.receipt_dir / 'allocation.json', receipt)
            control.emit(receipt)
        except BaseException:
            client.unassign(assigned.endpoint)
            raise
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        control.emit({'kind': 'bounded_allocation_error', **control.safe_error(error)})
        raise SystemExit(1)
