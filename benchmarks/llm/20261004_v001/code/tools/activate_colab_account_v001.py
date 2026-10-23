"""Activate staged Google credentials only while the architecture supervisor is stopped.

Old credential/session records stay in a private backup; archived scientific
history remains untouched. Clearing the old pending allocation prevents its
accidental attribution to an assignment on the newly authenticated account.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys

logging.disable(logging.CRITICAL)


def read(path):
    return json.loads(path.read_text())


def atomic(path, data):
    temp = path.with_name(path.name + '.account-switch-tmp')
    with temp.open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temp.chmod(0o600)
    os.replace(temp, path)


def main():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import AuthorizedSession
    from colab_cli.client import Client, Prod
    from filelock import ReadWriteLock

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-dir', type=Path, required=True)
    p.add_argument('--supervisor', type=Path, required=True)
    p.add_argument('--receipt-dir', type=Path, required=True)
    args = p.parse_args()
    root = Path(os.environ.get('STRASSEN_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
    folder = args.supervisor.resolve()
    receipt = args.receipt_dir.resolve()
    store = (Path.home() / '.config/colab-cli').resolve()
    staged = args.private_dir.resolve()
    assert folder.is_relative_to(root / 'runs') and receipt.is_relative_to(folder)
    assert staged.is_relative_to(store) and staged != store
    assert read(staged / 'completed.json')['state'] == 'authenticated_pending_activation'
    assert read(staged / 'preactivation-check.json')['different_from_previous_account'] is True
    os.umask(0o077)
    with (folder / 'start.lock').open('ab') as starter, (folder / 'supervisor.lock').open('ab') as supervisor:
        fcntl.flock(starter, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(supervisor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(folder / 'supervisor-state.json')
        assert state.get('runtime') is None and state.get('current_cohort') is None
        assert state['action'] == 'paused-user' and not state.get('finished')
        credentials = Credentials.from_authorized_user_file(str(staged / 'token.json'))
        account = read(staged / 'account.json')
        with AuthorizedSession(credentials, refresh_timeout=20) as session:
            response = session.get('https://www.googleapis.com/oauth2/v2/userinfo', timeout=30)
            response.raise_for_status()
            actual = response.json()
            assert actual.get('verified_email') is True and actual['id'] == account['subject']
            assignments = Client(Prod(), session).list_assignments()
            if assignments:
                raise RuntimeError('New account has an existing allocation; migration left inactive')
        receipt.mkdir(parents=True, exist_ok=False)
        backup = staged / 'previous-account-backup'
        backup.mkdir(mode=0o700, exist_ok=False)
        # Lock the same SDK session store used by the installed CLI. No cloud
        # runtimes are stopped and the old private store can be restored.
        with ReadWriteLock(str(store / 'sessions.json.lock'), is_singleton=False).write_lock():
            for name in ('token.json', 'sessions.json', 'active-account.json'):
                path = store / name
                if path.exists():
                    shutil.copyfile(path, backup / name)
                    (backup / name).chmod(0o600)
            atomic(store / 'token.json', json.loads(credentials.to_json()))
            atomic(store / 'sessions.json', {})
            atomic(store / 'active-account.json', {**account, 'private_authentication_dir':str(staged)})
        atomic(receipt / 'supervisor-state-before.json', state)
        history_hash = hashlib.sha256((folder / 'history.json').read_bytes()).hexdigest()
        old_request = state.get('pending_request')
        state.update(pending_request=None, allocation_requests=[], runtime_needs_release=False,
            action='authenticated-ready-to-resume', detail='New Google account authenticated by owner; old-account pending allocation and cached sessions privately separated.',
            account_transition=str(receipt.relative_to(root)))
        atomic(folder / 'supervisor-state.json', state)
        record = dict(completed_utc=datetime.now(timezone.utc).isoformat(), credentials_activated=True,
            different_account_verified=True, new_account_assignments_before_resume=[],
            private_previous_account_backup=True, old_pending_request=old_request,
            old_pending_request_cleared=True, history_sha256_unchanged=history_hash,
            owned_runtime=None, allocations_created=0, runtimes_terminated=0,
            measurement_contract_changed=False, ready_to_resume=True)
        atomic(receipt / 'activation.json', record)
        with (folder / 'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(utc=record['completed_utc'], event='authenticated_account_transition',
                evidence=str(receipt.relative_to(root)), runtime=None, ready_to_resume=True)) + '\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(state='account_activation_failed', error_type=type(error).__name__)), flush=True)
        sys.exit(1)
