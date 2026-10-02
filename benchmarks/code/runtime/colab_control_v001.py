#!/usr/bin/env python3
"""Small Colab controller with refreshed proxy tokens and sanitized output.

Use the existing .venv-colab interpreter. Credentials stay in the existing SDK
store. This module never initializes CLI logging or launches interactive auth.
All allocation/execution actions require an explicit subcommand.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
from urllib.parse import quote
import uuid

# The installed client logs HTTP headers/bodies at DEBUG. Do not enable it.
logging.disable(logging.CRITICAL)
_SECRETS: set[str] = set()
REMOTE_ROOT = PurePosixPath('/content/Strassen_MM_Focus')
DEFAULT_CONFIG = Path.home() / '.config/colab-cli/sessions.json'
DEFAULT_TOKEN = Path.home() / '.config/colab-cli/token.json'


def remember_secret(value):
    if isinstance(value, str) and len(value) >= 8:
        _SECRETS.add(value)


def safe_text(value):
    text = str(value)
    for secret in sorted(_SECRETS, key=len, reverse=True):
        text = text.replace(secret, '[REDACTED]')
    text = re.sub(r'(?i)(authorization[\"\']?\s*[:=]\s*[\"\']?(?:bearer\s+)?)[^\s\"\',}]+', r'\1[REDACTED]', text)
    text = re.sub(r'(?i)((?:colab-runtime-proxy-token|access_token|refresh_token|client_secret|id_token)[\"\']?\s*[:=]\s*[\"\']?)[^\s\"\'&,}]+', r'\1[REDACTED]', text)
    return text


def safe_value(value):
    if isinstance(value, dict):
        return {key: ('[REDACTED]' if str(key).lower() in
                     ('authorization', 'token', 'access_token', 'refresh_token',
                      'id_token', 'client_secret', 'colab-runtime-proxy-token')
                     else safe_value(item)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_value(item) for item in value]
    return safe_text(value) if isinstance(value, str) else value


def emit(value):
    print(json.dumps(safe_value(value), sort_keys=True, default=str), flush=True)


def safe_error(error):
    response = getattr(error, 'response', None)
    result = {'error_type': type(error).__name__,
              'http_status': getattr(response, 'status_code', None)}
    if type(error) in (RuntimeError, ValueError, FileExistsError, FileNotFoundError, IsADirectoryError):
        result['message'] = safe_text(str(error))
    return result


def remote_path(value):
    path = PurePosixPath(value)
    if '..' in path.parts or not path.is_absolute() or not path.is_relative_to(REMOTE_ROOT):
        raise ValueError('Remote files must be inside /content/Strassen_MM_Focus')
    return str(path)


def session_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', value):
        raise argparse.ArgumentTypeError('Use a short alphanumeric session name')
    return value


@contextmanager
def api(token_path):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import AuthorizedSession
    from colab_cli.client import Client, Prod

    credentials = Credentials.from_authorized_user_file(str(token_path))
    for field in ('token', 'refresh_token', 'id_token', 'client_secret'):
        remember_secret(getattr(credentials, field, None))

    class TimedSession(AuthorizedSession):
        def request(self, method, url, *args, **kwargs):
            kwargs.setdefault('timeout', 45)
            response = super().request(method, url, *args, **kwargs)
            remember_secret(self.credentials.token)
            return response

    with TimedSession(credentials, refresh_timeout=30) as session:
        yield Client(Prod(), session)


def store_at(path):
    from colab_cli.state import StateStore
    return StateStore(str(path))


def sanitized_assignment(assignment, name=None):
    return {'name': name, 'endpoint': assignment.endpoint,
            'accelerator': assignment.accelerator.value,
            'variant': assignment.variant.value,
            'machine_shape': assignment.machine_shape.value if hasattr(assignment, 'machine_shape') else None}


def refreshed_session(args, client):
    store = store_at(args.config)
    saved = store.get(args.session)
    if saved is None:
        raise RuntimeError('Named session absent from local SDK store')
    if args.expect_endpoint and saved.endpoint != args.expect_endpoint:
        raise RuntimeError('Saved endpoint changed; refusing runtime switch')
    assignments = client.list_assignments()
    matches = [a for a in assignments if a.endpoint == saved.endpoint]
    if len(matches) != 1:
        raise RuntimeError('The saved endpoint is not currently allocated')
    assigned = matches[0]
    if assigned.accelerator.value != 'V5E1':
        raise RuntimeError('This campaign controller requires V5E1')
    saved.url = assigned.runtime_proxy_info.url
    saved.token = assigned.runtime_proxy_info.token
    remember_secret(saved.token)
    # Preserve endpoint and existing kernel IDs; credentials remain in SDK store.
    store.add(saved)
    return store, saved


def content_request(saved, method, path, *, payload=None, content=False):
    import requests
    url = saved.url.rstrip('/') + '/api/contents/' + quote(path.strip('/'), safe='/')
    params = {'authuser': '0', 'colab-runtime-proxy-token': saved.token}
    if content:
        params['content'] = '1'
    response = requests.request(method, url, params=params, json=payload, timeout=(20, 180))
    response.raise_for_status()
    return response.json() if response.content else None


def get_content(saved, path):
    value = content_request(saved, 'GET', path, content=True)
    if value.get('type') != 'file':
        raise IsADirectoryError(path)
    if value.get('format') == 'base64':
        return base64.b64decode(value.get('content', ''))
    return str(value.get('content', '')).encode()


def heartbeat(args):
    started = time.monotonic()
    while time.monotonic() - started < args.max_hours * 3600:
        try:
            saved = store_at(args.config).get(args.session)
            if saved is None or saved.endpoint != args.expect_endpoint:
                return 1
            with api(args.token_file) as client:
                assignments = client.list_assignments()
                if not any(a.endpoint == saved.endpoint for a in assignments):
                    return 0
                client.keep_alive_assignment(saved.endpoint)
        except Exception as error:
            emit({'kind': 'heartbeat_error', **safe_error(error)})
        time.sleep(60)
    return 0


def launch_heartbeat(args, saved, store):
    if saved.keep_alive_pid:
        try:
            os.kill(saved.keep_alive_pid, 0)
            return saved.keep_alive_pid
        except ProcessLookupError:
            pass
    command = [sys.executable, str(Path(__file__).resolve()),
               '--config', str(args.config), '--token-file', str(args.token_file),
               'heartbeat', '--session', saved.name, '--expect-endpoint', saved.endpoint]
    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
    saved.keep_alive_pid = child.pid
    store.add(saved)
    return child.pid


def execute_file(args, saved, store):
    from colab_cli.runtime import ColabRuntime
    source = args.file.read_text()
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    for field in ('token',):
        remember_secret(getattr(saved, field))

    def on_kernel(kernel_id):
        saved.kernel_id = kernel_id
        store.add(saved)

    def on_session(session_id):
        saved.session_id = session_id
        store.add(saved)

    runtime = ColabRuntime(saved.url, saved.token, kernel_id=saved.kernel_id,
                           session_id=saved.session_id, on_kernel_started=on_kernel,
                           on_session_started=on_session)
    prefix = "import sys as _campaign_sys\n_campaign_old_argv = _campaign_sys.argv\n"
    body = (
        f"_campaign_sys.argv = {repr([str(args.file)] + args.script_arg)}\n"
        "try:\n"
        f"    exec(compile({source!r}, {str(args.file)!r}, 'exec'), "
        f"{{'__name__': '__main__', '__file__': {str(args.file)!r}, "
        f"'__source_text__': {source!r}, '__source_sha256__': {source_hash!r}}})\n"
        "finally:\n    _campaign_sys.argv = _campaign_old_argv\n"
    )
    seen = []

    def output_hook(output):
        seen.append(output)
        if output.get('output_type') == 'stream':
            emit({'kind': 'remote_stream', 'stream': output.get('name'), 'text': output.get('text', '')})
        elif output.get('output_type') == 'error':
            emit({'kind': 'remote_error', 'name': output.get('ename'),
                  'message': output.get('evalue'), 'traceback': output.get('traceback', [])})
        elif 'text/plain' in output.get('data', {}):
            emit({'kind': 'remote_display', 'text': output['data']['text/plain']})

    try:
        result = runtime.execute_code(prefix + body, output_hook=output_hook, timeout=args.timeout)
        errors = [o for o in result if o.get('output_type') == 'error']
        for error in errors:
            if error not in seen:
                output_hook(error)
        emit({'kind': 'execution_finished', 'success': not errors, 'endpoint': saved.endpoint,
              'source_sha256': source_hash})
        return 1 if errors else 0
    finally:
        runtime.stop(shutdown_kernel=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--token-file', type=Path, default=DEFAULT_TOKEN)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    for name in ('create', 'exec-file', 'upload', 'download', 'read', 'heartbeat'):
        command = commands.add_parser(name)
        command.add_argument('--session', required=True, type=session_name)
        command.add_argument('--expect-endpoint')
        if name == 'exec-file':
            command.add_argument('--file', type=Path, required=True)
            command.add_argument('--timeout', type=float, default=60)
            command.add_argument('--script-arg', action='append', default=[])
        if name in ('upload', 'download', 'read'):
            command.add_argument('--remote', required=True, type=remote_path)
        if name in ('upload', 'download'):
            command.add_argument('--local', required=True, type=Path)
        if name == 'read':
            command.add_argument('--max-bytes', type=int, default=65536)
        if name == 'heartbeat':
            command.add_argument('--max-hours', type=float, default=24)
    args = parser.parse_args()
    if args.command == 'heartbeat':
        return heartbeat(args)
    with api(args.token_file) as client:
        if args.command == 'list':
            try:
                entries = json.loads(args.config.read_text()) if args.config.exists() else {}
                names = {v['endpoint']: k for k, v in entries.items()}
            except (OSError, ValueError, KeyError):
                names = {}
            emit({'kind': 'allocations', 'assignments': [sanitized_assignment(a, names.get(a.endpoint))
                                                         for a in client.list_assignments()]})
            return 0
        if args.command == 'create':
            from colab_cli.client import Accelerator, Variant
            from colab_cli.state import SessionState
            store = store_at(args.config)
            existing = store.get(args.session)
            assignments = client.list_assignments()
            if existing:
                _, saved = refreshed_session(args, client)
                pid = launch_heartbeat(args, saved, store)
                emit({'kind': 'allocation', 'created': False, 'session': saved.name,
                      'endpoint': saved.endpoint, 'accelerator': 'V5E1', 'heartbeat_pid': pid})
                return 0
            # Avoid duplicate billing and accidental machine switching.
            if assignments:
                raise RuntimeError('Existing allocations present; explicitly resolve them before creation')
            if args.expect_endpoint:
                raise RuntimeError('Cannot create a new machine under an expected endpoint')
            assigned = client.assign(uuid.uuid4(), variant=Variant.TPU, accelerator=Accelerator.V5E1)
            saved = SessionState(name=args.session, token=assigned.runtime_proxy_info.token,
                                 url=assigned.runtime_proxy_info.url, endpoint=assigned.endpoint,
                                 variant='TPU', accelerator='V5E1')
            remember_secret(saved.token)
            store.add(saved)
            pid = launch_heartbeat(args, saved, store)
            emit({'kind': 'allocation', 'created': True, 'session': saved.name,
                  'endpoint': saved.endpoint, 'accelerator': 'V5E1', 'heartbeat_pid': pid})
            return 0
        store, saved = refreshed_session(args, client)
        if args.command == 'exec-file':
            return execute_file(args, saved, store)
        if args.command == 'upload':
            try:
                content_request(saved, 'GET', args.remote)
            except Exception as error:
                if getattr(getattr(error, 'response', None), 'status_code', None) != 404:
                    raise
            else:
                raise FileExistsError('Remote destination already exists')
            data = args.local.read_bytes()
            payload = {'name': PurePosixPath(args.remote).name, 'path': args.remote,
                       'type': 'file', 'format': 'base64',
                       'content': base64.b64encode(data).decode(), 'chunk': 1}
            content_request(saved, 'PUT', args.remote, payload=payload)
            emit({'kind': 'uploaded', 'remote': args.remote, 'bytes': len(data),
                  'sha256': hashlib.sha256(data).hexdigest(), 'endpoint': saved.endpoint})
        elif args.command == 'download':
            if args.local.exists():
                raise FileExistsError('Local destination already exists')
            data = get_content(saved, args.remote)
            with args.local.open('xb') as target:
                target.write(data)
            emit({'kind': 'downloaded', 'local': str(args.local), 'remote': args.remote,
                  'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'endpoint': saved.endpoint})
        elif args.command == 'read':
            data = get_content(saved, args.remote)
            emit({'kind': 'remote_file', 'remote': args.remote, 'endpoint': saved.endpoint,
                  'bytes': len(data), 'truncated': len(data) > args.max_bytes,
                  'text': data[:args.max_bytes].decode('utf-8', errors='replace')})
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        # Never echo exceptions/HTTP response objects, which can contain tokens.
        emit({'kind': 'controller_error', **safe_error(error)})
        raise SystemExit(1)
