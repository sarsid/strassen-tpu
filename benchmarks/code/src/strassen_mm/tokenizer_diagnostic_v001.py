"""Bounded, local-only Mistral tokenizer diagnostic; no package or model changes.

The standard phase adapter uses the existing isolated model-tools interpreter.
Only tokenizer/config files are hashed, never the checkpoint weight shards.
Tokenization of corpus text and model execution are deliberately absent.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata as md
import io
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

MODEL_ID = 'mistralai/Mistral-7B-v0.3'
REVISION = 'caa1feb0e54d415e2df31207e5f4e273e33509b1'
MANIFEST_SHA256 = '79c068765177d1741d4baea0869190169655a00e0f1efdddab87fb35c4f9c1f1'
CACHE = Path('/content/Strassen_MM_Focus/models/mistralai--Mistral-7B-v0.3') / REVISION
SMALL_FILES = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
               'tokenizer.model', 'tokenizer.model.v3', 'vocab.json', 'merges.txt', 'added_tokens.json'}
OPTIONAL_PACKAGES = ('protobuf', 'tiktoken', 'mistral-common', 'sentencepiece', 'transformers', 'tokenizers',
                     'huggingface-hub', 'torch', 'numpy', 'safetensors')


def sanitize(value):
    text = str(value)
    text = re.sub(r'(https?://)[^/@\s]+:[^/@\s]+@', r'\1[REDACTED]@', text)
    text = re.sub(r'(?i)((?:authorization|access_token|refresh_token|hf_token|client_secret)\s*[=:]\s*)[^\s,;]+',
                  r'\1[REDACTED]', text)
    return re.sub(r'\bhf_[A-Za-z0-9]{12,}\b', '[REDACTED_HF_TOKEN]', text)


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def installed():
    optional = {}
    for name in OPTIONAL_PACKAGES:
        try: optional[name] = md.version(name)
        except md.PackageNotFoundError: optional[name] = None
    all_packages = sorted([{'name': value.metadata.get('Name') or 'unknown', 'version': value.version}
                           for value in md.distributions()], key=lambda value: value['name'].lower())
    return {'optional_and_core_versions': optional, 'all_distributions': all_packages,
            'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix, 'base_prefix': sys.base_prefix}


def child_main(argv):
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-manifest', type=Path, required=True)
    parser.add_argument('--model-manifest-sha256', required=True)
    parser.add_argument('--venv-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output_dir; out.mkdir(parents=True, exist_ok=False)
    before = installed(); save(out / 'installed_before.json', before)
    stdout, stderr = io.StringIO(), io.StringIO()
    stage = 'input_verification'; verified = False; error = None; loaded = False; tokenizer_info = None
    try:
        if Path(sys.prefix).resolve() != args.venv_dir.resolve() or sys.prefix == sys.base_prefix:
            raise RuntimeError('Diagnostic did not start in the requested isolated model-tools environment')
        if args.model_manifest.resolve() != CACHE / 'manifest.json':
            raise ValueError('Use the exact frozen Mistral checkpoint manifest')
        if args.model_manifest_sha256 != MANIFEST_SHA256 or digest(args.model_manifest) != MANIFEST_SHA256:
            raise ValueError('Checkpoint manifest SHA256 differs from the sealed original input')
        manifest = json.loads(args.model_manifest.read_text())
        if (manifest.get('model_id'), manifest.get('revision'), manifest.get('cache_dir')) != (MODEL_ID, REVISION, str(CACHE)):
            raise ValueError('Checkpoint identity differs from the frozen Mistral input')
        checked = []
        for item in manifest['files']:
            if item['path'] not in SMALL_FILES: continue
            path = CACHE / item['path']
            if not path.resolve().is_relative_to(CACHE) or not path.is_file():
                raise ValueError('Missing or escaping tokenizer/config file')
            if path.stat().st_size != item['bytes'] or path.stat().st_size > 64 * 1024 ** 2:
                raise ValueError('Tokenizer/config file size differs from manifest or exceeds diagnostic bound')
            sha = digest(path)
            if sha != item['sha256']: raise ValueError('Tokenizer/config file SHA256 mismatch: ' + item['path'])
            checked.append({'path': item['path'], 'bytes': item['bytes'], 'sha256': sha})
        names = {item['path'] for item in checked}
        if not {'config.json', 'tokenizer_config.json'} <= names:
            raise ValueError('Required model/tokenizer configuration was not verified')
        tokenizer_config = json.loads((CACHE / 'tokenizer_config.json').read_text())
        save(out / 'input_provenance.json', {'model_id': MODEL_ID, 'revision': REVISION,
            'checkpoint_manifest_path': str(args.model_manifest), 'checkpoint_manifest_sha256': MANIFEST_SHA256,
            'checked_small_files': checked, 'weight_shards_read': False,
            'tokenizer_config_summary': {key: tokenizer_config.get(key) for key in
                ('tokenizer_class', 'legacy', 'from_slow', 'add_bos_token', 'add_eos_token', 'model_max_length')},
            'load_arguments': {'local_files_only': True, 'trust_remote_code': False, 'use_fast': True},
            'scope': 'Load tokenizer only; no corpus input, encoded token output or model inference'})
        verified = True
        with redirect_stdout(stdout), redirect_stderr(stderr):
            stage = 'import_AutoTokenizer'
            from transformers import AutoTokenizer
            stage = 'local_AutoTokenizer_from_pretrained'
            tokenizer = AutoTokenizer.from_pretrained(CACHE, local_files_only=True, trust_remote_code=False, use_fast=True)
            tokenizer_info = {'class': type(tokenizer).__name__, 'is_fast': bool(tokenizer.is_fast)}
            loaded = True
    except BaseException as problem:
        error = {'type': type(problem).__name__, 'message': sanitize(problem), 'stage': stage,
                 'traceback': sanitize(traceback.format_exc())}
        save(out / 'exception.json', error)
        with (out / 'exception.traceback.txt').open('x') as stream: stream.write(error['traceback'])
        # Corpus text and tokenizer outputs are never generated by this check.
        for label, buffer in (('stdout', stdout), ('stderr', stderr)):
            with (out / ('load.' + label + '.log')).open('x') as stream: stream.write(sanitize(buffer.getvalue()))
    finally:
        after = installed(); save(out / 'installed_after.json', after)
        unchanged = before == after
        summary = {'diagnostic_completed': verified and unchanged, 'tokenizer_loaded': loaded,
                   'stage': stage, 'exception_type': error['type'] if error else None,
                   'installed_distributions_unchanged': unchanged, 'tokenizer': tokenizer_info,
                   'status': 'completed' if verified and unchanged else 'failed',
                   'environment_modified': False, 'scope': 'Local-only tokenizer diagnostic; no quality/performance result'}
        save(out / 'status.json', summary)
        save(out / 'artifact_manifest.json', {'sha256': {str(path.relative_to(out)): digest(path)
            for path in sorted(out.rglob('*')) if path.is_file()}})
        print(json.dumps({'diagnostic_completed': summary['diagnostic_completed'],
                         'tokenizer_loaded': loaded, 'exception_type': summary['exception_type']}), flush=True)
    return 0 if summary['diagnostic_completed'] else 1


def main(argv=None):
    from . import benchmark_v001 as base
    from .application_tools_v002 import cleanup_group
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'phase', 'output-dir', 'expected-identity', 'allocation-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-wall-seconds', type=float, default=300)
    parser.add_argument('--venv-dir', type=Path, default=Path('/content/Strassen_MM_Focus/model-tools-v002'))
    parser.add_argument('--model-manifest', type=Path, required=True)
    parser.add_argument('--model-manifest-sha256', required=True)
    args = parser.parse_args(argv); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=False)
    journal = base.Journal(out, args.phase); started = time.monotonic(); status = 'failed'; error = None
    child = None; child_status = None; current = None
    cleanup = {'proven_stopped': True, 'reason': 'Diagnostic child not launched'}
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(signum, frame): raise InterruptedError('Diagnostic received ' + signal.Signals(signum).name)
    for sig in previous: signal.signal(sig, interrupted)
    try:
        if 'jax' in sys.modules: raise RuntimeError('Tokenizer diagnostic must not import JAX')
        if args.max_wall_seconds < 30: raise ValueError('Diagnostic requires at least 30 seconds')
        expected = json.loads(Path(args.expected_identity).read_text()); expected = expected.get('identity', expected)
        current = {'colab_endpoint': args.allocation_id, 'hostname': socket.gethostname(),
                   'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                   'versions': {name: md.version(name) for name in expected['versions']}}
        base.exclusive_json(out / 'environment.json', {'identity': current, 'qualification': 'Host/core identity only; no JAX import'})
        for key in current:
            if current[key] != expected[key]: raise ValueError('Diagnostic cohort mismatch: ' + key)
        command = [str(args.venv_dir / 'bin/python'), '-I', '-B', str(Path(__file__).resolve()), '--child',
                   '--output-dir', str(out / 'diagnostic'), '--model-manifest', str(args.model_manifest),
                   '--model-manifest-sha256', args.model_manifest_sha256, '--venv-dir', str(args.venv_dir)]
        base.exclusive_json(out / 'command.json', {'argv': command})
        env = {key: value for key, value in os.environ.items() if key not in
               ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACEHUB_API_TOKEN')}
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (out / 'child.stdout.log').open('xb') as stdout, (out / 'child.stderr.log').open('xb') as stderr:
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set(previous))
            try:
                child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
                base.exclusive_json(out / 'child_process.json', {'pid': child.pid, 'pgid': child.pid})
            finally: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            child.wait(timeout=min(180, args.max_wall_seconds - 15))
        child_status = json.loads((out / 'diagnostic/status.json').read_text())
        journal.emit('tokenizer_diagnostic', **child_status)
        if child.returncode or not child_status['diagnostic_completed']:
            raise RuntimeError('Diagnostic could not verify inputs or complete; inspect preserved exception')
        status = 'completed'
    except BaseException as problem:
        error = {'type': type(problem).__name__, 'message': sanitize(problem), 'traceback': sanitize(traceback.format_exc())}
        journal.emit('run_error', **error)
    finally:
        for sig in previous: signal.signal(sig, signal.SIG_IGN)
        if child is not None: cleanup = cleanup_group(child)
        base.exclusive_json(out / 'subprocess_cleanup.json', cleanup)
        core_unchanged = current is None or current['versions'] == {name: md.version(name) for name in current['versions']}
        if not cleanup['proven_stopped'] or not core_unchanged:
            status = 'failed'
            error = {'type': 'DiagnosticIsolationFailure', 'message': 'Child cleanup or unchanged core packages not proven'}
        summary = {'phase': args.phase, 'status': status, 'completed': status == 'completed',
                   'child_diagnostic': child_status, 'error': error, 'elapsed_seconds': time.monotonic() - started,
                   'subprocess_cleanup_all_proven': cleanup['proven_stopped'], 'core_versions_unchanged': core_unchanged,
                   'scope': 'Diagnostic success means error evidence captured; tokenizer_loaded is a separate outcome'}
        journal.emit('run_complete', **summary); journal.close(); base.exclusive_json(out / 'summary.json', summary)
        base.exclusive_json(out / 'artifact_manifest.json', {'sha256': {str(path.relative_to(out)): digest(path)
            for path in sorted(out.rglob('*')) if path.is_file()}, 'sealed_utc': base.utc_now()})
        print(json.dumps({'phase': args.phase, 'status': status, 'tokenizer_loaded': (child_status or {}).get('tokenizer_loaded')}), flush=True)
        for sig, handler in previous.items(): signal.signal(sig, handler)
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(child_main(sys.argv[2:]) if len(sys.argv) > 1 and sys.argv[1] == '--child' else main())
