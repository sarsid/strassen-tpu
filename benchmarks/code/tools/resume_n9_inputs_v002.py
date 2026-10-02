"""Resume sealed N9 inputs without downloading or replacing prior artifacts.

Run this file from the canonical project only, between TPU timing phases. Each
child uses run_phase_v004 and commits its immutable evidence before the next
child begins. Original checkpoints, corpus and failed v001 venv are preserved.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = '/content/Strassen_MM_Focus'
MODELS = [('qwen', 'Qwen/Qwen3-0.6B', 'c1899de289a04d12100db370d81485cdf75e47ca'),
          ('mistral', 'mistralai/Mistral-7B-v0.3', 'caa1feb0e54d415e2df31207e5f4e273e33509b1')]
CORPUS_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def utc(): return datetime.now(timezone.utc).isoformat()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def sealed_file(directory, relative):
    """Verify a local artifact against both the execution and preparation seals."""
    directory = Path(directory).resolve()
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError('Missing or escaping sealed input: ' + str(path))
    root_seal = json.loads((directory / 'artifact-manifest.json').read_text())
    entry = root_seal.get(relative)
    actual = {'sha256': digest(path), 'bytes': path.stat().st_size}
    if entry != actual:
        raise ValueError('Execution artifact hash/size mismatch: ' + str(path))
    prep = directory / 'artifacts/preparation-00'
    if path.is_relative_to(prep) and path.name != 'artifact_manifest.json':
        seal_path = prep / 'artifact_manifest.json'
        sealed_file(directory, 'artifacts/preparation-00/artifact_manifest.json')
        inner = json.loads(seal_path.read_text())['files']
        if inner.get(str(path.relative_to(prep))) != actual:
            raise ValueError('Preparation artifact hash/size mismatch: ' + str(path))
    return path


def verify_run(directory, endpoint, expected_completion_sha256=None):
    directory = Path(directory).resolve()
    if directory.parent != ROOT / 'runs':
        raise ValueError('Expected an immediate canonical project run directory')
    path = sealed_file(directory, 'completion.json')
    if expected_completion_sha256 and digest(path) != expected_completion_sha256:
        raise ValueError('Original bundle completion SHA256 changed')
    value = json.loads(path.read_text())
    if value.get('allocation_id') != endpoint:
        raise ValueError('Preparation input belongs to a different allocation')
    if value.get('status') != 'completed' or value.get('remote_may_still_be_running') is not False:
        raise ValueError('Preparation input is incomplete or its remote state is uncertain')
    status = json.loads(sealed_file(directory, 'artifacts/preparation-00/status.json').read_text())
    if status.get('status') != 'completed':
        raise ValueError('Inner preparation did not complete')
    return value


def validate_reuse(bundle_path, corpus_run, endpoint):
    """No network, imports of ML frameworks, or fabricated replacement inputs."""
    original = json.loads(Path(bundle_path).read_text())
    if original.get('cohort') != endpoint:
        raise ValueError('Original input bundle allocation does not match requested cohort')
    steps = {step['run_id']: step for step in original['steps']}
    if len(steps) != len(original['steps']): raise ValueError('Duplicate original step IDs')
    available = {item['model_id']: item for item in original['models']}
    if len(available) != len(original['models']): raise ValueError('Duplicate model IDs')
    models = []
    sources = []
    for slug, model_id, revision in MODELS:
        item = dict(available[model_id])
        if item.get('revision') != revision or item.get('status') != 'checkpoint_ready':
            raise ValueError('Resume requires the frozen completed checkpoint: ' + model_id)
        step = steps[item['download_run_id']]
        directory = ROOT / 'runs' / item['download_run_id']
        if Path(step['run_path']).resolve() != directory or step.get('status') != 'completed':
            raise ValueError('Original checkpoint step path/status mismatch')
        verify_run(directory, endpoint, step['completion_sha256'])
        local = sealed_file(directory, 'artifacts/preparation-00/model_manifest.json')
        if Path(item['local_model_manifest']).resolve() != local:
            raise ValueError('Checkpoint local manifest path does not match original run')
        manifest = json.loads(local.read_text())
        cache = BASE + '/models/' + model_id.replace('/', '--') + '/' + revision
        if (manifest.get('model_id'), manifest.get('revision'), manifest.get('cache_dir')) != (model_id, revision, cache):
            raise ValueError('Checkpoint manifest identity differs from frozen plan')
        if item.get('model_manifest') != cache + '/manifest.json':
            raise ValueError('Checkpoint remote manifest path differs from frozen cache')
        if not manifest.get('files'): raise ValueError('Checkpoint manifest has no files')
        for entry in manifest['files']:
            path = PurePosixPath(entry['path'])
            if path.is_absolute() or '..' in path.parts or not re.fullmatch('[0-9a-f]{64}', entry['sha256']):
                raise ValueError('Invalid checkpoint manifest member')
        item['model_manifest_sha256'] = digest(local)
        models.append(item)
        sources.append({'kind': 'model', 'slug': slug, 'path': str(local), 'sha256': digest(local),
                        'completion_sha256': step['completion_sha256']})
    corpus_run = Path(corpus_run).resolve()
    step = steps[corpus_run.name]
    if step.get('phase') != 'N9-corpus' or Path(step['run_path']).resolve() != corpus_run:
        raise ValueError('Supplied corpus run is not the original completed corpus step')
    verify_run(corpus_run, endpoint, step['completion_sha256'])
    local = sealed_file(corpus_run, 'artifacts/preparation-00/corpus_manifest.json')
    corpus = json.loads(local.read_text())
    expected = ('Salesforce/wikitext', CORPUS_REVISION, 'wikitext-2-raw-v1', 'test',
                BASE + '/models/corpus/' + CORPUS_REVISION)
    if tuple(corpus.get(key) for key in ('dataset_id', 'revision', 'configuration', 'split', 'cache_dir')) != expected:
        raise ValueError('Corpus differs from frozen dataset/revision/split')
    if not corpus.get('files'): raise ValueError('Corpus manifest has no files')
    sources.append({'kind': 'corpus', 'path': str(local), 'sha256': digest(local),
                    'completion_sha256': step['completion_sha256']})
    corpus_info = {'run_id': corpus_run.name, 'local_corpus_manifest': str(local),
                   'corpus_manifest': corpus['cache_dir'] + '/manifest.json',
                   'corpus_manifest_sha256': digest(local), 'revision': CORPUS_REVISION}
    blocked = available.get('google/gemma-3-1b-pt')
    if not blocked or blocked.get('status') != 'blocked_access':
        raise ValueError('Preserve the original explicit Gemma access limitation')
    models.append(dict(blocked))
    return original, models, corpus_info, sources


def validate_tokens(directory, item, corpus):
    manifest_path = sealed_file(directory, 'artifacts/preparation-00/tokens_manifest.json')
    manifest = json.loads(manifest_path.read_text())
    required = {'model_id': item['model_id'], 'model_revision': item['revision'],
                'model_manifest_sha256': item['model_manifest_sha256'],
                'dataset_id': 'Salesforce/wikitext', 'dataset_revision': CORPUS_REVISION,
                'corpus_manifest_sha256': corpus['corpus_manifest_sha256'],
                'token_array_path': 'token_ids.npy', 'shape': [32, 1024],
                'scored_positions': 32736, 'add_special_tokens': False}
    if any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError('Tokenization output provenance or registered token scope mismatch')
    tokens = sealed_file(directory, 'artifacts/preparation-00/token_ids.npy')
    if digest(tokens) != manifest['token_array_sha256']:
        raise ValueError('Token array SHA256 mismatch')
    return manifest_path, manifest


def validate_reference(directory, item, tokens):
    path = sealed_file(directory, 'artifacts/preparation-00/reference_manifest.json')
    manifest = json.loads(path.read_text())
    required = {'model_id': item['model_id'], 'revision': item['revision'],
                'model_manifest_sha256': item['model_manifest_sha256'],
                'tokens_sha256': tokens['token_array_sha256']}
    if any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError('Official reference input provenance mismatch')
    if manifest.get('versions', {}).get('transformers') != '4.56.2':
        raise ValueError('Official reference must use frozen Transformers version')
    for name in ('input_ids', 'logits', 'final_hidden'):
        info = manifest['files'][name]
        if info['path'] != name + '.npy': raise ValueError('Unexpected reference filename')
        raw = sealed_file(directory, 'artifacts/preparation-00/' + info['path'])
        if digest(raw) != info['sha256']: raise ValueError('Official reference array SHA256 mismatch')
    if manifest['files']['input_ids']['shape'] != [1, 64]: raise ValueError('Official reference must use 64 tokens')
    return path, manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('session', 'endpoint', 'expected-identity', 'controller-python', 'output-dir', 'input-bundle', 'corpus-run'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--venv-dir', default=BASE + '/model-tools-v002')
    args = p.parse_args(argv)
    if not (ROOT / '.git').exists():
        raise RuntimeError('Invoke the canonical project tool; archive snapshots may not orchestrate child runs')
    out = Path(args.output_dir).resolve()
    if not out.is_relative_to(ROOT / 'plans'):
        raise ValueError('New continuation evidence must be inside canonical project plans/')
    target = PurePosixPath(args.venv_dir)
    if target.parent != PurePosixPath(BASE) or not re.fullmatch(r'model-tools-v[0-9]{3}', target.name) or target.name == 'model-tools-v001':
        raise ValueError('Use a new versioned project model-tools environment')
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'plan.json', {'created_utc': utc(), 'arguments': vars(args),
        'policy': 'Reuse verified existing checkpoint/corpus bytes; sequential committed steps; never retry old partial venv'})
    records, models, sources = [], [], []
    corpus = None; model_tools = None; error = None; original = None; status = 'failed'

    def run(label, module, extra, budget):
        phase = 'N9-' + label
        command = [sys.executable, str(ROOT / 'tools/run_phase_v004.py'), '--phase', phase,
            '--session', args.session, '--endpoint', args.endpoint, '--expected-identity', args.expected_identity,
            '--controller-python', args.controller_python, '--benchmark-module', module, '--timeout-seconds', str(budget)]
        command += ['--runner-arg=' + value for value in extra]
        save(out / (label + '.command.json'), {'argv': command, 'started_utc': utc()})
        print('Starting ' + phase, flush=True)
        log = out / (label + '.controller.log'); start = time.monotonic()
        with log.open('xb') as stream:
            # run_phase_v004 owns a finite remote budget plus archival grace. Do
            # not kill that manager before it can preserve and commit evidence.
            child = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        paths = [line.split('=', 1)[1] for line in log.read_text(errors='replace').splitlines()
                 if line.startswith('EXECUTION_DIR=')]
        if len(paths) != 1: raise RuntimeError('Child execution path unavailable; stop to avoid overlapping uncertain remote work')
        directory = Path(paths[0]).resolve()
        if directory.parent != ROOT / 'runs': raise RuntimeError('Child run escaped canonical run directory')
        try:
            path = sealed_file(directory, 'completion.json')
            completion = json.loads(path.read_text())
            prep_path = directory / 'artifacts/preparation-00/status.json'
            prep_status = json.loads(sealed_file(directory, 'artifacts/preparation-00/status.json').read_text()).get('status') if prep_path.is_file() else 'missing'
        except (ValueError, KeyError, OSError) as problem:
            raise RuntimeError('Child state cannot be verified; no subsequent preparation will launch') from problem
        record = {'phase': phase, 'run_id': directory.name, 'run_path': str(directory), 'status': completion['status'],
                  'preparation_status': prep_status, 'exit_code': child.returncode,
                  'completion_sha256': digest(path), 'elapsed_seconds': time.monotonic() - start,
                  'remote_may_still_be_running': completion.get('remote_may_still_be_running')}
        records.append(record)
        with (out / 'steps.jsonl').open('a') as stream: stream.write(json.dumps(record, sort_keys=True) + '\n')
        print(json.dumps(record), flush=True)
        if completion.get('remote_may_still_be_running') is not False:
            raise RuntimeError('Remote execution may still be running; no subsequent preparation will launch')
        if module=='strassen_mm.application_tools_v002':
            try:
                summary=json.loads(sealed_file(directory,'artifacts/summary.json').read_text())
            except (ValueError,KeyError,OSError) as problem:
                raise RuntimeError('Installer descendant cleanup evidence missing; stop preparation') from problem
            if summary.get('subprocess_cleanup_all_proven') is not True:
                raise RuntimeError('Installer descendants may still be running; stop preparation')
        if child.returncode or completion['status'] != 'completed' or prep_status != 'completed': return None
        try:
            verify_run(directory, args.endpoint)
        except (ValueError, KeyError, OSError) as problem:
            raise RuntimeError('Completed child cohort or seals could not be verified; stop preparation') from problem
        return directory

    try:
        original, models, corpus, sources = validate_reuse(args.input_bundle, args.corpus_run, args.endpoint)
        save(out / 'original_input_bundle.json', original)
        save(out / 'reuse_provenance.json', {'input_bundle_path': str(Path(args.input_bundle).resolve()),
             'input_bundle_sha256': digest(args.input_bundle), 'verified_sources': sources,
             'remote_byte_verification': 'Frozen tokenize/reference helpers verify every checkpoint/corpus member SHA256; returned manifest hashes must match sealed originals'})
        for index, source in enumerate(sources):
            with (out / ('reused-manifest-%02d.json' % index)).open('xb') as stream:
                stream.write(Path(source['path']).read_bytes())
        tools_run = run('model-tools-v002', 'strassen_mm.application_tools_v002',
                        ['--venv-dir', args.venv_dir], 7200)
        if tools_run is None:
            for item in models:
                if item['status'] == 'checkpoint_ready': item['resume_status'] = 'blocked_model_tools_failed'
        else:
            result = json.loads(sealed_file(tools_run, 'artifacts/preparation-00/result.json').read_text())
            if result.get('venv_dir') != args.venv_dir or result.get('core_jax_environment_modified') is not False:
                raise ValueError('Isolated tool installation did not meet its target/core environment contract')
            model_tools = {'run_id': tools_run.name, 'venv_dir': args.venv_dir,
                           'local_manifest': str(tools_run / 'artifacts/preparation-00/result.json'),
                           'manifest_sha256': digest(tools_run / 'artifacts/preparation-00/result.json')}
            for (slug, _, _), item in zip(MODELS, models):
                try:
                    tokens_run = run('tokenize-' + slug + '-v002', 'strassen_mm.application_prep_v001',
                        ['--action', 'tokenize', '--venv-dir', args.venv_dir, '--model-manifest', item['model_manifest'],
                         '--corpus-manifest', corpus['corpus_manifest']], 3600)
                    if tokens_run is None: item['status'] = 'tokenization_failed'; continue
                    token_path, tokens = validate_tokens(tokens_run, item, corpus)
                    remote = BASE + '/runs/' + tokens_run.name + '/artifacts/preparation-00'
                    item.update(tokens_manifest=remote + '/tokens_manifest.json', local_tokens_manifest=str(token_path),
                                tokenize_run_id=tokens_run.name, tokens_manifest_sha256=digest(token_path), status='tokens_ready')
                    reference_run = run('reference-' + slug + '-v002', 'strassen_mm.application_prep_v001',
                        ['--action', 'reference', '--venv-dir', args.venv_dir, '--model-manifest', item['model_manifest'],
                         '--tokens', remote + '/token_ids.npy'], 7200)
                    if reference_run is None: item['status'] = 'reference_failed'; continue
                    reference_path, _ = validate_reference(reference_run, item, tokens)
                    item.update(status='inputs_ready', reference_run_id=reference_run.name,
                        reference_manifest=BASE + '/runs/' + reference_run.name + '/artifacts/preparation-00/reference_manifest.json',
                        local_reference_manifest=str(reference_path), reference_manifest_sha256=digest(reference_path))
                except ValueError as problem:
                    # Failed integrity never qualifies a model, but a completed
                    # child need not block the other independent checkpoint.
                    item.update(status='input_verification_failed', error={'type': type(problem).__name__, 'message': str(problem)})
        status = 'completed' if all(item['status'] in ('inputs_ready', 'blocked_access') for item in models) else 'partial'
    except BaseException as problem:
        error = {'type': type(problem).__name__, 'message': str(problem)}
    finally:
        bundle = {'created_utc': utc(), 'cohort': args.endpoint, 'models': models, 'steps': records,
                  'reused_sources': sources, 'corpus': corpus, 'model_tools': model_tools,
                  'resumed_from': {'path': str(Path(args.input_bundle).resolve()),
                                  'sha256': digest(args.input_bundle) if Path(args.input_bundle).is_file() else None},
                  'scope': 'Reproducible inputs and independent short reference only; no TPU quality result yet',
                  'status': status, 'error': error}
        save(out / 'input_bundle.json', bundle)
        save(out / 'status.json', {'status': status, 'error': error, 'finished_utc': utc()})
        save(out / 'artifact_manifest.json', {'files': {str(path.relative_to(out)): {'bytes': path.stat().st_size, 'sha256': digest(path)}
              for path in sorted(out.rglob('*')) if path.is_file()}})
        import archive_v002 as archive
        revision = archive.commit('Archive N9 input continuation v002 (' + status + ')')
        print('INPUT_BUNDLE=' + str(out / 'input_bundle.json'), flush=True)
        print(json.dumps({'status': status, 'commit': revision, 'error': error}), flush=True)
    return 0 if status == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
