"""Locally finalize preserved Qwen and repaired Mistral inputs from sealed runs.

Safe to execute from an archived source snapshot: --canonical-root explicitly
identifies the existing project. This tool performs no network access, launches
no subprocesses, imports no ML frameworks and never edits an existing artifact.
The outer archived-execution manager owns execution records and commits.
"""
from __future__ import annotations
import argparse
import ast
import copy
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import traceback

import resume_n9_inputs_v002 as prior

BASE = prior.BASE
VENV = BASE + '/model-tools-v003'
MODEL_HASHES = {
    'Qwen/Qwen3-0.6B': '277723ef331e9ab6c782ca382ae93baa79cb1a3d842b16cffb9e9233116427f7',
    'mistralai/Mistral-7B-v0.3': '79c068765177d1741d4baea0869190169655a00e0f1efdddab87fb35c4f9c1f1',
}
CORPUS_HASH = '1eeef67999c0b02db97648a3e492dfe4d91caa67ac79c23672759b42e4983d31'
VERSIONS = {'torch': '2.8.0+cpu', 'transformers': '4.56.2', 'tokenizers': '0.22.0',
    'huggingface-hub': '0.34.4', 'safetensors': '0.6.2', 'numpy': '2.2.6',
    'pyarrow': '21.0.0', 'sentencepiece': '0.2.1', 'accelerate': '1.10.1', 'protobuf': '6.32.1'}
PINS = {name + '==' + ('2.8.0' if name == 'torch' else version) for name, version in VERSIONS.items()}
IDENTITY_KEYS = ('colab_endpoint', 'hostname', 'boot_id', 'versions')


def load(path): return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition: raise ValueError(message)


def verify_seal(directory, name):
    """Validate every listed member, rejecting missing/escaping/unhashed inputs."""
    directory = Path(directory).resolve()
    path = directory / name
    value = load(path)
    members = value.get('files', value.get('sha256', value))
    require(isinstance(members, dict) and bool(members), 'Empty artifact seal: ' + str(path))
    for relative, expected in members.items():
        member = (directory / relative).resolve()
        require(not Path(relative).is_absolute() and member.is_relative_to(directory)
                and member.is_file() and member != path, 'Unsafe seal member: ' + str(relative))
        actual = prior.digest(member)
        if isinstance(expected, dict):
            require(expected == {'sha256': actual, 'bytes': member.stat().st_size},
                    'Seal hash/size mismatch: ' + str(member))
        else: require(expected == actual, 'Seal hash mismatch: ' + str(member))
    return {'path': str(path), 'sha256': prior.digest(path), 'member_count': len(members)}, members


class Evidence:
    def __init__(self, root, endpoint):
        self.root, self.endpoint = root, endpoint
        self.identity = None
        self.runs = {}

    def run(self, path, *, successful=True, expected_completion=None):
        path = Path(path).resolve()
        require(path.parent == self.root / 'runs', 'Run must be immediately inside canonical runs/: ' + str(path))
        if path.name in self.runs:
            record = self.runs[path.name]
            require(not successful or record['status'] == 'completed', 'Required successful run failed')
            require(expected_completion is None or expected_completion == record['completion_sha256'],
                    'Preserved completion hash differs')
            return record
        outer, members = verify_seal(path, 'artifact-manifest.json')
        required = {'completion.json', 'artifacts/artifact_manifest.json', 'artifacts/environment.json', 'artifacts/summary.json'}
        require(required <= set(members), 'Execution seal omits required evidence')
        completion_path = prior.sealed_file(path, 'completion.json')
        completion = load(completion_path)
        if expected_completion:
            require(prior.digest(completion_path) == expected_completion, 'Preserved completion SHA256 changed')
        require(completion.get('allocation_id') == self.endpoint, 'Run allocation differs from preserved cohort')
        require(completion.get('remote_may_still_be_running') is False, 'Remote work may still be running')
        if successful: require(completion.get('status') == 'completed', 'Required source run did not complete')
        canonical, inner = verify_seal(path / 'artifacts', 'artifact_manifest.json')
        require({'environment.json', 'summary.json'} <= set(inner), 'Canonical seal omits identity or summary')
        environment = load(prior.sealed_file(path, 'artifacts/environment.json'))
        identity = environment.get('identity', {})
        require(all(key in identity for key in IDENTITY_KEYS), 'Host/core identity fields are missing')
        require(identity['colab_endpoint'] == self.endpoint and bool(identity['boot_id']), 'Incomplete host cohort identity')
        current = {key: identity[key] for key in IDENTITY_KEYS}
        if self.identity is None: self.identity = current
        require(current == self.identity, 'Host/boot/core versions differ between input runs')
        summary = load(prior.sealed_file(path, 'artifacts/summary.json'))
        require(summary.get('status') == completion.get('status'), 'Outer and inner status disagree')
        if successful:
            require(summary.get('status') == 'completed', 'Canonical run summary is unsuccessful')
            remote = completion.get('remote_completion', {})
            require(remote.get('status') == 'completed' and remote.get('process_exit_code') == 0,
                    'Successful remote process completion is not proven')
        seals = {'execution': outer, 'canonical': canonical}
        for subdirectory in ('preparation-00', 'diagnostic'):
            child = path / 'artifacts' / subdirectory
            if not child.is_dir(): continue
            relative = subdirectory + '/artifact_manifest.json'
            require(relative in inner and 'artifacts/' + relative in members, 'Child seal is not itself sealed')
            seal, child_members = verify_seal(child, 'artifact_manifest.json')
            require('status.json' in child_members, 'Child status is not sealed')
            if successful: require(load(child / 'status.json').get('status') == 'completed', 'Child preparation did not complete')
            seals[subdirectory] = seal
        record = {'run_id': path.name, 'run_path': str(path), 'status': completion['status'],
                  'phase': summary.get('phase'), 'completion_sha256': prior.digest(completion_path),
                  'remote_may_still_be_running': False, 'seals': seals}
        self.runs[path.name] = record
        return record


def npy(path, expected_shape, expected_dtype, *, payload=False):
    """Read bounded NumPy headers without importing NumPy or unpickling objects."""
    with Path(path).open('rb') as stream:
        require(stream.read(6) == b'\x93NUMPY', 'Invalid NPY magic')
        version = tuple(stream.read(2))
        require(version in ((1, 0), (2, 0)), 'Unsupported NPY format version')
        width = 2 if version == (1, 0) else 4
        length = int.from_bytes(stream.read(width), 'little')
        require(0 < length <= 65536, 'NPY header exceeds fixed bound')
        header = ast.literal_eval(stream.read(length).decode('latin1'))
        require(header.get('shape') == tuple(expected_shape) and header.get('descr') == expected_dtype
                and header.get('fortran_order') is False, 'NPY shape/dtype/layout mismatch: ' + str(path))
        size = math.prod(expected_shape) * 4
        require(Path(path).stat().st_size == stream.tell() + size, 'NPY payload size differs from header')
        if payload:
            require(size <= 32 * 1024 * 4, 'Requested NPY payload exceeds token bound')
            return stream.read()


def validate_model(evidence, item):
    run = evidence.root / 'runs' / item['download_run_id']
    evidence.run(run)
    path = prior.sealed_file(run, 'artifacts/preparation-00/model_manifest.json')
    require(Path(item['local_model_manifest']).resolve() == path, 'Local checkpoint path changed')
    require(prior.digest(path) == item['model_manifest_sha256'] == MODEL_HASHES[item['model_id']],
            'Exact official checkpoint manifest changed')
    manifest = load(path)
    revision = dict((model, revision) for _, model, revision in prior.MODELS)[item['model_id']]
    cache = BASE + '/models/' + item['model_id'].replace('/', '--') + '/' + revision
    require((manifest.get('model_id'), manifest.get('revision'), manifest.get('cache_dir'))
            == (item['model_id'], revision, cache), 'Official model/revision/cache differs')
    require(item['revision'] == revision and item['model_manifest'] == cache + '/manifest.json', 'Model binding changed')
    require(manifest.get('access') == 'available' and bool(manifest.get('files')), 'Checkpoint is unavailable')
    for row in manifest['files']:
        relative = PurePosixPath(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and row['bytes'] > 0
                and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'Malformed official checkpoint member')
        require(row['canonical_url'] == 'https://huggingface.co/' + item['model_id'] + '/resolve/' + revision + '/' + row['path'],
                'Checkpoint source is not the frozen official revision')
        if row.get('official_lfs_sha256'):
            require(row['official_lfs_sha256'] == row['sha256'], 'Official LFS digest differs')
    config = prior.sealed_file(run, 'artifacts/preparation-00/config.json')
    row = next(row for row in manifest['files'] if row['path'] == 'config.json')
    require(prior.digest(config) == row['sha256'] and load(config) == manifest['config'], 'Official config differs')
    return manifest


def validate_pair(evidence, item, corpus, model, token_run, reference_run, venv):
    evidence.run(token_run); evidence.run(reference_run)
    token_path, tokens = prior.validate_tokens(token_run, item, corpus)
    reference_path, reference = prior.validate_reference(reference_run, item, tokens)
    for run, action in ((token_run, 'tokenize'), (reference_run, 'reference')):
        summary = load(prior.sealed_file(run, 'artifacts/summary.json'))
        require(summary.get('action') == action and len(summary.get('records', [])) == 1, 'Wrong preparation action/coverage')
        record = summary['records'][0]
        require(record.get('exit_code') == 0 and record['command'][0] == venv + '/bin/python', 'Wrong tool interpreter or failed action')
        require(record['status'].get('action') == action and record['status'].get('status') == 'completed', 'Preparation action failed')
    for name in ('numpy', 'pyarrow', 'tokenizers', 'transformers'):
        require(tokens['versions'].get(name) == VERSIONS[name], 'Tokenizer package pin differs')
    require(tokens.get('selection') == 'first 32768 contiguous tokens, nonoverlapping 1024-token windows'
            and tokens.get('full_corpus_token_count', 0) >= 32768, 'Token corpus selection differs')
    array = prior.sealed_file(token_run, 'artifacts/preparation-00/token_ids.npy')
    raw = npy(array, (32, 1024), '<i4', payload=True)
    require(prior.hashlib.sha256(raw).hexdigest() == tokens['raw_token_bytes_sha256'], 'Raw token bytes digest differs')
    values = [value[0] for value in struct.iter_unpack('<i', raw)]
    require(min(values) >= 0 and max(values) < model['config']['vocab_size'], 'Token IDs are outside model vocabulary')
    require(len(set(values)) == tokens['unique_token_ids'], 'Unique token count differs')
    require(reference.get('implementation') == 'official Transformers AutoModelForCausalLM, eager attention, CPU BF16, eval(), no KV cache',
            'Reference implementation contract differs')
    expected_class = 'Qwen3ForCausalLM' if item['model_id'].startswith('Qwen/') else 'MistralForCausalLM'
    require(reference.get('model_class') == expected_class, 'Official reference architecture differs')
    for name in ('torch', 'transformers', 'safetensors', 'numpy'):
        require(reference['versions'].get(name) == VERSIONS[name], 'Reference package pin differs')
    shapes = {'input_ids': [1, 64], 'logits': [1, 64, model['config']['vocab_size']],
              'final_hidden': [1, 64, model['config']['hidden_size']]}
    for name, shape in shapes.items():
        require(reference['files'][name]['shape'] == shape, 'Reference output shape metadata differs')
        path = prior.sealed_file(reference_run, 'artifacts/preparation-00/' + name + '.npy')
        data = npy(path, shape, '<i4' if name == 'input_ids' else '<f4', payload=name == 'input_ids')
        if name == 'input_ids': require(data == raw[:64 * 4], 'Reference does not use the first 64 frozen corpus tokens')
    return token_path, tokens, reference_path, reference


def validate_tools(evidence, directory):
    evidence.run(directory)
    summary = load(prior.sealed_file(directory, 'artifacts/summary.json'))
    result_path = prior.sealed_file(directory, 'artifacts/preparation-00/result.json')
    result = load(result_path)
    require(summary.get('subprocess_cleanup_all_proven') is True and summary.get('jax_imported') is False,
            'Installer cleanup or JAX isolation is not proven')
    require(load(prior.sealed_file(directory, 'artifacts/subprocess-cleanup.json')).get('proven_stopped') is True,
            'Installer process group remains uncertain')
    require(result.get('venv_dir') == VENV and result.get('python') == VENV + '/bin/python', 'Wrong repaired model-tools environment')
    require(result.get('core_jax_environment_modified') is False and result.get('global_core_before') == result.get('global_core_after'),
            'Installer modified core packages')
    require(all(result['global_core_after'].get(key) == value for key, value in evidence.identity['versions'].items()),
            'Installer core versions differ from preserved cohort')
    require(result.get('versions') == VERSIONS and set(result.get('pins', [])) == PINS, 'Model-tools pins differ, including protobuf')
    installed = load(prior.sealed_file(directory, 'artifacts/preparation-00/installed.json'))
    require(installed.get('prefix') == VENV and installed.get('base_prefix') != VENV
            and installed.get('jax_visible') is False and installed.get('torch_cuda_version') is None,
            'Installed tools are not isolated CPU-only tools')
    require(installed.get('versions') == VERSIONS, 'Installed module versions differ from result manifest')
    require(all(PurePosixPath(path).is_relative_to(VENV) for path in installed['module_paths'].values()),
            'A tool imported outside its isolated environment')
    return {'run_id': directory.name, 'venv_dir': VENV, 'local_manifest': str(result_path),
            'manifest_sha256': prior.digest(result_path), 'versions': VERSIONS}


def validate_diagnostic(evidence, directory, mistral):
    evidence.run(directory)
    summary = load(prior.sealed_file(directory, 'artifacts/summary.json'))
    status = load(prior.sealed_file(directory, 'artifacts/diagnostic/status.json'))
    require(summary.get('child_diagnostic') == status and summary.get('core_versions_unchanged') is True
            and summary.get('subprocess_cleanup_all_proven') is True, 'Diagnostic isolation/cleanup differs')
    require(status.get('diagnostic_completed') is True and status.get('tokenizer_loaded') is True
            and status.get('installed_distributions_unchanged') is True and status.get('environment_modified') is False
            and status.get('exception_type') is None, 'Tokenizer diagnostic did not load successfully')
    require(load(prior.sealed_file(directory, 'artifacts/subprocess_cleanup.json')).get('proven_stopped') is True,
            'Diagnostic child cleanup is uncertain')
    before = load(prior.sealed_file(directory, 'artifacts/diagnostic/installed_before.json'))
    after = load(prior.sealed_file(directory, 'artifacts/diagnostic/installed_after.json'))
    require(before == after and before.get('prefix') == VENV and before.get('base_prefix') != VENV,
            'Diagnostic used or changed the wrong environment')
    for name, value in before['optional_and_core_versions'].items():
        if name in VERSIONS: require(value == VERSIONS[name], 'Diagnostic package pin differs')
    require(before['optional_and_core_versions'].get('protobuf') == VERSIONS['protobuf'], 'Diagnostic lacks pinned protobuf')
    provenance = load(prior.sealed_file(directory, 'artifacts/diagnostic/input_provenance.json'))
    require(provenance.get('checkpoint_manifest_sha256') == MODEL_HASHES[mistral['model_id']]
            and provenance.get('model_id') == mistral['model_id'] and provenance.get('revision') == mistral['revision'],
            'Diagnostic checkpoint identity differs')
    return {'run_id': directory.name, 'tokenizer_loaded': True,
            'status_sha256': prior.digest(directory / 'artifacts/diagnostic/status.json'),
            'input_provenance_sha256': prior.digest(directory / 'artifacts/diagnostic/input_provenance.json')}


def finalize(args):
    original = load(args.input_bundle)
    plan_seal, members = verify_seal(args.input_bundle.parent, 'artifact_manifest.json')
    require(args.input_bundle.name in members, 'Preserved input bundle is absent from plan seal')
    require(original.get('status') == 'partial' and original.get('error') is None, 'Expected intact partial v002 input bundle')
    items = {item['model_id']: item for item in original['models']}
    require(len(items) == len(original['models']) == 3 and set(items) == set(MODEL_HASHES) | {'google/gemma-3-1b-pt'},
            'Original model inventory differs')
    qwen = items['Qwen/Qwen3-0.6B']; mistral = items['mistralai/Mistral-7B-v0.3']; gemma = items['google/gemma-3-1b-pt']
    require(qwen.get('status') == 'inputs_ready' and gemma.get('status') == 'blocked_access', 'Preserve ready Qwen and blocked Gemma')
    require(mistral.get('status') == 'tokenization_failed', 'Expected the preserved Mistral tokenizer failure')
    evidence = Evidence(args.canonical_root, original['cohort'])
    configs = {item['model_id']: validate_model(evidence, item) for item in (qwen, mistral)}
    for source in original['reused_sources']:
        path = Path(source['path']).resolve(); run = path.parents[2]
        evidence.run(run, expected_completion=source['completion_sha256'])
        require(prior.digest(prior.sealed_file(run, str(path.relative_to(run)))) == source['sha256'], 'Reused source digest changed')
    for step in original['steps']:
        run = args.canonical_root / 'runs' / step['run_id']
        require(Path(step['run_path']).resolve() == run, 'Original step path differs')
        evidence.run(run, successful=step['status'] == 'completed', expected_completion=step['completion_sha256'])
    corpus = original['corpus']; corpus_run = args.canonical_root / 'runs' / corpus['run_id']
    evidence.run(corpus_run)
    corpus_path = prior.sealed_file(corpus_run, 'artifacts/preparation-00/corpus_manifest.json')
    require(Path(corpus['local_corpus_manifest']).resolve() == corpus_path
            and prior.digest(corpus_path) == corpus['corpus_manifest_sha256'] == CORPUS_HASH, 'Exact corpus manifest changed')
    corpus_manifest = load(corpus_path)
    require(tuple(corpus_manifest.get(key) for key in ('dataset_id', 'revision', 'configuration', 'split'))
            == ('Salesforce/wikitext', prior.CORPUS_REVISION, 'wikitext-2-raw-v1', 'test'), 'Corpus scope differs')
    require(corpus['corpus_manifest'] == BASE + '/models/corpus/' + prior.CORPUS_REVISION + '/manifest.json', 'Corpus remote binding differs')
    qtoken, _, qreference, _ = validate_pair(evidence, qwen, corpus, configs[qwen['model_id']],
        args.canonical_root / 'runs' / qwen['tokenize_run_id'], args.canonical_root / 'runs' / qwen['reference_run_id'],
        original['model_tools']['venv_dir'])
    for name, path in (('tokens', qtoken), ('reference', qreference)):
        require(Path(qwen['local_' + name + '_manifest']).resolve() == path
                and qwen[name + '_manifest_sha256'] == prior.digest(path)
                and qwen[name + '_manifest'] == BASE + '/runs/' + path.parents[2].name + '/artifacts/preparation-00/' + path.name,
                'Preserved Qwen binding changed')
    tools = validate_tools(evidence, args.model_tools_run)
    diagnostic = validate_diagnostic(evidence, args.diagnostic_run, mistral)
    token_path, tokens, reference_path, _ = validate_pair(evidence, mistral, corpus, configs[mistral['model_id']],
        args.tokens_run, args.reference_run, VENV)
    updated = copy.deepcopy(mistral)
    updated.update(status='inputs_ready', tokenize_run_id=args.tokens_run.name, reference_run_id=args.reference_run.name,
        local_tokens_manifest=str(token_path), tokens_manifest_sha256=prior.digest(token_path),
        tokens_manifest=BASE + '/runs/' + args.tokens_run.name + '/artifacts/preparation-00/tokens_manifest.json',
        local_reference_manifest=str(reference_path), reference_manifest_sha256=prior.digest(reference_path),
        reference_manifest=BASE + '/runs/' + args.reference_run.name + '/artifacts/preparation-00/reference_manifest.json')
    prior.save(args.output_dir / 'original_input_bundle.json', original)
    provenance = {'source_bundle': {'path': str(args.input_bundle), 'sha256': prior.digest(args.input_bundle), 'plan_seal': plan_seal},
        'cohort_identity': evidence.identity, 'verified_runs': list(evidence.runs.values()),
        'tokenizer_diagnostic': diagnostic, 'model_tools': tools,
        'preservation': 'Qwen and Gemma model entries unchanged; original steps/failures retained; only repaired Mistral bindings added',
        'remote_bytes_scope': 'Local sealed manifests, arrays and command evidence verified; original remote preparation verified checkpoint/corpus members; no remote rehash or model execution here'}
    prior.save(args.output_dir / 'provenance.json', provenance)
    old_ids = {step['run_id'] for step in original['steps']}
    additional = [evidence.runs[path.name] for path in (args.model_tools_run, args.diagnostic_run, args.tokens_run, args.reference_run)]
    require(len({row['run_id'] for row in additional}) == 4 and not old_ids.intersection(row['run_id'] for row in additional),
            'Repair requires four distinct new runs')
    bundle = {**copy.deepcopy(original), 'created_utc': prior.utc(), 'status': 'completed', 'error': None,
        'models': [copy.deepcopy(qwen), updated, copy.deepcopy(gemma)],
        'steps': copy.deepcopy(original['steps']) + additional,
        'preserved_model_tools': copy.deepcopy(original['model_tools']), 'model_tools': tools,
        'tokenizer_diagnostic': diagnostic, 'finalized_from': provenance['source_bundle'],
        'finalization_provenance': 'provenance.json',
        'scope': 'Sealed reproducible Qwen/Mistral inputs and independent official short references; Gemma access blocker retained; no TPU quality or performance result'}
    prior.save(args.output_dir / 'input_bundle.json', bundle)
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('canonical-root', 'input-bundle', 'model-tools-run', 'diagnostic-run', 'tokens-run', 'reference-run', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args(argv)
    for name, value in vars(args).items(): setattr(args, name, value.resolve())
    root = args.canonical_root
    require((root / '.git').exists() and (root / 'runs').is_dir() and (root / 'plans').is_dir(), 'Explicit canonical project root is required')
    require(args.input_bundle.is_relative_to(root / 'plans') and args.output_dir.is_relative_to(root / 'plans')
            and args.output_dir != root / 'plans', 'Bundle and new output must be in canonical plans/')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prior.save(args.output_dir / 'plan.json', {'created_utc': prior.utc(), 'arguments': {key: str(value) for key, value in vars(args).items()},
        'source_file': str(Path(__file__).resolve()), 'source_sha256': prior.digest(__file__),
        'frozen_helper_sha256': prior.digest(prior.__file__), 'policy': 'Local sealed-evidence finalization; no execution, network, old-file edits or commits'})
    status = 'failed'; error = None
    try:
        finalize(args); status = 'completed'
    except Exception as problem:
        error = {'type': type(problem).__name__, 'message': str(problem), 'traceback': traceback.format_exc()}
    finally:
        prior.save(args.output_dir / 'status.json', {'status': status, 'error': error, 'finished_utc': prior.utc()})
        prior.save(args.output_dir / 'artifact_manifest.json', {'files': {str(path.relative_to(args.output_dir)):
            {'bytes': path.stat().st_size, 'sha256': prior.digest(path)} for path in sorted(args.output_dir.rglob('*')) if path.is_file()}})
    print(json.dumps({'status': status, 'input_bundle': str(args.output_dir / 'input_bundle.json') if status == 'completed' else None,
                      'error': error['message'] if error else None}), flush=True)
    return 0 if status == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
