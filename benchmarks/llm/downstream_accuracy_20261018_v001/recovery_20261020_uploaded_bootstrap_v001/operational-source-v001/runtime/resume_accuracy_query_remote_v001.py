"""Restore repaired-study checkpoints and invoke the exact frozen query worker.

This operational bootstrap is archived separately from the immutable scientific
bundle. It installs the same pinned environment and never changes model code.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import shutil
import sys
import tarfile

BASE = Path('/content/Strassen_MM_Focus')
Q14_SHA = 'e62cb2ff94ba447e13bb44a6fd802285ac556fcb9a61e8104c0cdd54c516c940'
WORKER = 'runtime/accuracy_query_block_recovery_v001.py'
WORKER_SHA = 'b34d852b4c3f7e3ec7bef799a3dd319262db1735df3bc7ecd6cbc98520ae8c23'
PINS = ['jax==0.11.2', 'jaxlib==0.11.2', 'libtpu==0.0.48', 'numpy==2.2.6',
        'scipy==1.15.3', 'transformers==5.5.0', 'tokenizers==0.22.2',
        'safetensors==0.8.0', 'requests==2.34.2', 'huggingface-hub==1.33.0']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def inspect_seed(path, expected, source_sha):
    if sha(path) != expected:
        raise ValueError('Recovery archive hash mismatch')
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() for m in members):
            raise ValueError('Seed requires unique regular files')
        receipt = json.loads(archive.extractfile('seed-manifest.json').read())
        if receipt['source_sha256'] != source_sha or set(names) != set(receipt['files']) | {'seed-manifest.json'}:
            raise ValueError('Seed membership/source mismatch')
        for name, expected_hash in receipt['files'].items():
            p = PurePosixPath(name)
            allowed = name == 'completed/qwen3_14b.tar.gz' or re.fullmatch(
                r'artifacts/qwen3_32b/(manifest.json|execution-contract.json|batches/(mmlu|ppl|gsm8k)-[0-9]{3}\.json|provenance/original-runtime-admission.json)', name)
            if not allowed or p.is_absolute() or '..' in p.parts:
                raise ValueError('Unexpected checkpoint path')
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != expected_hash:
                raise ValueError('Changed checkpoint content')
        if receipt['files']['completed/qwen3_14b.tar.gz'] != Q14_SHA:
            raise ValueError('Changed inherited Qwen14 archive')
        contract = json.loads(archive.extractfile('artifacts/qwen3_32b/execution-contract.json').read())
        manifest = json.loads(archive.extractfile('artifacts/qwen3_32b/manifest.json').read())
        if contract['bundle_sha256'] != source_sha or digest(contract) != manifest['execution_contract_sha256']:
            raise ValueError('Changed scientific contract')
        for name in names:
            if '/batches/' in name:
                batch = json.loads(archive.extractfile(name).read())
                if batch['execution_contract_sha256'] != digest(contract):
                    raise ValueError('Batch contract mismatch')
        admission = json.loads(archive.extractfile('artifacts/qwen3_32b/provenance/original-runtime-admission.json').read())
        if admission['status'] != 'passed' or not admission['all_logits_finite']:
            raise ValueError('Missing original passed admission')
    return receipt


def prepare(archive, source_sha, seed, seed_sha, run_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,90}', run_id):
        raise ValueError('Invalid run ID')
    if not archive.resolve().is_relative_to(BASE) or not seed.resolve().is_relative_to(BASE) or sha(archive) != source_sha:
        raise ValueError('Source path/hash mismatch')
    receipt = inspect_seed(seed, seed_sha, source_sha)
    with tarfile.open(archive) as handle:
        if hashlib.sha256(handle.extractfile(WORKER).read()).hexdigest() != WORKER_SHA:
            raise ValueError('Frozen repaired worker changed')
    run = BASE / 'runs' / run_id
    run.mkdir(parents=True, exist_ok=False)
    source = run / 'source'
    source.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(source, filter='data')
    with tarfile.open(seed) as handle:
        for name in receipt['files']:
            target = run / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(handle.extractfile(name).read())
    q14 = run / 'completed/qwen3_14b.tar.gz'
    shutil.copy2(q14, run / q14.name)
    with tarfile.open(q14) as handle:
        seen = set()
        for member in handle.getmembers():
            p = PurePosixPath(member.name)
            if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] != 'qwen3_14b' or not (member.isfile() or member.isdir()) or member.name in seen:
                raise ValueError('Invalid inherited model archive')
            seen.add(member.name)
            target = run / 'artifacts' / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(handle.extractfile(member).read())
    save(run / 'qwen3_14b.tar.ready.json', dict(path=str(run / q14.name), bytes=q14.stat().st_size, sha256=Q14_SHA))
    save(run / 'recovery-seed.json', receipt)
    inherited = dict(model='qwen3_14b', archive_sha256=Q14_SHA,
        source_bundle_sha256='162116e4ef16f4f7b24a713211e5d7b70c91b739b42d7a3d88734ae097bc091d',
        note='Completed original execution, copied unchanged; not recomputed')
    save(run / 'inherited-results.json', inherited)
    save(run / 'artifacts/inherited-results.json', inherited)
    environment_run = BASE / 'runs' / (run_id + '-environment')
    environment_run.mkdir(exist_ok=False)
    (environment_run / 'artifacts').mkdir()
    save(run / 'launch.json', dict(source_sha256=source_sha, seed_sha256=seed_sha,
        launch_mode='foreground_notebook', bootstrap='resume_accuracy_query_remote_v001.py',
        bootstrap_sha256=sha(Path(__file__)), frozen_worker=WORKER, frozen_worker_sha256=WORKER_SHA,
        models=['qwen3_32b', 'gemma4_31b'], completed_models=['qwen3_14b'],
        private_environment_run=str(environment_run)))
    save(run / 'status.json', dict(status='running', scope='three-model paired downstream subsets',
        stage='installing_identical_environment', completed_models=['qwen3_14b'],
        test_evaluations_completed=3, test_evaluations_total=9, model='qwen3_32b',
        updated_utc=datetime.now(timezone.utc).isoformat(), release_attempted=False))
    save(run / 'progress.json', dict(model='qwen3_32b', completed_tasks=['mmlu', 'ppl'],
        task='gsm8k', completed_examples=receipt['saved_gsm_pairs'], total_examples=32,
        batch=receipt['saved_gsm_pairs'] // 8 + 1, batches=4,
        latest_event=dict(kind='restoring_environment', utc=datetime.now(timezone.utc).isoformat())))
    return run, environment_run


def install_environment(run, environment_run):
    source = run / 'source'
    sys.path.insert(0, str(source / 'runtime'))
    from parent_exact_remote_v001 import execute
    private = BASE / '.runtime_private' / environment_run.name
    private.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'HF_TOKEN',
        'HUGGING_FACE_HUB_TOKEN', 'LIBTPU_INIT_ARGS', 'XLA_FLAGS', 'JAX_PLATFORMS') and not k.startswith(('PIP_', 'QWEN3_'))}
    env.update(PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1', JAX_PLATFORMS='tpu',
        HF_HOME=str(private / 'huggingface'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1')
    artifacts = environment_run / 'artifacts'
    python = private / 'environment/bin/python'
    if execute([sys.executable, '-m', 'venv', '--without-pip', str(python.parent.parent)], artifacts / 'venv.log', env, 120, source):
        raise RuntimeError('Could not create isolated environment')
    pip = [sys.executable, '-m', 'pip', '--isolated', '--python', str(python)]
    if execute(pip + ['install', '--disable-pip-version-check', '--only-binary=:all:', '--report', str(artifacts / 'installation.json'), *PINS], artifacts / 'install.log', env, 900, source):
        raise RuntimeError('Pinned environment installation failed')
    execute(pip + ['freeze', '--all'], artifacts / 'freeze.txt', env, 60, source)


def execute_frozen(run, environment_run):
    namespace = runpy.run_path(str(run / 'source' / WORKER), run_name='frozen_query_recovery')
    try:
        install_environment(run, environment_run)
        # Exact, byte-verified existing worker: original preflight and driver,
        # fresh admission, strict contract guard, and reuse of saved batches.
        namespace['worker'](run, environment_run)
    except BaseException as error:
        state = dict(status='failed', scope='three-model paired downstream subsets',
            stage='recovery_bootstrap_failed', error_type=type(error).__name__,
            completed_models=['qwen3_14b'], test_evaluations_completed=3, test_evaluations_total=9,
            finished_utc=datetime.now(timezone.utc).isoformat(), release_attempted=False)
        namespace['save'](run / 'status.json', state)
        if (environment_run / 'artifacts').is_dir():
            shutil.copytree(environment_run / 'artifacts', run / 'artifacts/bootstrap-environment', dirs_exist_ok=True)
        ready = namespace['archive'](run / 'artifacts', run / 'all-artifacts.tar.gz')
        namespace['save'](run / 'archive-ready.json', ready)
        raise


def main():
    parser = argparse.ArgumentParser()
    for name in ('archive', 'seed'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('sha256', 'seed-sha256', 'run-id'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run, environment_run = prepare(args.archive, args.sha256, args.seed, args.seed_sha256, args.run_id)
    print(json.dumps(dict(kind='accuracy_query_recovery_launched', run=str(run), pid=os.getpid(),
        source_sha256=args.sha256, seed_sha256=args.seed_sha256, launch_mode='foreground_notebook')), flush=True)
    execute_frozen(run, environment_run)


if __name__ == '__main__':
    main()
