"""Train-only loop-stable Native projection candidate; original experiment unchanged."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--diagnostic-id', required=True)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--prefix-length', type=int, choices=(64, 129, 257, 1024), default=64)
    args = parser.parse_args()
    base = Path('/content/Strassen_MM_Focus')
    if not args.run.resolve().is_relative_to(base / 'runs'):
        raise ValueError('Expected owned remote run')
    if not args.diagnostic_id.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Invalid diagnostic identifier')
    source = args.run / 'source'
    output = args.run / 'diagnostics' / args.diagnostic_id
    private = base / '.runtime_private' / args.run.name
    if not args.child:
        import fcntl
        state = json.loads((args.run / 'status.json').read_text())
        if state['status'] != 'failed' or state['model'] != 'qwen3_32b':
            raise ValueError('Diagnostic requires ended Qwen32 failure')
        lock = (base / 'active.lock').open('ab')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        script = output / 'probe.py'
        text = globals().get('__source_text__') or Path(__file__).read_text()
        script.write_text(text)
        contract = json.loads((args.run / 'artifacts/qwen3_32b/execution-contract.json').read_text())
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'LIBTPU_INIT_ARGS', 'XLA_FLAGS', 'JAX_PLATFORMS') and not k.startswith(('PIP_', 'QWEN3_'))}
        env.update(PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1', HF_HOME=str(private / 'huggingface'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', STRASSEN_SOURCE_SHA256=contract['bundle_sha256'])
        env.update({k: v for k, v in contract['environment'].items() if v is not None})
        command = [str(private / 'environment/bin/python'), str(script), '--run', str(args.run), '--diagnostic-id', args.diagnostic_id, '--prefix-length', str(args.prefix_length), '--child']
        save(output / 'launch.json', dict(started_utc=datetime.now(timezone.utc).isoformat(), scope='train-only diagnosis; no test evaluation', source_sha256=hashlib.sha256(text.encode()).hexdigest(), command=command, original_contract_sha256=hashlib.sha256((args.run / 'artifacts/qwen3_32b/execution-contract.json').read_bytes()).hexdigest()))
        sys.path.insert(0, str(source / 'runtime'))
        from parent_exact_remote_v001 import execute
        code = execute(command, output / 'execution.log', env, 1200, source)
        save(output / 'completion.json', dict(returncode=code, finished_utc=datetime.now(timezone.utc).isoformat(), tpu_released=False))
        print(json.dumps(dict(kind='admission_probe_finished', returncode=code, output=str(output))), flush=True)
        lock.close()
        if code:
            raise RuntimeError('Train-only diagnostic failed; retained TPU')
        return

    import numpy as np
    import jax
    import jax.numpy as jnp
    sys.path.insert(0, str(source / 'tools'))
    from accuracy_engine_v003 import Engine
    from accuracy_decoder_v001 import make_layer
    from accuracy_mm_v001 import rounded
    if jax.default_backend() != 'tpu' or len(jax.devices()) != 1:
        raise RuntimeError('Expected retained single TPU')
    def emit(value):
        row = dict(utc=datetime.now(timezone.utc).isoformat(), **value)
        with (output / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(row, allow_nan=False), flush=True)
    def stats(a, b):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        return dict(relative_l2=float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-30)), max_abs=float(np.abs(a-b).max()), equal=bool(np.array_equal(a, b)))
    pilot = json.loads((source / 'plans/llm_accuracy_20261018_v001/pilot_qwen3_32b_v001.json').read_text())
    if pilot['scope'] != 'train_generation_timing_pilot':
        raise ValueError('Only frozen training pilot inputs may be used')
    engine = Engine('qwen3_32b', private / 'weights-qwen3_32b', emit)
    def fixed_rows(lhs, weight):
        shape = lhs.shape
        flat = lhs.reshape(-1, shape[-1]); m = flat.shape[0]
        padded_m = max(256, 128 * ((m + 127) // 128))
        flat = rounded(jnp.pad(flat, ((0, padded_m - m), (0, 0))))
        weight = rounded(weight)
        blocks = flat.reshape(-1, 128, flat.shape[-1])
        result = jax.lax.map(lambda block: rounded(jnp.matmul(block, weight,
            precision=jax.lax.Precision.DEFAULT, preferred_element_type=jnp.float32)), blocks)
        return result.reshape(-1, weight.shape[-1])[:m].reshape(*shape[:-1], weight.shape[-1])
    engine.functions['native', 'qwen'] = make_layer(engine.config, 0, fixed_rows)
    emit(dict(kind='diagnostic_candidate', native_row_block=128, minimum_native_blocks=2, precision='DEFAULT', dtype='bfloat16', original_experiment_unchanged=True))
    if pilot['model'] != engine.repo or pilot['revision'] != engine.revision:
        raise ValueError('Changed train checkpoint')
    engine.prepare()
    length = args.prefix_length
    engine.reset(8, 128 * ((2 * length + 127) // 128))
    tokens = np.asarray([row[:length] for row in pilot['token_ids']], np.int32)
    if tokens.shape != (8, length):
        raise ValueError('Expected eight complete training prefixes')
    prefix = tokens.copy(); prefix[:, length-1] = 0
    padded = np.zeros((8, length), np.int32); padded[:, 0] = tokens[:, length-1]
    hidden = {name: jnp.asarray(np.asarray(engine.embedding[ids])) for name, ids in dict(full=tokens, prefix=prefix, cached=tokens[:, length-1:length], padded=padded).items()}
    pos_full = jnp.asarray(np.tile(np.arange(length, dtype=np.int32), (8, 1)))
    pos_one = jnp.full((8, 1), length-1, jnp.int32)
    pos_padded = jnp.asarray(np.tile(np.arange(length-1, 2*length-1, dtype=np.int32), (8, 1)))
    lengths_full = jnp.full(8, length, jnp.int32)
    lengths_prefix = jnp.full(8, length-1, jnp.int32)
    traces = []
    for index, host_params in enumerate(engine.weights):
        params = {key: jnp.asarray(value) for key, value in host_params.items()}
        ck, cv = (jnp.asarray(x) for x in engine.caches['native'][index])
        def run(name, k, v, pos, lengths):
            values = (hidden[name], params, k, v, pos, lengths)
            result = engine.executable('native', values, index)(*values)
            jax.block_until_ready(result)
            return result
        h_prefix, k_prefix, v_prefix = run('prefix', ck, cv, pos_full, lengths_prefix)
        h_full, _, _ = run('full', ck, cv, pos_full, lengths_full)
        h_cached, _, _ = run('cached', k_prefix, v_prefix, pos_one, lengths_full)
        h_padded, _, _ = run('padded', k_prefix, v_prefix, pos_padded, lengths_full)
        row = dict(kind='layer_comparison', layer=index+1,
            cached_vs_full=stats(h_cached[:, 0], h_full[:, length-1]),
            padded_vs_full=stats(h_padded[:, 0], h_full[:, length-1]),
            prefix_causal_max_abs=float(jnp.max(jnp.abs(h_prefix[:, :length-1].astype(jnp.float32)-h_full[:, :length-1].astype(jnp.float32)))))
        traces.append(row); emit(row)
        hidden = dict(prefix=h_prefix, full=h_full, cached=h_cached, padded=h_padded)
        del params, ck, cv, k_prefix, v_prefix
    logits = {name: np.asarray(engine.head_logits(hidden[name][:, position]), np.float32) for name, position in [('full', length-1), ('cached', 0), ('padded', 0)]}
    summary = dict(scope='train-only loop-stable Native candidate; no downstream scores', model=engine.repo, checkpoint_revision=engine.revision,
        prefix_length=length, native_row_block=128, minimum_native_blocks=2,
        train_example_ids=pilot['example_ids'], train_input_sha256=pilot['inputs_sha256'],
        cached_vs_full=stats(logits['cached'], logits['full']), padded_vs_full=stats(logits['padded'], logits['full']),
        greedy_agreement={name: float(np.mean(np.argmax(logits[name], axis=-1)==np.argmax(logits['full'], axis=-1))) for name in ('cached', 'padded')},
        original_admission_limit=0.005, experiment_unchanged=True, layers=traces)
    save(output / 'summary.json', summary)
    emit(dict(kind='probe_complete', **{k:v for k,v in summary.items() if k != 'layers'}))


if __name__ == '__main__':
    main()
