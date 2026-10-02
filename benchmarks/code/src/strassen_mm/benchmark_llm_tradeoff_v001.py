"""Held-out quality and resident full-model forwards with exact larger weights.

Separate tune/quality/resident processes preserve evidence and isolate memory.
No checkpoint reads, host layout conversion or H2D weight transfer in resident
timers. This is a prompt-forward/scoring benchmark, not a serving engine.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from . import benchmark_large_models_v001 as prior
from . import llm_error_metrics_v001 as errors
base = prior.base


def scalar_tree(value):
    return {k: v.item() for k, v in jax.device_get(value).items()}


def verify_environment(args, cfg):
    from .kernels_v002 import enable_qualified_mosaic_v7_compat
    env = base.capture_environment(args, cfg, enable_qualified_mosaic_v7_compat())
    name = env['identity']['device_kind'].lower().replace(' ', '')
    target = cfg['device']['target']
    allowed = {'v5e': ('v5e', 'v5lite'), 'v6e': ('v6e', 'v6lite')}
    if (target not in allowed or not any(s in name for s in allowed[target])
        or env['device_count'] != 1 or env['local_device_count'] != 1 or env['process_count'] != 1
        or env['backend'] != 'tpu'):
        raise RuntimeError('Expected exactly one '+target+' TPU; no hardware substitution')
    env['qualified_target'] = target
    expected = json.loads(args.expected_identity.read_text()); expected = expected.get('identity', expected)
    for key in ('colab_endpoint','hostname','boot_id','versions','devices'):
        if expected[key] != env['identity'][key]:
            raise RuntimeError('Allocation identity mismatch: '+key)
    base.exclusive_json(args.output_dir/'environment.json', env)
    return env


class Study(prior.Study):
    def load(self, bundle):
        self.model_key = bundle['model_key']
        for name, expected in bundle['input_sha256'].items():
            if base.digest_file(Path(bundle[name])) != expected:
                raise ValueError('Input manifest changed: '+name)
        manifest = json.loads(Path(bundle['model_manifest']).read_text())
        mod = gm if manifest['config']['model_type'] in ('gemma3','gemma3_text') else qm
        cp = mod.Checkpoint(bundle['model_manifest'])
        if (cp.manifest['model_id'], cp.manifest['revision']) != (bundle['model_id'], bundle['revision']):
            raise ValueError('Checkpoint identity changed')
        tm_path = Path(bundle['tokens_manifest']); tm = json.loads(tm_path.read_text())
        tokenpath = tm_path.parent/tm['token_array_path']
        if base.digest_file(tokenpath) != tm['token_array_sha256']:
            raise ValueError('Token payload changed')
        ids = np.load(tokenpath, allow_pickle=False)
        if list(ids.shape) != self.cfg['token_shape'] or ids.dtype != np.dtype('<i4'):
            raise ValueError('Unexpected token shape or dtype')
        rp = Path(bundle['reference_manifest']); self.reference_dir = rp.parent
        ref = json.loads(rp.read_text())
        if (ref['tokens_sha256'] != base.digest_file(tokenpath)
            or ref['model_manifest_sha256'] != bundle['input_sha256']['model_manifest']
            or tm['model_manifest_sha256'] != bundle['input_sha256']['model_manifest']):
            raise ValueError('Oracle/model/token provenance differs')
        base.exclusive_json(self.args.output_dir/'input_provenance.json', bundle)
        return cp, ids, ref

    def tune_model(self, cp, ids, ref, bundle):
        default = next(a for a in grid.candidates(self.cfg) if a['candidate_id'] == 'native_default')
        native = {k: default for k in ('gateup','down')}
        qualification = self.qualify(cp, ids, ref, native)
        operands, weights = self.real_operands(cp, ids, native)
        policies = self.tune(cp, operands[:4], weights)
        result = dict(model_id=bundle['model_id'], revision=bundle['revision'],
            model_key=self.model_key, policies=policies, qualification=qualification,
            input_sha256=bundle['input_sha256'], allocation_id=self.args.allocation_id,
            campaign_sha256=base.digest_file(self.args.campaign),
            tuning_windows=list(range(4)), quality_windows=self.cfg['heldout_windows'],
            policy_selection='M2048 screen winner frozen before three fresh-window confirmations')
        base.exclusive_json(self.args.output_dir/'policy_bundle.json', result)
        return result

    def selected(self, bundle):
        selection = json.loads(self.args.selection.read_text())
        for key in ('model_id','revision','model_key','input_sha256'):
            if selection[key] != bundle[key]:
                raise ValueError('Selection provenance mismatch: '+key)
        if (selection['allocation_id'] != self.args.allocation_id
            or selection['campaign_sha256'] != base.digest_file(self.args.campaign)):
            raise ValueError('Selection must come from this hardware allocation and frozen campaign')
        if not selection['qualification']['passed']:
            raise RuntimeError('Native failed independent official qualification')
        if set(selection['tuning_windows']) & set(self.cfg['heldout_windows']):
            raise ValueError('Tuning and evaluation windows overlap')
        base.exclusive_json(self.args.output_dir/'selection_provenance.json', dict(
            source=str(self.args.selection), sha256=base.digest_file(self.args.selection), selection=selection))
        return selection['policies']

    def functions(self, cp, length, policies, index):
        result = {arm: self.layer(cp, length, policy, index) for arm, policy in policies.items()}
        result['native_full_jit'] = self.full_jit(cp, length, index)
        return result

    def quality(self, cp, ids, policies):
        windows = self.cfg['heldout_windows']; length = ids.shape[1]
        arms = list(policies) + ['native_full_jit']
        initial = [jax.device_put(cp.embeddings(ids[w])) for w in windows]
        states = {arm: list(initial) for arm in arms}; del initial
        hm = errors.make_hidden_comparator(jax, jnp)
        layer_errors, resident = [], {arm: [] for arm in arms}
        rng = random.Random(self.cfg['order_seed'])
        for index in range(cp.config['num_hidden_layers']):
            self.check()
            weights = jax.device_put(cp.layer(index)); jax.block_until_ready(weights)
            fns = self.functions(cp, length, policies, index)
            incoming = states['native_default'][0]
            reference = fns['native_default'](incoming, weights); reference.block_until_ready()
            for arm, fn in fns.items():
                value = fn(incoming, weights); value.block_until_ready()
                row = dict(layer=index, arm_id=arm, scope='local_same_native_input',
                           metrics=scalar_tree(hm(reference, value)))
                layer_errors.append(row); self.emit('layer_error', **row)
                del value
            order = list(arms); rng.shuffle(order)
            for repeat in range(self.cfg['resident_timing']['repeats']):
                for arm in order[repeat % len(order):] + order[:repeat % len(order)]:
                    start = time.perf_counter_ns(); value = fns[arm](incoming, weights); value.block_until_ready()
                    elapsed = (time.perf_counter_ns()-start)/1e6; resident[arm].append(elapsed)
                    self.emit('resident_layer_sample', layer=index, arm_id=arm, repeat=repeat, elapsed_ms=elapsed)
                    del value
            del reference, incoming
            for arm, fn in fns.items():
                for wi in range(len(windows)):
                    states[arm][wi] = fn(states[arm][wi], weights); states[arm][wi].block_until_ready()
            for arm in arms:
                row = dict(layer=index, arm_id=arm, scope='accumulated_own_forward_state',
                           metrics=scalar_tree(hm(states['native_default'][0], states[arm][0])))
                layer_errors.append(row); self.emit('layer_error', **row)
            self.emit('quality_layer_complete', model_id=cp.manifest['model_id'], layer=index,
                      completed=index+1, total=cp.config['num_hidden_layers'])
            del weights, fns; gc.collect()
        norm = jax.device_put(np.array(cp.tensor('model.norm.weight'), copy=True)); head = jax.device_put(cp.head())
        compare = errors.make_comparator(jax, jnp)
        pieces = {arm: {} for arm in arms}
        for wi, window in enumerate(windows):
            for lo in range(0, length-1, 128):
                self.check(); hi = min(lo+128, length-1)
                reference = composed.head(cp, states['native_default'][wi][lo:hi], norm, head)
                targets = jax.device_put(ids[window, lo+1:hi+1])
                for arm in arms:
                    value = reference if arm == 'native_default' else composed.head(cp, states[arm][wi][lo:hi], norm, head)
                    raw = jax.device_get(compare(reference, value, targets))
                    raw['window'] = np.full(hi-lo, window, dtype=np.int32)
                    raw['position'] = np.arange(lo, hi, dtype=np.int32)
                    for key, val in raw.items():
                        pieces[arm].setdefault(key, []).append(np.asarray(val))
                    del value, raw
                del reference, targets
            self.emit('quality_window_complete', window=window, completed=wi+1, total=len(windows))
        quality = {}
        for arm in arms:
            data = {key: np.concatenate(value) for key, value in pieces[arm].items()}
            np.savez_compressed(self.args.output_dir/(arm+'_token_metrics.npz'), **data)
            quality[arm] = errors.summarize(data, cp.config['vocab_size'])
            if quality[arm]['positions'] != len(windows)*(length-1):
                raise ValueError('Scored token count mismatch')
            self.emit('quality_summary', arm_id=arm, metrics=quality[arm])
        base.exclusive_json(self.args.output_dir/'quality.json', quality)
        base.exclusive_json(self.args.output_dir/'layer_errors.json', layer_errors)
        return dict(quality=quality, resident_layer_timings={a: prior.timing_ci(v, self.cfg['order_seed']) for a,v in resident.items()},
                    quality_scope='Held-out text windows disjoint from kernel tuning; teacher-forced actual next-token targets',
                    layer_error_scope='Local and accumulated errors on first held-out window; full tensor reductions',
                    fully_resident_model_timed=False)

    def load_resident(self, cp):
        stats = jax.devices()[0].memory_stats() or {}
        limit = stats.get('bytes_limit')
        if not limit:
            raise RuntimeError('Device memory limit unavailable; cannot certify resident loading budget')
        widths = {'BF16': 2, 'F32': 4}
        layer_bytes = sum(math.prod(shape)*widths[dtype] for name, (_,_,shape,dtype) in cp.locations.items()
                          if name.startswith('model.layers.') or name == 'model.norm.weight')
        embed_shape = cp.locations['model.embed_tokens.weight'][2]
        extra = 2 * math.prod(embed_shape) * 2  # resident embedding table and separately laid-out head
        estimate = layer_bytes + extra
        reserve = self.cfg['resident_memory_reserve_bytes']
        budget = dict(estimated_weight_bytes=estimate, device_memory=stats, reserve_bytes=reserve,
                      allowed=estimate+reserve+stats.get('bytes_in_use', 0) <= limit)
        base.exclusive_json(self.args.output_dir/'resident_memory_preflight.json', budget)
        self.emit('resident_memory_preflight', **budget)
        if not budget['allowed']:
            raise MemoryError('Full resident model does not fit with frozen workspace reserve; no streamed substitution')
        layers, components = [], []
        for index in range(cp.config['num_hidden_layers']):
            self.check(); start = time.perf_counter_ns(); host = cp.layer(index)
            laid_out = time.perf_counter_ns(); value = jax.device_put(host); jax.block_until_ready(value)
            transferred = time.perf_counter_ns(); layers.append(value); del host
            row = dict(layer=index, host_read_layout_ms=(laid_out-start)/1e6,
                       h2d_ms=(transferred-laid_out)/1e6)
            components.append(row); self.emit('resident_layer_loaded', **row)
        start = time.perf_counter_ns()
        embedding = jax.device_put(np.array(cp.tensor('model.embed_tokens.weight'), copy=True))
        norm = jax.device_put(np.array(cp.tensor('model.norm.weight'), copy=True))
        head = jax.device_put(cp.head()); jax.block_until_ready((embedding, norm, head))
        payload = dict(layer_setup=components, embedding_norm_head_setup_ms=(time.perf_counter_ns()-start)/1e6,
                       loaded_device_bytes=sum(v.nbytes for v in jax.tree_util.tree_leaves((layers, embedding, norm, head))),
                       after_memory=jax.devices()[0].memory_stats(), setup_in_timed_forward=False)
        base.exclusive_json(self.args.output_dir/'resident_setup.json', payload)
        return layers, embedding, norm, head

    def resident(self, cp, ids, policies):
        layers, embedding, norm, head = self.load_resident(cp)
        gemma = cp.config['model_type'] == 'gemma3_text'
        scale = np.asarray(math.sqrt(cp.config['hidden_size']), dtype=prior.np.dtype(ml_dtypes.bfloat16)) if gemma else None
        @jax.jit
        def embed(tokens, table):
            x = table[tokens]
            return (x.astype(jnp.float32)*jnp.asarray(scale, jnp.float32)).astype(jnp.bfloat16) if gemma else x
        @jax.jit
        def score(hidden, targets, norm, weight):
            logits = composed.head(cp, hidden, norm, weight).astype(jnp.float32)
            return -jax.nn.log_softmax(logits, -1)[jnp.arange(len(targets)), targets]
        arms = list(policies)+['native_full_jit']; length = ids.shape[1]
        fns = [self.functions(cp, length, policies, index) for index in range(len(layers))]
        token_inputs = [jax.device_put(ids[w]) for w in self.cfg['timing_windows']]
        def forward(tokens, arm, scope):
            hidden = embed(tokens, embedding)
            for index, weights in enumerate(layers):
                hidden = fns[index][arm](hidden, weights)
            if scope == 'prompt_forward_last_token':
                return composed.head(cp, hidden[-1:], norm, head)
            values = []
            for lo in range(0, length-1, 128):
                hi = min(lo+128, length-1)
                values.append(score(hidden[lo:hi], tokens[lo+1:hi+1], norm, head))
            return values
        rng = random.Random(self.cfg['order_seed']); report = {}
        for scope in ('prompt_forward_last_token','teacher_forced_scoring'):
            for arm in arms:
                for _ in range(self.cfg['full_model_warmups']):
                    self.check(); value = forward(token_inputs[0], arm, scope); jax.block_until_ready(value); del value
                self.emit('resident_model_warmup_complete', scope=scope, arm_id=arm)
            samples = {a: [] for a in arms}
            for repeat in range(self.cfg['full_model_repeats']):
                for wi, tokens in enumerate(token_inputs):
                    order = list(arms); rng.shuffle(order)
                    for arm in order:
                        self.check(); start = time.perf_counter_ns()
                        value = forward(tokens, arm, scope); jax.block_until_ready(value)
                        elapsed = (time.perf_counter_ns()-start)/1e6; samples[arm].append(elapsed); del value
                        self.emit('resident_model_sample', scope=scope, arm_id=arm, repeat=repeat,
                                  input_index=wi, elapsed_ms=elapsed, token_count=length)
            timing = {a: prior.timing_ci(v, self.cfg['order_seed']) for a,v in samples.items()}
            comparisons = {}
            for arm in arms:
                comparisons[arm] = {}
                for reference in ('native','native_default','native_full_jit'):
                    candidate, baseline = np.asarray(samples[arm]), np.asarray(samples[reference])
                    # Resample entire rounds, preserving the three input windows per round.
                    matrix = np.stack([candidate, baseline], -1).reshape(self.cfg['full_model_repeats'],len(token_inputs),2)
                    ix = np.random.default_rng(922).integers(0,len(matrix),(2000,len(matrix)))
                    means = matrix[ix].mean((1,2)); ratios = means[:,0]/means[:,1]
                    comparisons[arm][reference] = dict(latency_ratio=float(candidate.mean()/baseline.mean()),
                        latency_reduction=float(1-candidate.mean()/baseline.mean()),
                        paired_round_ci95=np.quantile(ratios,[.025,.975]).tolist(),
                        scope='Same inputs and timing rounds; one hardware allocation; no multiplicity correction')
            report[scope] = dict(timings=timing, comparisons=comparisons,
                tokens_per_second={a: length*1000/v['mean_ms'] for a,v in timing.items()})
            base.exclusive_json(self.args.output_dir/(scope+'_timings.json'), report[scope])
        base.exclusive_json(self.args.output_dir/'resident_timings.json', report)
        self.emit('resident_model_timing_complete', scopes=list(report))
        return dict(fully_resident_model_timed=True, resident_timings=report,
            scope='Device token IDs through resident embedding/layers/final norm/head; Python dispatch and synchronization included',
            excluded='One-time checkpoint reading, layout, weight transfer, tokenization and compilation',
            limitation='B1 S2048 prompt forwards/scoring; no persistent KV cache, production serving or decode-latency claim',
            memory_after=jax.devices()[0].memory_stats())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','output-dir','expected-identity','model-input'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('phase','allocation-id'): p.add_argument('--'+name, required=True)
    p.add_argument('--max-wall-seconds', type=float, default=7200)
    p.add_argument('--action', choices=('tune','quality','resident'), required=True)
    p.add_argument('--selection', type=Path)
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.campaign.read_text()); journal = prior.Journal(args.output_dir, args.phase)
    problem = None; result = None
    try:
        if 'jax' in sys.modules: raise RuntimeError('Fresh device process required')
        os.environ.update(base.FIXED_ENVIRONMENT)
        global jax, jnp, np, ml_dtypes, qm, gm, composed, grid
        import jax
        import jax.numpy as jnp
        import numpy as np
        import ml_dtypes
        from . import model_n9_v001 as qm, model_gemma_v002 as gm, model_composed_v002 as composed
        from . import benchmark_mlsys_shapes_v001 as grid
        for module in (base, prior):
            module.jax, module.jnp, module.np, module.ml_dtypes = jax, jnp, np, ml_dtypes
        prior.qm, prior.gm, prior.composed, prior.grid = qm, gm, composed, grid
        grid.install_reference_policy(cfg)
        base.snapshot_sources(args.output_dir, args.campaign, args.campaign.parent/cfg['shape_manifest'], args.campaign.parent/cfg['distribution_manifest'])
        verify_environment(args, cfg); journal.emit('identity_check', status='matched')
        bundle = json.loads(args.model_input.read_text())
        if bundle['status'] != 'ready': raise ValueError('Model inputs unavailable')
        study = Study(args, cfg, journal); cp, ids, reference = study.load(bundle)
        if args.action == 'tune': result = study.tune_model(cp, ids, reference, bundle)
        else:
            policies = study.selected(bundle)
            result = getattr(study, args.action)(cp, ids, policies)
        result.update(model_id=bundle['model_id'], revision=bundle['revision'], action=args.action,
                      actual_model_execution=True, status='completed')
        base.exclusive_json(args.output_dir/'model_summary.json', base.json_safe(result))
    except BaseException as exc:
        problem = dict(type=type(exc).__name__, message=str(exc))
        journal.emit('run_error', **problem, traceback=traceback.format_exc())
    summary = dict(completed=problem is None, action=args.action, error=problem,
                   model=result, case_status_counts=dict(journal.status_counts))
    base.exclusive_json(args.output_dir/'summary.json', base.json_safe(summary)); journal.close()
    return 0 if problem is None else 1

if __name__ == '__main__': raise SystemExit(main())
