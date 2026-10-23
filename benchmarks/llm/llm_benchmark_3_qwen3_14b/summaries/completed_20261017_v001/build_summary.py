"""Rebuild the derived report from immutable published JSON; no TPU required."""
import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparisons(times):
    s = times['strassen_tuned']
    return {arm: {'speedup': times[arm] / s,
                  'saved_ms': times[arm] - s,
                  'time_saved_percent': 100 * (1 - s / times[arm])}
            for arm in ['native', 'native_tuned']}


def write_preserving(path, text):
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Refusing to overwrite different results: {path}')
    else:
        path.write_text(text)


def build(root, output):
    ops = root / 'operations/completed_20261017_v001'
    complete = read(ops / 'completion.json')
    plan = read(ops / 'remote/artifacts/plan.json')
    source = read(root / 'operations/initial_index_20261017_v001/MANIFEST.json')
    assert complete['status'] == 'completed'
    assert complete['release_exit_code'] == 0 and not complete['allocation_retained']
    assert read(ops / 'verification/release-verification.json')['verified_absent']
    stages = complete['remote_summary']['stages']
    assert len(stages) == 30 and all(x['returncode'] == 0 and x['attempt'] == 0 for x in stages)
    records = []
    arms = plan['arms']
    for b in [1, 4, 8, 16, 32, 64]:
        path = root / f'results/20261017_v001/b{b}'
        artifact = path / 'artifacts'
        archive_hash = digest(path / 'results.tar.gz')
        assert archive_hash == read(path / 'receipt.json')['sha256']
        assert read(artifact / f'b{b}-summary.json')['quality_pass']
        profile_path = artifact / f'b{b}-tune-a0/profile.json'
        profile = read(profile_path)
        assert (profile['model'], profile['revision'], profile['batch'], profile['sequence']) == (
            plan['model'], plan['revision'], b, 1024)
        qualify = read(artifact / f'b{b}-qualify-a0/summary.json')
        resident = read(artifact / f'b{b}-confirm-a0/summary.json')
        natural = read(artifact / f'b{b}-quality-a0/summary.json')
        prefill = read(artifact / f'b{b}-prefill-a0/summary.json')
        assert qualify['status'] == 'passed' and natural['all_pass']
        for a in arms:
            assert resident['repetitive_quality'][a]['passes'] and natural['results'][a]['passes']
            samples = prefill['samples_ms'][a]
            assert len(samples) == 3
            assert math.isclose(statistics.mean(samples), prefill['mean_ms'][a], rel_tol=1e-12)
        records.append({
            'batch': b, 'sequence': 1024, 'tokens': b * 1024,
            'archive_sha256': archive_hash,
            'profile_file_sha256': digest(profile_path), 'frozen_profile': profile,
            'qualification': qualify, 'resident': resident, 'natural_quality': natural,
            'offloaded_prefill': prefill,
            'resident_strassen_comparisons': comparisons(resident['aggregate_ms']),
            'offloaded_strassen_comparisons': comparisons(prefill['mean_ms']),
            'offloaded_sample_statistics': {
                a: {'min_ms': min(prefill['samples_ms'][a]),
                    'max_ms': max(prefill['samples_ms'][a]),
                    'sample_stddev_ms': statistics.stdev(prefill['samples_ms'][a])}
                for a in arms},
            'stage_wall_seconds': {s['stage']: s['seconds'] for s in stages if s['batch'] == b},
            'raw_artifacts': str(artifact.relative_to(root)),
        })
    elapsed = (datetime.fromisoformat(complete['remote_summary']['finished_utc']) -
               datetime.fromisoformat(complete['remote_summary']['started_utc'])).total_seconds()
    summary = {
        'schema': 'qwen14_completed_summary_v001', 'status': 'completed_released',
        'run_id': '20261017-qwen3-14b-v001', 'source_commit': source['source_commit'],
        'source_sha256': source['source_sha256'], 'full_archive_sha256': complete['result_sha256'],
        'plan': plan, 'completed_batches': 6, 'remote_worker_elapsed_seconds': elapsed,
        'finished_utc': complete['remote_summary']['finished_utc'],
        'resident_scope': 'Sum of 40 independently warmed decoder-layer means; excludes transfers, weight preparation, embeddings, final norm/head and compilation; not uninterrupted whole-model latency.',
        'offloaded_scope': 'Serialized cached-disk-offloaded prefill; common loader/packing/Strassen preparation and recurring H2D included for every arm; embedding/all layers/last-position head included; no initial network downloads, compilation or decode.',
        'inference_limits': ['One allocation, five samples per warmed layer; layer bootstrap is not independent whole-model repetition.',
                             'Three measured offloaded passes per arm; report raw variation, not a general serving speedup.',
                             'Corpus spans differ by batch; compare algorithms within each batch.',
                             'Whole implementation comparison does not isolate Strassen arithmetic from fusion and scheduling.'],
        'batches': records,
    }
    write_preserving(output / 'summary.json', json.dumps(summary, indent=2) + '\n')

    lines = [
        '# LLM Benchmark 3: Qwen3-14B — completed six-batch report', '',
        'All six batches completed on one v6e and passed the registered quality gates. '
        'Tuned Strassen S1 was the fastest of the four arms in the summed resident-layer measurement at every batch. '
        'It used **5.86% less time than Tuned Native at B1 and 23.08–29.83% less at B4–B64**. '
        'The maximum natural-text perplexity increase relative to Native was **0.1191%**.', '',
        'The remote worker ran from 17:53:18 to 22:03:15 UTC on October 17, 2026 '
        '(about 4 hours 10 minutes, including installation, tuning and checkpoint loading). '
        'All 30 stages succeeded on their first attempt. All archives were verified and the TPU was released; '
        'its absence and the exit of the supervisor/keepalive were independently checked.', '',
        '## Matched setup', '',
        'Qwen/Qwen3-14B at revision `40c069824f4251a91eefaf281ebe4c544efd3e18`; '
        '40 layers, hidden width 5120, FFN width 17408. Sequence length is 1024; '
        'batches 1, 4, 8, 16, 32 and 64 were executed in order 8, 16, 32, 64, 4, 1. '
        'Every batch used JAX/jaxlib 0.11.2 and libtpu 0.0.48, BF16 inputs/output, '
        'parent FP32 accumulation and DEFAULT precision.', '',
        'Fresh tuning was performed per batch and frozen before confirmation or held-out quality. '
        'Native VMEM and each custom arm’s gate/up and O/down tiles were selected independently. '
        'Q/K used the parent published tile, with BM clipped at B1; V remained native. '
        'The parent kernels and measurement contracts are unchanged. See [protocol](../../PROTOCOL.md) '
        'and [frozen code](../../code/initial_20261017_v001/source).', '',
        '## Resident decoder-layer computation', '',
        'Times below are **milliseconds summed over 40 separately warmed layer means**, '
        'with two warmups and five timed samples per layer. They exclude weight preparation/transfers, '
        'embedding, final norm/head and compilation. They are not uninterrupted whole-model latency. '
        'Each arm propagates its own activations; arm order rotates/reverses. '
        'Resident projections and block0 additionally have three warmups and 20 samples in the raw events.', '',
        '| Batch | Native ms | Tuned Native ms | Tuned Cubic ms | Strassen S1 ms | Less time vs Tuned Native | Speedup vs Tuned Native | Less time vs Native |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in records:
        t = r['resident']['aggregate_ms']; c = r['resident_strassen_comparisons']
        lines.append(f"| {r['batch']} | " + ' | '.join(f'{t[a]:.3f}' for a in arms) +
                     f" | {c['native_tuned']['time_saved_percent']:.2f}% | {c['native_tuned']['speedup']:.3f}× | {c['native']['time_saved_percent']:.2f}% |")
    lines += ['',
        'Tuned Native at B8 was slightly slower than default Native during confirmation. '
        'The preselected profile is retained without selecting again on confirmation results. '
        'Stored layer-bootstrap intervals describe variation among measured layers and are not '
        'confidence intervals from independent whole-model runs.', '',
        '## Cached-disk-offloaded prefill', '',
        'Means below are **seconds for three measured complete passes per arm**, after one warmup. '
        'These include cached weight reads, packing, H2D transfers, embedding, all layers and last-position logits. '
        'The common loader performs Strassen preparation for every arm; this is not a separately optimized '
        'native serving baseline. No decode, initial network downloads or compilation is timed.', '',
        '| Batch | Native s | Tuned Native s | Tuned Cubic s | Strassen S1 s | Less time vs Tuned Native | Less time vs Native |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in records:
        t = r['offloaded_prefill']['mean_ms']; c = r['offloaded_strassen_comparisons']
        lines.append(f"| {r['batch']} | " + ' | '.join(f'{t[a]/1000:.3f}' for a in arms) +
                     f" | {c['native_tuned']['time_saved_percent']:.2f}% | {c['native']['time_saved_percent']:.2f}% |")
    lines += ['',
        'Strassen’s mean offloaded time is lower than both native baselines at every batch, '
        'but most differences are small relative to the observed pass variation. Cubic has the lowest '
        'offloaded mean at B16. At B64, all three Strassen passes are below all three passes of each native baseline; '
        'the mean saving is 0.894 seconds (1.42%) versus Tuned Native and 2.247 seconds (3.49%) versus Native. '
        'Three passes on one allocation do not establish a general deployment gain.', '',
        'The large resident savings therefore carry only modestly into this offloaded harness. '
        'Recurring loading/preparation/transfers dominate its wall time. These are distinct measurement scopes, '
        'so their difference should not be reported as a direct measurement of transfer time.', '',
        'All individual offloaded pass times (seconds):', '',
        '| Batch | Native | Tuned Native | Tuned Cubic | Strassen S1 |',
        '|---:|---|---|---|---|',
    ]
    for r in records:
        lines.append(f"| {r['batch']} | " + ' | '.join(', '.join(f'{v/1000:.3f}' for v in r['offloaded_prefill']['samples_ms'][a]) for a in arms) + ' |')
    lines += ['', '## Numerical cost', '',
        'All four arms passed repetitive-prompt and natural-text gates at every batch. '
        'Natural evaluation uses four contiguous WikiText2 test segments; token hashes are recorded. '
        'Larger batches cover different corpus spans, so absolute perplexity across batches is not a quality scaling result. '
        'Top-1 agreement below is agreement with Native predictions, not ground-truth accuracy.', '',
        '| Batch | Native PPL | S1 PPL | S1 relative PPL change | S1 top-1 agreement | S1 logit relative L2 | S1 KL (nats) |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in records:
        q = r['natural_quality']['results']; s = q['strassen_tuned']
        lines.append(f"| {r['batch']} | {q['native']['perplexity']:.6f} | {s['perplexity']:.6f} | {100*s['relative_perplexity_delta']:+.4f}% | {100*s['top1_agreement']:.3f}% | {100*s['logit_l2_relative']:.3f}% | {s['mean_kl_nats']:.7f} |")
    lines += ['',
        'Passing the predictive gates does not mean identical logits. Natural-text S1 logit relative L2 '
        'is about 2.07–2.53%; repetitive-prompt logit differences are larger. The complete NLL, PPL, '
        'KL, agreement, finite checks, relative L2 and maximum errors for every arm are retained in '
        '[summary.json](summary.json), with projection/block RMSE and other diagnostics in the raw events.', '',
        'Thresholds are unchanged: repetitive absolute NLL difference ≤0.01, KL ≤0.005 and agreement ≥0.99; '
        'natural absolute NLL difference ≤0.01, KL ≤0.02 and agreement ≥0.97. '
        'Native capture reconstruction relative L2 must be ≤0.01. This follows the parent Qwen computation; '
        'it does not add an independent cross-framework model-equivalence claim.', '',
        '## Frozen tuning decisions', '',
        'Tiles below are `(BM, BN, BK)` as recorded by the parent implementation. '
        'The model-specific geometry was used throughout; Qwen32 tiles were not copied as tuning results. '
        'All candidate measurements, filtering/compilation events and profile identities remain in each batch’s raw archive.', '',
        '| Batch | Native VMEM MiB | S1 gate/up | S1 O | S1 down | Cubic gate/up | Cubic O | Cubic down | Fixed Q/K |',
        '|---:|---:|---|---|---|---|---|---|---|',
    ]
    for r in records:
        p = r['frozen_profile']['profile']; fmt = lambda t: '×'.join(map(str, t))
        native = p['native_options'].get('xla_tpu_scoped_vmem_limit_kib')
        vals = [p['product_tile'], p['strassen_sites']['o'], p['strassen_sites']['down'],
                p['cubic_tile'], p['cubic_sites']['o'], p['cubic_sites']['down'], p['qk_tile']]
        lines.append(f"| {r['batch']} | {native/1024:g} | " + ' | '.join(fmt(v) for v in vals) + ' |')
    lines += ['', '## Preservation and interpretation', '',
        'This is evidence that the parent fused Strassen implementation can improve LLM decoder computation '
        'using real checkpoint weights and propagated activations. It does not isolate arithmetic savings '
        'from fusion/scheduling, or measure future hardware benefits. All four controls remain available.', '',
        'The first allocation request received HTTP 502 and succeeded on retry. A preliminary local validation '
        'environment lacked `requests`; the unchanged checks passed in the existing validation environment. '
        'Neither issue caused a TPU measurement-stage retry or a change to scientific settings.', '',
        '- [Immutable batch results](../../results/20261017_v001): all raw timing/error events, profiles and per-batch tar files.',
        '- [Completion and release evidence](../../operations/completed_20261017_v001): full remote archive, environment freeze, installation report and independent release/process checks.',
        '- [Initial index snapshot](../../operations/initial_index_20261017_v001): the original launch README and manifest remain preserved.',
        '- `build_summary.py` regenerates this report and JSON using only published raw summaries and refuses to replace differing output.', '',
        f"Source archive SHA-256: `{source['source_sha256']}`. Full results archive SHA-256: `{complete['result_sha256']}`.", '',
        'The code, protocol, historical partial summaries and prior Qwen32/Gemma results remain unchanged. '
        'No other model, architecture or new experiment is queued.', '',
    ]
    write_preserving(output / 'README.md', '\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    build(args.benchmark_root, args.output)
