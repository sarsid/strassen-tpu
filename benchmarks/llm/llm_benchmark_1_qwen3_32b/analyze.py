"""Offline, standard-library reconstruction of LLM Benchmark 1."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import statistics
import tarfile
import io

ARMS = ('native', 'native_tuned', 'cubic_tuned', 'strassen_tuned')
LABELS = ('Native', 'Tuned Native', 'Tuned Cubic', 'Tuned Strassen S1')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8), (a, b)


def reconstruct(root):
    index = json.loads((root / 'evidence/index.json').read_text())
    def data(name):
        entry = index['files'][name]
        raw = gzip.decompress((root / entry['blob']).read_bytes())
        assert digest(raw) == entry['sha256'] and len(raw) == entry['bytes'], name
        return raw
    def read(name): return json.loads(data(name))
    def events(name): return [json.loads(line) for line in data(name).splitlines() if line.strip()]
    def one(rows, kind):
        vals = [r for r in rows if r.get('kind') == kind]
        assert len(vals) == 1, (kind, len(vals))
        return vals[0]
    report = dict(title='LLM Benchmark 1 (Qwen3-32B)', sequence=1024,
                  architecture='single TPU v6e', arms=list(ARMS), batches=[])
    for spec in index['selected_batches']:
        b, base = spec['batch'], spec['artifacts']
        def stage(s, filename): return f'{base}/b{b}-{s}-a0/{filename}'
        cs = read(stage('confirm', 'summary.json'))
        qs = read(stage('quality', 'summary.json'))
        ps = read(stage('prefill', 'summary.json'))
        ce = events(stage('confirm', 'events.jsonl'))
        qe = events(stage('quality', 'events.jsonl'))
        pe = events(stage('prefill', 'events.jsonl'))
        for s in ('confirm','quality','prefill'):
            assert read(stage(s, 'completion.json'))['status'] == 'completed'
        env = one(ce, 'environment')
        fp = one(ce, 'frozen_profile')
        for rows in (qe, pe):
            assert one(rows,'environment')['versions'] == env['versions']
            assert one(rows,'frozen_profile')['sha256'] == fp['sha256']
        assert digest((json.dumps(fp['profile'], indent=2)+'\n').encode()) == fp['sha256']
        layers = [e for e in ce if e.get('kind') == 'layer']
        assert [e['layer'] for e in layers] == list(range(64))
        projections = [e for e in ce if e.get('kind') == 'projection_confirmation']
        block = one(ce, 'block_confirmation')
        assert len(projections) == 6
        natural_layers = [e for e in qe if e.get('kind') == 'quality_layer']
        assert len(natural_layers) == 256 and all(e['finite'] for e in natural_layers)
        for e in layers + projections + [block]:
            assert e['finite']
            assert set(e['samples_ms']) == set(ARMS)
            for arm in ARMS:
                assert len(e['samples_ms'][arm]) == (5 if e['kind']=='layer' else 20)
                close(statistics.fmean(e['samples_ms'][arm]), e['mean_ms'][arm])
        passes = [e for e in pe if e.get('kind') == 'offloaded_prefill']
        assert len(passes) == 16 and all(e['finite'] for e in passes)
        for arm in ARMS:
            close(sum(e['mean_ms'][arm] for e in layers), cs['aggregate_ms'][arm])
            measured = [e['milliseconds'] for e in passes if e['arm']==arm and not e['warmup']]
            assert len(measured)==3 and measured==ps['samples_ms'][arm]
            close(statistics.fmean(measured), ps['mean_ms'][arm])
            q = qs['results'][arm]
            assert q['positions'] == 4*b*1023
            assert q['finite'] and q['passes']
            assert q['absolute_loss_delta']<=.01 and q['mean_kl_nats']<=.02 and q['top1_agreement']>=.97
            close(math.exp(q['nll']), q['perplexity'])
            assert cs['repetitive_quality'][arm]['passes']
        assert qs['all_pass']
        comparisons = {}
        for scope, times in [('resident',cs['aggregate_ms']),('offloaded_prefill',ps['mean_ms'])]:
            comparisons[scope] = {arm: {baseline: dict(speedup=times[baseline]/times[arm],
                saved_ms=times[baseline]-times[arm],time_reduction_percent=100*(1-times[arm]/times[baseline]))
                for baseline in ('native','native_tuned') if baseline!=arm} for arm in ARMS[1:]}
        for arm, baselines in cs['comparisons'].items():
            for baseline, metrics in baselines.items():
                close(metrics['speedup'], comparisons['resident'][arm][baseline]['speedup'])
                close(metrics['saved_ms'], comparisons['resident'][arm][baseline]['saved_ms'])
        report['batches'].append(dict(batch=b,tokens=b*1024,run=spec['run'],
            source_group='parent-stack' if b<64 else 'current-stack-extension',
            environment=env,profile=fp,token_provenance={'repetitive':one(ce,'tokens'),'natural':one(qe,'tokens')},
            resident_ms=cs['aggregate_ms'],resident_recorded_comparisons=cs['comparisons'],
            offloaded_prefill=ps,comparisons=comparisons,natural_quality=qs['results'],
            repetitive_quality=cs['repetitive_quality'],block=block,projections=projections,layers=layers,
            natural_layer_errors=[e for e in qe if e.get('kind')=='quality_layer'],prefill_passes=passes))
    assert [b['batch'] for b in report['batches']] == [1,4,8,16,32,64]
    return report


def render(r):
    lines = ['# LLM Benchmark 1 (Qwen3-32B)', '',
      'Completed October 16, 2026. Six batches × four algorithms, sequence length 1,024, one TPU v6e. All selected runs completed and all registered predictive-quality gates passed. The TPU was released; no further experiment is queued.', '',
      '**Strassen S1 reduces resident computation time versus Tuned Native at every tested batch size. Offloaded prefill gains are much smaller because serial weight loading and common preparation dominate.**', '',
      '**Keep B64 separate in cross-batch conclusions:** B1–B32 used JAX/jaxlib 0.7.2 and libtpu 0.0.21.1. B64 used JAX/jaxlib 0.11.2 and libtpu 0.0.48 after compiler stack-overflow failures. Every within-batch comparison uses the same stack. Do not interpret the B32→B64 change as a pure batch-scaling effect.', '',
      '## Resident transformer computation', '',
      'Milliseconds: sum of 64 individually warmed layer means, including attention, normalization, residuals and device-memory traffic. Excludes weight transfers, compilation, embeddings, final norm and vocabulary head. This is not uninterrupted fully resident model latency.', '',
      '| Batch | Tokens | Native | Tuned Native | Tuned Cubic | Strassen S1 | S1 speedup vs tuned | S1 time saved vs tuned |',
      '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for b in r['batches']:
        c=b['comparisons']['resident']['strassen_tuned']['native_tuned']
        lines.append(f"| {b['batch']}{'*' if b['batch']==64 else ''} | {b['tokens']:,} | " + ' | '.join(f"{b['resident_ms'][a]:,.3f}" for a in ARMS) + f" | {c['speedup']:.3f}× | {c['time_reduction_percent']:.2f}% |")
    lines += ['', '*B64 uses the separately registered compiler stack.', '', '## Serialized offloaded prefill', '',
      'Seconds, arithmetic mean of three measured passes per arm after one full warmup. Includes cached-checkpoint reads, common weight preparation/packing, recurring H2D transfers, embeddings, all layers, final norm and last-position logits. Excludes network downloads and compilation. No decode is measured. The common loader performs Strassen-related packing for every arm; this is not a comparison against an independently optimized native serving system.', '',
      '| Batch | Native | Tuned Native | Tuned Cubic | Strassen S1 | S1 time saved vs tuned |',
      '|---:|---:|---:|---:|---:|---:|']
    for b in r['batches']:
        c=b['comparisons']['offloaded_prefill']['strassen_tuned']['native_tuned']
        lines.append(f"| {b['batch']}{'*' if b['batch']==64 else ''} | " + ' | '.join(f"{b['offloaded_prefill']['mean_ms'][a]/1000:.3f}" for a in ARMS) + f" | {c['time_reduction_percent']:.2f}% |")
    lines += ['', 'Small offloaded differences are descriptive results from only three passes, not established deployment gains. Raw samples and ordering are preserved; do not infer an exact transfer overhead by subtracting the two different timing procedures.', '',
      '## Natural-text predictive quality', '',
      'Four contiguous WikiText-2 test segments per batch. Each arm sees identical tokens within a batch; larger batches score more tokens, so absolute perplexities are not quality rankings across batch sizes. These are next-token measurements against the matched Native implementation, not downstream task accuracy or an independent Transformers-equivalence test.', '',
      '| Batch | Positions | Native PPL | Tuned Native PPL | Cubic PPL | S1 PPL | S1 top-1 agreement | S1 KL (nats) | S1 abs NLL delta | S1 logit relative L2 |',
      '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for b in r['batches']:
        q=b['natural_quality'];s=q['strassen_tuned']
        lines.append(f"| {b['batch']} | {s['positions']:,} | "+' | '.join(f"{q[a]['perplexity']:.6f}" for a in ARMS)+f" | {100*s['top1_agreement']:.3f}% | {s['mean_kl_nats']:.6f} | {s['absolute_loss_delta']:.6f} | {100*s['logit_l2_relative']:.3f}% |")
    lines += ['', 'All arms pass the natural-text thresholds: absolute NLL delta ≤0.01 nats, KL ≤0.02 nats, top-1 agreement ≥97%, with finite outputs. All repetitive-text gates also pass (NLL ≤0.01, KL ≤0.005, top-1 ≥99%). Numerical differences are nonzero; near-identical perplexity is not identical predictions or proof of a quality improvement.', '',
      '## Frozen tuning choices', '', 'Tiles are `(BM, BN, BK)` in the parent implementation. Q/K uses the fixed published fused tile (BM clipped for B1); it was not independently tuned. V remains native. Cubic and Strassen use their own gate/up and O/down winners. Native tuning selects complete-block scoped VMEM; custom arms retain the parent 48 MiB XLA scope and separate Pallas limits.', '',
      '| Batch | Native VMEM MiB | S1 gate/up | Cubic gate/up | Fixed Q/K | S1 O | S1 down | Cubic O | Cubic down |',
      '|---:|---:|---|---|---|---|---|---|---|']
    for b in r['batches']:
        p=b['profile']['profile']['profile'];vm=p['native_options'].get('xla_tpu_scoped_vmem_limit_kib')
        tiles=[p['product_tile'],p['cubic_tile'],p['qk_tile'],p['strassen_sites']['o'],p['strassen_sites']['down'],p['cubic_sites']['o'],p['cubic_sites']['down']]
        lines.append(f"| {b['batch']} | {vm/1024 if vm else 'default'} | "+' | '.join('×'.join(map(str,t)) for t in tiles)+' |')
    lines += ['', '## Method, provenance and limitations', '',
      '- Official checkpoint: `Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137`; pinned parent code: `sarsid/strassen-tpu@95be1fb088656a89813b04492e1d77c66b36ccf9`.',
      '- Parent numerical contract: BF16 outputs, DEFAULT operand precision, parent FP32 accumulation and BF16 rounding boundaries. One-level product-aware fused SwiGLU and fused Q/K normalization/RoPE; no S2, padding or decode.',
      '- Parent kernel/math implementation is preserved. This is a four-arm batch-sweep extension of the reproduced parent study, not a byte-for-byte reproduction of the original benchmark. O-site tuning measures its actual plain-projection use; down includes residual. Gate/up tuning uses seeded synthetic operands; confirmation uses official weights and real propagated activations.',
      '- Tuning: two warmups/five samples, frozen before confirmation. Projection and block confirmation: three warmups/20 samples. Each layer: two warmups/five samples, with rotating/reversing arm order. Each arm propagates its own activations. Paired layer-bootstrap intervals are preserved in `summary.json`; they describe that sampling procedure, not variation across allocations.',
      '- B1/B4/B8/B16 come from the original completed sweep. B32 comes from the memory-repaired retry with its original frozen profile. B64 comes from the completed current-stack retry, reusing the exact SHA-verified profile recovered after runtime loss; no incomplete confirmation timings are pooled into a final result.',
      '- B64 tuning provenance is partial: the runtime was lost after tuning completed, and saved snapshots can omit candidate events. The exact selected profile was recovered with its original SHA-256 from two complete frozen-profile records. The package does not claim a complete B64 candidate ledger; final qualification, confirmation, quality and prefill records are complete on the replacement runtime.',
      '- B32 capture buffers were reduced; B64 uses a segmented BF16-boundary capture and qualification. These setup changes are versioned and preserved. Old compiler failures, controller timeout, runtime loss, and the pre-allocation superseded request remain in the failure/history evidence.',
      '- The 32B checkpoint does not fit fully on this single device. Resident-layer savings establish a compute benefit; the separately measured offload pipeline is the actual transfer-inclusive result. Do not present resident-layer totals as whole-model serving latency.', '',
      '## Contents and offline verification', '',
      '- [summary.json](summary.json): all four arms, raw timing samples, all 64 layer errors, projection/block results, natural/repetitive errors, prefill passes, selected tiles, versions and token hashes.',
      '- [evidence/index.json](evidence/index.json): original artifact paths mapped to lossless SHA-256-addressed gzip files. Includes all original batch results and failed attempts in the selected study roots; earlier unrelated LLM campaigns remain in their historical folders.',
      '- [provenance/index.json](provenance/index.json): exact source archives/manifests, run completions, environment freezes and result-selection records. Source snapshots are immutable historical evidence; do not execute archived controllers automatically.',
      '- [validation.json](validation.json): integrity, sample-count, finiteness, arithmetic, quality-gate and report-reconstruction checks. No TPU is required.', '',
      'From this directory run `python3 analyze.py --check`. To regenerate into a fresh directory, run `python3 analyze.py --output /tmp/qwen32-report`. To recover any raw file, find its `blob` in `evidence/index.json` and run `gzip -dc <blob> > <destination>`; the index records its original SHA-256 and size. Source archives are gzip-compressed original tar bytes.', '',
      'No credentials, checkpoint weights, private authentication state or virtual environments are included. No new experiment was run to build this report.', '']
    return '\n'.join(lines)


def verify(root):
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name,m in manifest['files'].items():
        data=(root/name).read_bytes()
        assert digest(data)==m['sha256'] and len(data)==m['bytes'], name
    index=json.loads((root/'evidence/index.json').read_text())
    for name,m in index['files'].items():
        data=gzip.decompress((root/m['blob']).read_bytes())
        assert digest(data)==m['sha256'] and len(data)==m['bytes'], name
    prov=json.loads((root/'provenance/index.json').read_text())
    for r in prov['runs']:
        raw=gzip.decompress((root/r['source_archive']).read_bytes())
        m=json.loads((root/r['source_manifest']).read_text())
        assert digest(raw)==m['source_sha256']
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            for path,sha in m['files'].items():
                assert digest(tf.extractfile(path).read())==sha,(r['run'],path)
    report=reconstruct(root)
    assert report==json.loads((root/'summary.json').read_text())
    assert render(report)==(root/'README.md').read_text()
    return dict(status='passed',batches=6,arms_per_batch=4,confirmed_layers=384,
                measured_prefill_passes=72,checks=['file_and_evidence_hashes','exact_frozen_sources',
                'stage_completion','frozen_profile_hashes','all_raw_sample_counts','timing_arithmetic',
                'all_registered_quality_gates','exact_summary_and_readme_reconstruction'])


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--output',type=Path)
    a=ap.parse_args()
    if a.check: print(json.dumps(verify(a.root)))
    if a.output:
        a.output.mkdir(parents=True,exist_ok=False)
        r=reconstruct(a.root);dump(a.output/'summary.json',r)
        (a.output/'README.md').write_text(render(r))
    if not a.check and not a.output:ap.error('choose --check or --output')
