"""Offline, provenance-bound timing scopes from completed BF16 LLM records.

No device allocation, benchmark execution, model loading or extrapolated serving
latency. Isolated projection screens remain separate from confirmation data.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ARMS = ('native_default', 'native_tuned', 'cubic_tuned', 's1', 's2')
LABEL = dict(zip(ARMS, ('Default Native', 'Tuned Native', 'Tuned Cubic', 'S1', 'S2')))
MODELS = ('qwen3_8b', 'qwen3_32b', 'qwen3_14b')
MODEL_LABEL = dict(zip(MODELS, ('Qwen3-8B', 'Qwen3-32B', 'Qwen3-14B')))
LAYERS = dict(zip(MODELS, (36, 64, 40)))
SCOPES = ('warmed_layer_aggregate', 'streamed_layer_compute', 'streamed_all_token_forward')
SCOPE_LABEL = dict(zip(SCOPES, ('Warmed transformer-layer aggregate', 'Transformer-layer compute during streaming', 'Full streamed forward, all-token logits')))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.inputs.read_text())
    assert digest(manifest['ledger']) == manifest['ledger_sha256']
    records = [json.loads(l) for l in Path(manifest['ledger']).read_text().splitlines() if l]
    records = [r for r in records if r['output_dtype'] == 'bfloat16' and r['model'] in MODELS]
    assert len(records) == manifest['expected_comparisons'] == 80
    groups = defaultdict(dict)
    for row in records:
        key = (row['model'], row['batch'], row['sequence'])
        assert row['algorithm'] not in groups[key] and row['status'] == 'completed'
        groups[key][row['algorithm']] = row
    assert Counter(k[0] for k in groups) == {'qwen3_8b': 10, 'qwen3_32b': 5, 'qwen3_14b': 1}
    index = np.random.default_rng(20261003).integers(0, 15, (10000, 15))
    event_inputs = {(r['model'],r['batch'],r['sequence']):r for r in manifest['events']}
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(), ledger_sha256=manifest['ledger_sha256'],
        active_completed_comparisons=80, expected_campaign_comparisons=200, groups=[],
        methods={
            'warmed_layer_aggregate':'Sum of individual layer medians; three warmups and 15 synchronized repeats per layer/algorithm. Each algorithm propagates its own hidden states. Weights remain on the device during each block of repeats. Excludes weight H2D, embeddings and vocabulary head. Includes attention, normalization, residuals, projections and device memory traffic. This is not an uninterrupted fully resident whole-model execution.',
            'streamed_layer_compute':'For each of 15 complete streamed rounds, sum directly recorded per-layer resident_compute_ms; then take the median of those sums. Excludes recorded layer H2D, embedding and head intervals. Not calculated by subtracting independently computed component medians. Cache/dispatch history differs from the separate warmed resident pass.',
            'streamed_all_token_forward':'Median of 15 measured complete-forward elapsed times, including serialized recurring layer/head H2D, embeddings and all prompt-token vocabulary logits. Excludes checkpoint loading, compilation, host packing, held-out scoring and error calculations. This is an offloaded all-token-logit prefill workload, not a measurement of generation decode or optimized last-token-only prefill.',
            'intervals':'Only matched streamed rounds are paired-bootstrap resampled (10000 draws, seed 20261003, percentile 95%, no multiple-comparison adjustment). No whole-model timing confidence interval is invented for the sum of individually warmed layer medians.',
            'projection_screening':'Selected candidates from eight-repeat tuning screens on layer-zero real activations; not independent confirmation. Only custom Cubic/S1/S2 projections were timed in isolation; Native screening used the whole layer. Never use Native whole-layer timing as an isolated-GEMM baseline.',
            'quality':'Preserve each policy’s separate fixed held-out WikiText2 next-token errors versus qualified Default Native. Re-scoping time does not remeasure or change these errors.'})
    projections = dict(scope=output['methods']['projection_screening'], groups=[])
    for key in sorted(groups, key=lambda k:(MODELS.index(k[0]),k[1],k[2])):
        model,batch,sequence = key
        group = groups[key]
        assert set(group)==set(ARMS)
        item = dict(model=model,batch=batch,sequence=sequence,layers=LAYERS[model],algorithms={})
        sample_sets = {}
        for name in ARMS:
            row,m = group[name],group[name]['measurement']
            rounds=m['samples'];assert len(rounds)==15
            assert m['confirmation_order']==group['native_default']['measurement']['confirmation_order']
            sums=[]
            for sample in rounds:
                assert len(sample['layers'])==LAYERS[model]
                assert [v['layer'] for v in sample['layers']]==list(range(LAYERS[model]))
                assert sample['eligible_for_inference_timing']
                value=sum(v['resident_compute_ms'] for v in sample['layers'])
                assert np.isclose(value,sample['resident_layer_sum_ms'],rtol=0,atol=1e-6)
                sums.append(value)
            elapsed=[r['elapsed_ms'] for r in rounds]
            assert np.isclose(np.median(elapsed),m['elapsed_median_ms'],rtol=0,atol=1e-6)
            assert np.isclose(np.median(sums),m['resident_layer_median_ms'],rtol=0,atol=1e-6)
            times={'warmed_layer_aggregate':None,'streamed_layer_compute':float(np.median(sums)),
                   'streamed_all_token_forward':float(np.median(elapsed))}
            resident=m.get('resident_pass')
            if resident:
                assert resident['warmups_per_layer']==3 and resident['repeats_per_layer']==15
                assert [r['layer'] for r in resident['per_layer']]==list(range(LAYERS[model]))
                for layer in resident['per_layer']:
                    assert layer['finite'] and len(layer['samples_ms'])==15 and np.isfinite(layer['samples_ms']).all()
                    assert np.isclose(np.median(layer['samples_ms']),layer['median_ms'],rtol=0,atol=1e-6)
                total=sum(float(np.median(r['samples_ms'])) for r in resident['per_layer'])
                assert np.isclose(total,resident['aggregate_ms'],rtol=0,atol=1e-6)
                times['warmed_layer_aggregate']=total
            sample_sets[name]={'streamed_layer_compute':np.array(sums),'streamed_all_token_forward':np.array(elapsed)}
            item['algorithms'][name]=dict(entry_id=row['entry_id'],evidence_run=row['evidence_run'],times_ms=times,
                warmed_per_layer=resident['per_layer'] if resident else None, streamed_layer_samples_ms=sums,
                full_forward_samples_ms=elapsed, profile=row['profile'], quality=m['quality'],
                hidden_errors=m['hidden_errors'], quality_gate_passed=row['quality_gate_passed'],
                perplexity_change_percent=100*(m['quality']['perplexity']/group['native_default']['measurement']['quality']['perplexity']-1),
                max_hidden_relative_l2=max((h['relative_l2'] for h in m['hidden_errors']),default=0),comparisons={})
        for name in ARMS:
            v=item['algorithms'][name]
            for scope in SCOPES:
                actual=v['times_ms'][scope]
                v['comparisons'][scope]={}
                if actual is None:continue
                for base in ('native_default','native_tuned'):
                    baseline=item['algorithms'][base]['times_ms'][scope]
                    result=dict(speedup=baseline/actual,time_reduction_percent=100*(1-actual/baseline))
                    if scope!='warmed_layer_aggregate':
                        dist=np.median(sample_sets[base][scope][index],axis=1)/np.median(sample_sets[name][scope][index],axis=1)
                        result['paired_speedup_95']=np.quantile(dist,[.025,.975]).tolist()
                    else:
                        before=item['algorithms'][base]['warmed_per_layer'];after=v['warmed_per_layer']
                        result['layers_nominally_faster']=sum(a['median_ms']<b['median_ms'] for a,b in zip(after,before))
                    v['comparisons'][scope][base]=result
        output['groups'].append(item)
        ev=event_inputs[key];assert digest(ev['path'])==ev['sha256']
        events=[json.loads(l) for l in Path(ev['path']).read_text().splitlines() if l]
        candidates=[r for r in events if r.get('kind')=='candidate' and r.get('output_dtype')=='bfloat16']
        screens=dict(model=model,batch=batch,sequence=sequence,events_file=ev['path'],events_sha256=ev['sha256'],
            isolated_native_available=False,candidate_status_counts=dict(Counter(r['status'] for r in candidates)),selected=[],
            native_whole_layer_screens=[r for r in events if r.get('kind')=='native_screen' and r.get('output_dtype')=='bfloat16'])
        for family in ('cubic_tuned','s1','s2'):
            for site,arm in group[family]['profile']['by_layer_type']['full_attention']['policy'].items():
                if arm['implementation']=='native':
                    screens['selected'].append(dict(family=family,site=site,status='native_fallback_no_isolated_timing'));continue
                matched=[r for r in candidates if r['family']==family and r['site']==site and r['arm']==arm and r['status']=='eligible']
                assert len(matched)==1,(key,family,site,len(matched))
                r=matched[0];assert len(r['samples_ms'])==8
                screens['selected'].append(dict(family=family,site=site,status='selected_screen_only',candidate_id=r['candidate_id'],
                    median_ms=float(np.median(r['samples_ms'])),samples_ms=r['samples_ms'],arm=r['arm'],metadata=r['metadata'],errors=r['errors']))
        projections['groups'].append(screens)
    folder=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';folder.mkdir(exist_ok=True)
    (folder/'measurements.json').write_text(json.dumps(output,indent=2)+'\n')
    (folder/'projection_screening.json').write_text(json.dumps(projections,indent=2)+'\n')
    make_report(output,projections,folder)
    make_figure(output,folder)
    availability={s:sum(g['algorithms']['s1']['times_ms'][s] is not None for g in output['groups']) for s in SCOPES}
    validation=dict(passed=True,comparisons=len(records),workloads=len(groups),workloads_by_scope=availability,
        all_quality_gates_passed=all(r['quality_gate_passed'] for r in records),event_files_verified=len(event_inputs),
        selected_projection_screens=sum(len(g['selected']) for g in projections['groups']),
        layer_sums_and_medians_reconstructed=True,no_new_hardware_measurements=True,
        unsupported=['Five-arm isolated Native/projection confirmation','Uninterrupted full-model resident latency','Last-token-only generation prefill','Decode throughput'])
    (folder/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    import matplotlib
    (folder/'software.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__),indent=2)+'\n')
    print(json.dumps(validation,indent=2))


def make_report(out,projections,folder):
    text=['# LLM timings separated by execution scope','',
        'Offline reconstruction of all 80 completed BF16 comparisons: Qwen3-8B (10 workloads), Qwen3-32B (5), and Qwen3-14B (1). No TPU was allocated or benchmark rerun. All recorded predictive-quality gates passed. The campaign remains paused.','',
        'The headline integration measurement below is the warmed transformer-layer aggregate. Removing host transfers does not remove device-memory reads/writes, projection epilogues, attention, normalization or residual work. It establishes integrated block-compute performance, not an uninterrupted resident-model serving latency.','',
        '| Measurement | Available workloads | Five algorithms? | Evidence |','|---|---:|---|---|',
        '| Isolated fused projections | 16 | Cubic/S1/S2 only | Eight-repeat tuning screens, not independent confirmation |',
        '| Warmed transformer-layer aggregate | 14 | Yes | 3 warmups + 15 repeats per layer, own propagated activations |',
        '| Layer compute during streaming | 16 | Yes | Direct sum of layer intervals in each of 15 streamed rounds |',
        '| Full streamed forward, all-token logits | 16 | Yes | 15 matched rounds, recurring H2D included |',
        '| Fully resident model / generation decode | 0 | No | Cannot reconstruct this deployment from these data |','',
        'Qwen8 B1/S512 and Qwen14 B1/S512 predate the separate warmed-layer pass. Their streaming-derived compute timings are retained separately, not substituted into that missing scope.','',
        '![Warmed layer speedups](warmed_layer_speedups.png)','',
        'Ratios above 1 mean faster. Both Default Native and Tuned Native baselines are shown; changing the baseline changes the claim. Resident points are sums of layer medians, with no invented whole-model confidence intervals.','']
    for scope in SCOPES:
        text += ['## '+SCOPE_LABEL[scope],'',out['methods'][scope],'']
        for model in MODELS:
            groups=[g for g in out['groups'] if g['model']==model and g['algorithms']['s1']['times_ms'][scope] is not None]
            if not groups:continue
            text += ['### '+MODEL_LABEL[model],'',
                '| Batch × tokens | Default Native ms | Tuned Native ms | Tuned Cubic ms | S1 ms | S2 ms | S1 / Default speedup | S1 / Tuned speedup | S2 / Tuned speedup |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for g in groups:
                a=g['algorithms'];values=' | '.join(f"{a[n]['times_ms'][scope]:.3f}" for n in ARMS)
                ratios=[a[n]['comparisons'][scope][b]['speedup'] for n,b in [('s1','native_default'),('s1','native_tuned'),('s2','native_tuned')]]
                text.append(f"| {g['batch']} × {g['sequence']} | {values} | "+' | '.join(f'{v:.4f}×' for v in ratios)+' |')
            text += ['']
    text += ['## Numerical costs remain attached to the policy','',
        'These held-out errors do not change when the timing scope changes. Perplexity shifts of either sign are not evidence of improved model quality; agreement is against Default Native rather than a downstream task score. Hidden-state and logit errors are diagnostic; they are not all pass/fail gates.','',
        '| Model | Algorithm | PPL change range | Top-1 agreement range | Logit relative L2 range | Worst layer relative L2 |',
        '|---|---|---:|---:|---:|---:|']
    for model in MODELS:
        groups=[g for g in out['groups'] if g['model']==model]
        for name in ARMS[1:]:
            rows=[g['algorithms'][name] for g in groups];ppl=[r['perplexity_change_percent'] for r in rows]
            top=[100*r['quality']['top1_agreement'] for r in rows];l2=[100*r['quality']['logit_relative_l2'] for r in rows]
            text.append(f"| {MODEL_LABEL[model]} | {LABEL[name]} | {min(ppl):+.3f}% to {max(ppl):+.3f}% | {min(top):.2f}%–{max(top):.2f}% | {min(l2):.2f}%–{max(l2):.2f}% | {100*max(r['max_hidden_relative_l2'] for r in rows):.2f}% |")
    text += ['', '## Available projection evidence','',out['methods']['projection_screening'],'',
        'See [projection screening tables](projection_screening.md) and [all selected samples, tiles, shapes and errors](projection_screening.json). Native whole-layer screening records remain in the JSON with their proper scope.','',
        '## Interpretation and limits','',
        '- Qwen8: S1 beats Tuned Native in 3/9 warmed-layer aggregates; S2 in 0/9. The strongest S1 time reduction is B4/S1024 (11.6%). At B8/S2048, S1 is 1.4106× Default Native but only 0.9247× Tuned Native. Transfers are not the explanation for that resident regression.',
        '- Qwen32: the stronger S1 warmed-compute reductions versus Tuned Native are 11.0% at B1/S2048, 5.2% at B1/S4096 and 8.9% at B4/S512. Against Default Native, B4/S512 reaches approximately 1.400×. These integrated-compute improvements remain valid even when the streamed-forward gain is small.',
        '- The historical parent B8/S1024 experiment is a separate workload/protocol. Its resident-layer result is not invalidated by this streamed-forward report. The current Qwen32 sweep has not completed B8/S1024; do not claim a matched replication from these five workloads.',
        '- A separate warmed pass and layer intervals during streaming need not match: their cache, dispatch and synchronization histories differ. Neither is a full resident model measurement.',
        '- Ordinary last-token-only prefill latency cannot be obtained by subtracting all-token head time. Its execution and memory behavior require a dedicated measurement. No decode timings are available.',
        '- Tuning screens and selected winners are not independent kernel confirmation. The current custom search optimizes isolated projections and reuses Native-selected whole-layer compiler options; this report does not remove that tuning limitation.','',
        out['methods']['intervals'],'',
        '[Machine-readable timings, samples, speedups against both baselines, intervals, profiles and all quality errors](measurements.json). [Integrity checks](validation.json).', '',
        'Input ledger SHA256: `'+out['ledger_sha256']+'`. Executed analysis source and input file hashes are preserved in the surrounding archived run.','']
    (folder/'README.md').write_text('\n'.join(text))
    lines=['# Isolated fused-projection tuning screens','',projections['scope'],'',
        'Every value below is a selected candidate’s median of eight calibration repeats, in milliseconds. These are layer-zero projection timings. They exclude host weight preparation and transfer and include the fused epilogue, activation padding/cropping and layout work described in each candidate’s metadata. Do not compare these values to Native whole-layer times or label their ratios as independently confirmed speedups.','']
    for g in projections['groups']:
        selected={(r['site'],r['family']):r for r in g['selected']}
        lines += [f"## {MODEL_LABEL[g['model']]} — B{g['batch']}/S{g['sequence']}",'',
                  '| Projection | M × K × N | Tuned Cubic ms | S1 ms | S2 ms |','|---|---|---:|---:|---:|']
        for site in ('q','k','v','o','gateup','down'):
            rows=[selected[(site,n)] for n in ('cubic_tuned','s1','s2')]
            shape=next(r['metadata']['shape_mkn'] for r in rows if 'metadata' in r)
            times=[f"{r['median_ms']:.6f}" if 'median_ms' in r else 'Native fallback; unavailable' for r in rows]
            lines.append('| '+site+' | '+' × '.join(map(str,shape))+' | '+' | '.join(times)+' |')
        lines += ['','Source events SHA256: `'+g['events_sha256']+'`.','']
    (folder/'projection_screening.md').write_text('\n'.join(lines))


def make_figure(out,folder):
    os.environ.setdefault('MPLCONFIGDIR',str(folder.parent/'mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,8.8),sharey=True)
    colors={'native_default':'#718096','native_tuned':'#202b3b','cubic_tuned':'#30977e','s1':'#2563c7','s2':'#d47b20'}
    for i,model in enumerate(('qwen3_8b','qwen3_32b')):
        groups=[g for g in out['groups'] if g['model']==model and g['algorithms']['s1']['times_ms']['warmed_layer_aggregate'] is not None]
        x=np.arange(len(groups));labels=[f"{g['batch']}×{g['sequence']}" for g in groups]
        for j,base in enumerate(('native_default','native_tuned')):
            ax=axes[i,j];ax.axhline(1,color='#9aa3b2',lw=1,ls='--')
            for name in ARMS:
                if name==base:continue
                values=[g['algorithms'][name]['comparisons']['warmed_layer_aggregate'][base]['speedup'] for g in groups]
                ax.plot(x,values,marker='o',lw=1.6,ms=4.5,color=colors[name],label=LABEL[name])
            ax.set_title(MODEL_LABEL[model]+' / '+LABEL[base],loc='left',fontweight='bold')
            ax.set_xticks(x,labels,rotation=35,ha='right');ax.set_ylim(.60,1.61)
            ax.set_yticks([.6,.8,1.,1.2,1.4,1.6],['0.6×','0.8×','1.0×','1.2×','1.4×','1.6×'])
            ax.grid(axis='y',alpha=.2);ax.set_xlabel('Batch × prompt tokens')
            if j==0:ax.set_ylabel('Warmed layer-compute speedup')
            ax.legend(loc='upper left',fontsize=8,ncol=2,frameon=False)
    fig.suptitle('Resident transformer-layer computation: two Native baselines',x=.065,ha='left',fontweight='bold',fontsize=17)
    fig.text(.065,.935,'Real weights and propagated activations · BF16 output · v6e · higher is faster',fontsize=11,color='#48576b')
    fig.text(.065,.025,'Sum of per-layer medians (3 warmups, 15 repeats). Excludes host transfers, embeddings and vocabulary head.\nIncludes device-memory traffic and all transformer-layer operations; not fully resident model-serving latency.',fontsize=10,color='#48576b')
    fig.subplots_adjust(top=.865,bottom=.16,left=.065,right=.985,hspace=.49,wspace=.16)
    fig.savefig(folder/'warmed_layer_speedups.png',dpi=160,facecolor='white')
    fig.savefig(folder/'warmed_layer_speedups.svg',facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    main()
