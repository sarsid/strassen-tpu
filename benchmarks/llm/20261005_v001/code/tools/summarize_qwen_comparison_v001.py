"""Offline summary of completed Qwen8 and five interim Qwen32 workloads.

Preserve streaming and separate resident scopes. Intervals resample matched
confirmation rounds, not independent devices or datasets. No hardware access.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ARMS = ('native_default', 'native_tuned', 'cubic_tuned', 's1', 's2')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ledger', type=Path, required=True)
    a = p.parse_args()
    raw = a.ledger.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    groups = defaultdict(dict)
    for row in rows:
        if row['model'] in ('qwen3_8b', 'qwen3_32b') and row['output_dtype'] == 'bfloat16':
            key = (row['model'], row['batch'], row['sequence'])
            assert row['algorithm'] not in groups[key]
            groups[key][row['algorithm']] = row
    assert Counter(k[0] for k in groups) == {'qwen3_8b': 10, 'qwen3_32b': 5}
    index = np.random.default_rng(20261003).integers(0, 15, size=(10000, 15))
    output = dict(as_of_utc=datetime.now(timezone.utc).isoformat(),
        ledger_sha256=hashlib.sha256(raw).hexdigest(),
        method='Paired bootstrap ratios of medians, 15 matched rounds, 10000 resamples, seed 20261003, percentile 95% intervals; no multiple-comparison adjustment.',
        timing_scope='Streamed prefill includes serialized H2D, embeddings and all-token logits. Resident pass separately sums per-layer medians; it is not fully resident whole-model latency.',
        transfer_share_scope='Median over individual rounds of (layer H2D + head H2D) / elapsed time. Components exclude transfer overlap because none is implemented.',
        quality_scope='Fixed held-out WikiText2 window, compared with Default Native; not downstream task accuracy.',
        groups=[], aggregate={})
    for (model, batch, sequence), group in sorted(groups.items()):
        assert set(group) == set(ARMS)
        measurements = {name: row['measurement'] for name, row in group.items()}
        samples = {name: np.array([s['elapsed_ms'] for s in m['samples']]) for name, m in measurements.items()}
        for name, values in samples.items():
            assert len(values) == 15 and np.isfinite(values).all()
            assert measurements[name]['confirmation_order'] == measurements['native_tuned']['confirmation_order']
            assert abs(np.median(values) - measurements[name]['elapsed_median_ms']) < 1e-6
        result = dict(model=model, batch=batch, sequence=sequence, algorithms={})
        for name in ARMS:
            row, m = group[name], measurements[name]
            transfer = [(s['layer_transfer_sum_ms']+s['head_transfer_ms'])/s['elapsed_ms'] for s in m['samples']]
            profile = row['profile']['by_layer_type']['full_attention']
            value = dict(entry_id=row['entry_id'], evidence_run=row['evidence_run'],
                full_forward_ms=m['elapsed_median_ms'], resident_ms=m.get('resident_pass', {}).get('aggregate_ms'),
                transfer_share_percent=100*float(np.median(transfer)),
                components_ms={k:float(np.median([s[k] for s in m['samples']])) for k in
                    ('embedding_ms','layer_transfer_sum_ms','resident_layer_sum_ms','head_transfer_ms','head_compute_ms')},
                quality=m['quality'], quality_gate_passed=row['quality_gate_passed'],
                perplexity_change_percent=100*(m['quality']['perplexity']/measurements['native_default']['quality']['perplexity']-1),
                max_hidden_relative_l2=max((v['relative_l2'] for v in m['hidden_errors']), default=0),
                compiler_options=profile.get('compiler_options', {}),
                selected_custom_sites=profile.get('selected_custom_sites', []),
                policies={site:{k:arm.get(k) for k in ('implementation','depth','tile','accumulator','buffers')}
                    for site,arm in profile['policy'].items()})
            for base in ('native_default', 'native_tuned'):
                b = measurements[base]
                ci = np.quantile(np.median(samples[base][index],axis=1)/np.median(samples[name][index],axis=1), [.025,.975])
                br = b.get('resident_pass', {}).get('aggregate_ms')
                value['vs_'+base] = dict(full_time_reduction_percent=100*(1-m['elapsed_median_ms']/b['elapsed_median_ms']),
                    speedup_95=ci.tolist(), resident_time_reduction_percent=None if br is None else 100*(1-value['resident_ms']/br))
            result['algorithms'][name] = value
        output['groups'].append(result)
    for model in ('qwen3_8b','qwen3_32b'):
        by_model = [g for g in output['groups'] if g['model']==model]
        output['aggregate'][model] = {}
        for name in ARMS:
            values = [g['algorithms'][name] for g in by_model]
            q = [v['quality'] for v in values]
            resident = [v for v in values if v['resident_ms'] is not None]
            output['aggregate'][model][name] = dict(workloads=len(values), all_quality_gates_passed=all(v['quality_gate_passed'] for v in values),
                full_point_wins=sum(v['vs_native_tuned']['full_time_reduction_percent']>0 for v in values),
                full_ci_wins=sum(v['vs_native_tuned']['speedup_95'][0]>1 for v in values),
                full_ci_losses=sum(v['vs_native_tuned']['speedup_95'][1]<1 for v in values),
                resident_comparable=len(resident), resident_point_wins=sum(v['vs_native_tuned']['resident_time_reduction_percent']>0 for v in resident),
                transfer_share_percent_range=[min(v['transfer_share_percent'] for v in values),max(v['transfer_share_percent'] for v in values)],
                perplexity_change_percent_range=[min(v['perplexity_change_percent'] for v in values),max(v['perplexity_change_percent'] for v in values)],
                quality_ranges={k:[min(v[k] for v in q),max(v[k] for v in q)] for k in
                    ('delta_nll','mean_kl','top1_agreement','logit_relative_l2','logit_rmse','logit_max_abs','logit_normalized_max')},
                max_hidden_relative_l2=max(v['max_hidden_relative_l2'] for v in values))
    folder=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts'
    folder.mkdir(exist_ok=True)
    (folder/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'aggregate':output['aggregate'],'qwen32':[
        {'batch':g['batch'],'sequence':g['sequence'],'algorithms':{n:{k:g['algorithms'][n][k] for k in
            ('full_forward_ms','resident_ms','vs_native_tuned')} for n in ARMS}}
        for g in output['groups'] if g['model']=='qwen3_32b']},indent=2))


if __name__=='__main__':
    main()
