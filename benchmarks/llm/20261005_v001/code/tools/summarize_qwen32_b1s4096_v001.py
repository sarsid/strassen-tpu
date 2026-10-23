"""Frozen five-arm summary for Qwen32 B1/S4096, with paired timing intervals."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ledger', type=Path, required=True)
    args = parser.parse_args()
    raw = args.ledger.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    rows = [r for r in rows if (r['model'], r['batch'], r['sequence'], r['output_dtype']) ==
            ('qwen3_32b', 1, 4096, 'bfloat16')]
    names = ('native_default', 'native_tuned', 'cubic_tuned', 's1', 's2')
    assert len(rows) == 5 and {r['algorithm'] for r in rows} == set(names)
    by_name = {r['algorithm']: r for r in rows}
    measurements = {name: row['measurement'] for name, row in by_name.items()}
    samples = {name: np.array([s['elapsed_ms'] for s in m['samples']]) for name, m in measurements.items()}
    for name, values in samples.items():
        assert len(values) == 15 and np.isfinite(values).all()
        assert measurements[name]['confirmation_order'] == measurements['native_tuned']['confirmation_order']
        assert abs(np.median(values) - measurements[name]['elapsed_median_ms']) < 1e-6
    index = np.random.default_rng(20261003).integers(0, 15, size=(10000, 15))
    result = dict(as_of_utc=datetime.now(timezone.utc).isoformat(), workload=dict(model='qwen3_32b', batch=1, sequence=4096),
        ledger_sha256=hashlib.sha256(raw).hexdigest(),
        selected_entries_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        all_quality_gates_passed=all(r['quality_gate_passed'] for r in rows),
        method='Paired bootstrap ratio of medians, 15 matched rounds, 10000 resamples, seed 20261003, percentile 95% intervals; no multiple-comparison adjustment.',
        scope='Full streamed forward includes H2D, embeddings and all-token logits; resident pass is a separate sum of layer medians excluding those costs.',
        algorithms={})
    for name in names:
        m = measurements[name]
        value = dict(entry_id=by_name[name]['entry_id'], evidence_run=by_name[name]['evidence_run'],
            full_forward_ms=m['elapsed_median_ms'], resident_pass_ms=m['resident_pass']['aggregate_ms'],
            quality=m['quality'], max_hidden_relative_l2=max((h['relative_l2'] for h in m['hidden_errors']), default=0),
            perplexity_change_percent=100*(m['quality']['perplexity']/measurements['native_default']['quality']['perplexity']-1),
            median_layer_transfer_ms=float(np.median([s['layer_transfer_sum_ms'] for s in m['samples']])),
            median_head_transfer_ms=float(np.median([s['head_transfer_ms'] for s in m['samples']])))
        for base in ('native_default', 'native_tuned'):
            b = measurements[base]
            interval = np.quantile(np.median(samples[base][index], axis=1)/np.median(samples[name][index], axis=1), [.025, .975])
            value['vs_'+base] = dict(speedup=b['elapsed_median_ms']/m['elapsed_median_ms'],
                speedup_95=interval.tolist(), time_reduction_percent=100*(1-m['elapsed_median_ms']/b['elapsed_median_ms']),
                resident_time_reduction_percent=100*(1-m['resident_pass']['aggregate_ms']/b['resident_pass']['aggregate_ms']))
        result['algorithms'][name] = value
    folder = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
    folder.mkdir(exist_ok=True)
    (folder/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
