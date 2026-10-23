"""Audit dimension-only candidate regions against preserved measurements."""
import hashlib
import itertools
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ['STRASSEN_PROJECT_ROOT'])
OUT = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
OUT.mkdir(exist_ok=False)
INPUT = ROOT / 'runs/20260921T020920Z-probe-comparison-workbook-v002-06f7d8/artifacts/normalized_results.json'
MANIFEST = ROOT / 'runs/20260920-region-grid-cohort-v001/source/configs/generated_region_grid_v001/sampled_shapes.json'
CONFIRM = {
    'current': ROOT / 'runs/20260920-region-grid-cohort-v001/phases/region_grid_v001_20260920-grid-confirm-69874c/artifacts/confirmation.json',
    'previous': ROOT / 'runs/20260920T054326Z-GRID-confirm-v5e-v004-00a15b/artifacts/confirmation.json',
}
data = json.loads(INPUT.read_text())
manifest = json.loads(MANIFEST.read_text())
reports = {k: json.loads(p.read_text()) for k, p in CONFIRM.items()}


def lattice(n):
    if type(n) is not int or n < 1:
        return False
    while n % 2 == 0:
        n //= 2
    return n in (1, 3)


def large_core(m, n, k):
    return all(lattice(d) for d in (m, n, k)) and min(m, n, k) >= 8192


def deep_k(m, n, k):
    return all(lattice(d) for d in (m, n, k)) and m >= 2048 and n >= 2048 and k >= 16384


rules = {
    'large_core': large_core,
    'deep_k': deep_k,
    'union': lambda m, n, k: large_core(m, n, k) or deep_k(m, n, k),
}
result = {
    'status': 'post_hoc_candidate_not_validated',
    'grid': 'Every dimension is 2^i or 3*2^i for a nonnegative integer i',
    'large_core': 'min(M,N,K) >= 8192',
    'large_core_strassen_tile_bm_bn_bk': [2048, 2048, 512],
    'deep_k': 'M >= 2048 and N >= 2048 and K >= 16384',
    'deep_k_tile_policy': 'Screen-selected per-shape configuration; no fixed tile policy established',
    'runtime_scope': 'Complete call, including padding and crop; frozen kernels_v002, BF16 input, FP32 output',
    'validation': 'Existing measurements only. No new measurements, no latency pooling, no holdout evaluation.',
    'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (INPUT, MANIFEST, *CONFIRM.values())},
    'support': {},
}
for name, rule in rules.items():
    result['support'][name] = {}
    for cohort, rows in data.items():
        matched = [r for r in rows if rule(r['m'], r['n'], r['k'])]
        assert matched
        evidence = []
        for r in matched:
            assert r['on_lattice'] and r['native_ci_low'] > 1
            contrast = [p for p in reports[cohort]['by_shape'][r['shape_id']]['headline_comparisons']
                        if p['scope'] == 'call' and p['candidate_id'] == r['strassen_candidate']
                        and p['reference_candidate_id'] == 'native_default']
            assert len(contrast) == 1 and contrast[0]['speedup_ci95'][0] > 1
            if name == 'large_core':
                assert r['strassen_candidate'] == 'strassen_2048_2048_512'
            evidence.append({**{k: r[k] for k in ('m', 'n', 'k', 'shape_id', 'allocation_id', 'strassen_candidate', 'strassen_ms', 'native_ms', 'native_default_ms', 'native_ci_low', 'native_ci_high')},
                             'default_ci_low': contrast[0]['speedup_ci95'][0],
                             'default_ci_high': contrast[0]['speedup_ci95'][1]})
        def saved(baseline):
            values = [100 * (1 - r['strassen_ms'] / r[baseline + '_ms']) for r in matched]
            return [min(values), max(values)]
        result['support'][name][cohort] = {
            'observations': len(matched), 'ci_wins_vs_tuned_native': len(matched),
            'ci_wins_vs_default_native': len(matched),
            'time_saved_percent_range_tuned_native': saved('native'),
            'time_saved_percent_range_default_native': saved('native_default'),
            'sampling_roles': dict(Counter(r['sampling_role'] for r in matched)),
            'tiles': sorted(set(r['strassen_candidate'] for r in matched)), 'evidence': evidence,
        }

# Map coverage of a finite first validation box without opening any holdout results.
old_reserved = {tuple(r[d] for d in ('m', 'n', 'k')) for r in manifest['old_holdout_shapes']}
new_reserved = {tuple(r[d] for d in ('m', 'n', 'k')) for r in manifest['shapes'] if r['id'] in manifest['new_holdout_shape_ids']}
box = []
for shape in itertools.product((8192, 12288, 16384), repeat=3):
    observed_in = [c for c, rows in data.items() if any(tuple(r[d] for d in ('m', 'n', 'k')) == shape for r in rows)]
    reserved = 'legacy' if shape in old_reserved else 'new' if shape in new_reserved else None
    assert not (observed_in and reserved)
    box.append(dict(zip(('m', 'n', 'k'), shape), observed_in=observed_in, reserved=reserved))
assert len(box) == 27 and sum(bool(r['observed_in']) for r in box) == 8
result['finite_validation_box'] = {
    'dimensions': [8192, 12288, 16384], 'total_shapes': 27, 'observed_unique_shapes': 8,
    'unmeasured_shapes': 19, 'protected_legacy_holdouts': 1, 'unmeasured_unreserved_shapes': 18,
    'shapes': box, 'policy': 'Inventory only. Protected holdouts remain excluded and unmeasured.',
}
for rule, expected in [('large_core', (8, 2)), ('deep_k', (16, 5)), ('union', (21, 6))]:
    assert tuple(result['support'][rule][c]['observations'] for c in ('current', 'previous')) == expected
(OUT / 'candidate_region.json').write_text(json.dumps(result, indent=2) + '\n')
lines = [
    '# A sufficient-region candidate for Strassen', '',
    'Use A[M,K] times B[K,N], with each dimension restricted to 2^i or 3*2^i.', '',
    'The primary candidate is **min(M,N,K) >= 8192**, using Strassen tile **(BM,BN,BK) = (2048,2048,512)**.', '',
    'A separate extension is **M,N >= 2048 and K >= 16384**. Its evidence uses several screen-selected Strassen tiles, so it is not yet a fixed-configuration rule.', '',
    '| Region | Current observations / CI wins | Previous observations / CI wins | Current time saved vs tuned native | Previous time saved vs tuned native |',
    '|---|---:|---:|---:|---:|',
]
for name in rules:
    current, previous = (result['support'][name][c] for c in ('current', 'previous'))
    fmt = lambda r: ' to '.join(f'{x:.2f}%' for x in r['time_saved_percent_range_tuned_native'])
    lines.append(f"| {name.replace('_', ' ')} | {current['observations']} / {current['observations']} | {previous['observations']} / {previous['observations']} | {fmt(current)} | {fmt(previous)} |")
lines += ['',
    'Every matched observation also has a pointwise 95% CI wholly favoring Strassen over default native. Core and deep-K regions overlap; use the union row instead of adding their counts.', '',
    'This formalizes an empirical candidate, not a mathematical runtime guarantee or a validated unseen-shape dispatch policy. Thresholds were chosen after examining results. Confidence intervals are pointwise, with no multiplicity correction, and do not measure across-machine variability. Cohorts remain separate.', '',
    'The current core evidence consists of two repeat anchors and six focused shapes; the previous core evidence consists only of 8192^3 and 16384^3. Those are repeated geometries, not ten independent shapes. All core observations have dimensions between8192 and16384. Applying the lower-threshold rule beyond that range is extrapolation.', '',
    'A precise first validation domain is **M,N,K in {8192,12288,16384}**:27 distinct grid shapes. Eight have been measured in these probes;19 remain unmeasured. Of those,12288^3 is a protected legacy holdout and18 are unreserved. This audit does not compile, time or otherwise inspect held-out outcomes.', '',
    'The weaker min(M,N,K)>=4096 rule has an observed counterexample:4096^3 loses in the previous probe. Volume alone also fails on sufficiently thin rectangles. A rule can deliberately miss other Strassen wins; high-confidence coverage is the goal.', '',
    'For kernel use, the core tile is supported directly. The deep-K branch still requires a tile-selection policy frozen before new validation. Selecting the best tile from new test measurements would evaluate an oracle rather than a deployable rule.', '',
    'Matched shape identities, complete-call times, paired CI bounds, allocation labels, source hashes and the finite-box inventory are preserved in candidate_region.json. Original tables and experiments are unchanged.',
]
(OUT / 'REGION_RULE.md').write_text('\n'.join(lines) + '\n')
print(json.dumps({'status': 'passed', 'rules': {k: {c: v['observations'] for c, v in cs.items()} for k, cs in result['support'].items()}, 'finite_box': {'total': 27, 'observed': 8, 'unmeasured': 19, 'protected': 1}}))
