"""Human-readable evidence for the frozen joint tuner, without reselection."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def configuration(candidate):
    if candidate['depth'] == 0:
        return candidate['arm_id']
    bm, bn, bk = candidate['tile']
    return (f"S{candidate['depth']} / {candidate['accumulator']} / "
            f"{bm} × {bn} × {bk} / {candidate['buffers']} buffers")


def ratio(comparison):
    if not comparison or not comparison.get('eligible'):
        return 'ineligible or unavailable'
    lo, hi = comparison['ci95']
    return f"{comparison['speedup']:.4f}× [{lo:.4f}, {hi:.4f}]"


def interpretation(comparison):
    if not comparison or not comparison.get('eligible'):
        return 'No eligible timing conclusion; inspect the preserved failure or error record.'
    lo, hi = comparison['ci95']
    if lo > 1:
        return 'The selected configuration is faster in this matched comparison.'
    if hi < 1:
        return ('The alternative is faster in confirmation. The tuner keeps its frozen '
                'choice; this is evidence of screening uncertainty, not a reason to reselect.')
    return 'The interval includes 1; this comparison does not resolve a speed difference.'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    decisions = json.loads(args.decisions.read_text())
    if not results['completed'] or len(results['results']) != 12:
        raise ValueError('A complete six-shape, two-output-precision study is required.')
    fixture_only = any(r.get('identity', {}).get('test_fixture')
                       or 'FAKE' in r.get('identity', {}).get('device_kind', '')
                       for r in results['results'])
    output = args.output_dir or Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
    output.mkdir(parents=True, exist_ok=False)
    lines = [
        '# Why the joint tuner made these choices', '',
        'This document explains the executed tuner and each frozen decision. It does not '
        'change the recommendation table after looking at confirmation results.', '',
        'Matrix dimensions are **M × K × N**. Kernel tiles are **BM × BN × BK**: '
        'output rows, output columns, then contraction-panel length. Full contraction means '
        'BK = K; it does not mean the entire output matrix occupies one tile.', '',
        '## Design decisions made before measurement', '',
        '1. **Tune the complete configuration jointly.** Output tile, K-panel length, '
        'accumulator strategy and buffer count share the same memory and scheduling budget. '
        'Choose the fastest eligible combination rather than optimizing each knob in isolation.',
        '2. **Keep both accumulator strategies.** Products retains seven FP32 outer products '
        'and reconstructs at the end. Outputs updates four outer result quadrants as products '
        'arrive: directly in the FP32 output tile, or in four FP32 scratch quadrants before a '
        'final BF16 store. Smaller scratch storage is a candidate advantage, not a speed guarantee.',
        '3. **Preserve known controls and bound search cost.** Use the declared output-tile '
        'menu, full K, selected aligned divisors near K/2, K/4 and 1024, and every historical '
        'control panel. Record omitted divisors. Reject misaligned/nondivisible candidates; '
        'prune rough footprints above 160 MiB except historical controls. The common custom '
        'kernel allowance is 112 MiB. The rough estimate is deliberately not a compiler '
        'feasibility test; actual compilation failures are retained separately.',
        '4. **Match precision and timing scope.** FP32 output and BF16 output are independent '
        'comparisons. Inputs and Strassen pre-adds are BF16; products, accumulation and '
        'reconstruction are FP32. Final output conversion is included. Timings describe '
        'warmed, resident matrix multiplication calls, excluding compilation and transfers.',
        '5. **Make screening auditable.** Randomize batches of up to twelve custom candidates '
        'with a repeated Native_default anchor. Rank seven-round mean latency divided by '
        'that batch anchor. Exclude failed, incomplete and numerically ineligible records. '
        'Use a stable candidate ID to break exact ties. Normalization mitigates drift but '
        'cannot eliminate screening noise.',
        '6. **Freeze before confirmation.** Confirm Native, overall, each depth, and each '
        'depth/storage/short-or-full stratum on three fresh Gaussian inputs with thirty '
        'paired rounds per input. Include historical controls, a matched cubic kernel, '
        'and each selected depth with only storage or buffer count changed. Confirmation '
        'explains the choice and gates the saved recommendation; it does not select another finalist.',
        '7. **Use a declared acceptance rule.** Recommend the frozen custom winner only '
        'when every confirmation error gate passes, mean speedup over the frozen tuned '
        'Native is at least 1.01, and the paired 95% lower bound is above one. Otherwise '
        'recommend Native. This 1% speedup-ratio margin is a practical policy choice.',
        '8. **Limit the claim to the evidence.** These are six development shapes with '
        'Gaussian BF16 operands. Errors use sampled 128 × 128 output entries, with all '
        'K terms computed in FP64, plus full-output finiteness. The gate is relative L2 '
        '≤ 2% and maximum absolute error ≤ 0.001 + 0.05 × maximum absolute reference. '
        'Intervals are pointwise, conditional on screening, without multiplicity correction. '
        'The saved policy is a research artifact, not a general installed dispatcher or '
        'an LLM-quality guarantee.', '',
        '## Measured choices', '',
        'All speedups below are reference latency / selected latency. A 1.50× speedup '
        'means one-third less latency, not 50% less latency. Error values are the worst '
        'observed value across the three fresh confirmation inputs.', ''
    ]
    summary = {}
    for dtype in ('float32', 'bfloat16'):
        rows = [r for r in results['results'] if r['shape']['output_dtype'] == dtype]
        counts = Counter(r['recommendation']['status'] for r in rows)
        summary[dtype] = dict(recommendations=dict(counts), depth_choices={})
        lines += [f'### {dtype} output', '',
                  '| M × K × N | Native ms | S1 ms | S1 speedup | S2 ms | S2 speedup | Recommended |',
                  '|---|---:|---:|---:|---:|---:|---|']
        for r in rows:
            methods = r['methods']
            cells = [' × '.join(str(r['shape'][d]) for d in ('m', 'k', 'n')),
                     f"{methods['native']['mean_ms']:.4f}"]
            for role in ('s1', 's2'):
                m = methods.get(role)
                cells += ([f"{m['mean_ms']:.4f}", ratio(m['comparisons'].get('native'))]
                          if m and m['mean_ms'] is not None else ['unavailable', 'unavailable'])
            candidate = r['recommendation'].get('candidate')
            cells.append(configuration(candidate) if candidate else 'unavailable')
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines.append('')
        for depth in (1, 2):
            chosen = [r['methods'][f's{depth}']['candidate'] for r in rows
                      if f's{depth}' in r['methods']]
            summary[dtype]['depth_choices'][str(depth)] = dict(
                accumulators=dict(Counter(a['accumulator'] for a in chosen)),
                buffers=dict(Counter(a['buffers'] for a in chosen)))
    for r in results['results']:
        shape = r['shape']; methods = r['methods']; decision = r['recommendation']
        ledger = decisions[shape['id']]
        scores = ledger['candidate_scores']
        registry = ledger['registry']
        dispositions = Counter(c['disposition'] for c in registry['candidates'])
        exclusions = Counter(v['status'] for v in scores.values())
        lines += [f"## {shape['id']}", '',
                  'Matrix ' + ' × '.join(str(shape[d]) for d in ('m', 'k', 'n')) + '.', '',
                  f"Final rule: **{decision['status']}** — `{decision['reason']}`.", '',
                  f"Registered custom candidates: {dict(dispositions)}. Screening records "
                  f"including Native variants: {dict(exclusions)}. Offered K panels: "
                  f"{registry['offered_k_panels']}; omitted aligned K divisors: "
                  f"{registry['omitted_k_divisors']}.", '',
                  '| Algorithm | Relative L2 % | Max absolute | RMSE | Mean absolute | p50 absolute | p99 absolute | Normwise |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for role in ('native', 's1', 's2'):
            if role not in methods:
                continue
            errors = methods[role]['errors']
            values = [errors.get(key) for key in ('relative_l2', 'max_abs_error', 'rmse',
                      'mean_abs_error', 'p50_abs_error', 'p99_abs_error', 'normwise_error')]
            if values[0] is not None:
                values[0] *= 100
            lines.append('| ' + role + ' | ' + ' | '.join(
                'unavailable' if v is None else f'{v:.6g}' for v in values) + ' |')
        lines.append('')
        for depth in (1, 2):
            role = f's{depth}'
            if role not in methods:
                lines += [f'### S{depth}: no eligible screening candidate', '']; continue
            method = methods[role]; candidate = method['candidate']
            pool = sorted((v for v in scores.values() if v['status'] == 'eligible'
                           and v['candidate']['depth'] == depth),
                          key=lambda v: (v['score'], v['candidate']['arm_id']))
            lines += [f'### S{depth}: {configuration(candidate)}', '',
                      f"The tuner chose this combination because it had the lowest eligible "
                      f"batch-normalized screening latency among {len(pool)} candidates "
                      f"at this depth. BK={candidate['tile'][2]} gives "
                      f"{shape['k'] // candidate['tile'][2]} K panel(s). "
                      'This establishes the empirical selection reason; it does not isolate '
                      'a hardware cause for the chosen tile or K-panel length.', '',
                      '| Leading screening candidates | Normalized latency (lower is better) |',
                      '|---|---:|']
            for value in pool[:3]:
                lines.append(f"| {configuration(value['candidate'])} | {value['score']:.6f} |")
            lines += ['', 'Screening differences have no confirmation confidence interval. '
                      'A close screen ranking is not evidence that the first entry is '
                      'reliably faster than every other candidate.', '']
            for suffix, label in [('other_accumulator', 'Accumulator storage'),
                                  ('other_buffers', 'Buffer count')]:
                reference = role + '_' + suffix
                comparison = method['comparisons'].get(reference)
                other = methods.get(reference)
                lines.append(f"- **{label}:** alternative "
                             f"{configuration(other['candidate']) if other else 'unavailable'}. "
                             f"Selected speedup: {ratio(comparison)}. {interpretation(comparison)}")
            comparison = method['comparisons'].get('matched_' + role)
            lines.append(f"- **Same-tile cubic control:** {ratio(comparison)}. "
                         'This isolates the arithmetic-family comparison at that tile and '
                         'buffer count; tuned Native remains the main baseline.')
            historical = [(key, value) for key, value in methods.items()
                          if key.startswith('prior_') and value['candidate']['depth'] == depth]
            lines += ['', '| Replayed historical control | Selected speedup over control | Same configuration? |',
                      '|---|---|---|']
            for key, value in historical:
                same = candidate['arm_id'] == value['candidate']['arm_id']
                lines.append(f"| {key}: {configuration(value['candidate'])} | "
                             f"{ratio(method['comparisons'].get(key))} | {'yes' if same else 'no'} |")
            lines += ['', 'These comparisons use the controls rerun in this allocation. '
                      'An identical selected/control configuration is retained knowledge, '
                      'not a newly discovered kernel improvement.', '']
        lines += ['### Short versus full contraction finalists', '',
                  '| Depth / storage / panel regime | Frozen configuration | Confirmed ms | Speedup over Native |',
                  '|---|---|---:|---|']
        for depth in (1, 2):
            for mode in ('products', 'outputs'):
                for panel in ('short', 'full'):
                    role = f's{depth}_{mode}_{panel}'
                    method = methods.get(role)
                    if not method:
                        lines.append(f'| {role} | no eligible screening candidate | — | — |'); continue
                    ms = method['mean_ms']
                    displayed_ms = '—' if ms is None else f'{ms:.4f}'
                    lines.append(f"| {role} | {configuration(method['candidate'])} | "
                                 f"{displayed_ms} | {ratio(method['comparisons'].get('native'))} |")
        lines += ['', 'Each regime was allowed to choose its own tile and buffers. '
                  'This comparison answers which complete configuration worked better '
                  'within each regime, not the isolated causal effect of changing BK.', '']
    lines += ['## Evidence and reproducibility', '',
              f"Terminal outcome counts: `{results['case_status_counts']}`.", '',
              'The companion TUNER_DESIGN.md states the protocol, candidate_decisions.json '
              'contains every offered/pruned candidate and screen observation, results.json '
              'contains all confirmation roles and comparisons, and tuner_policy.json '
              'records the frozen recommendation and hardware/software identity. '
              'Compilation and timing archives retain unsuccessful candidates as well as winners.', '',
              'To attribute an observed benefit specifically to overlap, stalls, spills or '
              'matrix-unit utilization would require targeted compiler/profiler evidence. '
              'The matched measurements here support configuration choices without making '
              'those stronger causal claims.', '']
    if fixture_only:
        lines[:0] = ['# SYNTHETIC REPORT VALIDATION — NO TPU MEASUREMENTS', '',
                     'All numbers below are fabricated fixtures for checking report generation. '
                     'They are not experimental evidence.', '']
    (output / 'CHOICE_EXPLANATIONS.md').write_text('\n'.join(lines))
    provenance = dict(source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (args.results, args.decisions)}, summary=summary,
                      synthetic_fixture_only=fixture_only)
    (output / 'explanation_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(completed=True,output=str(output),summary=summary)))


if __name__ == '__main__':
    main()
