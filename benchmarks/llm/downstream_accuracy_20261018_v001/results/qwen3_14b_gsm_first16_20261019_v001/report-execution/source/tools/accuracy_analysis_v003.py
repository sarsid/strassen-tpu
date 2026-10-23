"""Small-subset paired accuracy/PPL analysis; refuses missing or mismatched evaluation pairs."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from scipy.stats import beta

ALPHA = 0.05
MARGINS_PP = {'mmlu': 0.5, 'gsm8k': 1.0}
PPL_MARGIN = 0.01


def strict_gsm8k(text):
    match = re.search(r'#### (\-?[0-9\.\,]+)', text)
    return None if match is None else normalize_answer(match.group(1))


def normalize_answer(value):
    return re.sub(r'\.$', '', str(value).replace(',', '').replace('$', '').strip().lower())


def cp_bounds(k, n, tail):
    return (0.0 if k == 0 else float(beta.ppf(tail, k, n-k+1)),
            1.0 if k == n else float(beta.ppf(1-tail, k+1, n-k)))


def preservation(ci, margin, higher_is_better):
    lo, hi = ci
    if higher_is_better:
        return 'preserved' if lo > -margin else 'regression beyond margin' if hi < -margin else 'inconclusive'
    return 'preserved' if hi < margin else 'regression beyond margin' if lo > margin else 'inconclusive'


def paired_accuracy(baseline, strassen, predictions_baseline, predictions_strassen, margin):
    b, s = np.asarray(baseline, bool), np.asarray(strassen, bool)
    if b.shape != s.shape or b.ndim != 1 or not b.size:
        raise ValueError('Expected nonempty paired correctness arrays')
    n = b.size
    gain, loss = int(np.sum(~b & s)), int(np.sum(b & ~s))
    gl, gu = cp_bounds(gain, n, ALPHA / 4)
    ll, lu = cp_bounds(loss, n, ALPHA / 4)
    ci = [100 * (gl-lu), 100 * (gu-ll)]
    changed = [x != y for x, y in zip(predictions_baseline, predictions_strassen, strict=True)]
    if len(changed) != n:
        raise ValueError('Prediction and correctness lengths differ')
    return dict(n=int(n), baseline_score=100*float(b.mean()), strassen_score=100*float(s.mean()),
                difference=100*(gain-loss)/n, difference_unit='percentage points', confidence_interval=ci,
                confidence_method='paired discordant-cell simultaneous Clopper-Pearson, four tails alpha/4',
                confidence_level=0.95, margin_pp=margin, criterion=preservation(ci, margin, True),
                both_correct=int(np.sum(b & s)), both_incorrect=int(np.sum(~b & ~s)),
                baseline_correct_strassen_incorrect=loss, baseline_incorrect_strassen_correct=gain,
                different_predictions=sum(changed), prediction_disagreement_rate=sum(changed)/n)


def paired_ppl(rows_b, rows_s, draws=20000, block_windows=8, seed=20261018):
    n = len(rows_b)
    if not n or n != len(rows_s):
        raise ValueError('Expected nonempty paired text windows')
    nb = np.array([x['nll_sum'] for x in rows_b], float)
    ns = np.array([x['nll_sum'] for x in rows_s], float)
    count = np.array([x['scored_tokens'] for x in rows_b], np.int64)
    other = np.array([x['scored_tokens'] for x in rows_s], np.int64)
    if not np.array_equal(count, other) or np.any(count <= 0) or np.any(nb < 0) or np.any(ns < 0):
        raise ValueError('Invalid or mismatched token counts/NLL')
    if not np.all(np.isfinite(nb)) or not np.all(np.isfinite(ns)):
        raise ValueError('Nonfinite NLL')
    loss_b, loss_s = nb.sum()/count.sum(), ns.sum()/count.sum()
    pb, ps = math.exp(loss_b), math.exp(loss_s)
    rng = np.random.default_rng(seed)
    width = min(block_windows, n)
    groups = math.ceil(n/width)
    # Circular moving blocks: draw starting positions, concatenate eight-window
    # blocks, and truncate to the original window count. Every window has equal
    # marginal sampling probability; both algorithms share identical indices.
    absolute, relative = np.empty(draws), np.empty(draws)
    for first in range(0, draws, 256):
        stop = min(draws, first+256)
        starts = rng.integers(n, size=(stop-first, groups, 1))
        idx = ((starts + np.arange(width)) % n).reshape(stop-first, -1)[:, :n]
        tokens = count[idx].sum(axis=1)
        lb, ls = nb[idx].sum(axis=1)/tokens, ns[idx].sum(axis=1)/tokens
        absolute[first:stop] = np.exp(ls)-np.exp(lb)
        relative[first:stop] = np.expm1(ls-lb)
    q = [ALPHA/2, 1-ALPHA/2]
    ci_abs, ci_rel = np.quantile(absolute, q).tolist(), np.quantile(relative, q).tolist()
    enough = n // block_windows >= 20
    return dict(n_windows=n, scored_tokens=int(count.sum()), baseline_score=pb, strassen_score=ps,
                difference=ps-pb, difference_unit='perplexity', confidence_interval=ci_abs,
                relative_change=(ps-pb)/pb, relative_confidence_interval=ci_rel,
                delta_nll_nats=float(loss_s-loss_b), confidence_level=0.95,
                confidence_method='paired circular moving-block bootstrap of token-weighted NLL',
                bootstrap_draws=draws, block_windows=block_windows, seed=seed,
                relative_margin=PPL_MARGIN, criterion=preservation(ci_rel,PPL_MARGIN,False) if enough else 'inconclusive',
                insufficient_text_blocks=not enough)


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    result = {}
    for row in rows:
        key = (row['task'], row['example_id'])
        if key in result:
            raise ValueError('Duplicate task/example ID: ' + str(key))
        result[key] = row
    return result


def analyze(manifest, baseline, strassen):
    if manifest.get('admission_status') != 'passed':
        raise ValueError('Inference admission has not passed; refuse scientific verdicts')
    b, s = read_rows(baseline), read_rows(strassen)
    expected = {(x['task'], x['example_id']):x for x in manifest['examples']}
    if len(expected) != len(manifest['examples']) or not expected or set(b) != set(s) or set(b) != set(expected):
        raise ValueError('Incomplete, duplicate or unequal evaluation example sets')
    tasks = defaultdict(list)
    for key, spec in expected.items():
        for rows, arm in ((b,'native'),(s,'strassen')):
            row = rows[key]
            for field in ('input_sha256','gold','subject'):
                if row.get(field) != spec.get(field):
                    raise ValueError('Changed ' + field + ': ' + str(key))
            if row['arm'] != arm or row['protocol_sha256'] != manifest['protocol_sha256']:
                raise ValueError('Wrong arm or protocol')
            if row['checkpoint_revision'] != manifest['checkpoint_revision'] or row['execution_contract_sha256'] != manifest['execution_contract_sha256']:
                raise ValueError('Changed checkpoint or execution contract')
        if key[0] == 'ppl':
            for rows in (b,s):
                if rows[key]['scored_tokens'] != spec['scored_tokens'] or rows[key]['window_index'] != spec['window_index']:
                    raise ValueError('Changed PPL mask/token count/order')
        elif key[0] == 'gsm8k':
            for rows in (b,s):
                expected_prediction = None if rows[key].get('stop_reason') == 'max_new_tokens' else strict_gsm8k(rows[key]['generated_text'])
                if rows[key]['prediction'] != expected_prediction:
                    raise ValueError('GSM8K strict extraction mismatch')
        elif key[0] != 'mmlu':
            raise ValueError('Unregistered task')
        tasks[key[0]].append(key)
    report = dict(model=manifest['model'], checkpoint_revision=manifest['checkpoint_revision'],
                  protocol_sha256=manifest['protocol_sha256'], tasks={}, subjects={}, disagreements=[])
    def accuracy(keys, task):
        gold = [b[k]['gold'] for k in keys]
        bp, sp = [b[k]['prediction'] for k in keys], [s[k]['prediction'] for k in keys]
        return paired_accuracy([x==y for x,y in zip(bp,gold)], [x==y for x,y in zip(sp,gold)], bp,sp,MARGINS_PP[task])
    for task, keys in tasks.items():
        if task == 'ppl':
            keys.sort(key=lambda k:b[k]['window_index'])
            if [b[k]['window_index'] for k in keys] != list(range(len(keys))):
                raise ValueError('PPL window ordering has gaps')
            report['tasks'][task] = paired_ppl([b[k] for k in keys],[s[k] for k in keys],block_windows=1)
            continue
        report['tasks'][task] = accuracy(keys,task)
        if task == 'mmlu':
            # The pilot samples two questions per subject; a pooled binomial
            # interval does not represent this stratified sampling design.
            # Hoeffding applies to bounded paired differences and independent
            # uniform sampling within strata (without replacement is conservative).
            result = report['tasks'][task]
            result['pooled_binomial_interval_descriptive_only'] = result['confidence_interval']
            half = 100*math.sqrt(2*math.log(2/ALPHA)/len(keys))
            ci = [max(-100.,result['difference']-half),min(100.,result['difference']+half)]
            result.update(confidence_interval=ci,confidence_method='distribution-free Hoeffding bound for equally stratified paired differences; conservative',criterion=preservation(ci,MARGINS_PP[task],True))
            for subject in sorted({b[k]['subject'] for k in keys}):
                result = accuracy([k for k in keys if b[k]['subject']==subject],task)
                result.pop('criterion'); result['scope']='descriptive subject interval, unadjusted'
                report['subjects'][subject] = result
        for k in keys:
            if b[k]['prediction'] != s[k]['prediction']:
                report['disagreements'].append(dict(task=task,example_id=k[1],subject=b[k].get('subject'),gold=b[k]['gold'],
                    baseline_prediction=b[k]['prediction'],strassen_prediction=s[k]['prediction'],
                    baseline_correct=b[k]['prediction']==b[k]['gold'],strassen_correct=s[k]['prediction']==b[k]['gold']))
    report['input_sha256'] = {name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [('native',baseline),('strassen',strassen)]}
    return report


def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True);p.add_argument('--strassen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    value=analyze(json.loads(a.manifest.read_text()),a.native,a.strassen)
    with a.output.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':main()
