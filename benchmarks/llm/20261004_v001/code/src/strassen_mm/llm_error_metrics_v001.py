"""Prediction and numerical errors, with per-token evidence and paired intervals.

JAX is imported only when the caller installs the device comparison function.
All probability metrics compare the same vocabulary and true next-token IDs.
"""
import math
import numpy as np


def make_comparator(jax, jnp):
    @jax.jit
    def compare(reference, candidate, targets):
        p_logits = reference.astype(jnp.float32)
        q_logits = candidate.astype(jnp.float32)
        lp = jax.nn.log_softmax(p_logits, axis=-1)
        lq = jax.nn.log_softmax(q_logits, axis=-1)
        p, q = jnp.exp(lp), jnp.exp(lq)
        lm = jnp.logaddexp(lp, lq) - jnp.log(2.)
        ix = jnp.arange(targets.shape[0])
        pv, pi = jax.lax.top_k(p_logits, 5)
        qv, qi = jax.lax.top_k(q_logits, 5)
        p1, q1 = jnp.argmax(p_logits, -1), jnp.argmax(q_logits, -1)
        diff = q_logits - p_logits
        return dict(
            target=targets, reference_prediction=p1, candidate_prediction=q1,
            reference_nll=-lp[ix, targets], candidate_nll=-lq[ix, targets],
            reference_top1_correct=p1 == targets, candidate_top1_correct=q1 == targets,
            reference_top5_correct=jnp.any(pi == targets[:, None], -1),
            candidate_top5_correct=jnp.any(qi == targets[:, None], -1),
            agreement=p1 == q1, reference_confidence=p[ix, p1], candidate_confidence=q[ix, q1],
            reference_margin=pv[:, 0] - pv[:, 1], candidate_margin=qv[:, 0] - qv[:, 1],
            kl_forward=jnp.sum(p * (lp - lq), -1), kl_reverse=jnp.sum(q * (lq - lp), -1),
            js=.5 * jnp.sum(p * (lp - lm) + q * (lq - lm), -1),
            total_variation=.5 * jnp.sum(jnp.abs(p - q), -1),
            reference_brier=jnp.sum(p * p, -1) - 2 * p[ix, targets] + 1,
            candidate_brier=jnp.sum(q * q, -1) - 2 * q[ix, targets] + 1,
            logit_squared_error=jnp.sum(diff * diff, -1),
            logit_absolute_error=jnp.sum(jnp.abs(diff), -1),
            logit_max_absolute_error=jnp.max(jnp.abs(diff), -1),
            reference_squared_norm=jnp.sum(p_logits * p_logits, -1),
            candidate_squared_norm=jnp.sum(q_logits * q_logits, -1),
            logit_dot_product=jnp.sum(p_logits * q_logits, -1),
            finite=jnp.all(jnp.isfinite(p_logits) & jnp.isfinite(q_logits), -1))
    return compare


def make_hidden_comparator(jax, jnp):
    @jax.jit
    def compare(reference, candidate):
        r, c = reference.astype(jnp.float32), candidate.astype(jnp.float32)
        d = c - r
        r2, c2, e2 = jnp.sum(r*r), jnp.sum(c*c), jnp.sum(d*d)
        return dict(relative_l2=jnp.sqrt(e2/jnp.maximum(r2, 1e-30)),
                    rmse=jnp.sqrt(jnp.mean(d*d)), mean_abs_error=jnp.mean(jnp.abs(d)),
                    max_abs_error=jnp.max(jnp.abs(d)),
                    cosine_similarity=jnp.sum(r*c)/jnp.maximum(jnp.sqrt(r2*c2), 1e-30),
                    finite=jnp.all(jnp.isfinite(r)) & jnp.all(jnp.isfinite(c)))
    return compare


def expected_calibration_error(confidence, correct, bins=15):
    confidence, correct = np.asarray(confidence), np.asarray(correct)
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    result, rows = 0., []
    for i in range(bins):
        mask = index == i
        if not mask.any():
            continue
        conf, accuracy = float(confidence[mask].mean()), float(correct[mask].mean())
        count = int(mask.sum())
        result += count / len(confidence) * abs(conf - accuracy)
        rows.append(dict(bin=i, count=count, confidence=conf, accuracy=accuracy))
    return dict(value=float(result), bins=bins, bin_statistics=rows)


def paired_window_ci(values, windows, seed=922):
    """Resample whole text windows, never pretend adjacent tokens are IID."""
    unique = np.unique(windows)
    means = np.asarray([np.mean(values[windows == w]) for w in unique])
    rng = np.random.default_rng(seed)
    estimates = means[rng.integers(0, len(means), (2000, len(means)))].mean(1)
    return dict(mean=float(np.mean(values)), ci95=np.quantile(estimates, [.025, .975]).tolist(),
                window_count=len(unique), scope='Paired equal-length text-window bootstrap; one corpus and hardware session')


def summarize(data, vocab_size):
    d = {k: np.asarray(v) for k, v in data.items()}
    count = len(d['target'])
    if count == 0 or any(len(v) != count for v in d.values()):
        raise ValueError('Empty or inconsistent per-token metrics')
    finite = bool(d['finite'].all())
    output = dict(positions=count, all_finite=finite, nonfinite_positions=int((~d['finite']).sum()),
                  vocabulary_size=int(vocab_size))
    if not finite:
        return dict(output, passed=False, reason='Nonfinite logits; retain token evidence without valid aggregate claims')
    nll = float(d['candidate_nll'].mean()); ref_nll = float(d['reference_nll'].mean())
    e2 = float(d['logit_squared_error'].astype(np.float64).sum())
    r2 = float(d['reference_squared_norm'].astype(np.float64).sum())
    c2 = float(d['candidate_squared_norm'].astype(np.float64).sum())
    ref_correct, cand_correct = d['reference_top1_correct'], d['candidate_top1_correct']
    delta = d['candidate_nll'] - d['reference_nll']
    output.update(candidate_nll=nll, reference_nll=ref_nll, nll_delta=nll-ref_nll,
        candidate_perplexity=math.exp(nll), reference_perplexity=math.exp(ref_nll),
        candidate_top1_accuracy=float(cand_correct.mean()), reference_top1_accuracy=float(ref_correct.mean()),
        candidate_top5_accuracy=float(d['candidate_top5_correct'].mean()),
        reference_top5_accuracy=float(d['reference_top5_correct'].mean()),
        top1_agreement=float(d['agreement'].mean()),
        correct_to_wrong=int((ref_correct & ~cand_correct).sum()),
        wrong_to_correct=int((~ref_correct & cand_correct).sum()),
        mean_kl=float(d['kl_forward'].mean()), mean_reverse_kl=float(d['kl_reverse'].mean()),
        mean_js=float(d['js'].mean()), mean_total_variation=float(d['total_variation'].mean()),
        mean_brier=float(d['candidate_brier'].mean()), reference_mean_brier=float(d['reference_brier'].mean()),
        calibration=expected_calibration_error(d['candidate_confidence'], cand_correct),
        reference_calibration=expected_calibration_error(d['reference_confidence'], ref_correct),
        relative_l2=math.sqrt(e2/max(r2, 1e-30)), rmse=math.sqrt(e2/(count*vocab_size)),
        mean_abs_error=float(d['logit_absolute_error'].astype(np.float64).sum()/(count*vocab_size)),
        max_abs_error=float(d['logit_max_absolute_error'].max()),
        logit_cosine=float(d['logit_dot_product'].astype(np.float64).sum()/max(math.sqrt(r2*c2), 1e-30)),
        nll_delta_quantiles=dict(zip(['p01','p05','p50','p95','p99'], np.quantile(delta,[.01,.05,.5,.95,.99]).tolist())),
        token_max_logit_error_quantiles=dict(zip(['p50','p95','p99'], np.quantile(d['logit_max_absolute_error'],[.5,.95,.99]).tolist())),
        paired_nll_delta=paired_window_ci(delta, d['window']),
        paired_accuracy_delta=paired_window_ci(cand_correct.astype(float)-ref_correct.astype(float), d['window']))
    thresholds=dict(max_abs_nll_delta=.01, max_mean_kl=.02, min_top1_agreement=.97)
    output.update(thresholds=thresholds,
        passed=abs(output['nll_delta']) <= .01 and output['mean_kl'] <= .02 and output['top1_agreement'] >= .97)
    return output
