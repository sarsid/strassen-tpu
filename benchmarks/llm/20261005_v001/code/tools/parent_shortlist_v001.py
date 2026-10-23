"""Selection helpers for a separate, measured refinement of a frozen screen."""
import math
import statistics


def trimmed_mean(values):
    """Symmetric one-sample trimming proposes candidates; it never reports speedup."""
    if len(values) < 4 or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError('Expected positive finite latency samples')
    return statistics.fmean(sorted(values)[1:-1])


def robust_score(values):
    """Median of eight means, each spanning four interleaved measurement rounds."""
    if len(values) != 32 or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError('Expected 32 positive finite latency samples')
    return statistics.median(statistics.fmean(values[i:i+4]) for i in range(0,32,4))


def shortlist(events,site,dtype,profile):
    """Union of two raw-time leaders, two robust-ratio leaders, and prior winner.

    Native is added by the caller. Selection happens only in a new, interleaved
    experiment: these ranking heuristics are not accepted final tuning scores.
    """
    result={}
    for family in ('cubic','s1','s2'):
        rows=[e for e in events if e.get('kind')=='candidate' and e['site']==site
              and e.get('eligible') and e['arm']['output_dtype']==dtype and e['arm']['family']==family]
        for key in (lambda e:statistics.fmean(e['samples_ms']),
                    lambda e:trimmed_mean(e['samples_ms'])/trimmed_mean(e['native_control_samples_ms'])):
            for row in sorted(rows,key=key)[:2]:
                candidate=row['arm'];result[candidate['arm_id']]=candidate
        previous=profile['profiles'][dtype][family][site]
        if previous['implementation']!='native':result[previous['arm_id']]=previous
    return [result[k] for k in sorted(result)]
