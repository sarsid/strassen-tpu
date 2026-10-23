"""Exercise the progress feed against overlapping, partial and session evidence."""
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'status'))
from llm_progress_v001 import EventHistory, summarize


def main():
    checks = []
    def check(condition, label):
        assert condition, label
        checks.append(label)
    event = dict(kind='confirmation', utc='2026-10-03T12:00:00+00:00', output_dtype='bfloat16',
                 algorithm='s1', round=12, elapsed_ms=23., host_preparation_seconds=4.)
    later = dict(event, utc='2026-10-03T12:01:20+00:00', round=13, elapsed_ms=25.)
    fp32 = dict(later, utc='2026-10-03T12:02:40+00:00', output_dtype='float32', round=0)
    snapshot = lambda events, case=None: {'tails': {('artifacts/cases/' + case if case else 'artifacts') + '/main-prefill/events.jsonl': 'truncated line\n' + '\n'.join(json.dumps(e) for e in events)}}
    with tempfile.TemporaryDirectory() as folder:
        run = Path(folder) / 'run';(run / 'control').mkdir(parents=True)
        payload = json.dumps(snapshot([event, later])) + '\n'
        path = run / 'control/poll-a.log'
        path.write_text('\n'.join(json.dumps(dict(kind='remote_stream', stream='stdout', text=t)) for t in (payload[:40], payload[40:])))
        history = EventHistory()
        events = history.read(run, snapshot([later]))
        check(len(events) == 2, 'overlapping snapshots deduplicated, chunked stdout joined')
        check(len(history.read(run, snapshot([event]))) == 2, 'repeated or out-of-order polls preserve history')
        result = summarize(events, {'model':'qwen', 'batch':1, 'sequence':512}, {'layers':40}, {}, now=1791029000)
        check(result['observed_rounds'] == 2, 'highest round index never fabricates earlier samples')
        check(result['tracks'][0]['arms'][3]['median_ms'] == 24., 'provisional medians use observed samples')
        check(result['sample_cadence_seconds'] == 80., 'cadence uses successive timestamps')
        result = summarize(history.read(run, snapshot([fp32])), {}, {}, {})
        check([t['observed_rounds'] for t in result['tracks']] == [2,1], 'BF16 history survives FP32 transition')
        path.write_text(path.read_text() + '\n' + json.dumps(dict(kind='remote_stream', stream='stdout', text=json.dumps(snapshot([dict(fp32, round=1)])))))
        check(len(history.read(run, {})) == 4, 'growing poll file is re-read')
        check(history.read(run, snapshot([later], 'case-a'), 'case-b') == [], 'session cases isolated')
        check(len(history.read(run, snapshot([later], 'case-b'), 'case-b')) == 1, 'session case events selected')
        run2 = Path(folder) / 'retry';run2.mkdir()
        check(history.read(run2, {}) == [], 'new runtime retry does not inherit prior rounds')
        check(summarize([], {}, {}, {})['event_age_seconds'] is None, 'missing event freshness is unknown')
        result = summarize([dict(event, round=99), dict(event, round=-1), dict(event, algorithm='other')], {}, {}, {})
        check(result['observed_rounds'] == 0, 'invalid algorithms and round indices excluded')
        quality = dict(kind='heldout_quality', utc=event['utc'], output_dtype='bfloat16', algorithm='s1', quality={'finite':True}, hidden_errors=[{'large':'payload'}])
        resident = dict(kind='resident_layer', utc=later['utc'], output_dtype='bfloat16', algorithm='s1', layer=0, median_ms=.1)
        result = summarize([quality,resident], {}, {}, {})
        check(result['tracks'][0]['arms'][3]['quality'] == {'finite':True}, 'quality reports preserved separately from completion')
        check(result['observed_rounds'] == 0 and result['tracks'][0]['arms'][3]['resident_layers_observed'] == 1, 'resident records never count as full-forward samples')
        check('hidden_errors' not in result['recent_events'][1], 'large per-layer arrays excluded from feed')
        check(result['expected_rounds'] == 150, 'five algorithms times fifteen rounds times two precisions')
    return checks


if __name__ == '__main__':
    print(json.dumps({'status':'completed', 'checks':main()}))
