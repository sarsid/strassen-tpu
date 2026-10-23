"""Read-only workload progress reconstructed from saved controller snapshots.

Poll tails overlap and may start mid-line. Deduplicate observed events; never
turn a highest round index into an invented sample count. Session case paths
isolate workloads and run paths isolate retries. This does not contact a TPU.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import PurePosixPath
import statistics
import threading

ALGORITHMS = ('native_default', 'native_tuned', 'cubic_tuned', 's1', 's2')
DTYPES = ('bfloat16', 'float32')
ROUNDS = 15  # Frozen 700-comparison campaign protocol.


def json_lines(text):
    for line in text.splitlines():
        try:
            item = json.loads(line)
            if isinstance(item, dict):
                yield item
        except (ValueError, TypeError):
            pass


def timestamp(value):
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return dt.timestamp() if dt.tzinfo else None
    except (AttributeError, ValueError, TypeError):
        return None


class EventHistory:
    def __init__(self):
        self.lock = threading.Lock()
        self.run = None
        self.files = {}
        self.events = {}

    def ingest(self, snapshot):
        for path, text in snapshot.get('tails', {}).items():
            if not path.endswith('/main-prefill/events.jsonl') or not isinstance(text, str):
                continue
            parts = PurePosixPath(path).parts
            case = parts[parts.index('cases') + 1] if 'cases' in parts else None
            for event in json_lines(text):
                if not event.get('kind') or timestamp(event.get('utc')) is None:
                    continue
                key = (case, json.dumps(event, sort_keys=True))
                self.events[key] = event

    def read(self, run, latest, case_id=None):
        with self.lock:
            if self.run != run:
                self.run, self.files, self.events = run, {}, {}
            for path in (run / 'control').glob('poll-*.log'):
                try:
                    stat = path.stat()
                    signature = (stat.st_size, stat.st_mtime_ns)
                    if self.files.get(path) == signature:
                        continue
                    text = path.read_text()
                except OSError:
                    continue
                # A command may deliver its stdout in more than one chunk.
                chunks = [row.get('text', '') for row in json_lines(text)
                          if row.get('kind') == 'remote_stream' and row.get('stream') == 'stdout']
                for snapshot in json_lines(''.join(chunks)):
                    self.ingest(snapshot)
                self.files[path] = signature
            self.ingest(latest)
            return sorted((event for (case, _), event in self.events.items() if case == case_id),
                          key=lambda event: timestamp(event['utc']))


def summarize(events, job, model, remote, now=None, session=False):
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    confirmations = {}
    for event in events:
        index = event.get('round')
        if (event.get('kind') == 'confirmation' and event.get('algorithm') in ALGORITHMS
                and event.get('output_dtype') in DTYPES and isinstance(index, int)
                and not isinstance(index, bool) and 0 <= index < ROUNDS):
            confirmations[(event['output_dtype'], event['algorithm'], index)] = event
    latest = events[-1] if events else None
    tracks = []
    for dtype in DTYPES:
        subset = [e for e in events if e.get('output_dtype') == dtype]
        arms = []
        for algorithm in ALGORITHMS:
            samples = [e for (dt, alg, _), e in confirmations.items() if (dt, alg) == (dtype, algorithm)]
            samples.sort(key=lambda e: timestamp(e['utc']))
            quality = next((e.get('quality') for e in reversed(subset)
                            if e['kind'] == 'heldout_quality' and e.get('algorithm') == algorithm), None)
            resident = {e['layer'] for e in subset if e['kind'] == 'resident_layer'
                        and e.get('algorithm') == algorithm and isinstance(e.get('layer'), int)}
            saved = next((e for e in reversed(subset) if e['kind'] == 'comparison_complete'
                          and e.get('algorithm') == algorithm), None)
            arms.append(dict(algorithm=algorithm, observed_rounds=len(samples), expected_rounds=ROUNDS,
                             last_round=samples[-1]['round'] + 1 if samples else None,
                             last_ms=samples[-1].get('elapsed_ms') if samples else None,
                             median_ms=statistics.median(e['elapsed_ms'] for e in samples) if samples else None,
                             quality=quality, resident_layers_observed=len(resident),
                             result_emitted=bool(saved), quality_gate_passed=saved.get('quality_gate_passed') if saved else None))
        tracks.append(dict(output_dtype=dtype, arms=arms,
                           observed_rounds=sum(a['observed_rounds'] for a in arms), expected_rounds=ROUNDS * 5,
                           tuning_started=any(e['kind'] == 'tuning_started' for e in subset),
                           profiles_frozen=any(e['kind'] == 'profiles_frozen' for e in subset),
                           candidates_observed=sum(e['kind'] == 'candidate' for e in subset),
                           quality_checks_observed=sum(a['quality'] is not None for a in arms)))
    samples = sorted(confirmations.values(), key=lambda e: timestamp(e['utc']))
    gaps = [timestamp(b['utc']) - timestamp(a['utc']) for a, b in zip(samples, samples[1:])
            if a['output_dtype'] == b['output_dtype']]
    last_sample = samples[-1] if samples else None
    gates = {e['layer']: e for e in events if e['kind'] == 'official_checkpoint_layer_gate'}
    reused = next((e for e in reversed(events) if e['kind'] == 'official_checkpoint_gate_reused'), None)
    # Whitelist small display fields; layer error arrays stay in original evidence.
    fields = ('kind', 'utc', 'algorithm', 'family', 'output_dtype', 'round', 'layer', 'site',
              'elapsed_ms', 'resident_layer_sum_ms', 'host_preparation_seconds', 'host_cache',
              'median_ms', 'compile_seconds', 'status', 'passed', 'quality_gate_passed',
              'bitwise_full_model_equal', 'cached_host_activation_seconds', 'original_host_setup_seconds')
    compact = lambda e: {k: e[k] for k in fields if k in e}
    first = events[0]['utc'] if events else None
    return dict(job_id=job.get('job_id'), model=job.get('model'), model_name=model.get('name', job.get('model')),
                batch=job.get('batch'), sequence=job.get('sequence'), layers=model.get('layers'), session=session,
                reported_stage=remote.get('active'), started_utc=first if session else remote.get('started_utc'),
                start_scope='measurement worker' if session else 'workload runtime',
                last_event=compact(latest) if latest else None,
                event_age_seconds=max(0, now - timestamp(latest['utc'])) if latest else None,
                last_measurement=compact(last_sample) if last_sample else None,
                sample_cadence_seconds=statistics.median(gaps[-10:]) if gaps else None,
                tracks=tracks, observed_rounds=len(confirmations), expected_rounds=ROUNDS * 10,
                official_layers_observed=len(gates), official_layers_failed=sum(not e.get('passed') for e in gates.values()),
                official_layers_reused=reused.get('layers') if reused else None,
                events_observed=len(events), recent_events=[compact(e) for e in reversed(events[-12:])],
                recent_measurements=[compact(e) for e in reversed(samples[-20:])],
                history_scope='Observed events recovered from local poll logs; gaps are possible between snapshots.')
