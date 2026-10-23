"""Append dated, reviewable decisions without rewriting prior entries."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def record(decision_id, decision, rationale, evidence, implications):
    directory = ROOT / 'decisions'
    directory.mkdir(exist_ok=True)
    value = {'utc': datetime.now(timezone.utc).isoformat(), 'id': decision_id,
             'decision': decision, 'rationale': rationale, 'evidence': evidence,
             'implications': implications}
    with (directory / 'N5_N9_v001.jsonl').open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    with (directory / 'N5_N9_v001.md').open('a') as stream:
        stream.write(f"\n## {decision_id} — {value['utc']}\n\nDecision: {decision}\n\nWhy: {rationale}\n\nEvidence: {evidence}\n\nConsequences: {implications}\n")
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('id', 'decision', 'rationale', 'evidence', 'implications'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(record(args.id, args.decision, args.rationale, args.evidence, args.implications)))
