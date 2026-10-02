"""Append status events without changing previous entries."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def emit(kind, **fields):
    event = {"utc": datetime.now(timezone.utc).isoformat(), "kind": kind, **fields}
    with (ROOT / "status/events.jsonl").open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return event

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind")
    parser.add_argument("--name", default="campaign")
    parser.add_argument("--state", required=True)
    parser.add_argument("--detail", required=True)
    args = parser.parse_args()
    print(json.dumps(emit(args.kind, name=args.name, state=args.state, detail=args.detail)))
