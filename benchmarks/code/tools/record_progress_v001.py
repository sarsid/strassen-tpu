"""Append a factual work update without changing prior progress records."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("activity", "agent", "model", "milestone", "decision"), required=True)
    parser.add_argument("--id", default="current")
    parser.add_argument("--state", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--detail", required=True)
    parser.add_argument("--next-step")
    parser.add_argument("--completed", type=int)
    parser.add_argument("--total", type=int)
    parser.add_argument("--evidence", action="append", default=[], help="Project-relative file")
    args = parser.parse_args()
    if args.completed is not None and args.completed < 0:
        parser.error("completed must be nonnegative")
    if args.total is not None and (args.total <= 0 or (args.completed or 0) > args.total):
        parser.error("total must be positive and at least completed")
    links = []
    for name in args.evidence:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or any(p.startswith('.') for p in path.relative_to(ROOT).parts):
            parser.error("evidence must be an existing nonprivate project file")
        links.append({"label": path.name, "path": str(path.relative_to(ROOT))})
    row = dict(kind=args.kind, id=args.id, name=args.title, title=args.title, state=args.state,
               detail=args.detail, utc=datetime.now(timezone.utc).isoformat(), evidence=links)
    if args.next_step:
        row['next_step'] = args.next_step
    if args.completed is not None:
        row['completed'] = args.completed
    if args.total is not None:
        row['total'] = args.total
    with (ROOT / 'status/work_progress_v001.jsonl').open('a') as handle:
        handle.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row))


if __name__ == '__main__':
    main()
