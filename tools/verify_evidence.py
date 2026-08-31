"""Verify every promoted Qwen3 artifact against its published SHA-256."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_DIR = REPOSITORY_ROOT / "evidence" / "qwen3"
INDEX = EVIDENCE_DIR / "README.md"
ROW = re.compile(
    r"`(?P<name>strassen_qwen3_[^`]+\.jsonl)`\s*\|\s*"
    r"`(?P<digest>[0-9a-f]{64})`"
)


def main() -> int:
    entries = ROW.findall(INDEX.read_text(encoding="utf-8"))
    documented = {name for name, _ in entries}
    actual = {path.name for path in EVIDENCE_DIR.glob("*.jsonl")}
    failures = []

    if len(documented) != len(entries):
        failures.append("duplicate artifact names in evidence index")
    for name, expected in entries:
        path = EVIDENCE_DIR / name
        if not path.is_file():
            failures.append(f"missing: {name}")
            continue
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            failures.append(
                f"hash mismatch: {name}: expected {expected}, got {observed}"
            )
    for name in sorted(actual - documented):
        failures.append(f"unindexed: {name}")
    for name in sorted(documented - actual):
        failures.append(f"indexed but absent: {name}")

    if failures:
        print("\n".join(failures))
        return 1
    print(f"{len(entries)} Qwen3 artifacts verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
