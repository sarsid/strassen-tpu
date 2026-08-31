"""Check local Markdown links in the public-facing documentation."""

from __future__ import annotations

from pathlib import Path
import re
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "docs" / "RESULTS.md",
    REPOSITORY_ROOT / "docs" / "COLAB_RUNBOOK.md",
    REPOSITORY_ROOT / "experiments" / "qwen3" / "README.md",
    REPOSITORY_ROOT / "evidence" / "qwen3" / "README.md",
    REPOSITORY_ROOT / "archive" / "initial-snapshot" / "README.md",
)
LINK = re.compile(r"\[[^]]*\]\(([^)]+)\)")


def main() -> int:
    failures = []
    for document in DOCUMENTS:
        for target in LINK.findall(document.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            path = (document.parent / target.split("#", 1)[0]).resolve()
            if not path.exists():
                failures.append(f"{document.relative_to(REPOSITORY_ROOT)}: {target}")
    if failures:
        print("broken local links:\n" + "\n".join(failures))
        return 1
    print(f"local links verified in {len(DOCUMENTS)} documents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
