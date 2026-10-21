"""Execute a hash-verified uploaded bootstrap with a real remote __file__."""
import argparse
import hashlib
from pathlib import Path
import runpy
import sys

BASE = Path("/content/Strassen_MM_Focus")


def execute(path, expected_sha, forwarded):
    path = path.resolve()
    if not path.is_relative_to(BASE.resolve()) or not path.is_file():
        raise ValueError("Expected an uploaded bootstrap inside the experiment root")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("Uploaded bootstrap hash mismatch")
    previous = sys.argv
    try:
        sys.argv = [str(path), *forwarded]
        return runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", required=True, type=Path)
    parser.add_argument("--bootstrap-sha256", required=True)
    args, forwarded = parser.parse_known_args()
    execute(args.bootstrap, args.bootstrap_sha256, forwarded)


if __name__ == "__main__":
    main()
