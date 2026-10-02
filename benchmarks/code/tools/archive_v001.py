"""Immutable execution directories with source and result commits.

Run local commands from a Git snapshot, not the mutable working directory.
Remote orchestration uses begin()/finish() with the same archive contract.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1]

def utc():
    return datetime.now(timezone.utc).isoformat()

def exclusive_json(path, obj):
    with Path(path).open("x") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

def commit(message):
    subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        subprocess.run(["git", "commit", "-m", message], cwd=ROOT, check=True)
    return git("rev-parse", "HEAD")

def begin(label, command=None, scope="local_validation"):
    source_commit = commit("Freeze source before " + label)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + label + "-" + uuid.uuid4().hex[:6]
    path = ROOT / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    source = path / "source"
    source.mkdir()
    roots = [name for name in git("ls-tree", "--name-only", "HEAD").splitlines()
             if name not in ("runs", "status")]
    archive = path / "source.tar"
    with archive.open("xb") as stream:
        subprocess.run(["git", "archive", source_commit, "--", *roots], cwd=ROOT,
                       stdout=stream, check=True)
    with tarfile.open(archive) as tf:
        tf.extractall(source, filter="data")
    manifest = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(source.rglob("*")) if p.is_file()}
    exclusive_json(path / "source-manifest.json", {"source_commit": source_commit,
                  "source_sha256": manifest,
                  "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()})
    exclusive_json(path / "execution.json", {"run_id": run_id, "label": label,
                  "created_utc": utc(), "source_commit": source_commit,
                  "scope": scope, "command": command})
    return path

def finish(path, status, **details):
    path = Path(path)
    exclusive_json(path / "completion.json", {"status": status, "finished_utc": utc(), **details})
    manifest = {str(p.relative_to(path)): {"bytes": p.stat().st_size,
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(path.rglob("*")) if p.is_file()}
    exclusive_json(path / "artifact-manifest.json", manifest)
    return commit("Archive " + path.name + " (" + status + ")")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        parser.error("a command is required after --")
    run = begin(args.label, cmd)
    print("EXECUTION_DIR=" + str(run), flush=True)
    env = dict(os.environ, PYTHONPATH=str(run / "source/src"), PYTHONUNBUFFERED="1")
    code = -1
    error = None
    try:
        with (run / "execution.log").open("x") as log:
            child = subprocess.Popen(cmd, cwd=run / "source", env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
            exclusive_json(run / "process.json", {"pid": child.pid, "started_utc": utc()})
            code = child.wait()
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    revision = finish(run, "completed" if code == 0 else "failed", exit_code=code, error=error)
    print(json.dumps({"run_id": run.name, "exit_code": code, "commit": revision}), flush=True)
    raise SystemExit(0 if code == 0 else 1)

if __name__ == "__main__":
    main()
