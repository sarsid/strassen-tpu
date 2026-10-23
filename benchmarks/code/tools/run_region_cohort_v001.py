"""Freeze and run a separate region-grid cohort without changing old evidence.

The existing transport and TPU launcher are reused with an exclusive archive
adapter. This deliberately bypasses their global git-add and dashboard helpers.
All device phases use the SAME source archive. No automatic machine replacement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "configs/generated_region_grid_v001/campaign_region_grid_v001.json"
PHASES = ("GRID-smoke", "GRID-screen", "GRID-confirm")


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def seal(directory, name="artifact-manifest.json"):
    write(directory / name, {"created_utc": utc(), "sha256": {
        str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}})


def verify_frozen(cohort):
    frozen = read(cohort / "frozen.json")
    meta = read(cohort / "cohort.json")
    for field in ("cohort_id", "allocation_id", "session"):
        if meta.get(field) != frozen.get(field):
            raise ValueError("Cohort identity changed: " + field)
    if sha(cohort / "source.tar") != frozen["archive_sha256"]:
        raise ValueError("Frozen archive changed")
    for name, value in frozen["source_sha256"].items():
        if sha(cohort / "source" / name) != value:
            raise ValueError("Frozen source changed: " + name)
    return frozen


def freeze(cohort):
    """Capture committed baseline code plus an explicit new-file allowlist."""
    meta = read(cohort / "cohort.json")
    if (cohort / "frozen.json").exists() or (cohort / "source").exists():
        raise FileExistsError("This cohort already has a source snapshot")
    baseline = meta["historical_git_head_at_start"]
    source = cohort / "source"
    source.mkdir()
    # An immutable Git revision protects against simultaneous working-tree edits.
    baseline_tar = cohort / "baseline-source.tar"
    with baseline_tar.open("xb") as stream:
        subprocess.run(["git", "archive", baseline, "--", "src", "tools", "runtime", "configs"],
                       cwd=ROOT, stdout=stream, check=True)
    with tarfile.open(baseline_tar) as archive:
        archive.extractall(source, filter="data")
    additions = [
        "src/strassen_mm/benchmark_region_grid_v001.py",
        "tools/build_region_grid_v001.py", "tools/fit_region_margin_v001.py",
        "tools/run_region_cohort_v001.py", "tools/summarize_region_cohort_v001.py",
        "protocols/REGION_GRID_v001.md", "docs/REGION_MARGIN_MODEL_v001.md",
        "docs/REGION_GRID_RUNNER_REVIEW_v001.md",
        "tests/test_region_grid_v001.py", "tests/test_region_margin_v001.py",
        "tests/test_region_runner_v001.py", "tests/test_region_cohort_v001.py",
    ]
    additions += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "configs/generated_region_grid_v001").iterdir()) if p.is_file()]
    for relative in additions:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with (ROOT / relative).open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
    config = read(source / CAMPAIGN)
    write(source / "region-cohort.json", {**meta, "campaign_id": config["campaign_id"]})
    with (cohort / "source.tar").open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for item in sorted(source.iterdir()):
                archive.add(item, arcname=item.name)
    frozen = {"created_utc": utc(), "baseline_commit": baseline,
              "new_files": additions, "campaign_relative": CAMPAIGN,
              "archive_sha256": sha(cohort / "source.tar"),
              "source_sha256": {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()},
              "cohort_id": meta["cohort_id"], "allocation_id": meta["allocation_id"], "session": meta["session"],
              "pooling": "forbidden; one allocation only"}
    write(cohort / "frozen.json", frozen)
    print(json.dumps({"status": "frozen", "cohort": str(cohort), "source_archive_sha256": frozen["archive_sha256"]}), flush=True)


class CohortArchive:
    """Only these adapter methods are consumed by run_phase_v004."""
    def __init__(self, cohort, phase):
        self.cohort, self.phase, self.ROOT = cohort, phase, cohort
        self.frozen = verify_frozen(cohort)
        self.run = None

    utc = staticmethod(utc)
    exclusive_json = staticmethod(write)

    def begin(self, label, command=None, scope=None):
        parent = self.cohort / "phases"
        parent.mkdir(exist_ok=True)
        run_id = self.frozen["cohort_id"] + "-" + self.phase.lower() + "-" + uuid.uuid4().hex[:6]
        run = parent / run_id
        run.mkdir(exist_ok=False)
        with (run / "source.tar").open("xb") as dst, (self.cohort / "source.tar").open("rb") as src:
            shutil.copyfileobj(src, dst)
        with tarfile.open(run / "source.tar") as packed:
            packed.extractall(run / "source", filter="data")
        write(run / "execution.json", {"run_id": run_id, "phase": self.phase, "created_utc": utc(),
              "cohort_id": self.frozen["cohort_id"], "allocation_id": self.frozen["allocation_id"],
              "source_archive_sha256": self.frozen["archive_sha256"], "source_commit": self.frozen["baseline_commit"],
              "new_source_hashes": "../../frozen.json", "scope": scope})
        self.run = run
        write(self.cohort / (self.phase + "-started.json"), {"run": str(run), "run_id": run_id, "started_utc": utc()})
        return run

    def finish(self, path, status, **details):
        write(path / "completion.json", {"status": status, "finished_utc": utc(), **details})
        seal(path)
        write(self.cohort / (self.phase + "-finished.json"), {"status": status, "run": str(path),
              "run_id": path.name, "finished_utc": utc(), "completion_sha256": sha(path / "completion.json")})
        return "sha256:" + sha(path / "artifact-manifest.json")


def run_phase(cohort, phase, controller_python, expected_identity, timeout):
    meta = read(cohort / "cohort.json")
    adapter = CohortArchive(cohort, phase)
    tools = cohort / "source/tools"
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location("region_transport", tools / "run_phase_v004.py")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    transport.archive = adapter
    transport.emit = lambda *a, **k: print(json.dumps({"kind": "region_progress", **k}), flush=True)
    argv = ["run_phase_v004.py", "--phase", phase, "--session", meta["session"], "--endpoint", meta["allocation_id"],
            "--expected-identity", expected_identity, "--controller-python", controller_python,
            "--campaign-relative", CAMPAIGN, "--benchmark-module", "strassen_mm.benchmark_region_grid_v001",
            "--timeout-seconds", str(timeout)]
    if phase == "GRID-confirm":
        prior = read(cohort / "GRID-screen-finished.json")
        if prior["status"] != "completed":
            raise ValueError("Screening must complete and be retrieved before confirmation")
        argv += ["--runner-arg=--selection", "--runner-arg=/content/Strassen_MM_Focus/runs/" + prior["run_id"] + "/artifacts/selections.json"]
    original = sys.argv
    try:
        sys.argv = argv
        return transport.main()
    finally:
        sys.argv = original


def run(cohort, controller_python, expected_identity, analysis_python):
    verify_frozen(cohort)
    meta = read(cohort / "cohort.json")
    write(cohort / "controller-started.json", {"pid": os.getpid(), "started_utc": utc(), "phases": PHASES,
          "device_timeout_seconds": {"GRID-smoke": 1800, "GRID-screen": 36000, "GRID-confirm": 21600},
          "cohort_id": meta["cohort_id"], "allocation_id": meta["allocation_id"]})
    status, error = "failed", None
    try:
        for phase, timeout in zip(PHASES, (1800, 36000, 21600)):
            cmd = [sys.executable, str(cohort / "source/tools/run_region_cohort_v001.py"), "phase",
                   "--cohort", str(cohort), "--phase", phase, "--controller-python", controller_python,
                   "--expected-identity", expected_identity, "--timeout", str(timeout)]
            with (cohort / (phase + "-controller.log")).open("x") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(phase + " failed; inspect its exclusive controller log")
            record = read(cohort / (phase + "-finished.json"))
            summary = read(Path(record["run"]) / "artifacts/summary.json")
            if not summary.get("completed"):
                raise RuntimeError(phase + " did not complete all planned groups")
            if phase == "GRID-smoke" and any(k != "ok" and v for k, v in summary["case_status_counts"].items()):
                raise RuntimeError("Smoke has non-ok outcomes; stop before screening")
        # Every device artifact has now been copied and verified locally.
        release = [controller_python, str(cohort / "source/runtime/release_allocation_v001.py"),
                   "--session", meta["session"], "--expect-endpoint", meta["allocation_id"]]
        with (cohort / "release.log").open("x") as log:
            result = subprocess.run(release, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            raise RuntimeError("Measurements archived, but allocation release needs attention")
        write(cohort / "device-work-completed.json", {"finished_utc": utc(), "released": True, "allocation_id": meta["allocation_id"]})
        summary_cmd = [analysis_python, str(cohort / "source/tools/summarize_region_cohort_v001.py"), "--cohort", str(cohort)]
        with (cohort / "analysis.log").open("x") as log:
            result = subprocess.run(summary_cmd, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Device measurements complete; local analysis needs attention")
        status = "completed"
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    write(cohort / "controller-completion.json", {"status": status, "finished_utc": utc(), "error": error,
          "historical_measurements_pooled": False, "cohort_id": meta["cohort_id"]})
    print(json.dumps({"status": status, "error": error}), flush=True)
    return 0 if status == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "phase"))
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--controller-python", default=sys.executable)
    parser.add_argument("--analysis-python", default=sys.executable)
    parser.add_argument("--expected-identity")
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--timeout", type=int, default=21600)
    args = parser.parse_args()
    cohort = args.cohort.resolve()
    if args.action == "freeze":
        freeze(cohort)
        return 0
    if not args.expected_identity:
        parser.error("--expected-identity is required for device work")
    if args.action == "phase":
        if not args.phase:
            parser.error("--phase is required")
        return run_phase(cohort, args.phase, args.controller_python, args.expected_identity, args.timeout)
    return run(cohort, args.controller_python, args.expected_identity, args.analysis_python)


if __name__ == "__main__":
    raise SystemExit(main())
