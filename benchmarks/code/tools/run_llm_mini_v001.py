"""One frozen, sequential v5e campaign with scoped commits and honest progress.

This controller never allocates, replaces or retries a runtime. Provisioning and
the initial setup identity are caller-owned. The existing transport enforces
the remote execution lock and retains uncertain executions for recovery.
Keep the outer controller's stdout/stderr log outside the cohort directory,
including any downstream tee target; the final cohort seal is immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tarfile
import time

import run_region_cohort_v001 as legacy


ROOT = Path(os.environ.get("STRASSEN_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
CAMPAIGN = "configs/llm_large_shapes_v001/campaign.json"
MODULE = "strassen_mm.benchmark_power_grid_v001"
PHASES = ("GRID-smoke", "GRID-screen", "GRID-confirm")
TIMEOUTS = dict(zip(PHASES, (1800, 7200, 3600)))
SOURCE_PATHS = ("src", "runtime", "tools", "configs/llm_large_shapes_v001", "plans/llm_large_shapes_v001")
MODELS = {"qwen": "Qwen3-8B", "mistral": "Mistral-7B-v0.3", "gemma": "Gemma3-12B (text)"}
write, read, sha, utc = legacy.write, legacy.read, legacy.sha, legacy.utc


def checked_cohort(cohort):
    cohort = Path(cohort).resolve()
    if not cohort.is_relative_to(ROOT / "runs") or cohort == ROOT / "runs":
        raise ValueError("Cohort must be a dedicated directory beneath the canonical project runs/")
    meta = read(cohort / "cohort.json")
    for name in ("cohort_id", "session", "allocation_id"):
        if not isinstance(meta.get(name), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", meta[name]):
            raise ValueError("Invalid cohort field: " + name)
    return cohort, meta


def scoped_commit(cohort, paths, message):
    """Never stage or commit another task's working-tree files."""
    relative = []
    for path in paths:
        path = Path(path).resolve()
        if not path.is_relative_to(cohort) or not path.exists():
            raise ValueError("Commit path must exist within this cohort")
        relative.append(str(path.relative_to(ROOT)))
    subprocess.run(["git", "add", "--", *relative], cwd=ROOT, check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *relative], cwd=ROOT)
    if changed.returncode not in (0, 1):
        raise RuntimeError("Cannot inspect scoped staged changes")
    if changed.returncode:
        subprocess.run(["git", "commit", "--only", "-m", message, "--", *relative], cwd=ROOT, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def freeze(cohort):
    cohort, meta = checked_cohort(cohort)
    if (cohort / "source").exists() or (cohort / "source.tar").exists() or (cohort / "frozen.json").exists():
        raise FileExistsError("Cohort already contains a source freeze")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source = cohort / "source"
    source.mkdir()
    with (cohort / "source.tar").open("xb") as stream:
        subprocess.run(["git", "archive", revision, "--", *SOURCE_PATHS], cwd=ROOT, stdout=stream, check=True)
    with tarfile.open(cohort / "source.tar") as packed:
        packed.extractall(source, filter="data")
    for relative in (CAMPAIGN, "tools/run_llm_mini_v001.py", "runtime/release_allocation_v001.py"):
        if not (source / relative).is_file():
            raise ValueError("Required source is not committed at HEAD: " + relative)
    # Validate progress denominators before any device phase starts.
    progress_totals(read(source / CAMPAIGN))
    write(cohort / "frozen.json", {
        "created_utc": utc(), "baseline_commit": revision, "campaign_relative": CAMPAIGN,
        "benchmark_module": MODULE, "source_paths": list(SOURCE_PATHS),
        "archive_sha256": sha(cohort / "source.tar"),
        "source_sha256": {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()},
        **{key: meta[key] for key in ("cohort_id", "allocation_id", "session")},
        "pooling": "forbidden; one newly provisioned allocation only",
    })
    legacy.verify_frozen(cohort)
    revision = scoped_commit(cohort, [cohort / name for name in ("cohort.json", "source", "source.tar", "frozen.json")],
                             "Freeze large LLM MM cohort " + meta["cohort_id"])
    print(json.dumps({"status": "frozen", "cohort": str(cohort), "commit": revision}), flush=True)


class CohortArchive(legacy.CohortArchive):
    def finish(self, path, status, **details):
        path = Path(path)
        legacy.CohortArchive.finish(self, path, status, **details)
        # Parent's live controller log is intentionally excluded until closed.
        return scoped_commit(self.cohort, [path, self.cohort / (self.phase + "-started.json"),
                                          self.cohort / (self.phase + "-finished.json")],
                             "Archive LLM MM phase " + path.name + " (" + status + ")")


def finished_run(cohort, phase):
    record = read(cohort / (phase + "-finished.json"))
    path = Path(record["run"]).resolve()
    if not path.is_relative_to(cohort / "phases") or path.name != record["run_id"]:
        raise ValueError("Phase receipt points outside this cohort")
    completion = read(path / "completion.json")
    if sha(path / "completion.json") != record["completion_sha256"]:
        raise ValueError("Phase completion hash mismatch")
    if completion.get("allocation_id") not in (None, read(cohort / "cohort.json")["allocation_id"]):
        raise ValueError("Phase allocation changed")
    return record, path, completion


def smoke_identity(cohort):
    record, path, completion = finished_run(cohort, "GRID-smoke")
    summary = read(path / "artifacts/summary.json")
    counts = summary.get("case_status_counts", {})
    if (record["status"] != "completed" or completion.get("remote_may_still_be_running") is not False
            or not summary.get("completed") or not counts or counts.get("ok", 0) <= 0
            or any(name != "ok" and value for name, value in counts.items())):
        raise RuntimeError("Smoke must finish, be retrieved, and have only ok outcomes")
    environment = path / "artifacts/environment.json"
    identity = read(environment).get("identity", {})
    if identity.get("colab_endpoint") != read(cohort / "cohort.json")["allocation_id"]:
        raise ValueError("Smoke identity belongs to another allocation")
    return "/content/Strassen_MM_Focus/runs/" + record["run_id"] + "/artifacts/environment.json"


def run_phase(cohort, phase, controller_python, expected_identity):
    cohort, meta = checked_cohort(cohort)
    adapter = CohortArchive(cohort, phase)
    if phase != "GRID-smoke":
        canonical = smoke_identity(cohort)
        if expected_identity != canonical:
            raise ValueError("Post-smoke phases must use this cohort's canonical smoke environment")
    tools = cohort / "source/tools"
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location("llm_mini_transport", tools / "run_phase_v004.py")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    transport.archive = adapter
    transport.emit = lambda *a, **k: print(json.dumps({"kind": "llm_mini_transport", **k}), flush=True)
    argv = ["run_phase_v004.py", "--phase", phase, "--session", meta["session"], "--endpoint", meta["allocation_id"],
            "--expected-identity", expected_identity, "--controller-python", controller_python,
            "--campaign-relative", CAMPAIGN, "--benchmark-module", MODULE,
            "--timeout-seconds", str(TIMEOUTS[phase])]
    if phase == "GRID-confirm":
        prior, path, completion = finished_run(cohort, "GRID-screen")
        if prior["status"] != "completed" or completion.get("remote_may_still_be_running") is not False:
            raise ValueError("Screening must be completed and retrieved before confirmation")
        if not (path / "artifacts/selections.json").is_file():
            raise ValueError("Frozen screening selections are missing")
        argv += ["--runner-arg=--selection", "--runner-arg=/content/Strassen_MM_Focus/runs/" +
                 prior["run_id"] + "/artifacts/selections.json"]
    original = sys.argv
    try:
        sys.argv = argv
        return transport.main()
    finally:
        sys.argv = original


def progress_totals(campaign):
    """Screen batches match the existing runner; confirmation is top-one only."""
    screen = campaign["experiments"]["GRID-screen"]
    confirm = campaign["experiments"]["GRID-confirm"]
    policy = {"top_k": screen.get("shortlist_top_k", 3),
              "max_confirm_per_family": screen.get("shortlist_max_per_family", 6),
              **screen.get("selection_policy", {})}
    if policy["top_k"] != 1 or policy["max_confirm_per_family"] != 1:
        raise ValueError("Mini campaign requires exactly one confirmed candidate per family")
    batches = max(len(value) for value in screen["candidate_families"].values())
    result = {model: {"GRID-screen": 0, "GRID-confirm": 0} for model in MODELS}
    for phase, experiment, multiplier in (("GRID-screen", screen, batches), ("GRID-confirm", confirm, 1)):
        ids = experiment["shape_ids"]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate campaign shape ID")
        for name in ids:
            model = name.split("_", 1)[0]
            if model not in result:
                raise ValueError("Unrecognized model shape prefix: " + name)
            result[model][phase] += multiplier
    if any(not all(counts.values()) for counts in result.values()):
        raise ValueError("Each model requires screen and confirmation shapes")
    return result


class ModelProgress:
    def __init__(self, cohort):
        self.cohort = cohort
        campaign = read(cohort / "source" / CAMPAIGN)
        self.totals = progress_totals(campaign)
        self.ids = {phase: set(campaign["experiments"][phase]["shape_ids"]) for phase in PHASES[1:]}
        self.offsets, self.completed, self.failures, self.last = {}, set(), Counter(), {}

    def emit(self, model, phase, path, final=None):
        count = sum(name == model for _, _, name in self.completed)
        total = sum(self.totals[model].values())
        state = ("complete" if final and count == total else "attention") if final is not None else "running"
        value = (state, phase, count, total, self.failures[model])
        if self.last.get(model) == value:
            return
        self.last[model] = value
        detail = (f"{phase}: {count}/{total} screen and confirmation groups completed; "
                  f"{self.failures[model]} non-ok case outcomes. Group completion is separate from numerical/speedup eligibility.")
        row = {"kind": "model", "id": model, "name": MODELS[model], "title": MODELS[model] + " large-shape MM",
               "state": state, "completed": count, "total": total, "detail": detail, "utc": utc(),
               "evidence": [{"label": "MM execution journal", "path": str(path.relative_to(ROOT))}]}
        target = ROOT / "status/work_progress_v001.jsonl"
        target.parent.mkdir(exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()

    def update(self, phase):
        if phase == "GRID-smoke":
            return
        marker = self.cohort / (phase + "-started.json")
        if not marker.exists():
            return
        run = Path(read(marker)["run"]).resolve()
        if not run.is_relative_to(self.cohort / "phases"):
            raise ValueError("Progress journal is outside cohort")
        path = run / "results.jsonl"
        if not path.exists():
            return
        offset = self.offsets.get(phase, 0)
        if path.stat().st_size < offset:
            raise ValueError("Append-only progress journal was truncated")
        with path.open("rb") as stream:
            stream.seek(offset)
            while True:
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    break
                self.offsets[phase] = stream.tell()
                row = json.loads(line)
                name = row.get("shape_id") or row.get("group_id", "").split("__", 1)[0]
                if name not in self.ids[phase]:
                    continue
                model = name.split("_", 1)[0]
                if row.get("event") == "case_result" and row.get("status") != "ok":
                    self.failures[model] += 1
                if row.get("event") == "group_complete":
                    self.completed.add((phase, row["group_id"], model))
                if row.get("event") in ("group_start", "group_complete"):
                    self.emit(model, phase, path)

    def finish(self, success):
        for model in MODELS:
            self.emit(model, "Campaign completed" if success else "Campaign needs attention",
                      self.cohort / "controller-completion.json", final=success)


def stop_local_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)


def require_external_controller_log(cohort):
    """Reject direct redirection that would modify the final sealed cohort."""
    descriptors = []
    for descriptor in (1, 2):
        try:
            info = os.fstat(descriptor)
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode):
            descriptors.append((info.st_dev, info.st_ino))
    if not descriptors:
        return
    for path in cohort.rglob("*"):
        if path.is_file():
            info = path.stat()
            if (info.st_dev, info.st_ino) in descriptors:
                raise ValueError("Keep the outer controller stdout/stderr log outside the sealed cohort")


def run(cohort, controller_python, expected_identity):
    cohort, meta = checked_cohort(cohort)
    require_external_controller_log(cohort)
    legacy.verify_frozen(cohort)
    progress = ModelProgress(cohort)
    write(cohort / "controller-started.json", {"pid": os.getpid(), "started_utc": utc(),
          "phases": list(PHASES), "device_timeout_seconds": TIMEOUTS,
          **{key: meta[key] for key in ("cohort_id", "allocation_id", "session")}})
    scoped_commit(cohort, [cohort / "controller-started.json"], "Start LLM MM controller " + meta["cohort_id"])
    status, problem, release_error = "failed", None, None
    attempted, receipts = [], []
    released, remote_uncertain = False, False
    progress_errors = []

    def refresh(phase):
        if progress_errors:
            return
        try:
            progress.update(phase)
        except Exception as error:
            progress_errors.append({"type": type(error).__name__, "message": str(error), "utc": utc()})

    try:
        for phase in PHASES:
            if phase != "GRID-smoke":
                expected_identity = smoke_identity(cohort)
            command = [sys.executable, str(cohort / "source/tools/run_llm_mini_v001.py"), "phase",
                       "--cohort", str(cohort), "--phase", phase, "--controller-python", controller_python,
                       "--expected-identity", expected_identity]
            log_path = cohort / (phase + "-controller.log")
            command_path = cohort / (phase + "-command.json")
            write(command_path, {"argv": command, "created_utc": utc()})
            attempted.append(phase)
            child, child_error, returncode = None, None, None
            try:
                with log_path.open("x") as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, env=dict(os.environ, STRASSEN_PROJECT_ROOT=str(ROOT),
                                                        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1"))
                    deadline = time.monotonic() + TIMEOUTS[phase] + 1800
                    while child.poll() is None:
                        refresh(phase)
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Local phase controller exceeded its bounded orchestration allowance")
                        time.sleep(3)
                    returncode = child.returncode
            except BaseException as error:
                child_error = {"type": type(error).__name__, "message": str(error)}
                if child is not None:
                    stop_local_child(child)
            finally:
                # The child's stdout handle is closed before this parent receipt/commit.
                refresh(phase)
                receipt = cohort / (phase + "-controller-receipt.json")
                write(receipt, {"finished_utc": utc(), "returncode": returncode, "error": child_error,
                               "command_sha256": sha(command_path), "log_sha256": sha(log_path) if log_path.exists() else None})
                paths = [command_path, receipt] + ([log_path] if log_path.exists() else [])
                revision = scoped_commit(cohort, paths, "Archive closed LLM MM controller log " + phase)
                receipts.append({"phase": phase, "receipt_sha256": sha(receipt), "commit": revision})
            if child_error or returncode:
                raise RuntimeError(phase + " controller failed; inspect immutable phase evidence")
            record, path, completion = finished_run(cohort, phase)
            summary = read(path / "artifacts/summary.json")
            if (record["status"] != "completed" or completion.get("remote_may_still_be_running") is not False
                    or not summary.get("completed") or summary.get("not_completed_group_ids")):
                raise RuntimeError(phase + " did not complete and retrieve every planned group")
            if phase == "GRID-smoke":
                expected_identity = smoke_identity(cohort)
        status = "completed"
    except BaseException as error:
        problem = {"type": type(error).__name__, "message": str(error)}
    finally:
        # A missing receipt is uncertainty, never permission to delete a runtime.
        for phase in attempted:
            try:
                _, _, completion = finished_run(cohort, phase)
                if completion.get("remote_may_still_be_running") is not False:
                    remote_uncertain = True
            except Exception:
                remote_uncertain = True
        if not remote_uncertain:
            release_path = cohort / "release.log"
            release = [controller_python, str(cohort / "source/runtime/release_allocation_v001.py"),
                       "--session", meta["session"], "--expect-endpoint", meta["allocation_id"]]
            try:
                write(cohort / "release-command.json", {"argv": release, "created_utc": utc()})
                with release_path.open("x") as log:
                    result = subprocess.run(release, stdout=log, stderr=subprocess.STDOUT, timeout=180)
                responses = [json.loads(line) for line in release_path.read_text().splitlines() if line.strip()]
                released = result.returncode == 0 and any(
                    row.get("kind") == "allocation_released" and row.get("verified_absent") is True
                    and row.get("endpoint") == meta["allocation_id"] for row in responses)
                if not released:
                    raise RuntimeError("Allocation release was not verified")
            except BaseException as error:
                release_error = {"type": type(error).__name__, "message": str(error)}
        if not released:
            status = "failed"
        write(cohort / "device-work-completed.json", {"finished_utc": utc(), "allocation_id": meta["allocation_id"],
              "released": released, "remote_may_still_be_running": remote_uncertain,
              "recovery_required": not released, "release_error": release_error})
        write(cohort / "controller-completion.json", {"status": status, "finished_utc": utc(), "error": problem,
              "release_error": release_error, "released": released, "remote_may_still_be_running": remote_uncertain,
              "phase_receipts": receipts, "progress_errors": progress_errors,
              "historical_measurements_pooled": False, "cohort_id": meta["cohort_id"]})
        try:
            progress.finish(status == "completed")
        except Exception:
            pass  # Dashboard availability cannot change immutable measurement outcomes.
        legacy.seal(cohort, "controller-artifact-manifest.json")
        revision = scoped_commit(cohort, [cohort], "Archive LLM MM cohort " + meta["cohort_id"] + " (" + status + ")")
        print(json.dumps({"status": status, "released": released, "remote_may_still_be_running": remote_uncertain,
                          "error": problem, "release_error": release_error, "commit": revision}), flush=True)
    return 0 if status == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "phase", "run"))
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--controller-python", default=sys.executable)
    parser.add_argument("--expected-identity")
    parser.add_argument("--phase", choices=PHASES)
    args = parser.parse_args()
    cohort, _ = checked_cohort(args.cohort)
    if args.action == "freeze":
        freeze(cohort)
        return 0
    if not args.expected_identity or not Path(args.expected_identity).is_absolute():
        parser.error("Device work requires an absolute remote --expected-identity path")
    if args.action == "phase":
        if args.phase is None:
            parser.error("phase requires --phase")
        return run_phase(cohort, args.phase, args.controller_python, args.expected_identity)
    return run(cohort, args.controller_python, args.expected_identity)


if __name__ == "__main__":
    raise SystemExit(main())
