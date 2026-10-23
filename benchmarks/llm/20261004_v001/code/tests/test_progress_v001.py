"""Offline progress-accounting and evidence-access checks; no server/jobs start."""
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


_SOURCE = Path(__file__).resolve().parents[1] / "status" / "server_v008.py"
_SPEC = importlib.util.spec_from_file_location("progress_server_v008", _SOURCE)
progress = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(progress)

NOW = datetime(2026, 9, 20, 22, 0, 0, tzinfo=timezone.utc)
RECENT = "2026-09-20T21:59:45Z"
OLD = "2026-09-20T20:00:00Z"


class ProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="strassen-progress-")
        self.addCleanup(temporary.cleanup)
        # Resolve macOS /var and /tmp aliases before path-boundary assertions.
        self.base = Path(temporary.name).resolve()
        self.root = self.base / "study"
        self.root.mkdir()
        reader_patch = mock.patch.object(progress, "READER", progress.JournalReader())
        reader_patch.start()
        self.addCleanup(reader_patch.stop)
        # state() may read git history, but these tests never execute git.
        git_patch = mock.patch.object(
            progress.subprocess, "run",
            return_value=subprocess.CompletedProcess(["git"], 0, stdout="", stderr=""),
        )
        git_patch.start()
        self.addCleanup(git_patch.stop)

    def write_json(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return path

    def append(self, path, row):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write((json.dumps(row) + "\n").encode("utf-8"))

    def grid_run(self, timestamp=RECENT):
        run = self.root / "runs" / "example-grid"
        run.mkdir(parents=True)
        self.write_json(progress.COHORT + "/GRID-confirm-started.json",
                        {"run": str(run), "started_utc": timestamp})
        return run

    def test_group_start_never_counts_as_completed_and_reads_are_idempotent(self):
        path = self.root / "journal.jsonl"
        reader = progress.JournalReader()
        self.append(path, {"event": "group_start", "index": 1, "total": 4,
                           "group_id": "first", "utc": RECENT})
        first = reader.read(path)
        self.assertEqual(first["completed"], 0)
        self.assertEqual(first["total"], 4)
        self.append(path, {"event": "group_complete", "index": 1, "total": 4,
                           "group_id": "first", "utc": RECENT})
        self.append(path, {"event": "group_start", "index": 2, "total": 4,
                           "group_id": "second", "utc": RECENT})
        after = reader.read(path)
        self.assertEqual(after["completed"], 1)
        self.assertEqual(after["current_group"], "second")
        self.assertEqual(reader.read(path), after)

    def test_partial_json_append_is_not_consumed_until_newline(self):
        path = self.root / "journal.jsonl"
        reader = progress.JournalReader()
        self.append(path, {"event": "group_complete", "index": 1, "total": 3})
        offset = reader.read(path)["offset"]
        payload = json.dumps({"event": "group_complete", "index": 2, "total": 3}).encode()
        split = len(payload) // 2
        with path.open("ab") as handle:
            handle.write(payload[:split])
        partial = reader.read(path)
        self.assertEqual(partial["completed"], 1)
        self.assertEqual(partial["offset"], offset)
        with path.open("ab") as handle:
            handle.write(payload[split:])
        self.assertEqual(reader.read(path)["completed"], 1)
        with path.open("ab") as handle:
            handle.write(b"\nmalformed JSON\n[]\n")
        complete = reader.read(path)
        self.assertEqual(complete["completed"], 2)
        self.assertEqual(complete["offset"], path.stat().st_size)
        self.assertEqual(reader.read(path), complete)

    def test_truncation_and_replacement_reset_cached_progress_and_error(self):
        path = self.root / "journal.jsonl"
        reader = progress.JournalReader()
        self.append(path, {"event": "group_complete", "index": 7, "total": 8, "utc": OLD})
        self.append(path, {"event": "run_error", "type": "old_failure"})
        self.assertEqual(reader.read(path)["completed"], 7)
        path.write_bytes(b"")
        reset = reader.read(path)
        self.assertEqual(reset["completed"], 0)
        self.assertIsNone(reset["total"])
        self.assertNotIn("error", reset)
        self.assertNotIn("updated_utc", reset)
        self.append(path, {"event": "group_start", "index": 1, "total": 2, "utc": RECENT})
        self.assertEqual(reader.read(path)["completed"], 0)
        replacement = self.root / "replacement.jsonl"
        self.append(replacement, {"event": "group_complete", "index": 1, "total": 3,
                                  "utc": RECENT, "group_id": "replacement-with-longer-row"})
        replacement.replace(path)
        replaced = reader.read(path)
        self.assertEqual(replaced["completed"], 1)
        self.assertEqual(replaced["total"], 3)
        self.assertNotIn("error", replaced)

    def test_missing_evidence_defaults_to_unknown_and_null_model_totals(self):
        data = progress.state(self.root, now=NOW)
        self.assertEqual(data["activity"]["state"], "unknown")
        self.assertEqual(data["grid"]["state"], "unknown")
        self.assertEqual(data["gemma"]["state"], "unknown")
        self.assertEqual(len(data["models"]), 3)
        for model in data["models"]:
            self.assertEqual(model["state"], "unknown")
            self.assertIsNone(model["completed"])
            self.assertIsNone(model["total"])

    def test_stale_execution_is_not_running_and_completion_wins_over_age(self):
        run = self.grid_run()
        journal = run / "results.jsonl"
        self.append(journal, {"event": "group_start", "index": 1, "total": 5, "utc": OLD})
        stale = progress.grid_state(self.root, NOW)
        self.assertEqual(stale["state"], "stale")
        self.assertEqual(stale["completed"], 0)
        self.assertGreater(stale["age_seconds"], 120)
        self.assertIn("does not prove", stale["detail"])
        self.write_json(progress.COHORT + "/GRID-confirm-finished.json", {"status": "completed"})
        self.assertEqual(progress.grid_state(self.root, NOW)["state"], "complete")
        # Both the cohort marker and the per-run completion record are supported.
        (self.root / progress.COHORT / "GRID-confirm-finished.json").unlink()
        self.write_json("runs/example-grid/completion.json", {"status": "completed"})
        self.assertEqual(progress.grid_state(self.root, NOW)["state"], "complete")

    def test_invalid_execution_pointer_does_not_report_running(self):
        relative = progress.COHORT + "/GRID-confirm-started.json"
        for pointer in (None, "", 17):
            with self.subTest(pointer=pointer):
                entry = {"started_utc": RECENT}
                if pointer is not None:
                    entry["run"] = pointer
                self.write_json(relative, entry)
                result = progress.grid_state(self.root, NOW)
                self.assertEqual(result["state"], "unknown")
                self.assertIsNone(result["completed"])
                self.assertIsNone(result["total"])

    def test_recent_start_without_complete_journal_evidence_is_unknown(self):
        run = self.grid_run()
        journal = run / "results.jsonl"
        self.assertEqual(progress.grid_state(self.root, NOW)["state"], "unknown")
        for contents in (b"", b'{"event": "group_start", "utc": '):
            with self.subTest(contents=contents):
                journal.write_bytes(contents)
                result = progress.grid_state(self.root, NOW)
                self.assertEqual(result["state"], "unknown")
                self.assertIsNone(result["total"])

    def test_gemma_pass_count_excludes_failed_missing_and_nonboolean_flags(self):
        self.write_json(progress.GEMMA + "/artifacts/summary.json", {
            "passed": False,
            "checks": {"final_hidden": {"passed": False, "relative_l2": 0.032},
                       "full_logits": {"passed": True, "relative_l2": 0.016},
                       "next_token_logits": {"passed": True, "top1_agreement": 62 / 63},
                       "unreported": {}, "not_boolean": {"passed": 1}},
        })
        result = progress.gemma_state(self.root)
        self.assertEqual(result["state"], "attention")
        self.assertEqual(result["passed"], 2)
        self.assertEqual(result["total"], 5)
        self.assertFalse(result["checks"][0]["passed"])

    def test_not_started_model_retains_unknown_total(self):
        self.append(self.root / progress.JOURNAL, {
            "kind": "model", "id": "qwen", "state": "not_started",
            "detail": "Execution has not started.", "utc": RECENT,
        })
        qwen = progress.state(self.root, now=NOW)["models"][0]
        self.assertEqual(qwen["state"], "not_started")
        self.assertIsNone(qwen["completed"])
        self.assertIsNone(qwen["total"])

    def test_public_evidence_rejects_escape_hidden_cache_and_symlink_escape(self):
        public = self.write_json("reports/result.json", {"safe": True})
        self.assertEqual(progress.public_file(self.root, "reports/result.json"), public)
        external = self.base / "outside.json"
        external.write_text("{}\n", encoding="utf-8")
        hidden = self.write_json(".runtime_private/checkpoint.json", {"private": True})
        self.write_json("runs/.hidden/cache.json", {"private": True})
        (self.root / "runs/outside.json").symlink_to(external)
        (self.root / "runs/cache.json").symlink_to(hidden)
        for relative in ("../outside.json", "runs/../../outside.json", str(external),
                         "runs/../reports/result.json", str(public),
                         ".runtime_private/checkpoint.json", "runs/.hidden/cache.json",
                         "runs/outside.json", "runs/cache.json"):
            with self.subTest(relative=relative):
                self.assertIsNone(progress.public_file(self.root, relative))
        unsupported = self.root / "reports/binary.npy"
        unsupported.write_bytes(b"not public evidence")
        self.assertIsNone(progress.public_file(self.root, "reports/binary.npy"))


if __name__ == "__main__":
    unittest.main()
