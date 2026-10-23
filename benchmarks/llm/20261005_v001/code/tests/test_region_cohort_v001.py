"""Offline safety checks for the new cohort's archive and machine separation."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_region_cohort_v001 as controller
import summarize_region_cohort_v001 as summary


class CohortTests(unittest.TestCase):
    def frozen(self, directory):
        controller.write(directory / "cohort.json", {"cohort_id": "new_cohort", "allocation_id": "new_allocation", "session": "new_session"})
        source = directory / "source"
        source.mkdir()
        (source / "payload.py").write_text("print('frozen')\n")
        with tarfile.open(directory / "source.tar", "w") as packed:
            packed.add(source / "payload.py", arcname="payload.py")
        controller.write(directory / "frozen.json", {
            "cohort_id": "new_cohort", "allocation_id": "new_allocation", "session": "new_session", "baseline_commit": "baseline_revision",
            "archive_sha256": controller.sha(directory / "source.tar"),
            "source_sha256": {"payload.py": controller.sha(source / "payload.py")}})

    def test_archive_never_runs_git_and_preserves_prior_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            cohort = Path(temporary)
            self.frozen(cohort)
            with mock.patch.object(controller.subprocess, "run", side_effect=AssertionError("Archive must not run git")):
                adapter = controller.CohortArchive(cohort, "GRID-screen")
                run = adapter.begin("ignored", scope="test")
                self.assertEqual((run / "source.tar").read_bytes(), (cohort / "source.tar").read_bytes())
                self.assertEqual((run / "source/payload.py").read_text(), "print('frozen')\n")
                revision = adapter.finish(run, "completed", allocation_id="new_allocation")
                self.assertTrue(revision.startswith("sha256:"))
                before = (run / "completion.json").read_bytes()
                with self.assertRaises(FileExistsError):
                    adapter.finish(run, "failed")
                self.assertEqual((run / "completion.json").read_bytes(), before)

    def test_modified_source_or_archive_refuses_phase_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            cohort = Path(temporary)
            self.frozen(cohort)
            (cohort / "source/payload.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Frozen source changed"):
                controller.CohortArchive(cohort, "GRID-screen")
            with (cohort / "source.tar").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Frozen archive changed"):
                controller.CohortArchive(cohort, "GRID-screen")
            self.assertFalse((cohort / "phases").exists())

    def test_summary_rejects_other_allocation_before_reading_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            cohort = Path(temporary)
            controller.write(cohort / "cohort.json", {"cohort_id": "new_cohort", "allocation_id": "new_machine"})
            for name, allocation in (("screen", "new_machine"), ("confirmation", "old_machine")):
                (cohort / name).mkdir()
                controller.write(cohort / name / "environment.json", {"identity": {"allocation_id": allocation}})
            with self.assertRaisesRegex(ValueError, "Mixed runtime identities"):
                summary.describe(cohort, cohort / "screen", cohort / "confirmation")

    def test_changed_cohort_identity_refuses_frozen_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            cohort = Path(temporary)
            self.frozen(cohort)
            meta = controller.read(cohort / "cohort.json")
            meta["allocation_id"] = "another_machine"
            (cohort / "cohort.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "Cohort identity changed"):
                controller.verify_frozen(cohort)

    def test_summary_rejects_pointer_to_another_cohort(self):
        with tempfile.TemporaryDirectory() as temporary:
            cohort = Path(temporary) / "new"
            cohort.mkdir()
            controller.write(cohort / "GRID-screen-finished.json", {"status": "completed", "run": str(Path(temporary) / "old")})
            with self.assertRaisesRegex(ValueError, "outside this cohort"):
                summary.verified_artifacts(cohort, "GRID-screen")


if __name__ == "__main__":
    unittest.main()
