"""Offline contract checks; no server, subprocess, TPU, or model execution."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

STATUS = Path(__file__).resolve().parents[1] / "status"
sys.path.insert(0, str(STATUS))
spec = importlib.util.spec_from_file_location("campaign_progress_v011", STATUS / "server_v011.py")
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)
NOW = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
RECENT = "2026-09-21T11:59:55Z"
OLD = "2026-09-21T11:40:00Z"


class CampaignProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "runs/cohort").mkdir(parents=True)
        self.feed = self.root / "runs/cohort/progress.json"

    def snapshot(self, **changes):
        payload = {"schema_version": 1, "campaign_id": "test", "state": "running",
                   "heartbeat_utc": RECENT, "worker_last_data_utc": RECENT,
                   "stages": [], "selected_results": [], **changes}
        self.feed.write_text(json.dumps(payload))
        return dashboard.campaign_state(self.root, NOW, self.feed)

    def stage(self, **changes):
        return {"id": "LLM-screen", "label": "LLM screen", "kind": "measurement",
                "state": "running", "completed": 2, "expected": 10,
                "succeeded": 1, "failed": 1, "measured": 1, **changes}

    def test_missing_or_invalid_feed_is_unknown(self):
        missing = dashboard.campaign_state(self.root, NOW, self.feed)
        self.assertEqual(missing["state"], "unknown")
        self.assertIsNone(missing["completed"])
        self.feed.write_text('{"schema_version":1,"mean":NaN}')
        self.assertEqual(dashboard.campaign_state(self.root, NOW, self.feed)["state"], "unknown")

    def test_preparation_is_excluded_and_scopes_do_not_double_counts(self):
        data = self.snapshot(stages=[self.stage(id="setup", kind="preparation", completed=999, expected=999), self.stage()])
        self.assertIsNone(data["stages"][0]["completed"])
        self.assertEqual((data["completed"], data["expected"], data["measured"]), (2, 10, 1))
        self.assertEqual(data["stages"][1]["failed"], 1)

    def test_inconsistent_counters_hide_progress(self):
        data = self.snapshot(stages=[self.stage(succeeded=8)])
        self.assertIsNone(data["completed"])
        self.assertEqual(data["stages"][0]["state"], "attention")
        self.assertTrue(data["warnings"])

    def test_unknown_denominator_stays_unknown(self):
        data = self.snapshot(stages=[self.stage(), self.stage(id="shape-screen", state="not_started", expected=None, completed=0, succeeded=0, failed=0, measured=0)])
        self.assertIsNone(data["expected"])
        self.assertEqual(data["completed"], 2)

    def test_controller_freshness_is_distinct_from_worker_evidence(self):
        data = self.snapshot(worker_last_data_utc=OLD)
        self.assertEqual(data["state"], "running")
        self.assertFalse(data["heartbeat_stale"])
        self.assertTrue(data["worker_data_quiet"])
        data = self.snapshot(heartbeat_utc=OLD)
        self.assertEqual(data["state"], "stale")
        self.assertEqual(data["reported_state"], "running")

    def test_terminal_completion_retains_failures_and_old_timestamp(self):
        data = self.snapshot(state="completed", heartbeat_utc=OLD, stages=[self.stage(state="completed", expected=2)])
        self.assertEqual(data["state"], "completed")
        self.assertEqual(data["stages"][0]["failed"], 1)
        incomplete = self.snapshot(state="completed", stages=[self.stage(state="completed")])
        self.assertEqual(incomplete["state"], "attention")

    def test_five_methods_and_unavailable_means(self):
        rows = [{"shape_id": "qwen", "shape_mkn": [2048, 4096, 12288], "scope": "call",
                 "method": method, "mean_ms": 0 if method == "strassen2" else 2.5,
                 "eligible": method != "strassen2", "selection_status": "confirmed"}
                for method in dashboard.METHODS]
        data = self.snapshot(selected_results=rows)
        self.assertEqual(len(data["selected_results"]), 5)
        self.assertIsNone(data["selected_results"][-1]["mean_ms"])
        self.assertIs(data["selected_results"][-1]["eligible"], False)

    def test_generic_agents_are_recorded_not_invented(self):
        records = dashboard.agent_records([{"id": "llm_campaign", "name": "LLM agent", "state": "working", "utc": OLD},
                                           {"id": "shape_campaign", "state": "complete", "utc": OLD}], NOW)
        self.assertEqual(records[0]["id"], "llm_campaign")
        self.assertEqual(records[0]["state"], "stale")
        self.assertEqual(records[1]["state"], "complete")
        self.assertEqual(dashboard.agent_records([], NOW), [])

    def test_paths_reject_escape_and_private_evidence(self):
        with self.assertRaises(ValueError):
            dashboard.feed_path(self.root, self.root.parent / "outside.json")
        private = self.root / ".cache/secret.json"
        private.parent.mkdir()
        private.write_text("{}")
        with self.assertRaises(ValueError):
            dashboard.feed_path(self.root, private)
        self.assertEqual(dashboard.links(self.root, [{"path": ".cache/secret.json"}, {"path": "runs/../../outside.json"}]), [])


if __name__ == "__main__":
    unittest.main()
