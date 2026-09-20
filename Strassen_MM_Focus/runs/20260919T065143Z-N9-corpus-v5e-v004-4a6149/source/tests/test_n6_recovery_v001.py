"""Offline recovery isolation tests; no JAX import or TPU execution."""
import copy
import json
from pathlib import Path
import unittest
from unittest import mock

from strassen_mm import benchmark_n6_recovery_v001 as recovery


class RecoveryTests(unittest.TestCase):
    def test_only_profile_mode_changes_and_input_remains_unchanged(self):
        path = Path(__file__).resolve().parents[1] / "configs/campaign_n5_n9_v1.json"
        campaign = json.loads(path.read_text())
        before = copy.deepcopy(campaign)
        effective = recovery.profiling_override(campaign)
        self.assertEqual(campaign, before)
        expected = copy.deepcopy(before)
        expected["experiments"]["N6"]["profiler"]["tpu_trace_mode"] = "TRACE_COMPUTE_AND_SYNC"
        self.assertEqual(effective, expected)
        with self.assertRaises(ValueError):
            recovery.profiling_override(effective)

    def test_options_constructor_explicitly_enables_device_and_preserves_type(self):
        class Options:
            device_tracer_level = 0
        options = recovery.enabled_device_options(Options)
        self.assertIsInstance(options, Options)
        self.assertEqual(options.device_tracer_level, 1)
        self.assertEqual(Options.device_tracer_level, 0)

    def test_n7_cannot_enter_recovery_or_modify_original_runner(self):
        previous = recovery.original.ProfileRunner
        with mock.patch.object(recovery, "validate_original") as validate:
            with self.assertRaises(SystemExit):
                recovery.main(["--phase", "N7-evaluate"])
            validate.assert_not_called()
        self.assertIs(recovery.original.ProfileRunner, previous)

    def test_profile_constructor_is_restored_after_parent_failure(self):
        class Options:
            pass
        class Profiler:
            ProfileOptions = Options
        class FakeJax:
            profiler = Profiler()
        runner = object.__new__(recovery.RecoveryProfileRunner)
        runner.journal = type("Journal", (), {"captures": []})()
        with mock.patch.object(recovery.base, "jax", FakeJax(), create=True):
            with mock.patch.object(recovery.OriginalProfileRunner, "run_profile_group", side_effect=RuntimeError("capture failed")):
                with self.assertRaisesRegex(RuntimeError, "capture failed"):
                    runner.run_profile_group({})
            self.assertIs(FakeJax.profiler.ProfileOptions, Options)


if __name__ == "__main__":
    unittest.main(verbosity=2)
