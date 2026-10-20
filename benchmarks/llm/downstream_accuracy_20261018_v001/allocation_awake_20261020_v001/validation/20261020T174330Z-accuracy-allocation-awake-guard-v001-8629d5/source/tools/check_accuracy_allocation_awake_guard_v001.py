"""Verify AC-only lifecycle behavior without creating actual power assertions."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
from accuracy_allocation_awake_guard_v001 import Guard, on_ac_power


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.child = Mock(pid=123)
        self.child.poll.return_value = None
        self.spawn = Mock(return_value=self.child)
        self.guard = Guard(456, self.spawn)

    def test_battery_never_creates_assertion(self):
        self.assertTrue(self.guard.step(True, False, False))
        self.spawn.assert_not_called()
        self.assertFalse(on_ac_power("Now drawing from 'Battery Power'\nAC Power"))

    def test_ac_starts_one_bounded_process(self):
        self.assertTrue(on_ac_power("Now drawing from 'AC Power'\n100%; charged"))
        self.guard.step(True, True, False)
        self.guard.step(True, True, False)
        self.spawn.assert_called_once()
        self.assertEqual(self.spawn.call_args.args[0],
                         ["/usr/bin/caffeinate", "-i", "-s", "-w", "456", "-t", "60"])

    def test_unplug_releases_only_owned_child(self):
        self.guard.step(True, True, False)
        self.assertTrue(self.guard.step(True, False, False))
        self.child.terminate.assert_called_once()
        self.child.wait.assert_called_once_with(timeout=2)
        self.assertIsNone(self.guard.child)

    def test_allocation_hands_off_and_controller_exit_stops(self):
        for matches, allocated in [(True, True), (False, False)]:
            with self.subTest(matches=matches, allocated=allocated):
                self.setUp()
                self.guard.step(True, True, False)
                self.assertFalse(self.guard.step(matches, True, allocated))
                self.child.terminate.assert_called_once()
                self.assertIsNone(self.guard.child)

    def test_expired_assertion_is_renewed(self):
        self.guard.step(True, True, False)
        self.child.poll.return_value = 0
        self.guard.step(True, True, False)
        self.assertEqual(self.spawn.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
