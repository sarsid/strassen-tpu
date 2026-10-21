"""Check overlapping bounded assertions and cleanup without changing real power state."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from accuracy_allocation_awake_guard_v004 import Guard, on_ac_power


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.children = []

        def spawn(command, **kwargs):
            child = Mock(pid=123+len(self.children))
            child.command = command
            child.poll.return_value = None
            self.children.append(child)
            return child

        self.spawn = Mock(side_effect=spawn)
        self.guard = Guard(456, self.spawn, lambda: self.now)

    def test_battery_does_not_start_assertion(self):
        self.assertTrue(self.guard.step(True, False, False))
        self.spawn.assert_not_called()
        self.assertFalse(on_ac_power("Now drawing from 'Battery Power'\nAC Power"))
        self.assertTrue(on_ac_power("Now drawing from 'AC Power'\n100%; charged"))

    def test_bounded_assertion_tied_to_supervisor(self):
        self.guard.step(True, True, False)
        self.guard.step(True, True, False)
        self.assertEqual(self.spawn.call_count, 1)
        self.assertEqual(self.children[0].command,
            ['/usr/bin/caffeinate','-i','-s','-w','456','-t','60'])

    def test_old_assertion_survives_replacement_start(self):
        self.guard.step(True, True, False)
        self.now = 39
        self.guard.step(True, True, False)
        self.assertEqual(self.spawn.call_count, 1)
        self.now = 40
        self.guard.step(True, True, False)
        self.assertEqual(self.spawn.call_count, 2)
        self.assertEqual(len(self.guard.children), 2)
        for child in self.children:child.terminate.assert_not_called()
        self.now = 60
        self.children[0].poll.return_value = 0
        self.guard.step(True, True, False)
        self.assertEqual(self.guard.children, [self.children[1]])
        self.children[0].terminate.assert_not_called()

    def test_unplug_releases_all_overlapping_owned_children(self):
        self.guard.step(True, True, False)
        self.now = 40
        self.guard.step(True, True, False)
        self.assertTrue(self.guard.step(True, False, False))
        for child in self.children:
            child.terminate.assert_called_once()
            child.wait.assert_called_once_with(timeout=2)
        self.assertIsNone(self.guard.child)
        self.assertEqual(self.guard.children, [])

    def test_handoff_or_supervisor_exit_releases_all(self):
        for matches, allocated in [(True,True),(False,False)]:
            with self.subTest(matches=matches,allocated=allocated):
                self.setUp()
                self.guard.step(True, True, False)
                self.now=40
                self.guard.step(True, True, False)
                self.assertFalse(self.guard.step(matches,True,allocated))
                for child in self.children:child.terminate.assert_called_once()
                self.assertEqual(self.guard.children, [])

    def test_unexpected_child_exit_is_replaced(self):
        self.guard.step(True,True,False)
        self.children[0].poll.return_value=1
        self.guard.step(True,True,False)
        self.assertEqual(self.spawn.call_count,2)
        self.assertEqual(self.guard.children,[self.children[1]])


if __name__=='__main__':unittest.main(verbosity=2)
