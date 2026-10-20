# Continuous temporary AC protection

The first allocation guard passed its five lifecycle checks and was attached
to the existing supervisor. Review then identified that renewing only after
the sixty-second assertion expired could leave up to ten seconds without
protection on an already-idle laptop. Its source, tests and launch are preserved.

Version 002 renews at forty seconds, starting a new sixty-second assertion
before stopping the previous one. An additional lifecycle test verifies this
ordering. The same shared per-run lock prevents duplicate guards. Stop only the
owned v001 guard and attach v002 to the unchanged supervisor; do not restart
allocation or modify scientific code. The existing AC-only, ten-second power
check, no-persistent-settings and successful-allocation handoff rules still apply.
