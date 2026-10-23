# Temporary AC power guard during allocation

The existing supervisor enables its runtime sleep guard only after allocation.
On October 20, macOS idle sleep repeatedly suspended allocation requests,
including while plugged into AC power. This is a local orchestration issue and
does not explain the preceding provider-side runtime disappearance.

Attach `accuracy_allocation_awake_guard_v001.py` to the existing v009 supervisor.
It validates the exact recorded PID and command and uses a per-run lock. It
creates a temporary idle-sleep assertion only while on AC power, checks power
every ten seconds, and releases its own child on battery power, supervisor exit,
identity mismatch, or successful allocation. Each assertion also expires after
sixty seconds if the guard unexpectedly dies. On allocation, the existing
supervisor's runtime guard takes over. No persistent power settings change;
this does not override lid closure, deliberate sleep, or service resource limits.

No allocation controller is restarted, no second request is submitted, and no
scientific source, checkpoint, contract, numerical setting, or scoring changes.
This operational helper is archived separately from the frozen scientific bundle.
