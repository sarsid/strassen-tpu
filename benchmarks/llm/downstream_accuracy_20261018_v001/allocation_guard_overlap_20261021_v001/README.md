# Temporary assertion overlap during allocation waits

The previous guard terminated its old sleep-prevention process immediately
after starting a replacement. Process creation does not acknowledge that its
new assertion is registered, leaving a possible renewal gap.

Version 004 renews after 40 seconds and lets each 60-second assertion expire
naturally. It tracks both owned processes and stops them on battery power,
supervisor exit or allocation handoff. Assertions stay bounded and tied to the
supervisor; permanent power settings and intentional sleep remain unchanged.
The allocation controller and scientific experiment are unchanged.

Six CPU-only lifecycle checks passed. They cover bounded process creation,
continued overlap during renewal, natural expiry, battery cleanup, supervisor
exit/allocation handoff, and replacement of an unexpectedly exited child.
Run them from this directory:

```sh
python source/tools/check_accuracy_allocation_awake_guard_v003.py
```

`tests.txt` preserves the passing unit-test output. This folder contains source
and test results only. Machine power logs, process identifiers, local execution
paths and diagnostic receipts remain local. It contains no new model accuracy
or TPU performance measurements.
