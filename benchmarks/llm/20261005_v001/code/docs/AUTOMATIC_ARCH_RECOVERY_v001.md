# Automatic recovery for the architecture studies

The user authorized continuing through failures without manual restarts on
September 27, 2026. Run v6e first, then the historical v5e implementation with
FP32 and BF16 outputs separately. Kernels, search menus, seeds, numerical gates
and statistical rules stay unchanged.

The supervisor persists its state, allocation requests, complete comparison
index and every failed attempt. It has an exclusive process lock and owns one
runtime at a time. Provider lookup failure means wait, never allocate blindly.
An uncertain allocation response is reconciled before another request; a sole
late assignment can be adopted only after an empty pre-request inventory and
an idle-kernel check. Allocation requests are capped at four per hour.

Lost launch acknowledgments cause reads of the existing worker. Downloads get
five checksum-verified attempts, preserving failed files. If the controller
still exits, the supervisor checks the existing worker and execution lock,
waits for active work, and retrieves any completed archive. It never launches
a duplicate while the old worker could still be active.

After a runtime disappears, the next runtime uses the original software pins
and reruns the architecture smoke checks. Reuse only complete screen/confirm
pairs with identical device identity. An incomplete pair restarts as a whole;
partial or retry samples are never pooled. This applies to one shape in v6e
and the historical fourteen-shape block within one output dtype in v5e.
All original failed runs and excluded candidate outcomes remain archived.

Each completed phase is exported and committed to the dedicated results
folder. The final report includes per-comparison allocation provenance.
Different shapes may come from different allocations, but every algorithm
comparison stays on one allocation. FP32 and BF16 results remain separate.

Recovery waits use exponential backoff, capped at fifteen minutes. Missing
capacity or credentials cannot be manufactured by retries: the live log
reports the problem and keeps retrying without launching duplicate devices.
The supervisor prevents idle sleep across controller failures. A closed or
offline laptop still pauses local orchestration; it reconciles and continues
when the machine wakes or reconnects. The independent task heartbeat checks
for an unexpectedly stopped supervisor and can repair orchestration issues.

Latest incident: the arch-060-screen worker completed at 16:50:34 UTC with
653 successful candidate cases and 107 recorded memory exclusions. Archive
download returned TransportError at 16:51:06 UTC; no HTTP status was provided.
The allocation was absent when checked at 21:21 UTC. Mac sleep began after
the controller stopped, so the sleep log does not establish the cause of the
initial download failure. Sixty whole shape comparisons are already saved;
the interrupted screen is retained as partial local evidence and rerun.
