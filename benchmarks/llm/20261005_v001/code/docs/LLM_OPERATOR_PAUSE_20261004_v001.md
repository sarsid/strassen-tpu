# Operator pause, October 4

The user requested pausing the LLM experiment and releasing the TPU to revisit
the strategy in the morning. Explicit new instructions are required before any
resumption. The queue supervisor, watchdog and scheduled heartbeat are paused.

There are 80 completed active BF16 comparisons: Qwen3-8B 50, Qwen3-32B 25 and
Qwen3-14B 5. All have passed their existing quality gates. Qwen32 B4/S512 was
the last complete workload. B4/S1024 was interrupted during preparation/tuning;
its diagnostics are evidence, not another completed five-algorithm group.
Never combine its partial work with timing rounds from a new runtime.

The remote worker records `KeyboardInterrupt` and `failed` because its existing
shutdown path uses these fields for an interrupted session. The operator-stop
receipt distinguishes this requested shutdown from an experimental failure.
Completed cases remain complete. Original compression of the whole session was
slow; a versioned shutdown helper retained the same artifacts in a gzip level-1
archive for verified download before release. No measurement source changed.

Queue snapshots, the release receipt and archive verification are kept under
`results/v6e/llm_campaign_20261003_v001/queue_v001/operator_pause_20261004_v001`.
The existing GitHub snapshot at commit `ff5a820` contains the preceding 75
completed comparisons. The additional B4/S512 group and shutdown evidence are
preserved locally with this pause; do not claim that they are in that immutable
GitHub snapshot.

The Gemma4 checkpoint qualification gate remains closed. The newly drafted
`llm_host_pool_gemma4_v001.py` and `llm_resident_gemma4_v001.py` were not executed
or deployed before this pause. They need validation before any future use.
