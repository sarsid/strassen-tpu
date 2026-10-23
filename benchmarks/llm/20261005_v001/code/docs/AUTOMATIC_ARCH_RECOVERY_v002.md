# Automatic recovery with a v6e-only completion boundary

This September 29, 2026 scope update supersedes the automatic v5e transition
in `AUTOMATIC_ARCH_RECOVERY_v001.md`. All existing recovery and measurement
rules remain unchanged within the current v6e study.

The active supervisor is `supervise_arch_studies_v004.py`, selected through
`runs/20260927-arch-supervised-v001/active-source.json`. The guarded starter
remains `tools/start_arch_supervisor_v003.py`. Never restart an older supervisor
that contains the v5e transition.

After all 168 v6e screen/confirm comparisons are verified, release the owned
runtime, finalize and commit the v6e report, record completion with scope
`v6e_only`, and stop. A failed final report must remain incomplete and retry
without acquiring more TPU time. No v5e or LLM work is scheduled by this process.
The scope guard rejects other hardware before any allocation request.

The heartbeat named "Finish v6e shapes and stop" remains active only to recover
unexpected interruption of this study. Once the final archive is verified,
the runtime is released and the supervisor is finished, it deletes itself.
It must not create replacement automations or launch another experiment.

The supervisor handover changes only the local supervisor. The detached
measurement controller, current worker and TPU continue with their existing
frozen sources. Migration records verify unchanged controller PID and allocation.
