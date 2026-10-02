# Launch receipt — 2026-09-21 UTC

The frozen campaign is `runs/20260921-llm-large-shapes-cohort-v001`.
Its source freeze commit is `3e04c8767ac37a0edc45bcf2086ce469db1eeab2`.
Device phases use the source archive recorded in that cohort's `frozen.json`.

- Allocation: `tpu-v5e1-s-kkb-usw1c0-3n8jv21cfhqx2` (single TPU v5 lite).
- Session: `strassen-large-llm-v5e-20260921-v001`.
- Setup: `runs/20260921T023732Z-llm-mini-setup-v001-40e332`.
- Initial configuration validation passed: `runs/20260921T024200Z-llm-mini-config-check-v001-63ffa3`.
- Orchestration envelope: `runs/20260921T024242Z-llm-mini-controller-v001-ba8918`. Its live log is outside the cohort to avoid sealing a log that is still being written.
- TPU smoke completed 2026-09-21T02:43:00Z: 2/2 groups, 12/12 outcomes `ok`. It includes the maximum-output Gemma concatenated projection.
- Screening started on the same allocation after verified smoke retrieval. Automatic confirmation uses the frozen per-shape family winners and a fresh input seed.

The controller archives and commits closed phases, then releases this exact
allocation after verified retrieval. Uncertain remote execution retains the
allocation for recovery. Per-model dashboard groups exclude smoke and distinguish
group completion from numerical eligibility.

Additional local controller fixtures were introduced after the initial check.
The preserved v002 validation failed because its macOS temporary-directory path
used a symlink while the controller requires a canonical path. The v003 fixture
resolves that path and passed. This was a local test-fixture issue, not a TPU
measurement failure; the running controller and benchmark source were unchanged.

This receipt records the launch, not final results. Consult the cohort's phase
receipts and `controller-completion.json` for final execution and release status.
