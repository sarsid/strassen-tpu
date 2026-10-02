# Resume the unchanged tuning plan on the user's idle v5e

On 2026-09-21 the user reported 35.15 available Colab compute units. A read-only
allocation listing found one live V5E1 endpoint:
`tpu-v5e1-s-kkb-usw1c1-192tu3wxx1p0f`. The user explicitly confirmed the
associated notebook is not running any job after we asked whether it could
be used for this pending experiment.

Register that exact existing endpoint under a fresh local session. Do not
create another TPU. This changes the acquisition step in PROTOCOL.md, which
originally specified a new allocation. All shapes, eight candidate tiles,
screening/confirmation seeds, selection rules, numerical gates, precision,
deadlines and kernels remain unchanged. Use the same verified endpoint for
both phases. Prior timeouts and the original blocked status remain preserved.

The current user-reported balance rules out an exhausted paid balance now; it
does not establish the cause of the earlier request timeouts. No further
allocation diagnostic is needed to run the authorized experiment on this
available idle machine. After verified evidence retrieval, release the runtime
as specified in the protocol to avoid continued compute consumption.
