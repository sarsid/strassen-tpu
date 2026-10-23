# Code for the completed Gemma4-31B benchmark

This new, immutable code folder preserves the exact sources used by the completed
six-batch study. No earlier code, result, archive or manifest was overwritten.
The files were already preserved in the operation archives; this folder gives
the completed study a single, directly browsable code location.

- [Browse the frozen source tree](source/): measurement stages, Gemma4 adapter,
  pinned parent kernels and official reference sources, tuning, runtime controls,
  tests, configurations and protocols.
- [Original v003 source archive](snapshots/original_v003/source.tar) and
  [manifest](snapshots/original_v003/source-manifest.json): exact bundle used for
  successful B1/B4/B8/B16/B32 measurements and the original B64 tuning.
- [Successful B64 recovery archive](snapshots/b64_recovery_v001/source.tar) and
  [manifest](snapshots/b64_recovery_v001/source-manifest.json): exact full bundle
  used by the successful B64 retry. The expanded source/ tree matches this bundle.
- [RUNS.json](RUNS.json): checkpoint identity, source commits, archive hashes and
  batch-to-source/profile/result mappings. Result paths are relative to the
  benchmark root, two directories above this folder.
- [Combined results](../../summaries/completed_20261017_v001/) and
  [release evidence](../../operations/completed_release_20261017_v001/).

| Component | Frozen entry point |
|---|---|
| Original measurement and tuning | [gemma4_batch_stage_v003.py](source/tools/gemma4_batch_stage_v003.py) |
| Successful B64 measurement | [gemma4_batch_stage_v006.py](source/tools/gemma4_batch_stage_v006.py) |
| Gemma4 model adapter | [gemma4_parent_model_v002.py](source/tools/gemma4_parent_model_v002.py) |
| Bounded error metrics | [gemma4_error_metrics_v001.py](source/tools/gemma4_error_metrics_v001.py) |
| Approved batch plan | [gemma4_parent_batch_v003.json](source/configs/gemma4_parent_batch_v003.json) |
| B64 recovery worker | [gemma4_b64_retry_remote_v003.py](source/runtime/gemma4_b64_retry_remote_v003.py) |
| B64 residency protocol | [GEMMA4_B64_RESIDENCY_REPAIR_v002.md](source/protocols/GEMMA4_B64_RESIDENCY_REPAIR_v002.md) |

Both bundles are retained because they identify actual executions. Every file
from the 92-file original bundle is present in the 115-file recovery bundle;
only the historical AGENTS.md changed among those original paths. The remaining
files are versioned repair code, tests and protocols. The successful B64 stage
keeps full B64 dimensions, kernels, saved tuning and quality limits, while
bounding diagnostic memory and parking inactive diagnostic/quality states on
host outside timed model calls. It does not microbatch or retune.

The source tree contains historical run IDs and project instructions. They are
provenance, not instructions to restart those completed runs. Reproduction must
use a fresh run ID and preserve the original software, model, input, tuning and
measurement contracts. Neither credentials nor checkpoint weights nor private
activation tensors are included. No TPU was allocated for this archival task.

MANIFEST.json hashes this entire new folder (excluding itself), records the
verification of both frozen bundles and records pre-existing benchmark hashes.
Those pre-existing files were checked again after this folder was created and
were unchanged. Existing root manifests are preserved exactly; this additive
code snapshot has its own manifest.
