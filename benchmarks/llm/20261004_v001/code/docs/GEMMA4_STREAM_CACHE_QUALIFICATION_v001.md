# Gemma4 streamed forward and private cache

The isolated Gemma4 streaming and cache adapters passed four local CPU tests.
They are not deployed to the active Qwen runtime. Gemma4 remains behind the
official-checkpoint qualification gate; no additional TPU was allocated.

Sources: `src/strassen_mm/llm_stream_gemma4_v001.py`,
`src/strassen_mm/llm_weight_cache_gemma4_v001.py` and
`tests/test_gemma4_stream_cache_v001.py`, committed before execution in
`a6d210cb3`. Archived run:
`runs/20261004T013441Z-gemma4-stream-cache-qualification-v001-a1ab04`,
archive commit `911f5394a`. The four tests took 3.325 seconds; their metrics
are retained in `artifacts/gemma4-stream-cache-checks.json`.

## Preserved experiment boundaries

The stream retains the existing serialized one-layer-at-a-time H2D schedule,
all-token vocabulary output, token chunking and full-forward timing definition.
Embedding scaling, direct Gemma4 final RMS scale and the soft-capped tied head
use the qualified Gemma4 semantics. BF16 is the only supported campaign output.
Each attention type gets its actual projection policy. Global layers cannot
inherit a fictitious V matrix multiplication from local-layer metadata.

Scoring excludes the final position of each sequence, including chunks crossing
batch boundaries. Capture, scoring and saved-logit passes are explicitly marked
ineligible for inference timing. No CPU timing here is a performance claim.

Canonical cache entries include all four layer norms, Q/K norms and the
persistent scalar, retaining its stored FP32 or BF16 dtype. Layer keys derive
from each layer's geometry. The cache identity includes checkpoint revision,
file hashes, configuration and hashes of the Gemma4 packing, checkpoint,
geometry, fused adapter and fusion/kernel sources. Packed layout keys retain
padding, interleave tile width and rotary head width. Cache entries remain
private read-only mappings with bounded disk usage and content verification.

## Evidence and limits

The synthetic checkpoint contains two distinct attention types, nonunit learned
normalization scales, an FP32 nonunit scalar and a BF16 scalar. Batch 2,
sequence 17, window 8 and seven-token logit chunks exercise sequence boundaries,
the local window and the final short chunk. Thirty-two positions are scored.

Against the independent pinned official model, explicit attention produced
identical logits. XLA attention produced 1.4776% logit relative L2 error,
delta NLL +0.000460, mean KL 0.000001488 and 100% top-1 agreement. Both passed
unchanged finite-output, absolute delta NLL <=0.01, KL <=0.02 and agreement
>=97% gates, plus the predeclared 3% native logit qualification limit.

Cached versus original XLA Native execution produced bit-identical hidden
states and every logit chunk, with zero delta NLL/KL/logit error. Host packing
matched device preparation and cache layouts exactly for cubic, S1 and S2 in
both layer types. A reopened cache verified content, and a deliberately mutated
cache byte was rejected. Invalid output contracts and token workloads were
also rejected. Temporary synthetic weights and cache bytes were not archived.

These checks cover Native complete-model execution and custom layout exactness;
custom whole-layer numerical checks are recorded separately in
`GEMMA4_WHOLE_FUSION_QUALIFICATION_v001.md`. Neither substitutes for the real
31B checkpoint, representative-activation tuning, TPU numerical qualification,
or validation of the launch/recovery bundle. Next work is a tuner that respects
the two attention geometries and the shared global K/V site, then integration
with the official input oracle and model-session runtime.
