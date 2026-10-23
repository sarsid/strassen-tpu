# First complete-model workload and tuner decisions

The first bounded execution is Qwen3-8B, B1/S512, all five algorithms and both
store tracks: at most 10 of the 700 main entries. It qualifies the entire new
execution path before expansion. It does not automatically launch the remaining
690 entries or imply downstream-task/decode completion.

The complete streamed CPU path passed 115 independent checks in
`runs/20261003T020828Z-llm-stream-qualification-v001-8d5ca1`. Those generated
small-model fixtures are correctness checks, not real-weight performance data.
The remote run additionally qualifies every real checkpoint layer against
official Transformers BF16 on 64 calibration tokens before tuning.

The tuner records the complete offered/pruned menu and limits compilation to
12 candidates per site, family, store and attention class. This is a bounded
search, not proof of a global optimum. It first preserves compatible parent
Qwen32B tiles, then covers full/partial K contraction, accumulator strategy and
buffer count. Within these groups it favors 256-wide leaves, less padding,
tiles near 1024, and lower estimated VMEM. These rankings spend the compilation
budget; measured times select the winner. The rough VMEM bound is not a claim
that other candidates cannot compile.

Native searches all six declared whole-layer VMEM options. Custom kernels get
the same selected whole-layer options. Candidate screening uses real layer-0
calibration activations (Gemma also uses the first global-attention layer),
three warmups and eight measurements, with projection error eligibility.
Projection shapes are constant across layers of an attention class; the frozen
policy is reused across those layers. Full held-out model scoring measures the
accumulated error. Native fallbacks are explicit, and missing candidates never
turn a wholly Native implementation into a claimed Strassen measurement.

Whole-layer compilation and finite-output gates precede profile freeze. The
first version does not search composite combinations after site selection;
confirmation, not isolated screening, determines whether the full model wins.
Native retains model BF16 intermediate boundaries, while custom epilogues may
consume FP32 accumulators. Both Native output-store labels remain distinct in
metadata, but XLA may eliminate conversions. A label is not proof of an HBM
store. Consumer conversion is included in the measured function.

One algorithm's prepared host matrices are held at a time. Switching algorithms
rebuilds host layouts outside timing, retains compiled functions, and performs
the same three complete warmups before each recorded sample. This avoids five
copies of large checkpoints in host memory. Each recorded forward includes
embedding lookup, serialized layer transfers, every layer, final normalization,
head transfer and all token logits. It excludes checkpoint download, host
packing, compilation and scoring. Fifteen randomized rounds pair the five arms
on one runtime; raw samples and a paired bootstrap interval are retained.

Calibration uses the first token window of WikiText2 raw train; held-out quality
uses the corresponding first test window, at corpus revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. Tokenizer, token IDs and hashes are
saved. A B1/S512 case scores 511 next-token positions; its perplexity is a small
fixed-window estimate, not whole-corpus perplexity. Quality passes are separately
instrumented and cannot be used as inference latency. Full hidden/logit errors,
NLL, perplexity, KL and top-1 agreement are measured against qualified Native.

The first controller is bounded to six hours including setup, downloads and
archival. Candidate failures are recorded and independent candidates continue.
A failed real-model semantic gate blocks dependent tuning. The controller
downloads and verifies evidence, records the outcome, releases the owned TPU,
and commits the run. It does not claim a model is ready when a prerequisite
fails. Gemma's official full-checkpoint extraction and secondary KV decoding
still require integration before their dedicated executions.
