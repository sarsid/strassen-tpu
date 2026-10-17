# Gemma4 numerical diagnosis

User authorized finding the cause of the remaining numerical error and reviewing
the parent implementation. This run performs no tuning or performance sweep.

Run the pinned real 31B checkpoint at B1/S32 through official Torch BF16 on CPU
and through the unchanged v002 Native formulation independently on JAX CPU and
v6e. At layers 0,1,5,39,52,53,58,59 compare each operation on identical official
inputs: projections, RMS norms, GeGLU, RoPE, attention and residual addition.
Spot-check differing GEMM entries with FP64 dot products to distinguish rounding
from an incorrect formula. Preserve the captured public-prompt BF16 activations
(excluding weights) for follow-up diagnosis. Capture independent propagated and
isolated errors at all 60 layers and final prediction disagreement margins.

Audit the first token/BOS identity and the causal mask. Do not change the input,
production code, thresholds or performance plan as part of diagnosis. CPU results
are numerical controls, never performance evidence. The completed diagnostic is
not a qualified benchmark; its summary has all_pass=null. Archive all results
and release the owned v6e. The live dashboard identifies this as diagnosis only.

Parent public-preview remains 95be1fb. The newer public-v2 branch ab432324
contains Gemma3 reference_fidelity.py and BOS/causal-mask bug evidence. Its
fidelity test is a single real layer with both formulations on CPU in FP32;
its algorithm quality checks use same-run Native XLA as the baseline. These
are distinct from our full-depth BF16 Torch-versus-JAX qualification gate.
Gemma3-specific weight-offset norms must not be copied into Gemma4.
