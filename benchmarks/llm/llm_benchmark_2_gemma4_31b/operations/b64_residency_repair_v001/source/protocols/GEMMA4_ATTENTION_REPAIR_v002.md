# Gemma4 attention rounding repair and restart

The user authorized fixing the numerical error and restarting on October 16.
Preserve the failed v001 run and its frozen sources. Continue the existing
Gemma4-31B four-arm protocol in a new run and result folder.

The v001 official-checkpoint check completed all 60 layers, but propagated
hidden relative L2 exceeded 2% at layers 51–57 (zero-based), peaking at 2.67%.
Final logit, NLL, KL and top-choice checks passed. No tuning or batch timings ran.

The wrapper used JAX dot_product_attention, which retains FP32 QK scores before
softmax. Official Transformers 5.5.0 eager Gemma produces BF16 scores, computes
softmax in FP32 and casts its probabilities to BF16. Version 002 restores those
explicit rounding boundaries, with DEFAULT matmul precision and FP32 accumulation.
Grouped-query einsums retain compact K/V tensors at large batches. All four arms
share the repaired attention. Parent GEMM/fusion kernels are unchanged.

Before allocation, independently compare local/global attention with the official
Torch implementation at real head widths, plus the existing layer, fused-kernel
and candidate-legality checks. Validate the repaired launcher/controller wiring.
The new full-checkpoint oracle also propagates the historical v001 path as a
diagnostic control and checks corrected isolated outputs at every layer. Old-path
metrics never enter tuning or timing. The original accuracy thresholds remain
unchanged. Only a successful official-checkpoint gate permits batch tuning.

Batches remain 8,16,32,64,4,1 at sequence 1024, BF16 output, Native, Tuned Native,
Tuned Cubic and parent Tuned Strassen S1. Runtime versions, tuning candidates,
repeat counts and measurement scopes remain unchanged from protocol v001.
Oracle diagnostics use batch 1, sequence 32 and now label that actual sequence
in progress events. Release the owned TPU after completion/failure and archive
every attempt. Persistent allocation waiting and process-scoped idle-sleep
prevention remain enabled. Dashboard stays at port 8790 and points at the new run.
