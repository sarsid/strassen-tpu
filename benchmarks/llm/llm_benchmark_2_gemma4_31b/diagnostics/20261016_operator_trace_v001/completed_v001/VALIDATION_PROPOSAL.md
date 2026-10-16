# Proposed Gemma4 validation split — awaiting user decision

This is a reviewable proposal, not an active protocol. Existing failed BF16
Torch-versus-JAX gates remain failed. No performance queue is started by this file.

The completed 60-layer control gives peak propagated relative L2 of 2.29134%
for JAX CPU and 2.37206% for JAX TPU against Torch CPU. All isolated layers remain
below 0.282%. The single TPU top-1 disagreement is at position 16: official
logits tie token IDs 818 and 236743 at 23.875, while TPU logits are respectively
23.875 and 24.0. Thirty of 31 agreements is 96.774%, below the existing 97% gate.
The real-weight FP32 control at layers 0, 5 and 53 gives relative L2 between
1.0344e-7 and 4.9391e-7. No additional formula mismatch was identified.

Proposed changes for a new, separately versioned benchmark attempt:

1. Qualify model formulas separately against official Transformers in CPU FP32.
   Require finite output and relative L2 at most 1e-5 on the declared local/global
   real-weight cases. The same tolerance was used in the FP32 control's local
   validation. Preserve the BF16 cross-backend diagnostics as reported evidence;
   do not relabel them as passing or hide the accumulated drift or tie.
2. Compare each algorithm with Native BF16 on the same v6e and the same token
   inputs, matching the parent's separation of model fidelity from kernel quality.
   Keep all existing per-projection/block qualification checks, four quality
   segments, and task-quality limits unchanged: repetitive NLL delta <=0.01,
   KL <=0.005 and top-1 agreement >=0.99; natural-text NLL delta <=0.01,
   KL <=0.02 and top-1 agreement >=0.97. Continue reporting hidden/logit error,
   perplexity alongside agreement.
3. Correct natural-text tokenization before quality measurements: every
   independent sequence begins with BOS and contains sequence_length-1 distinct
   contiguous corpus tokens. Hash and record the exact inputs and BOS policy.
   The current repetitive oracle already includes BOS; this correction addresses
   a separate future-quality issue, not the failure diagnosed here.
4. Preserve Native, Tuned Native, Tuned Cubic and Tuned Strassen S1, BF16 output,
   sequence 1024, batch order 8/16/32/64/4/1, compiler pins, tuning and all timing
   scopes. Validate the new harness before starting any performance stage.

The parent evidence is public-v2 at ab4323242f56ef0c482458bce47ec3b076034a69:
tools/reference_fidelity.py, experiments/gemma3/benchmark_gemma3_streamed.py,
and evidence/gemma3/README.md. Its Gemma3 formulas are not substituted for Gemma4.

Keep the currently owned TPU allocated for the decision and any authorized repair.
This proposal requires an explicit decision because it replaces the additional
cross-backend stopping rule rather than repairing a newly discovered formula bug.
