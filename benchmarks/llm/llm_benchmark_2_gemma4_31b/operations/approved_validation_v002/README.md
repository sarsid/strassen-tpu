# Approved parent-style validation and Gemma v003 restart

The user approved the validation split on October 16. Model fidelity uses the
completed CPU FP32 real-weight control at layers 0, 5 and 53, requiring finite
outputs and relative L2 <=1e-5. Reuse is explicit and guarded by checkpoint,
official Transformers and qualified model/control source hashes. Historical
BF16 cross-framework failures remain preserved; this is a separately versioned
protocol, not a relabeling of old results.

The four BF16 algorithms still use the same TPU Native reference and unchanged
projection/block checks and repetitive/natural-text quality limits. Tuning and
timing functions are unchanged. Natural sequences now each have BOS plus 1023
contiguous corpus tokens, with exact input and policy hashes recorded.

All five local check groups passed: BOS/source-integrity/unchanged scientific
contracts; local/global FP32 semantic controls; controller fault injection;
bundle launch integrity; retained-runtime supervisor recovery. The control
keeps the TPU on failed computations, failed quality checks and unverified
result downloads. It releases only after all six batches complete and pass.

The previous retained allocation disappeared from Colab at 14:33:47 UTC; no
controller release occurred. The new v003 supervisor started at 14:43 UTC and
waits for a replacement v6e, then runs only this authorized Gemma sweep.
This checkpoint includes no new TPU performance measurements. Numerical metrics
and source are public; activation tensors, checkpoint weights and credentials
are not included. See source/protocols/GEMMA4_PARENT_VALIDATION_v002.md.
