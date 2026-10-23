# Pre-execution qualification decisions

Frozen before the first adapter execution on 2026-09-20.

- Use the exact pinned Gemma 3 1B BF16 checkpoint and Transformers 4.56.2
  PyTorch 2.8.0 eager CPU forward as an independent oracle.
- Run CPU qualification while the separate region-grid campaign occupies the
  v5e TPU. Do not share, reset, or reconfigure that TPU.
- Check a small deterministic official random model with nonzero RMS offsets,
  multiple custom-kernel reduction panels and tile tails. Also check a separate
  six-layer model at 529 tokens with the actual 512-token attention window.
- For actual weights, compare the complete 64-token native model, including all
  vocabulary logits. Separately compare local layer 4 and global layer 5 using
  their official 529-token input activations. The latter is a layer check, not
  a full 529-token model qualification.
- Use a relative L2 limit of 0.02 for hidden/layer outputs, stricter than the
  0.03 logit limit because these checks precede the additional output projection.
  Primitive limits remain relative L2 0.01 and maximum absolute error 0.04.
  For next-token logits also require mean KL <= 0.01, absolute mean NLL
  difference <= 0.05, and top-1 agreement >= 0.90. All outputs must be finite.
  These are prospective engineering qualification limits, not universal
  numerical-accuracy standards. Preserve failures without relaxing the limits.
- Compare both common-input layers (isolating each implementation) and
  propagated layers (exposing accumulated differences).
- Custom cubic and Strassen paths initially receive CPU Pallas interpreter
  checks on random weights. Real-checkpoint custom quality and compiled TPU
  correctness/performance require their own subsequent experiments.
- Keep gated checkpoint payloads and credentials private. Archive source,
  immutable revisions, hashes, input selection, outputs, failures and decisions.
  Commit only the files belonging to this execution; preserve the other active
  campaign's changes.
