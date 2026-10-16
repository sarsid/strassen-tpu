# Gemma4 numerical diagnosis — completed

This is numerical diagnosis, not a Strassen speed or quality result. All four-arm
performance stages remain unstarted. The existing BF16 qualification failures
remain failed, and the TPU is retained for the user's validation decision.

The 60-layer comparison uses official Torch CPU BF16 and the unchanged Native
formulation in JAX CPU and JAX v6e. Inputs include BOS. It records 189 matched
operation comparisons at eight selected layers, all-layer isolated and propagated
errors, and final prediction differences.

| Metric versus official Torch CPU BF16 | JAX CPU | JAX v6e |
|---|---:|---:|
| Largest propagated hidden relative L2 | 2.29134% | 2.37206% |
| Largest isolated-layer relative L2 | 0.27412% | 0.28188% |
| Final logits relative L2 | 1.26908% | 1.28006% |
| Absolute NLL difference | 0.00703594 | 0.00952243 |
| Mean KL, nats | 0.00092427 | 0.00125595 |
| Top-1 agreement on 31 positions | 31/31 | 30/31 |

Both JAX backends exceed the additional 2% propagated-hidden stopping rule.
The single TPU prediction disagreement occurs at position 16: the official
reference ties token IDs 818 and 236743 at logits 23.875. TPU logits are 23.875
and 24.0. The lowest-index tie-breaking rule chooses 818 for the reference;
the TPU chooses 236743. One disagreement produces 96.774%, below the 97% rule.

All matched operations are finite. The largest matched-operation relative L2
is 0.00324718% on CPU and 0.01131696% on TPU. In 424 deliberately selected
GEMM disagreements, 321 TPU values and 87 Torch values match nearest-BF16
rounding of an FP64 dot product. These are the first differing entries rather
than a random sample and must not be used as a general accuracy estimate.

A separate real-weight FP32 control executes the same formulation and official
Transformers on CPU at layers 0, 5 and 53. Relative L2 is respectively
4.93904e-7, 1.03439e-7 and 4.18844e-7. This supports formula fidelity for both
attention types and the late global layer; it does not turn the BF16 checks
into passing results or establish task accuracy for Strassen.

The evidence points to accumulated cross-framework BF16 arithmetic differences
and an exact-tie-sensitive short prediction gate, rather than another identified
formula defect. The parent separates FP32 model-fidelity checks from same-device
BF16 algorithm-quality checks. See [the proposed validation split](VALIDATION_PROPOSAL.md),
which awaits user approval. Existing algorithm-quality limits would remain
unchanged. Natural-text BOS handling is a separate correction needed before
quality measurements; the current oracle already includes BOS.

`remote/artifacts/oracle-a0/summary.json` contains the complete diagnostic.
The captured activation NPZ files are preserved locally and excluded from this
publication following automatic approval review. Their hashes and sizes are
recorded in `local-activation-manifest.json`; no checkpoint weights are published.
`fp32-control/remote/artifacts/summary.json` contains the FP32 control.
The original result archive was downloaded and SHA-256 verified; its receipt
is in `completion.json`. This directory publishes its numerical evidence and
provenance, with the activation tensors excluded. The package manifest identifies
exactly which files are published.
