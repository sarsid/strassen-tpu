# Full 168-shape v6e experiment

Authorized 2026-09-23. Cohort: `runs/20260923-v6e-suite-v001`. Live dashboard: http://127.0.0.1:8778/.

All 168 original development shapes are retained in their original order. This is a synthetic matrix-multiplication study, not full LLM inference or prediction accuracy. It does not establish performance on unseen shapes. Only Native, classical Pallas and Strassen recursion depths 1 and 2 are in scope.

## Hardware and implementation

One v6e allocation, `tpu-v6e1-s-kkb-euw4a0-2z99a9s1u71hn`, session `strassen-v6e-suite-20260923`. JAX/jaxlib 0.11.2, libtpu 0.0.48, xprof-nightly 2.24.2a20260922. Exact endpoint, boot identity, compiler versions and devices are checked in every phase. No concurrent measurements or silent mixing across allocations.

The initial setup control connection dropped, but installation and the TPU probe completed remotely. The complete archive was recovered and all 11 listed files hash-verified. Setup was not rerun. Both the transport failure and successful recovery remain archived.

The already qualified kernels are unchanged. Revised S1 uses panel or deferred reconstruction; revised S2 uses hybrid outer deferral. Both retain double buffering. Original S1/S2 implementations remain candidates, particularly for small shapes where reconstruction overhead may favor them. All custom kernels retain the 112 MiB compiler VMEM allowance.

## Frozen search

The common pool contains 20 tiles, all valid for depths 1 and 2. Per shape, tiles are ranked by padded matrix volume, then decreasing tile volume, then tile tuple. The first four are offered when any dimension is below 1024; otherwise the first six. This ranking uses dimensions only and was frozen before new timings.

| Family | Four-tile shapes | Six-tile shapes |
|---|---:|---:|
| Native: default and four scoped-VMEM settings | 5 | 5 |
| Classical Pallas: all offered tiles | 4 | 6 |
| S1: panel/deferred on all tiles, original on first two | 10 | 14 |
| S2: hybrid on all tiles, original on first two | 6 | 8 |
| Total | 25 | 33 |

There are 69 four-tile shapes and 99 six-tile shapes: 4,992 screening cases. Each case has three warmups and eight paired rotated timing rounds on common operands. Search budgets differ by family because the available reconstruction modes differ; this bounded search does not establish a globally optimal implementation.

Large padding costs on tiny matrices remain visible. No matrix is omitted because padding makes Strassen unattractive. All padding and cropping happen inside the timed complete call.

## Qualification and confirmation

Qualification covers `(M,K,N)` equal to `(1,1,1)`, `(16,4096,4096)`, `(1025,1025,1025)`, `(2049,4097,1025)` and `(8192,8192,8192)`. At each shape, default Native and six custom implementations at the first ranked tile must match exact ternary-integer algebra: 35 checks. Any qualification failure blocks the broad screen.

For each shape, every family independently freezes its fastest numerically eligible screening mean. If none passes, the fastest finite measured candidate can be retained as an explicitly ineligible diagnostic; if none executes, the absent result remains recorded. No selection is changed using confirmation results.

Confirmation adds default Native and classical controls at the exact selected S1 and S2 tiles, deduplicating identical candidates. Each selected arm receives three fresh Gaussian inputs, five warmups and 30 paired rotated timing rounds per input. The upper bound is 3,528 confirmation cases, or 105,840 timed calls. The realized count depends on deduplication and candidate availability.

Twelve batches each contain a 14-shape screen followed by fresh confirmation. Each phase is archived and hash-verified before advancement. Complete-call timings and separate first-input device traces are retained. Profiles contain 20 device observations per arm; low-level instrumentation is disabled and profiling observations never enter call-time intervals.

## Error and reporting contract

Inputs are Gaussian A/sqrt(K) and Gaussian B, quantized once to identical BF16 operands. Strassen preadditions round in BF16; all methods accumulate and output FP32. Host FP64 references use the exact quantized inputs and all K. Small references are full; larger ones sample 128 rows × 128 columns. Finiteness is checked over the complete output.

Unchanged gates require finite output, relative L2 ≤2%, and sampled/full maximum absolute error ≤0.001 + 0.05·max|reference|. Raw relative L2, maximum absolute error, RMSE, MAE, median/p99 absolute error and normwise error are retained. Passing these gates is not a model-quality guarantee.

The report compares both depths against default Native, tuned Native, independently tuned cubic and exact tile-matched cubic. It includes padding ratios, separate device times and categories for small/skinny, larger 2048-aligned and other larger dimensions. Pointwise 95% hierarchical paired intervals are conditional on frozen screening selection, without multiplicity adjustment; losses, numerical failures and missing outcomes remain visible.

## Validation and lifecycle

- Plan validation: `runs/20260923T204457Z-check-v6e-suite-plan-v001-29d1a3`. Verified complete shape coverage, 12-batch equivalence, matching classical controls, distinct confirmation inputs and absent-candidate handling.
- Reporter regression: `runs/20260923T204635Z-check-v6e-suite-report-v001-af5886`. Reproduced the completed ten-shape experiment's speedups, errors and 9/5 qualified win counts. Its fixture is explicitly labeled and contains no new TPU measurements.
- Setup recovery: `runs/20260923T204906Z-recover-v6e-suite-setup-v001-237c9e`.

The detached controller archives every batch, generates device analysis and the final report, then releases the owned allocation. It also releases a known-idle runtime after failure; uncertain remote execution is preserved for inspection. Phase advancement and retrieval require this Mac to remain powered and online. The controller prevents idle sleep.

Final output destination: `runs/20260923-v6e-suite-v001/operations/report/artifacts/RESULTS.md`. The existence of this protocol is not evidence that measurement or final reporting has completed; consult the cohort's live progress and immutable phase receipts.
