# Recovery of the 168-shape v6e experiment

On return from the laptop disconnection, Colab reported no active allocations. The old local controller had stopped after five failed progress queries in batch 10 screening. Its last queries explicitly reported that the saved endpoint was no longer allocated. The available logs do not establish the reason Colab released it.

The first nine screen/confirmation batches were already archived locally: **126 confirmed shapes**. All 18 phase archives were hash-verified before recovery. The partial original batch-10 screen remains preserved as an interrupted execution and is excluded from scientific comparisons. No confirmed shape is rerun.

Original cohort: `runs/20260923-v6e-suite-v001`.
Original allocation: `tpu-v6e1-s-kkb-euw4a0-2z99a9s1u71hn`.

Recovery cohort: `runs/20260924-v6e-suite-recovery-v001`.
Replacement allocation: `tpu-v6e1-s-kkb-euw4a2-2hl4rl9ox1zzk`.
Session: `strassen-v6e-suite-recovery-20260924`.
Live progress: http://127.0.0.1:8779/.

The recovery plan runs the same 35 exact-integer qualification checks on the replacement TPU, then batches 10–12: **42 shapes, 1,386 screening cases**, followed by fresh paired confirmation. Kernel and scientific configuration files were verified byte-for-byte against the original frozen source. Compiler pins remain JAX/jaxlib 0.11.2 and libtpu 0.0.48. The replacement setup and single-v6e probe passed.

Each recovered shape's screen, selected Native controls, Strassen choices, matching classical controls and fresh confirmation all run on the replacement allocation. No incomplete original screen measurements are reused for selection. No paired timing samples are pooled across allocations.

The final combined report joins distinct completed shape rows from the two allocations: 126 original and 42 recovery. It records the allocation and cohort for every row. Separate device traces are analyzed after confirmation, and the replacement runtime is automatically released after report retrieval. These remain synthetic MM measurements, not a new full-model experiment.

Verification evidence:

- `runs/20260924T003947Z-check-v6e-suite-recovery-v001-eea80c/artifacts/summary.json`: 18 sealed archives verified, exact disjoint coverage of 126 + 42 shapes, unchanged scientific sources.
- `runs/20260924T003902Z-setup-v6e-suite-recovery-v001-97ae50/artifacts/summary.json`: successful replacement setup and identity capture.
- `runs/20260924-v6e-suite-recovery-v001/plan.json`: frozen continuation and automatic release plan.

Expected combined report destination: `runs/20260924-v6e-suite-recovery-v001/operations/combined-report/artifacts/report/RESULTS.md`. A destination path or protocol document is not evidence of completed execution; the live progress and sealed phase receipts determine current completion.
