# Larger LLM MM mini study — execution decisions

Recorded before measurement on 2026-09-21 UTC.

- The previous region-grid campaign completed and released its TPU. A fresh allocation inventory returned no assignments. This study uses a new single V5E1 endpoint throughout; it does not claim the previous campaign's physical machine or pool timings across allocations.
- Compare synthetic matrix products at dimensions verified from Qwen3-8B, Mistral-7B-v0.3, and Gemma3-12B's text configuration. Each supplies concatenated gate/up and down shapes at M=2048,8192,16384 (18 shapes). These are flattened token-row counts, not full-model runs or claims about context length.
- Preserve the existing BF16 input/pre-add, FP32 accumulate/output contract and numerical thresholds. Gemma's H=3840 stays the logical dimension; complete-call measurements include any necessary padding and crop.
- Reuse the frozen power-grid runner and kernels. The region-grid wrapper is inappropriate because it imposes another campaign's shape-role and holdout requirements.
- Independently tune cubic and one-level Strassen with the same six tile candidates per shape. Native XLA uses its ordinary compiler default. The six-candidate budget is a bounded search, not proof of a global optimum.
- After two-shape smoke, screen with two warmups/seven timing repeats. Freeze one eligible winner per family and shape, then confirm with five warmups/30 repeats and a fresh input seed. Confirmation is on already tuned shapes, not held-out dimensions.
- The smoke includes a small Qwen concatenated gate/up shape and the largest Gemma concatenated gate/up shape. Every smoke case must pass before screening.
- Count smoke separately. The planned main study has 108 screening groups and 18 confirmation groups: 42 groups per model. A completed group can contain a failed candidate; the dashboard must show completion and eligibility separately.
- Preserve failed candidates and numerical errors. Relative speed claims require eligible measurements in both compared arms. Report complete-call latency as the headline and prepared-kernel latency as a secondary view.
- Retain 48 MiB kernel VMEM and an 8 GiB estimated live-device budget. Preflight estimates exclude unmeasured compiler/runtime peaks and do not guarantee successful compilation.
- Run serially on one verified allocation, using an identical source archive across smoke/screen/confirmation. Pin JAX 0.7.2, jaxlib 0.7.2, libtpu 0.0.21.1. Check full smoke environment identity on subsequent phases.
- Archive source/configuration, chronological measurements, errors, hashes, phase receipts and logs with scoped commits. Do not stage the unrelated region-grid analysis files. Release this exact allocation after verified retrieval; retain it for recovery if remote completion is uncertain.
- Two-level Strassen, real weights/activations, full-model quality/inference and v6e replication remain separate experiments.
