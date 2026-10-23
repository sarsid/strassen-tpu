# Current publication: LLM Benchmark 1 (Qwen3-32B)

Start with `benchmarks/llm/llm_benchmark_1_qwen3_32b/README.md` for the completed
October 16, 2026 study. Six batches at sequence 1,024, four algorithms (Native,
Tuned Native, Tuned Cubic, Tuned Strassen S1), BF16 output on v6e. All selected
measurements and quality gates completed, results were preserved, and the TPU
was released. This is an offline publication; do not launch any older queue or
new model automatically. The user will discuss the next model separately.

Keep B64's JAX 0.11.2/libtpu 0.0.48 extension distinct from B1-B32 on JAX
0.7.2/libtpu 0.0.21.1. Keep resident layer totals distinct from serialized
offloaded prefill. The B64 winning profile is exactly recovered, while its
historical candidate ledger is partial. Final measurements are complete.
Run `python3 analyze.py --check` inside the benchmark directory to verify and
reconstruct the report offline. Preserve existing source/evidence snapshots.

The October 5 handoff below is historical and its five-arm campaign remains
paused. It does not override the current four-arm benchmark methodology.

# Faster Strassen TPU research handoff (historical October 5)

For the user's benchmark/LLM research project, start with
`benchmarks/llm/20261005_v001/README.md`. The parent repository's root kernels and
older `Strassen_MM_Focus/` and `benchmarks/code/` trees are historical or parent
implementations, not the current LLM research snapshot.

The LLM experiment is operator-paused: 80/200 completed BF16 comparisons across
16 whole workloads. The previous 75-comparison publication is historical.
Migration and environment setup do not authorize TPU allocation, checkpoint
downloads, queue/watchdog launch, monitor resumption, or new experiments.
Wait for explicit user resumption; retain the Gemma4 qualification barrier.

Read the newest operator-pause and scope instructions before the chronological
history in the snapshot's `code/AGENTS.md`. Maintain a progress log using its
`status/PROGRESS_WORKFLOW_v002.md`. Do not treat localhost URLs or archived
absolute paths as cloud-accessible services.

All snapshots and measured source versions are evidence. Create new versions
for changes and record source/environment/timing/error provenance. Keep v5e and
v6e tuning separate, preserve all five algorithms, and distinguish isolated
projection, warmed layer, streamed compute, and full streamed-forward timings.
Never label CPU checks as TPU performance or synthetic Gemma4 checks as official
checkpoint qualification. Keep secrets and model weights out of Git.

For initial cloud setup, run the snapshot's `verify.py` and CPU-only offline
report reconstruction as documented. No TPU credentials are needed for this.
