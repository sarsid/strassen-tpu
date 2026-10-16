# LLM Benchmark 2 (Gemma4-31B)

Preparation checkpoint, October 16, 2026. The authorized run is queued and
waiting for a Colab v6e allocation. **No TPU timing or real-checkpoint
qualification result is claimed by this folder yet.** Future results will be
added as separate immutable run artifacts.

The four arms are Native, Tuned Native, Tuned Cubic and Tuned Strassen S1,
using the unchanged pinned parent kernels with a Gemma4 dense-text adapter.
BF16 output; sequence length 1,024; batches 1, 4, 8, 16, 32 and 64. Execution
order is 8, 16, 32, 64, 4, 1. Local/global attention layers receive independent
tuning. All batches use JAX/jaxlib 0.11.2 and libtpu 0.0.48.

The official checkpoint reference comparison is a mandatory barrier before
performance measurements. Preserve projection/block timing, the sum of warmed
resident layer means, transfer-inclusive offloaded prefill, and accuracy as
separate measurements. See [the frozen protocol](PROTOCOL.md) and [plan](plan.json).

Local validation passed: both attention types against the official Transformers
reference, parent fused projections in the Pallas CPU interpreter, legal tile
menus, and controller timeout/checkpoint-integrity recovery. These are preparation
evidence, not real-weight TPU results. The held-out corpus contains enough Gemma
tokens for the largest registered quality workload.

`source/source.tar` contains the frozen execution tree; its manifest records
all file hashes and the local source commit. Extract into an isolated directory.
The supervisor requires the existing local Colab controller environment and
explicit `--start-authorized`; no credentials or model weights are included.
`validation/` preserves the preflight checks. `dashboard/` contains the new local
live-log view; it reads public run status files and is not a hosted service.

The live dashboard on the experiment laptop is <http://127.0.0.1:8790/>. It shows
current batch, stage, algorithm, layer type, tuning candidate/tile and heartbeat,
plus a six-batch progress table. It does not imply the laptop's local server is
reachable from another machine.

[LLM Benchmark 1 (Qwen3-32B)](../llm_benchmark_1_qwen3_32b/) is preserved separately.
