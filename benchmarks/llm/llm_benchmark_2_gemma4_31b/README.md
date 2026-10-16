# LLM Benchmark 2 (Gemma4-31B)

The initial v001 run completed all 60 official-checkpoint comparison layers but
failed the propagated hidden-state relative-L2 gate: seven layers exceeded 2%,
with a maximum of 2.67%. Final logits and prediction checks passed on the short
32-token qualification input. No tuning or batch timing ran. The owned TPU was
released. [Full failure evidence](runs/20261016-gemma4-31b-v001/) is preserved.

Version 002 restored BF16 attention scores before FP32 softmax, matching official
eager Torch. Local attention tests improved by 52x/88x, but the new real-checkpoint
run still failed full-depth qualification: maximum propagated hidden error was
2.372% versus the 2% limit (historical path: 2.670%). All 60 isolated layers passed,
with maximum error 0.282%. Final logit error, NLL and KL passed, while top-choice
agreement was 30/31 = 96.774%, below 97%. On this short 31-position comparison,
that threshold requires 31/31 matches. No threshold has been relaxed.

The attention repair addressed one discrepancy but the remaining arithmetic
differences have not been localized. **Neither run reached tuning or performance
measurement.** Both owned TPUs were released and both failed attempts preserved.
See [v002 sources, validation and full failure evidence](runs/20261016-gemma4-31b-v002/).
A matched-intermediate numerical diagnosis is needed before another full sweep.

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
