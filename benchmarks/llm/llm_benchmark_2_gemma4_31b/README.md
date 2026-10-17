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

A [controlled numerical diagnostic](diagnostics/20261016_operator_trace_v001/) is prepared to compare identical real inputs operation by operation in official Torch, JAX CPU and JAX v6e. Its local validation and review of the parent public-v2 fidelity method are preserved; no real diagnostic result is claimed by this preparation checkpoint.
The user subsequently requested retaining the TPU after errors and troubleshooting on the same machine. [The validated retention controller](operations/retain_on_error_v001/source/protocols/GEMMA4_RETAIN_ON_ERROR_v001.md) supersedes automatic release during diagnosis, including after diagnostic evidence collection. The original remote numerical code is unchanged; the replacement supervisor reuses the same tracked allocation request. Both failed local test-fixture attempts and the passing fault-injection run are preserved.

The [completed numerical diagnosis](diagnostics/20261016_operator_trace_v001/completed_v001/) found comparable accumulated BF16 drift on JAX CPU and TPU, plus one exact-tie-sensitive prediction disagreement. Independent real-weight FP32 formula checks agree within 0.00005% relative L2. The existing failed gates remain failed; the parent-style validation split was subsequently approved by the user. The prior TPU disappeared from Colab without a controller release; see the separate v003 restart below.

The [approved validation split and v003 restart](operations/approved_validation_v002/) use hash-verified completed CPU FP32 fidelity evidence, unchanged same-TPU BF16 algorithm gates, and BOS per natural-text sequence. Local checks passed. A replacement-allocation supervisor started at 14:43 UTC October 16; no new performance result is claimed by this checkpoint. This protocol supersedes the earlier cross-backend BF16 stopping rule for new attempts only.

[The v003 qualification checkpoint](checkpoints/20261016_v003_qualification/) records successful source-checked reuse of the approved FP32 control on a new owned v6e. B8 tuning was running at 15:00 UTC October 16; batch accuracy and performance outcomes are still pending.

[B8 tuning and block qualification](checkpoints/20261016_v003_b8_tune_qualify/) completed successfully. Both representative attention blocks passed the unchanged numerical limits; resident confirmation was active at 15:33 UTC. Full-model quality and speedup conclusions remain pending.

[B8 resident confirmation and full-depth quality](checkpoints/20261016_v003_b8_confirm_quality/) are complete: Strassen 935.69 ms summed resident-layer time versus 1207.60 ms Native and 1066.00 ms Tuned Native, with all repetitive/natural quality gates passed. Transfer-inclusive prefill was still running at 16:06 UTC; these resident sums are not total offloaded inference latency.

**[B8 is fully complete](results/20261016_v003/b8/)** with all quality gates passed and raw artifacts preserved. Strassen retains the 22.52% resident-layer time saving versus Native, while serialized offloaded prefill is effectively unchanged (239.586 s versus 240.251 s; only 0.276% difference with overlapping sample ranges). B16 tuning started automatically on the same owned TPU.

[B16 resident confirmation and quality](checkpoints/20261016_v003_b16_confirm_quality/) passed all unchanged gates: Strassen 1858.854 ms, Native 2445.893 ms, Tuned Native 2091.712 ms summed resident-layer time (24.0% and 11.1% less time). Its transfer-inclusive prefill test was running at 18:16 UTC.

**[B16 is fully complete](results/20261016_v003/b16/)** with all quality gates passed and raw artifacts preserved. Strassen resident time is 24.0% lower than Native; serialized offloaded means are 241.168 s versus 242.562 s (0.575% difference, with overlapping sample ranges). B32 tuning has started on the same owned TPU; two of six batches are complete.

[B32 resident confirmation and quality](checkpoints/20261016_v003_b32_confirm_quality/) passed all unchanged gates: Strassen 3769.584 ms, Native 4796.475 ms, Tuned Native 4151.352 ms summed resident-layer time (21.4% and 9.2% less time). Transfer-inclusive prefill was running at 20:26 UTC.

**[B32 is fully complete](results/20261016_v003/b32/)** with all quality gates passed and raw artifacts preserved. Strassen resident time is 21.4% lower than Native and 9.2% lower than Tuned Native. Serialized offloaded means are 246.267 s versus 247.072 s Native (0.326% difference, with overlapping sample ranges). B64 tuning has started automatically on the same owned TPU; three of six batches are complete.

[B64 tuning and block qualification passed, but confirmation failed in untimed error reporting](results/20261016_v003/b64_failed_v001/): a 5.25 GiB FP32 difference allocation exceeded free device memory. Both failures are preserved. The TPU remains allocated and B4/B1 continue. A [locally validated bounded-memory diagnostic repair](operations/b64_metric_memory_repair_v001/) is prepared for a B64-only retry afterward with the saved tuning; no repaired TPU performance result is claimed yet.

[B4 resident confirmation and natural quality](checkpoints/20261016_v003_b4_confirm_quality/) passed all unchanged gates: Strassen 466.894 ms versus Native 562.608 ms and Tuned Native 515.937 ms summed resident-layer time (17.0% and 9.5% less time). Natural perplexity changes by +0.000355%. Transfer-inclusive prefill is still running; B1 and the prepared B64 retry follow.

**[B4 is fully complete](results/20261016_v003/b4/)** with all quality gates passed and the full raw archive preserved. Strassen retains 17.0% resident time savings versus Native and 9.5% versus Tuned Native. Offloaded means are 242.184 s Strassen versus 242.690 s Native (0.209% difference, with overlapping sample ranges). B1 tuning is running; the validated B64-only recovery follows on the retained TPU.

[B1 resident confirmation and natural quality](checkpoints/20261016_v003_b1_confirm_quality/) passed all unchanged gates: Strassen 121.924 ms versus Native 137.584 ms and Tuned Native 131.293 ms summed resident-layer time (11.4% and 7.1% less time). Natural perplexity changes by +0.0425%. Offloaded prefill is running; the prepared B64 retry follows.
