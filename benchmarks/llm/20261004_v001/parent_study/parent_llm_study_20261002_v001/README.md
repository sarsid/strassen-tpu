# Qwen3-32B parent replication and our fused-kernel comparison

**The parent's headline result is reproducible. Our current kernels beat Native,
but do not beat the parent implementation.** The workload is Qwen3-32B on one
v6e, batch 8, sequence 1024, with the pinned real checkpoint and parent token
inputs. Parent source is frozen at `95be1fb088656a89813b04492e1d77c66b36ccf9`.

The reported time is the sum of resident compute for 64 streamed layers.
Checkpoint I/O, compilation, weight preparation/transfer, embedding lookup and
final scoring are excluded. These are not end-to-end serving latency numbers;
the entire 32B model is not resident on this single device.

| Implementation / pass | Native reference ms | Candidate ms | Speedup vs Native |
|---|---:|---:|---:|
| Parent, original JAX 0.7.2 / libtpu 0.0.21.1 | 1130.351 | 872.050 | 1.2962x |
| Parent, current JAX 0.11.2 / libtpu 0.0.48 | 1129.029 | 832.429 | 1.3563x |
| Our fused cubic, BF16 (revised-S1 pass control) | 1127.106 | 931.773 | 1.2096x |
| Our S1, BF16, initial pass including timing stall | 1126.742 | 924.635 | 1.2186x |
| Our S1, BF16, unchanged-profile repeat | 1126.743 | 887.782 | 1.2692x |
| Our S1, BF16, revised down memory cap/accumulator | 1127.106 | 884.971 | 1.2736x |
| Our S2, BF16 | 1126.508 | 931.269 | 1.2096x |
| Our fused cubic, FP32 (S1 pass control) | 1127.164 | 990.611 | 1.1378x |
| Our S1, FP32 | 1127.164 | 959.115 | 1.1752x |
| Our S2, FP32 | 1127.846 | 990.124 | 1.1391x |

The published original-stack target was **1.2950x**. We reproduced the block
target too: **1.2093x**. All 64 original-stack per-layer error records and the
repetitive-text task metrics match the published artifact exactly. The newer
stack's extra speed is a software-environment effect, not an improvement in our
kernels. The parent's current-stack early-finalization sub-test fails: its
standard fused block was slightly faster than its product-aware schedule.

The parent and our full-model runs used separate allocations, with their own
matched Native references. The independent same-machine block comparison also
puts the parent ahead: **13.020 ms parent S1 versus 13.896 ms revised S1**; our
latency is about **6.7% higher**. Default
Native was 21.230 ms and tuned Native was 17.824 ms in that block comparison.
The parent's block and streamed scripts use different down tiles; the stronger
streamed policy is the direct block control above.

The initial S1/BF16 pass contains one **197.153625 ms** call at layer 51, beside
four calls of about 13.84 ms. Its five-sample mean remains in the original
result, and its speed-versus-cubic gate failed. Exactly one unchanged-profile
repeat was declared after inspecting this anomaly. It passed that gate and
reproduced every per-layer error record and repetitive-text task metric exactly.
Neither pass replaces the other. The timing audit flags no comparable excursion
in the other seven streamed passes. The stall's cause is not established.

The revised S1/BF16 policy is **1.0529x** the equally fused cubic control.
S2 adds essentially no speed over that control: BF16 is statistically tied;
FP32's formal gate passes by only **0.044%**. Thus fusion/tuning accounts for
much of the Native-to-candidate gain, and S1 supplies a further useful increment.
S2 is not an attractive additional recursion level for this fixed workload.

| Current-stack policy | WikiText PPL | Absolute NLL delta | KL (nats) | Top-1 agreement |
|---|---:|---:|---:|---:|
| Native | 8.457450 | 0 | 0 | 100% |
| Parent S1 | 8.456219 | 0.00014554 | 0.00151429 | 98.2527% |
| Our revised S1 BF16 | 8.455338 | 0.00024982 | 0.00139212 | 98.3168% |
| Our S2 BF16 | 8.454008 | 0.00040712 | 0.00190903 | 98.0022% |
| Our S1 FP32 | 8.455130 | 0.00027438 | 0.00126517 | 98.3016% |
| Our S2 FP32 | 8.458034 | 0.00006903 | 0.00184837 | 98.0602% |

All evaluated policies pass the declared natural-text quality gates over
32,736 scored positions; repetitive-text scoring covers another 8,184 positions
per pass. All reported layer outputs are finite. Small perplexity decreases do
not establish improved model accuracy. Agreement is against the parent Native
implementation, not an independent official-model qualification.

Errors are not zero or uniformly tiny. Revised S1/BF16 has WikiText logit
relative L2 **0.02910** and logit max-normalized error **0.05936**. Its worst
layer-mean hidden relative L2 on WikiText is **0.04975**; on repetitive text the
worst layer reaches **0.20420**. The raw ledgers retain per-layer hidden errors,
logit errors, NLL, perplexity, KL and agreement. WikiText's parent harness records
mean per-layer L2 across segments; the repetitive-text harness additionally
records maximum absolute error, RMSE and max-normalized hidden error.

BF16/FP32 labels denote custom projection stores. Both feed BF16 model consumers,
and conversion is timed. Native fallback sites retain the parent BF16 behavior.
Our Q/K norm uses `HIGHEST` dot precision; the parent uses `DEFAULT`. Consequently,
parent-versus-ours differences include normalization precision, layout and tile
policy, not just Strassen arithmetic. This study does not isolate their individual
shares of the remaining performance gap.

The complete screen considered 364 candidates: 352 measured, 10 compile failures,
and two estimate-based exclusions. A 66-candidate interleaved shortlist corrected
noise-sensitive Native normalization before model evaluation. The separate
down-cap audit attempted 12 parent-budget candidates plus the old S1 control;
nine compiled and four failed. It recovered the S1 BF16 product-accumulator tile
at 120 MiB. FP32 Native remained faster than successfully compiled alternatives;
some S2/product candidates still exceeded their cap or physical VMEM.

Only S1/BF16 down changed: `(bm,bn,bk)=(2048,2560,1024)`, two buffers, product
accumulators, 120 MiB. Its full-model time is about 0.32% below the unchanged
repeat; independent block confirmation did not show a corresponding improvement,
so this is a small, context-dependent gain. Other site choices remain frozen.
The profile permits Native at V and O. Complete tiles, caps, accumulator choices,
output dtypes, compiler options and errors are preserved in the profile files.

Evidence is separated into [exact parent reproduction](../parent_llm_replication_20261002_v001/README.md),
[main comparison](../parent_llm_comparison_20261002_v001/README.md), and
[declared follow-up](../parent_llm_followup_20261002_v001/README.md).
The [tuner design record](../../../docs/PARENT_LLM_TUNER_DESIGN_v001.md) explains
the search, precision contracts, corrections and limits. `summary.json` combines
these groups without pooling measurements; `provenance.json` records input hashes.

All 20 top-level experiment stages executed, archives and token/profile identities
were checked, and the owned TPU was released with verified absence. No v5e or
additional LLM work is queued.
