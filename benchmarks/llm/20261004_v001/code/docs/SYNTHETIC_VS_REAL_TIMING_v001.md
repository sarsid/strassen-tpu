# Why the synthetic and real-model timing comparisons differ

This is a review of archived measurements and executed source, not a new TPU run.

If a replacement changes only the targeted matrix multiplications, with identical
surrounding work, then `T_model = T_target_MM + T_unchanged` and the absolute change
in model latency should approximately equal the change in targeted MM latency.
Unchanged work dilutes a percentage gain; it cannot, by itself, turn a true MM
latency reduction into a full-model latency increase.

Our dense kernels have no value-dependent sparsity skipping or routing. Merely
replacing random entries with checkpoint weights should not change their work
count at a fixed shape, dtype, layout and compiled kernel. The archives support
this distinction, rather than a claim that trained values inherently run slower.

## Same Qwen3-8B down-projection dimensions across the archives

All three rows have `(M,K,N) = (2048,12288,4096)`, measuring an isolated complete
device MM call with BF16 inputs and FP32 output. Native is independently tuned.

| Inputs and runtime | Native ms | Strassen 2 ms | Strassen / Native latency |
|---|---:|---:|---:|
| Random matrices, v5e | 1.387752 | 1.149567 | 0.8284 |
| Actual checkpoint/activations, v5e | 1.403724 | 1.168205 | 0.8322 |
| Actual checkpoint/activations, v6e | 0.509934 | 0.597820 | 1.1723 |

Random and actual operands give similar v5e results. The reversal is observed
between the v5e and v6e configurations: both methods get faster, but Native gets
much faster relative to Strassen. These are separate allocations and tuning runs,
not a hardware-only causal experiment. The two v5e Strassen tiles are both
`(BM,BN,BK)=(2048,1024,1024)`; the selected v6e tile is `(1024,2048,1024)`.
Compiler choices, allocations and selected tiles must not be attributed to weight
values. A paired test on one allocation with one fixed executable would isolate
the random-versus-real effect more tightly.

Evidence:

- [Synthetic shape summary](../runs/20260921T212710Z-strassen2-interim-scan-v001-6331e5/artifacts/shape_results.json), row `qwen_3_8b_down_m2048`.
- [Real v5e MM confirmation](../runs/20260922-large-real-v001/phases/20260922-large-real-v001-qwen-actual-bd2c37/artifacts/m2048_down_confirmation_statistics.json).
- [Real v6e MM confirmation](../runs/20260922-llm-tradeoff-proplus-v6e-v001/phases/20260922-llm-tradeoff-proplus-v6e-v001-qwen-tune-700a14/artifacts/m2048_down_confirmation_statistics.json).

## The work surrounding the target MM

The composed Native and Strassen arms share the attention prefix, activation,
residual suffix, embeddings, final normalization and vocabulary head. Only MLP
gate/up and down products are replaced. Attention's matrix products stay Native.
The same five-call layer structure is used by these composed arms. See
[Layer.__call__](../src/strassen_mm/model_composed_v002.py) and
[resident forward timing](../src/strassen_mm/benchmark_llm_tradeoff_v001.py).

The resident timings exclude checkpoint reads, host layout conversion, weight
transfers, tokenization and compilation. Those costs cannot explain the latest
resident result. The older streamed forward measurements had a different scope.

For Qwen3-8B on v6e, isolated gate/up plus down calls take 1.240868 ms with tuned
Native and 1.402538 ms with Strassen 1. Their difference times 36 layers is
5.820134 ms. The observed resident full-model difference is 5.354900 ms
(78.819184 minus 73.464284 ms). Strassen 2 gives a corresponding estimate of
13.251879 ms versus an observed 14.703680 ms. These agree in direction and scale:
the targeted MM is already slower before integrating it into the model.

This multiplication is only a consistency check. Isolated timings include their
own dispatch/synchronization, use first-layer weights, and are not an in-model
device trace. It does not measure or prove that every non-MM component takes
exactly the same time. JAX dispatch is asynchronous, so standalone synchronized
timings need not sum exactly to a multi-operation forward. See the
[official dispatch documentation](https://docs.jax.dev/en/latest/async_dispatch.html).

## Best Native and matched-structure Native answer different questions

The fastest measured Native control can compile a whole layer together, whereas
the composed arms compile the two replaceable projections independently. Thus a
comparison against the best Native control also measures differences in fusion
and dispatch structure. It answers the practical implementation comparison, not
the isolated effect of swapping one MM implementation.

For Mistral, whole-layer Native takes 64.928 ms, composed tuned Native 73.912 ms,
Strassen 1 81.841 ms and Strassen 2 89.588 ms. Consequently the matched-composition
slowdowns are 10.73% and 21.21%; the best-Native slowdowns are 26.05% and 37.98%.
Both comparisons should remain explicit.

[Full resident timing and accuracy report](../runs/20260922T175926Z-llm-tradeoff-analysis-v002-b19e5f/artifacts/RESULTS.md).

The existing evidence supports a hardware/configuration-dependent reversal and
an already slower target MM in the v6e runs. It does not support explaining the
reversal as an intrinsic cost of real weights or simply as unchanged LLM work
outside the MM. Further attribution would require same-executable operand swaps
and paired in-model device traces, without retuning between operand types.
