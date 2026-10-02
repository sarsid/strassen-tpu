# Review of the v5e to v6e performance gap

Reviewed 2026-09-23 from archived results and executed source. No new TPU run.
Future comparisons are limited to Native and Strassen depths 1 and 2.

We have performed bounded tile and compiler-memory tuning on v6e. We have not
completed a v6e-specific redesign or profile-guided optimization of the Strassen
kernels. The current evidence cannot separate recoverable implementation overhead
from a hardware balance that makes Strassen less advantageous.

## Matched Mistral projection dimensions

Real checkpoint weights and activations, BF16 inputs and FP32 outputs; independently
selected configurations on separate hardware allocations. These compare the same
matrix dimensions and timing scope, but are not a hardware-only controlled test.

| Projection (M,K,N) | Device | Tuned Native ms | Strassen 1 ms | Strassen 2 ms |
|---|---|---:|---:|---:|
| Down (2048,14336,4096) | v5e | 1.579318 | 1.436695 | 1.293687 |
| Down (2048,14336,4096) | v6e | 0.549972 | 0.592244 | 0.652303 |
| Gate/up (2048,4096,28672) | v5e | 2.766691 | 2.613884 | 2.397855 |
| Gate/up (2048,4096,28672) | v6e | 0.802257 | 0.977188 | 1.154971 |

For down projection, Native improves 2.87x between these configurations, Strassen 1
2.43x, and Strassen 2 1.98x. Strassen 2 changes from 18.09% lower latency than
Native on v5e to 18.61% higher latency on v6e. The reversal already occurs inside
the isolated MM; surrounding LLM operations are unnecessary to explain it.

Evidence: `m2048_down_confirmation_statistics.json` and
`m2048_gateup_confirmation_statistics.json` under:

- [v5e Mistral artifacts](../runs/20260923-mistral-resident-v5e-v001/phases/20260923-mistral-resident-v5e-v001-mistral-tune-ed1cf5/artifacts/)
- [v6e Mistral artifacts](../runs/20260922-llm-tradeoff-proplus-v6e-v001/phases/20260922-llm-tradeoff-proplus-v6e-v001-mistral-tune-3f85e4/artifacts/)

## Hardware interpretation and limits

The [JAX hardware reference](https://docs.jax.dev/en/latest/pallas/tpu/hardware.html)
lists approximately 197 versus 920 TFLOP/s BF16 compute and 820 versus 1640 GB/s
HBM bandwidth for v5e and v6e. Compute grows about 4.7x while HBM bandwidth grows
about 2x. Both list 128 MiB VMEM.

Strassen reduces leaf multiplication work while adding operand sums, output
reconstruction and scratch accesses. Those costs need not shrink in proportion
to MXU compute time. Consequently, saved multiplication time can be outweighed
by overhead on v6e. This is a plausible mechanism, not a measured bottleneck:
HBM specifications do not establish VMEM/vector-unit performance or prove that
our kernel is bandwidth-bound. The two-level kernel explicitly uses VMEM scratch
and does not materialize intermediate products in HBM.

## What tuning covered

- The real-model v6e search tested 16 tiles per custom family per projection,
  with a fixed 48 MiB custom VMEM allowance, and four Native compiler settings.
  See [the frozen configuration](../configs/llm_tradeoff_v6e_v001/campaign.json).
- The latest large-matrix search tested three tiles each for Strassen 1 and 2,
  with a 112 MiB allowance, and four Native settings. Both selected 2048-cubed
  tiles on all three shapes. See [the configuration](../configs/v6e_depth_v001/campaign.json)
  and [confirmed measurements](V6E_DEPTH_RESULTS_v001.md).
- Depths 1 and 2 retained their existing kernels. The compiler targets v6e, but
  the source fixes product order, output updates and scratch strategy. These
  searches did not compare alternative schedules, operand reuse strategies or
  pipeline designs on v6e. See [one-level kernels](../src/strassen_mm/kernels_v001.py)
  and [two-level kernel](../src/strassen_mm/kernels_two_level_v001.py).

Baseline tuning matters: on the latest 8192-cubed case, Native default averaged
1.805619 ms, tuned Native 1.582889 ms, and Strassen 1 1.640126 ms. Strassen 1 would
look 9.16% faster against default Native, yet is 3.62% slower than tuned Native.
The same confirmed inputs and timing protocol were used for these comparisons.

The latest larger cases leave Strassen 1 3.6–5.4% slower and Strassen 2 13.6–18.3%
slower than tuned Native. These quantify the current implementation gap, not a
proof that all v6e implementations lose.

The next useful experiment would profile Native, Strassen 1 and Strassen 2 on
identical shapes, inspect MXU utilization, vector work and memory stalls, then
test targeted tile, operand-reuse and scheduling changes while retaining the
same precision, error gates and fresh-input confirmation. Restoring a speedup
is an open experimental question.
