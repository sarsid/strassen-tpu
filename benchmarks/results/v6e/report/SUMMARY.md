# Final v6e results: 168 shapes

All 168 geometries are complete in FP32 and BF16 output. Measurements, tuning choices and numerical errors are retained separately for each output precision.

Wins and losses require the paired pointwise 95% interval to lie entirely above or below 1. Intervals crossing 1 are unresolved. There is no multiple-comparison adjustment; these Gaussian inputs and shapes are development data.

| Output | Method | Wins vs tuned Native | Losses | Unresolved |
|---|---|---:|---:|---:|
| float32 | cubic | 22 | 98 | 48 |
| float32 | s1 | 50 | 78 | 40 |
| float32 | s2 | 11 | 132 | 25 |
| bfloat16 | cubic | 23 | 99 | 46 |
| bfloat16 | s1 | 52 | 73 | 43 |
| bfloat16 | s2 | 17 | 124 | 27 |

## Numerical error

Each entry summarizes the worst of three fresh seeds per shape, relative to FP64 on exact BF16 operands. The reference uses the full output or 128 × 128 sampled entries over all K; finiteness is checked over the full output. Numerical eligibility does not establish LLM prediction quality.

| Output | Method | Median relative L2 % | Maximum relative L2 % |
|---|---|---:|---:|
| float32 | native | 0.000012 | 0.000042 |
| float32 | cubic | 0.000011 | 0.000023 |
| float32 | s1 | 0.440145 | 0.453157 |
| float32 | s2 | 0.981844 | 1.017440 |
| bfloat16 | native | 0.167049 | 0.278719 |
| bfloat16 | cubic | 0.167049 | 0.278719 |
| bfloat16 | s1 | 0.471232 | 0.483375 |
| bfloat16 | s2 | 0.996532 | 1.019354 |

[Full timings and decisions](RESULTS.md) · [Tuner design](TUNER_DESIGN.md).
