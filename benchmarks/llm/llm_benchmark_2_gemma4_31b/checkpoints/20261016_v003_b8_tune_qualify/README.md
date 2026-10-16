# B8 tuning and numerical block qualification

Gemma4-31B, BF16, sequence 1024, v6e: tuning finished in 2279.38 seconds;
block qualification finished in 196.63 seconds. Both representative blocks
passed unchanged finite-output and relative-L2 checks against same-TPU Native.

| Algorithm | Local block relative L2 | Global block relative L2 |
|---|---:|---:|
| Tuned Native | 0.008339% | 0.000912% |
| Tuned Cubic | 0.011975% | 0.028548% |
| Tuned Strassen S1 | 0.574793% | 0.170212% |

The per-block limit is 2%; segmented capture reconstruction also passed its 1%
limit. These two-block checks do not establish full-model prediction quality.
Resident confirmation was active at this checkpoint; full-depth quality and
completed performance comparisons remain pending. The frozen profiles include
separate local/global tiles and Native compiler settings. Full candidate traces
will be preserved with the batch archive. This checkpoint is not a speedup claim.
