# Independent interpretation of the completed power/midpoint study

The data support a geometry-dependent Strassen benefit: the two largest tested squares and long-K products beat both tuned native XLA and tuned full cubic, while simply making M or N very large does not produce the same result. Independently tuning native materially narrows the apparent Strassen advantage. No universal size threshold, rectangular tile range, hardware mechanism or held-out generalization follows from this campaign.

All shapes below use **(M,N,K)** for A[M,K]B[K,N]. Tiles use **(BM,BN,BK)**. Speedup is reference mean / Strassen mean, so values above one favor Strassen. Unless stated otherwise, observations are complete device calls and intervals are individual paired 95% bootstrap intervals without multiplicity correction.

## Evidence reviewed

The canonical [descriptive findings](../runs/20260920T065050Z-power-grid-final-summary-v002-f8205b/artifacts/findings.json) have SHA256 `f321adb018e7d6acc2c62f5eddbbe1fad23612152746d68dcf158ef6c1af7a87`; this review independently matched that hash to the directory's seal. The source [screen](../runs/20260920T034417Z-GRID-screen-v5e-v004-8046c8/artifacts/selections.json) freezes choices before the [confirmation journal](../runs/20260920T054326Z-GRID-confirm-v5e-v004-00a15b/artifacts/results.jsonl).

The summary reconciles 1,440 screening groups/6,240 scoped outcomes and 489 confirmation groups/2,176 scoped outcomes. Every scoped outcome is `ok`, all recorded numerical gates pass, and the failure inventory is empty. Those counts include two timing scopes and repeated reference controls; they are not counts of independent experiments or unique shapes. There are 60 deliberately chosen exploratory shapes, including ten off-lattice boundary probes, and twelve reserved shapes with no observations.

For a bounded independent check, this review reread the confirmation journal for five examples: 4096³, 8192³, 16384³, (2048,2048,131072), and 4097³. All 40 selected headline scoped means matched their 1,200 raw samples within 1e-12 ms. This is a raw-mean spot check and interpretation review, not a claim to have independently repeated the campaign or its full bootstrap audit.

## The practical baseline changes the conclusion

| Strassen compared with | Wins | Inconclusive | Losses | Median observed ratio |
|---|---:|---:|---:|---:|
| Frozen tuned native | 8 | 13 | 39 | 0.9583× |
| Native library defaults | 18 | 15 | 27 | 0.9803× |
| Independently tuned full cubic | 35 | 19 | 6 | 1.0224× |

These are nominal per-shape classifications for the deliberate sample, not win prevalence over the 39,304-shape lattice. Seven shapes beat both tuned native and cubic; the eighth native-relative win, (768,3072,12288), is inconclusive against cubic.

Native's frozen compiler-option choice beats default native in 35 complete-call comparisons, with no nominal losses. The remaining 25 include ten identical choices measured through the same route. Native tuning means a bounded default/32/48/64 MiB compiler-option screen, not recovery or direct control of its internal tile triple. The example (768,3072,12288) falls from 1.1726× versus default native to 1.0251× versus tuned native, making the distinction important in the headline.

## Square scaling is favorable at the largest measured points

| Square extent | Tuned native ms | Cubic ms | Strassen ms | Native / Strassen [CI95] | Cubic / Strassen |
|---|---:|---:|---:|---|---:|
| 2048 | 0.332139 | 0.361814 | 0.350599 | 0.9473× [0.9276,0.9676] | 1.0320× |
| 4096 | 0.966067 | 1.013124 | 1.001385 | 0.9647× [0.9150,0.9959] | 1.0117× |
| 8192 | 6.012079 | 6.150261 | 5.638621 | 1.0662× [1.0637,1.0687] | 1.0907× |
| 16384 | 45.568623 | 46.077647 | 41.976937 | 1.0856× [1.0839,1.0874] | 1.0977× |

The largest two squares retain the same direction in prepared scope: native/Strassen is 1.0625× and 1.0839×. This supports the user's large-square hypothesis locally. It does not locate a precise crossover between tested points or establish monotonic improvement for all larger shapes. The final manifest includes 2048³; draft planning discussions that described it as unmeasured are superseded by the final manifest and evidence.

The 4096³ complete/prepared ranking changes: complete call is 0.9647× against native while prepared is 1.0402× [1.0313,1.0488]. Strassen's complete-call mean is 1.001385 ms, median 0.968070 ms and maximum 1.702100 ms. Preserve the registered mean and interval, including that sample; do not replace them with a favorable median or omit the point. Both chosen custom tiles are aligned, so this particular scope difference cannot simply be called measured padding overhead. Compiler behavior, scheduling and timing variability were not isolated.

## Equal volume does not imply the same tradeoff

The following four shapes have the same MNK, use the same Strassen tile (2048,2048,512), require no Strassen padding, and have identical source-estimated dot and vector work. Their measured results nevertheless differ.

| Shape (M,N,K) | Strassen ms | Tuned native / Strassen [CI95] | Logical output tiles × sequential K panels |
|---|---:|---|---:|
| (8192,8192,8192) | 5.638621 | 1.0662× [1.0637,1.0687] | 16 × 16 |
| (131072,2048,2048) | 6.190263 | 0.9660× [0.9630,0.9693] | 64 × 4 |
| (2048,131072,2048) | 6.086874 | 0.9793× [0.9766,0.9819] | 64 × 4 |
| (2048,2048,131072) | 5.463647 | 1.0939× [1.0904,1.0972] | 1 × 256 |

In the K-axis series with M=N=2048, complete-call native/Strassen grows from 1.0540× at K=16384 to 1.0758×, 1.0815× and 1.0939× at K=32768,65536,131072. In contrast, the corresponding long-M and long-N points lose to tuned native. Their results support retaining directional dimensions, output-grid size, K-panel length and buffer footprints in a future model. Logical grid counts are not measured physical concurrency, occupancy or evidence of vector/MXU overlap.

The tile-local algorithm uses seven instead of eight half-tile products repeatedly; at fixed tile, both implementations remain Θ(MNK). Increasing the global matrix does not make the repeated pre-add work disappear. The equal-volume example is especially useful because even the elementary source-operation counts are held constant: a FLOP-count-only model cannot explain all of the observed variation. No fitted predictor or causal hardware explanation was produced in this study.

## Padding boundaries are informative but not isolated causal tests

At 4097³, both frozen custom choices use tile (1536,1536,512), pad to 4608³, and have padded-volume ratio 1.4228. Strassen still beats cubic at 1.0413×, but loses substantially to tuned native: 1.709425 versus 1.034112 ms, speedup 0.6049× [0.6005,0.6098]. The nearby aligned 4096³ Strassen mean is 1.001385 ms, but its chosen tile differs, so the boundary comparison is not an experiment changing only padding.

At 1025³, the selected cubic padding ratio is 1.4197 and Strassen's is 2.5238. Thus the Strassen/cubic padding ratio is 16/9, and its estimated padded dot work is **14/9≈1.556 times** cubic's despite the seven-product formula. Cubic/Strassen measured speedup is 0.9199× [0.8998,0.9403]. This is a concrete counterexample to treating an unqualified 12.5% arithmetic saving as the executed-work saving for independently selected padded shapes.

Prepared scope cannot stand in for complete-call performance. For example, (2048,2048,8191) changes from a 0.8829× native-relative complete-call loss to a 1.0830× prepared win. Conversely, (16384,4096,256) wins complete call at 1.0459× but loses prepared at 0.8551×. Keep both; their difference is not a directly measured preparation or dispatch cost.

## Competitive tile sets are discrete and useful

For **both 8192³ and 16384³**, four Strassen tuples are within 5% of the frozen screening winner in confirmation, and each recorded pointwise CI upper bound also lies below 1.05:

| Tuple (BM,BN,BK) | 8192³ candidate / frozen-reference latency [CI95] | 16384³ candidate / frozen-reference latency [CI95] |
|---|---|---|
| (2048,2048,512) | 1.0000 [1.0000,1.0000] | 1.0000 [1.0000,1.0000] |
| (2048,1024,512) | 1.0304 [1.0231,1.0429] | 1.0272 [1.0262,1.0282] |
| (1024,2048,512) | 1.0224 [1.0140,1.0282] | 1.0281 [1.0269,1.0291] |
| (1024,1024,1024) | 1.0327 [1.0302,1.0353] | 1.0434 [1.0421,1.0445] |

Ratios in this table are candidate/reference latency, so greater than one means slower—the reverse of the speedup tables. The winner's [1,1] interval is an identity comparison, not independent evidence of certainty. Each non-winner is paired with the frozen reference in its own registered group; reference observations from separate groups are not pooled.

These examples give a useful measured alternative to a single tile choice. They do **not** establish the entire box BM∈[1024,2048], BN∈[1024,2048], BK∈[512,1024]. Even restricting those axes to the two displayed discrete values would yield eight Cartesian combinations, whereas only the four listed tuples have this measured qualification. Some other combinations were not tested, and confirmation shortlists were capped at six per family.

At the 4097³ boundary, three Strassen tuples are descriptively within 5%, but only the frozen (1536,1536,512) tuple satisfies the pointwise upper-bound criterion. Thus a descriptive near set and a confidence-qualified tolerance set need not coincide. Neither is a simultaneous equivalence region or a guarantee relative to the unknown global optimum.

## Numerical scope and remaining limits

Every recorded screen and confirmation case passed the fixed finite/relative-L2/max-absolute-error gate. The largest confirmed Strassen relative L2 is 0.0044944 (about 0.449%), and its largest absolute error is 0.033069. Confirmed native and cubic maximum relative L2 values are about 6.98e-7 and 4.90e-7 respectively. Strassen therefore passes the stated Gaussian gate with appreciably larger numerical error; the result should not be called equal accuracy.

Large-shape references check saved output samples using every K term, alongside whole-output finiteness. This does not certify every output entry's error, cancellation robustness, arbitrary activations or model accuracy. The absence of failures in this bounded feasible sample does not establish that the full lattice fits or compiles. Twelve holdouts remain untouched, and thirty timing rounds within a cohort are not independent hardware replications.

This review created no new measurements, source/configuration changes or commits. The supported outcome is a useful measured shape-dependent performance and tile-tolerance map, with clear large-square/deep-K successes, substantial native-relative losses elsewhere, and explicit numerical and statistical limits.
