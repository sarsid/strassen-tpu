# Main v5e comparison and preliminary LLM shape study

The user authorized these measurements on September 21, 2026. The campaign is
a development dataset for a later inexpensive selector. These are not unseen
test shapes; a future experiment will evaluate any resulting rule on new shapes.
No AlphaTensor training or comparison is in this campaign.

Run the 18 earlier Qwen, Mistral, and Gemma synthetic projection shapes first,
alongside the separately recorded real-weight/model execution stage. Then run
all 168 distinct M,K,N tuples from the combined CSV. The source has 180 rows;
duplicates retain every original shape ID, sampling label, and CSV row number.
No old timing samples enter these new comparisons. All five methods use one
qualified v5e allocation; device changes require a new explicitly labeled cohort.
v6e replication is deferred.

## Candidates and tuning budget

Compare tuned Native, cubic, one-level Strassen, two-level Strassen, and default
Native. Each custom family has sixteen fixed candidate tiles. Eight come from
the recent depth-tuning probe, including the successful `(2048,1024,1024)` tile;
eight cover small/skinny geometries and useful prior configurations. The extra
tiles are family-specific and satisfy each kernel's alignment. The exact tuples
are the source of truth in `configs/mlsys_shapes_v001/campaign.json` and the
identical search in `configs/mlsys_llm_shapes_v001/campaign.json`.

Native has four per-compile settings: default and scoped VMEM of 32, 48, and
64 MiB. Native's physical tiling remains compiler-managed. Equal custom attempt
counts do not imply equal compilation time or identical searches. This is a
bounded, prior-informed tuning grid; results must not be described as global
optima or an exhaustive search. Every unsupported option, compile error, memory
skip, incomplete result, and numerical failure stays in the raw record.

This gives 52 screening attempts per shape: 8,736 main attempts and 936 LLM
shape attempts, excluding smoke and confirmation. The earlier region campaign
used 52 candidates and both call/prepared scopes: its 120-shape screening took
6.50 hours and confirmation took 4.31 hours. This campaign measures complete-call
scope only, caches repeated screening inputs/references, and compiles each
confirmation finalist once per shape. Two-level compilation can be slower;
elapsed-time estimates are not guaranteed completion deadlines.

## Screening and independent confirmation

Screen with three warmups and ten rotating/interleaved rounds on a common
Gaussian input per shape. Fix the fastest eligible arithmetic mean independently
per family; exact ties use candidate ID. Eligibility requires complete samples,
finite output, and the unchanged numerical thresholds below.

Retain the top three eligible candidates plus all candidates within five percent
of the fastest, capped at four per family in measured-mean order. Record every
candidate omitted by the cap. If none passes the numerical threshold, preserve
the fastest finite measured candidate solely for accuracy reporting; it is not
a valid speedup choice. If none executed, record the family as unavailable.

Confirm all retained candidates and default Native in one group per shape.
Compile each finalist once, then run three fresh Gaussian input seeds with five
warmups and thirty rotating rounds per seed. The original screening winner
remains the headline choice: do not reselect the fastest confirmation sample.
If default Native was selected as tuned Native, execute it once and give that
same observation both labels; do not invent independent samples.

Report hierarchical paired bootstrap intervals with 4,000 resamples, drawing
input seeds then paired rounds within inputs. These are pointwise 95% intervals
conditional on the frozen shortlist, not simultaneous confidence across all
searched candidates. Preserve a discrete list of tiles consistent with equal
latency and tiles whose upper latency-ratio bound is within five percent of the
frozen winner. Axis ranges are not confidence intervals and do not certify a
Cartesian region of good tiles. Headline uncertainty uses the same fresh paired
measurements. Statistical comparisons with failing numerical eligibility must
remain clearly marked ineligible.

## Accuracy and timing contract

Generate host inputs once and quantize to BF16. All candidates see the exact same
BF16 operands within a case. All dot operations use DEFAULT precision with FP32
accumulation and output; pre-additions at each Strassen level round to BF16.
Complete-call latency includes device padding, multiplication, and cropping,
and excludes compilation and host transfers. Record compile times and compiled
memory estimates separately.

Upgrade the common numerical reference to NumPy FP64 on these exact quantized
operands. Use a full reference when the existing output-size and work limits
allow it; otherwise sample up to 128 rows by 128 columns, always including the
edges, and sum over every K. This makes reference scope explicit and removes
FP32-reference noise from Native accuracy measurements. Max error is the maximum
on the reference sample, not an unsampled full-output maximum. Finiteness is
checked on the entire output.

Keep relative L2, maximum absolute error, reference magnitude, RMSE, mean
absolute error, absolute-error percentiles, normwise error, finiteness, sample
indices, all-K count, and per-input variation. Preserve metrics for failures.
The fixed eligibility gate is finite output, relative L2 <= 0.02, and maximum
absolute error <= 0.001 + 0.05 * maximum absolute reference value. It is an
experimental eligibility threshold, not evidence of acceptable model quality.
Synthetic Gaussian geometry tests are distinct from real model execution;
the application stage reports model-output/quality differences separately.

## Archival and controller interface

Root freezes and commits source/configuration before any TPU work. Each phase
has an immutable directory, source/configuration hashes, environment identity,
planned cases, raw timings and errors, summary, result-manifest checks and a
scoped commit. Twelve main chunks of fourteen shapes may screen and confirm
sequentially on the same allocation. Global manifest indices fix seeds, so
chunking cannot change the input cohorts. A partial phase never counts as
complete and is not silently resumed into an existing result directory.

Runner: `python -m strassen_mm.benchmark_mlsys_shapes_v001`, accepting
`--campaign`, `--phase` ending in `-smoke`, `-screen`, or `-confirm`,
`--output-dir`, `--expected-identity`, `--allocation-id`, `--shape-start`,
`--shape-count`, `--max-wall-seconds`, and `--selection` for confirmation.
The pure `plan_groups` function gives group/input/arm counts before execution.
`tools/audit_mlsys_shapes_v001.py --artifacts PATH` checks each completed phase;
with NumPy installed it also exactly replays confidence intervals. No aggregation
should mix these development samples with untouched validation or older runs.
