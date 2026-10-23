# Large Gemma: Native, one-level and two-level Strassen

User request: one large Gemma check with independent tuning of all three
algorithms. Use Gemma 3 12B text gate/up dimensions: (M,K,N)=(16384,3840,30720).
These are synthetic Gaussian operands with model-derived dimensions, not a
checkpoint or end-to-end model measurement. No model download is needed.

Use one fresh Colab v5e allocation for screening and confirmation. No concurrent
TPU work. Pinned software and fixed runtime flags match the prior depth study;
check the complete recorded setup identity in each phase. Release after verified
retrieval. Each device phase has a 30-minute bound, with no automatic broad sweep.

Screen eight (BM,BN,BK) choices independently for each Strassen depth:
(1024,1024,512), (1024,2048,512), (2048,1024,512), (1024,2048,1024),
(2048,1024,1024), (2048,2048,512), (2048,2048,1024), (2048,2048,2048).
The existing one-level output-accumulator and two-level scratch kernels are
unchanged. Their per-kernel scoped VMEM limit remains 48 MiB.

Native has four choices: XLA default plus per-compilation
`xla_tpu_scoped_vmem_limit_kib` values 32768, 49152 and 65536, as in the previous
registered grid study. This is a bounded compiler-option search; Native tiles
remain compiler-managed. It is not exhaustive Native tuning or an equal-count
search across algorithms. An accepted option may produce no scheduling change;
do not assume an observed small timing difference proves a compiler improvement.
Unsupported options remain recorded failures. No precision settings change.

Screening has nine groups: eight Strassen pairs and one four-candidate Native
group, shuffled deterministically. All 20 candidates receive identical BF16
operands generated from seed 2026092161. Use three warmups and ten rotating
timing rounds per executable. Record compilation failures and times. The
compiled-executable memory estimate must not exceed 8 GiB; it is not measured
peak memory. Select the fastest eligible arithmetic mean separately for each
family; exact ties use candidate ID. Require all ten samples. Freeze choices
before confirmation, without changing the search after seeing results.

Confirmation compares the three selected candidates together, plus default
Native if another Native candidate won. Use seeds 2026092261, 2026092361 and
2026092461, five warmups and 30 rotating timing rounds per arm and seed.
Compile once per selected configuration and reuse across seeds. Preserve
seed/round pairing in comparisons; do not reselect based on confirmation.
Report paired per-seed ratios and a hierarchical paired bootstrap interval
for the pooled ratio, conditional on these three input seeds and one allocation.

Every algorithm takes BF16 inputs and returns FP32 with FP32 accumulation and
DEFAULT dot precision. Strassen pre-additions are BF16 at each level. All
measurements include the complete device call, including necessary padding
and cropping; exclude compilation, random generation, references and transfers.
K=3840 must pad to 4096 for the selected Strassen grid. N may pad for tiles
with BN=2048. Record actual padded dimensions and include this cost in timings.

Use the exact quantized inputs and host FP32 reference, all K terms, a seeded
128x128 output cross-product including edges, and whole-output finiteness.
The existing gates remain relative L2 <= 0.02 and max absolute error <=
0.001 + 0.05 * max absolute reference. Failing rows cannot support speedup
claims. Report the actual error for each method: passing this gate does not
establish equal accuracy or unchanged LLM quality.

Preserve all executed versions, raw timings, errors, environment identity,
configuration, source hashes and frozen selection. Commit source before local
validation, and archive/commit every execution with the scoped runner. Confirm
only after retrieving and checking the screening artifacts. Publish a compact
three-method chart and clearly distinguish selected Native from its default.
