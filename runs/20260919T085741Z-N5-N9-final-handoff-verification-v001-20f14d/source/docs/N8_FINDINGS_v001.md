# N8: application epilogues, layout, fusion and early finalization

N8 supports standard epilogue fusion within the tested custom kernels, but its benefit depends strongly on the application and timing scope. Packing alone makes every tested SwiGLU case slower. Fusion improves every matched packed SwiGLU control, yet fused Strassen still loses to native XLA in all eight complete-call SwiGLU comparisons. With prepared operands reused, that native comparison becomes five wins, one inconclusive result and two losses. Residual fusion is more favorable: standard fused Strassen beats native in six of eight groups in both scopes. Early finalization mostly regresses and is never chosen by the application-selection artifact.

All **312 planned numerical/timing checks passed**. No coverage, archive-integrity or selection-provenance blocker was found in this independent read-only review. These are synthetic-input microbenchmarks on the existing v5e cohort, not model-quality or full-inference results.

Run: `20260919T073511Z-N8-v5e-v004-0a55e8`.

Archive commit: `83712dacc9b305940f591eea23595d056a34e89c`.

Source commit: `b84f55e95579bc3dcd586ea666b85b1b1d3fab3e`.

Journal SHA256: `415af55ad3b5392e877970136abe6432cb71ea3da9ee87a924ef160e3a5df357`.

Application selections SHA256: `b178f8b5daa9ed8a4d3822e2bd1f6e70c1b79928db689f0cffa51bbd28ea3f3e`.

## Coverage and common contract

The frozen campaign contains 16 groups: eight SwiGLU and eight residual projections. The original 12 groups at M=1024/2048 are retained. The preregistered D009 refinement adds M=512 SwiGLU and residual groups for Mistral and Qwen32, allowing exact N5-shape controls where available.

| Geometry label | M values | SwiGLU projection (K,N) | Residual projection (K,N) |
|---|---|---|---|
| Qwen3-0.6B | 1024,2048 | 1024,6144 | 3072,1024 |
| Mistral7B | 512,1024,2048 | 4096,28672 | 14336,4096 |
| Qwen32B | 512,1024,2048 | 5120,51200 | 25600,5120 |

For SwiGLU, N is the combined gate/up width; the final output has N/2 columns. Residual groups compute a projection followed by residual addition. Inputs are NumPy PCG64 Gaussian samples: A scaled by 1/sqrt(K), B and the residual standard normal, then quantized to BF16. Each group uses the registered seed plus its group index. These geometry labels do not mean checkpoint weights or real activations were used.

Every main SwiGLU group has 11 arms: native joint graph; unpacked-unfused, packed-unfused and standard-fused variants of full cubic, quadrant cubic and Strassen; and early Strassen. Each residual group has eight arms because no packing transform is needed. The two M=512 SwiGLU groups each add two N5-selected unfused controls. Thus there are 156 group–arm combinations, each measured in complete-call and prepared-kernel scopes, giving 312 results.

Matched ablations use tile (BM,BN,BK)=(1024,1024,512), plain full cubic, and interleaved quadrant cubic/Strassen. Native XLA owns optimization of its joint graph. All arms use BF16 operands and FP32 accumulation, round each projection to BF16 before the epilogue, and produce BF16 output. SwiGLU also rounds SiLU output to BF16 before its multiply. Early SwiGLU changes the last-panel FP32 accumulation order as explicitly documented; it is not a pure reordering with guaranteed bitwise equality.

Complete-call scope includes device preparation, layout transformation/padding where required, multiplication, epilogue and output finishing. Prepared-kernel scope reuses prepared operands. Neither includes compilation or host-to-device input transfer. Five preparation samples per group–arm are recorded separately; their means must not simply be added to or subtracted from separately compiled complete-call timings as a causal decomposition.

## Matched ablations: gains and regressions together

The table reports candidate wins / inconclusive results / losses. Speedup is reference mean divided by candidate mean. A win requires its individual paired 95% bootstrap interval wholly above 1, a loss wholly below 1; an interval containing 1 is inconclusive, not equivalence. There are 30 paired rounds and 2,000 bootstrap resamples per comparison, without multiplicity correction.

| Application and change | Candidate family | Groups | Complete call W/I/L | Prepared kernel W/I/L |
|---|---|---:|---|---|
| SwiGLU: unpacked-unfused → packed-unfused | Full cubic | 8 | 0/0/8 | 0/0/8 |
| SwiGLU: unpacked-unfused → packed-unfused | Quadrant cubic | 8 | 0/0/8 | 0/0/8 |
| SwiGLU: unpacked-unfused → packed-unfused | Strassen | 8 | 0/0/8 | 0/0/8 |
| SwiGLU: packed-unfused → standard-fused | Full cubic | 8 | 8/0/0 | 8/0/0 |
| SwiGLU: packed-unfused → standard-fused | Quadrant cubic | 8 | 8/0/0 | 8/0/0 |
| SwiGLU: packed-unfused → standard-fused | Strassen | 8 | 8/0/0 | 8/0/0 |
| Residual: unfused → standard-fused | Full cubic | 8 | 4/2/2 | 4/4/0 |
| Residual: unfused → standard-fused | Quadrant cubic | 8 | 6/2/0 | 7/1/0 |
| Residual: unfused → standard-fused | Strassen | 8 | 7/1/0 | 8/0/0 |
| Both: standard-fused → early | Strassen | 16 | 1/3/12 | 1/2/13 |

Packing changes gate/up column layout and the corresponding output-layout handling. Its regression remains even after input preparation is excluded: the prepared packed/unpacked speedup ranges are 0.592–0.711 for full cubic, 0.608–0.721 for quadrant cubic and 0.580–0.697 for Strassen. Therefore the loss cannot be attributed only to preparing weights on each call. No hardware profile in N8 establishes which layout, memory or compiler effect dominates.

Standard fusion improves Strassen versus its matched packed-unfused SwiGLU control by 1.191–1.601× in complete-call scope and 1.492–2.163× with prepared operands. Those positive ablation ratios are not comparisons against the original unpacked implementation or native XLA. Reporting only them would conceal the large full-call regressions below.

Residual Strassen fusion improves its unfused control by 1.024–1.071× in complete-call mean ratios. The one inconclusive call case still has a mean ratio above 1; all eight prepared comparisons have intervals above 1. This isolates a useful improvement in the tested residual path without establishing that every problem size beats native.

Early finalization has only one nominal complete-call win, Qwen0.6B M=1024 SwiGLU: 1.0424×, CI [1.0246,1.0630], versus standard fused. Its prepared interval contains 1. The only prepared win is Qwen0.6B M=2048 SwiGLU: 1.0311×, CI [1.0041,1.0570]; its call interval contains 1. Twelve call and thirteen prepared comparisons lose. There is no broad performance case here for choosing early finalization over standard fusion.

## Native and matched cubic comparisons

These are all registered native comparisons for the principal fixed-tile Strassen arms, separated by application so that opposing effects are visible.

| Application | Strassen candidate | Complete call W/I/L | Prepared kernel W/I/L |
|---|---|---|---|
| SwiGLU | Unpacked-unfused | 0/1/7 | 1/1/6 |
| SwiGLU | Standard-fused | 0/0/8 | 5/1/2 |
| SwiGLU | Early | 0/0/8 | 6/0/2 |
| Residual | Unpacked-unfused | 5/1/2 | 4/2/2 |
| Residual | Standard-fused | 6/0/2 | 6/0/2 |
| Residual | Early | 5/1/2 | 4/2/2 |

One representative scope reversal is Qwen32B M=1024 SwiGLU. Native takes 3.243736 ms per complete call; fused Strassen takes 14.041268 ms. With prepared operands, the corresponding means are 3.234665 and 3.028500 ms. Separately measured preparation for that fused Strassen arm averages about 8.439 ms. This demonstrates that including repeated relayout changes the practical comparison. It does not justify subtracting that preparation mean to infer a device-only gain. Reusing persistent prepacked model weights could be relevant, but must be an explicit application scope with preparation/storage costs accounted for.

For residuals, standard fused Strassen beats native at both M=1024/2048 for all three geometries. The strongest call ratio is Mistral7B M=1024: native 1.066791 ms, Strassen 0.858741 ms, speedup 1.2423× with CI [1.2262,1.2585]. Both M=512 residual cases lose: native/Strassen is 0.8423× for Mistral and 0.6165× for Qwen32. The fixed BM=1024 pads these small-M custom cases, but this experiment does not isolate padding as the only cause.

Against matched standard-fused cubic implementations over all 16 groups, standard-fused Strassen records 15 wins/one inconclusive/no losses versus full cubic and 14 wins/two inconclusive/no losses versus quadrant cubic in complete-call scope. This is a useful within-custom-kernel result, but the native comparisons above remain essential. These elementary implementations and their bounded tuning controls do not establish superiority over every possible optimized classical kernel.

## N5-selected anchor controls

The two M=512 SwiGLU groups match N5 training geometries exactly. Their extra controls preserve the original N5 screen's candidate, tile and variant; they are not newly tuned on N8. Both cubic controls use output_accumulator with tile (512,1024,1024). Mistral Strassen uses interleaved_output_accumulator with tile (512,1024,512); Qwen32 Strassen uses interleaved with that same tile.

| Anchor | Native call, ms | N5 cubic call, ms | N5 Strassen call, ms | Native / Strassen, paired 95% CI | Cubic / Strassen, paired 95% CI |
|---|---:|---:|---:|---|---|
| Mistral7B M=512 SwiGLU | 0.961188 | 0.965094 | 0.958599 | 1.0027× [0.9606,1.0627] | 1.0068× [0.9857,1.0284] |
| Qwen32B M=512 SwiGLU | 1.736859 | 2.035764 | 2.089773 | 0.8311× [0.8251,0.8368] | 0.9742× [0.9685,0.9801] |

The Mistral call comparisons are inconclusive. Its prepared Strassen/native ratio is 0.9839×, CI [0.9740,0.9940], a small nominal loss; its prepared cubic comparison is inconclusive. Qwen32 Strassen loses to native and selected cubic in both scopes.

The N5 choices substantially reduce latency relative to the fixed-tile unfused controls: descriptive call ratios are 1.935×/1.862× for Mistral cubic/Strassen and 1.817×/1.694× for Qwen32 cubic/Strassen. These particular fixed-versus-N5 ratios do not have registered paired intervals in the archive, so they are reported only as descriptive means. Tile and accumulator variant change together; the improvement must not be assigned solely to one optimization. More importantly, the stronger anchor controls remove a potential misleading conclusion from the poor fixed BM=1024 small-M baselines.

## Application choices and provenance

The application artifact selects the lowest complete-call mean among numerically eligible members of each family separately. It is not a rule that requires the custom family to beat native. All 48 choices—three families for each of 16 groups—were independently reconstructed from the final rows, and every selected arm and mean matches. All selected arms also pass their prepared-scope checks in this run.

For Strassen, it selects standard-fused for all eight residual groups, ordinary unpacked-unfused for the six M=1024/2048 SwiGLU groups, and N5-selected unfused for the two M=512 SwiGLU anchors. It selects no packed-unfused or early arm. Cubic selects nine ordinary unpacked-unfused, four quadrant-fused, one full-fused and two N5-selected unfused arms. Native uses its joint graph throughout.

Selection uses the same N8 observations summarized here. The selected-family minima are therefore selection results, not an independent confirmation of a new application policy. N9 must qualify those frozen choices independently on actual weights/activations and model-quality measurements, with its own timing scope. Prepared-scope wins cannot be silently substituted for the registered complete-call selection objective.

The copied N5 selections have SHA256 `e8243a48d32ce6326173d2a3706a21a8129f29b06159a34c1223f9d172cf9943`, matching the original N5 screen artifact byte for byte and the frozen N8 configuration. Exact geometry matches and both controls' complete selection-provenance objects were verified. Application-selection campaign SHA256 matches the archived N8 config (`af364369337857a5920a67eb162d75bc95da509bb585c9e388b3138e38c7592e`). The selection, copied N5 artifact and N8 environment all have identical cohort identity.

Canonical and outer seals bind the selection output to the source/config snapshots, raw results and summary. Every listed file hash and the committed outer-manifest bytes were checked; the full root source snapshot and relevant source/config bytes at the recorded source commit also match. The allocation is unchanged: `tpu-v5e1-s-kkb-usw1c1-3iv9vtqwssy8q`, JAX/JAXlib 0.7.2, libtpu 0.0.21.1. Physical chip serial is unavailable, so the evidence establishes logical allocation/runtime continuity.

## Numerical checks and audit limits

All 312 rows meet the unchanged finite-output, relative-L2 ≤0.02 and maximum-error ≤`0.001 + 0.05 × max_abs_reference` gates. Reference outputs use 32 sampled rows ×128 sampled columns, FP64 NumPy dots across all K with the exact BF16 operands, and the same BF16 epilogue rounding boundaries. Finiteness is checked over the complete output. Maximum recorded relative L2 is 0.00861222, for Mistral M=1024 unpacked-unfused Strassen SwiGLU. The largest native and cubic relative L2 values are about 0.003349. Passing the common tolerance does not mean bitwise equivalence, equal error, arbitrary-input safety or preserved model quality; N4 cancellation failures remain relevant.

This review verified exact planned group/arm/scope coverage, no duplicates or omissions, all **9,360 positive finite raw timing samples**, rounds 0–29 once per result, raw-array agreement and arithmetic means. It checked all 520 registered paired contrasts for reference coverage, numerical eligibility, paired rounds and ratios of means; interval endpoints were taken from the archived bootstrap, not independently resampled. Each arm has five positive finite preparation samples—780 separate observations, repeated in both scope records rather than counted twice. Recorded numerical gates were recomputed from their metrics. Input matrices and FP64 references were not regenerated.

N8 contains no hardware traces or full-model measurements. The retained evidence supports scope-specific layout/fusion findings and an unfavorable overall result for early finalization. It does not establish MXU/vector overlap, a causal performance mechanism, an optimal tile for each application, or LLM quality preservation. Individual intervals are unadjusted for many comparisons; one input seed per group and one v5e cohort further limit generalization.

No benchmarks or tests were run, no frozen files were edited and no commits were made for this review.

Primary evidence: [planned cases](../runs/20260919T073511Z-N8-v5e-v004-0a55e8/artifacts/planned_cases.json), [raw journal](../runs/20260919T073511Z-N8-v5e-v004-0a55e8/artifacts/results.jsonl), [summary and contrasts](../runs/20260919T073511Z-N8-v5e-v004-0a55e8/artifacts/summary.json), [application selections](../runs/20260919T073511Z-N8-v5e-v004-0a55e8/artifacts/application_selections.json), [effective campaign](../runs/20260919T073511Z-N8-v5e-v004-0a55e8/artifacts/effective_campaign.json), and [environment](../runs/20260919T073511Z-N8-v5e-v004-0a55e8/artifacts/environment.json).
