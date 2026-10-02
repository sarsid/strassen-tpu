# N5 independent confirmation review

The selected one-level Strassen implementation has a clear advantage on some larger matrix products, but it is not uniformly faster than native XLA. Across the 16 training geometries, complete-call comparisons give **six wins, four inconclusive comparisons and six losses against native XLA**, and **six wins, eight inconclusive comparisons and two losses against independently selected cubic**. These are per-comparison paired 95% intervals, without multiplicity correction.

This supports a geometry-dependent MM performance study and conservative dispatch policy. It does not support an unconditional Strassen replacement, a universal optimal tile, or a model-quality claim.

## Evidence and checks

Run: `20260919T063644Z-N5-confirm-v5e-v004-155d9f`.

Archive commit: `7a8cb57ee4a028d39b38a3225c3665cfd3ee2de2`.

Journal SHA256: `11cd8f82134e3a7d7aa7652e3e4debd71734fa7d3ac12c2aeca92196d771d014`.

The confirmation references the exact screening selection SHA256 `e8243a48d32ce6326173d2a3706a21a8129f29b06159a34c1223f9d172cf9943`. Its complete recorded environment identity matches the screen's new v5e allocation. The Gaussian seed is 20260920, distinct from screening seed 20260919. Tiles and variants were frozen before this run.

All 96 planned shape–arm–scope results pass: 16 shapes × three arms × two scopes. Independent journal inspection found 30 positive finite raw samples per result, rounds 0–29 exactly once, and agreement between raw arithmetic means and reported means: 2,880 samples in total. Recorded numerical passes agree with the unchanged finite/relative-L2/maximum-error gate. The largest selected Strassen relative L2 error is 0.004437, below the fixed 0.02 threshold. This review checked recorded metrics and provenance; it did not rerun matrix products or regenerate their numerical references.

The comparisons below use the retained paired bootstrap intervals: ratio of arithmetic means, 2,000 resamples of aligned timing rounds. Define **speedup = reference mean / candidate mean**; values greater than 1 favor the candidate. A win requires the entire interval above 1; a loss requires it below 1. An interval containing 1 is inconclusive, not proof of equivalence.

## Complete-call versus prepared-input outcomes

| Candidate versus reference | Scope | Wins | Inconclusive | Losses | Median observed ratio |
|---|---|---:|---:|---:|---:|
| Strassen versus selected cubic | Complete call | 6 | 8 | 2 | 1.0035× |
| Strassen versus selected cubic | Prepared kernel | 5 | 9 | 2 | 1.0109× |
| Strassen versus native XLA | Complete call | 6 | 4 | 6 | 0.9985× |
| Strassen versus native XLA | Prepared kernel | 7 | 5 | 4 | 1.0143× |
| Selected cubic versus native XLA | Complete call | 3 | 3 | 10 | 0.9802× |
| Selected cubic versus native XLA | Prepared kernel | 4 | 5 | 7 | 0.9805× |

Complete call is the primary scope: it includes device-side padding/layout preparation and output cropping when required. Prepared kernel uses already prepared resident buffers. Both scopes exclude compilation, initial host-to-device transfer and reference computation; complete call is not an end-to-end application measurement.

The two scopes are separate measurements, not independent replications or evidence that a compiler optimization caused the difference. For example, `tail_wide` loses to native in complete-call timing, while its prepared-kernel interval is inconclusive. The different call/prepared outcome on `vary_k_8192` against cubic is also close to the decision boundary: the complete-call interval begins at only 1.00075×, whereas the prepared interval contains 1. Retain both rather than treating every nominal win as equally strong.

## Most informative positive results

Matrix dimensions are ordered (M,K,N). These examples are selected for discussion from the full registered shape set; their extremeness is not an additional validation.

| Shape | Native ms | Selected cubic ms | Strassen ms | Strassen/native ratio [95% CI] | Strassen/cubic ratio [95% CI] |
|---|---:|---:|---:|---|---|
| (8192,8192,8192) | 6.322582 | 6.132881 | 5.624041 | 1.1242× [1.1196,1.1290] | 1.0905× [1.0860,1.0949] |
| Qwen-shaped (8192,5120,51200) | 22.807863 | 23.175248 | 21.295279 | 1.0710× [1.0694,1.0728] | 1.0883× [1.0863,1.0906] |
| (4096,4096,4096) | 1.012318 | 1.020855 | 0.960859 | 1.0536× [1.0416,1.0662] | 1.0624× [1.0517,1.0722] |

The largest square and the large Qwen-shaped anchor show gains against both controls, with the same direction in the prepared scope. These are useful central MM results because their advantage survives comparison with native XLA and with separately tuned classical code.

The (512,8192,2048) product also beats native at 1.1117× [1.0554,1.1724], but selected cubic beats native there too, and Strassen versus cubic is inconclusive at 1.0032× [0.9681,1.0414]. Thus that native-relative gain cannot be attributed specifically to Strassen's arithmetic saving. Similarly, the small 512-cube Strassen-versus-cubic win does not establish a native-relative win, whose interval contains 1.

## Negative results that constrain the claim

| Shape | Strassen/native ratio [95% CI] | Strassen/cubic ratio [95% CI] |
|---|---|---|
| Gemma-shaped (512,3584,28672) | 0.8955× [0.8880,0.9035] | 1.0017× [0.9926,1.0113] |
| Qwen-shaped (512,5120,51200) | 0.8960× [0.8916,0.9008] | 0.9758× [0.9684,0.9833] |
| Mistral-shaped (512,4096,28672) | 0.9027× [0.8871,0.9143] | 0.9955× [0.9767,1.0127] |
| Small-M (8,2048,2048) | 0.9504× [0.9239,0.9802] | 0.9569× [0.9311,0.9869] |

The three wide M=512 model-shaped products all lose to native, and those native-relative losses persist in the prepared scope. They therefore cannot be dismissed solely as complete-call padding costs. At identical Qwen K=5120 and N=51200, increasing M from 512 to 8192 changes the observed direction. This is useful evidence that matrix shape and available work matter; N5 does not isolate the particular hardware mechanism responsible.

Against cubic, the two complete-call losses are the small-M case and Qwen-shaped M=512 case. The other two wide model-shaped examples are inconclusive against cubic even though both custom implementations lose to native. Keeping native in the baseline set materially changes the practical conclusion.

## Publication scope and remaining uncertainty

These are training geometries, intentionally exposed to screening, confirmation and subsequent selector fitting. Fresh timing and a fresh Gaussian seed reduce reuse of screening noise, but they do not make these shapes held out. N7 must evaluate the already frozen selector on reserved geometries; it must not feed those outcomes back into this rule while being reported as a held-out evaluation.

The 30 timing rounds share one process/allocation per experiment. Their paired intervals measure within-run uncertainty; they do not quantify between-allocation, between-day or TPU-generation variability. The listed intervals are individual 95% intervals, not simultaneous familywise guarantees across shapes, references and scopes. The nominal win/loss counts have no multiplicity adjustment. Marginal intervals and selected best examples should not carry the central claim; retain every registered outcome and later fixed-policy replication.

The candidate families received equal attempted budgets, not an exhaustive search or equal feasible budgets. Classical code is optimized within the registered kernel/tile family, and native XLA is retained as a practical compiler baseline. This does not prove that either custom family is the strongest possible TPU implementation. Joint tile/variant selection also prevents attributing the confirmed advantage entirely to the reduction from eight to seven block products.

Finally, all N5 operands are Gaussian synthetic inputs, including the LLM-shaped matrices. Fixed BF16 inputs/pre-adds and FP32 accumulation/output define this numerical contract. N4's cancellation failures remain relevant; these N5 passes do not certify arbitrary activations, real-model accuracy or an automatically safe shape-only dispatch rule. Native fallback outside the validated input scope remains appropriate. N6 device profiling is a separate mechanism study; its first failed device captures cannot explain these speedups.

Primary evidence: [confirmation journal](../runs/20260919T063644Z-N5-confirm-v5e-v004-155d9f/artifacts/results.jsonl), [confirmation summary](../runs/20260919T063644Z-N5-confirm-v5e-v004-155d9f/artifacts/summary.json), [selection provenance](../runs/20260919T063644Z-N5-confirm-v5e-v004-155d9f/artifacts/selection_input_provenance.json), and [screening review](N5_SCREEN_REVIEW_v001.md).
