# Qwen3-32B Native admission failure and training-only diagnosis

This is failure and diagnostic evidence, not downstream accuracy results. No Qwen3-32B test examples were scored. Qwen3-14B's completed results remain in their separate immutable results folder.

The original Native cached/full-prefix check failed twice at relative L2 0.013896857653672283 (limit 0.005), maximum absolute difference 0.3125, greedy agreement 1.0, and finite logits. The same allocated v6e was retained for diagnosis. The frozen checkpoint, training prompts, BF16 precision, admission threshold and scoring rules were not changed.

| Training diagnostic | Cached/full logit relative L2 | Result |
| --- | ---: | --- |
| Original projection, interleaved 64-token diagnostic | 0.0234584072 | Mismatch; not quantitatively identical to the original admission execution |
| Fixed 128-row Native blocks, 64 tokens | 0 | Exact |
| Fixed 128-row Native blocks, 129 tokens | 0.0204748154 | Candidate fails to generalize |
| Minimum two 128-row Native blocks, 129 tokens | 0.0227013038 | Candidate still fails |

In every diagnostic, padding the single query to the same query width as the full-prefix path produced exact equality. Prefix causal hidden states also matched. These observations implicate execution-shape-dependent numerics; they do not identify a specific compiler pass.

The 129-token fixed-row operator trace reproduces that candidate's layer-six mismatch exactly even when returning intermediate tensors. Corresponding inputs and intermediate outputs through the MLP product are identical. The Native feed-forward down projection is the first differing operator. All three traced final layer outputs exactly reproduce their uninstrumented counterparts. The trace includes small activation samples, not checkpoint weights.

The original source bundle SHA256 is `162116e4ef16f4f7b24a713211e5d7b70c91b739b42d7a3d88734ae097bc091d`. An initial preservation receipt used the wrong manifest key and recorded a null source digest; it is retained unchanged, with a separate source-verification receipt correcting that reporting error. The original archive and execution contract agree.

All candidates in this snapshot are diagnostics only. The original scientific experiment was not modified or resumed by them. A later common-query-block candidate is under separate validation and is not claimed as a completed repair here. No failed result was discarded, and no threshold was relaxed.
