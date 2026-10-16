# Evidence index, Gemma 3 27B

Raw JSONL and JSON emitted by the Gemma harnesses, indexed by SHA-256.
`tools/verify_evidence.py` checks every digest here.

Gemma is the generalisation arm. Qwen3 establishes the result; Gemma asks
whether it survives a block that is deliberately harder for this kernel --
four norms instead of two, `(1 + w)` RMSNorm, GeGLU, and a 5:1
sliding/global attention pattern with two RoPE bases. Read the Qwen3 index
first; this one only records what changes.

All runs use `google/gemma-3-27b-it` at revision
`005ad3404e59d6023443cb575daa05336842228a`, 8 x 1024 tokens, one chip.

## Scoring under the published harness

These are the only numbers here produced by lm-evaluation-harness rather
than by our own scoring code, and they are the most load-bearing in the
file. HellaSwag, `--limit 200`, 256-token contexts, `acc_norm` is the
metric the published tables use.

| Arm | acc | acc_norm | Artifact | SHA-256 |
|---|---:|---:|---|---|
| Native XLA | 0.580 | 0.745 | `strassen_lmeval_gemma3-27b_regular_xla_hellaswag_bos.json` | `d442c5158b9391790985d184e7c9a4dd6151dfe6db865d81d8659e8e7030f6fa` |
| Strassen | 0.580 | 0.745 | `strassen_lmeval_gemma3-27b_gated_strassen_hellaswag_bos.json` | `84e3fd676d07b2de10302ec63dcdd599ce409100c0da40303943b592e6e5a431` |

Zero delta on both metrics: routing the projections through the kernel
changes no answer on the task.

The two files below are the same measurement **without the leading BOS
token**, retained because they are the reason the protocol was fixed. Gemma
is trained with a BOS and collapses toward a flat distribution without one;
`acc_norm` drops from 0.745 to 0.450. Both arms omitted it
identically, so every agreement-based gate in this repository cancelled the
defect exactly and reported nothing. Self-consistency gates catch kernel
defects; only an external harness catches protocol defects.

| Arm | acc | acc_norm | Artifact | SHA-256 |
|---|---:|---:|---|---|
| Native XLA, no BOS | 0.370 | 0.460 | `strassen_lmeval_gemma3-27b_regular_xla_hellaswag.json` | `3d0408bcffa5ed4632161a1291029664d5ae21b904a39c637e091729e6dc315b` |
| Strassen, no BOS | 0.365 | 0.450 | `strassen_lmeval_gemma3-27b_gated_strassen_hellaswag.json` | `4cf07cb0caffe1278e7a7e215674596a4a7f305a7a2f1ce4c8a294ff82fe0fa8` |

## Pure GEMM

| Gate/up, no epilogue | `strassen_gemma3_27b_pure_gemm_v5e.jsonl` | `d4a0d0dfc31aa5e5d7c54adf84e9f58a673b226eae0b35207cc7e4de3a0f8170` |

The shape recorded in this file is `[8192, 5376, 43008]`, which is Gemma's.
Its `verdict` string reads "pure Qwen3-32B gate/up GEMM" because the stage
reuses the Qwen3 harness and that label is hardcoded there. The artifact is
published as emitted rather than corrected by hand; the metadata block is
authoritative over the verdict string.

## Layer-0 block, real weights

| Configuration | v5e | Artifact | SHA-256 |
|---|---:|---|---|
| Gate/up + GeGLU, fused q/k | `1.1577x` | `strassen_gemma3_27b_block_v5e_on.jsonl` | `bc1209dadb28cebf3542108e0d06ca01a1bc4731f65969cb81b77da54619974b` |
| Gate/up + GeGLU only | `1.0781x` | `strassen_gemma3_27b_block_v5e_off.jsonl` | `356e4470c1a740d238ac70fa2c4304420db48ff95d94abedcdc0b9591ea2eb16` |

Fusing `q_norm`+RoPE into the q/k kernel is worth 8 points here, the same
as on Qwen3, on a model whose per-head norm sits in the same place.

| Configuration | v6e | Artifact | SHA-256 |
|---|---:|---|---|
| Gate/up + GeGLU, fused q/k | `1.1299x` | `strassen_gemma3_27b_block_v6e_t768.jsonl` | `ca045073fabbf659f18bc3fd3928b1af11ecf455697a2a8ecdddb27cadc1e94e` |

### A refuted hypothesis, retained

`o` and `down` carry a fusable residual in Qwen3 and win there. In Gemma a
norm sits between each projection and its residual add, so the fusable unit
becomes norm+residual+add. We built a `norm_residual_add` epilogue for it.
It loses 7.3 points:

| Configuration | v5e | Artifact | SHA-256 |
|---|---:|---|---|
| o/down left to XLA | `1.1568x` | `strassen_gemma3_27b_block_v5e_normres_off.jsonl` | `1131584f897b8304b6fc499422d972d6cb8596a108ae4074d18c24516214f82b` |
| o/down routed, norm_residual_add | `1.0841x` | `strassen_gemma3_27b_block_v5e_normres_on.jsonl` | `f837a57a720181e580a924419db24cf89def62eee1e6f66f6d6477bda92832c8` |

The epilogue's reduction spans the output width, which forces `bn == n` and
gives up tile freedom worth more than the traffic saved. The ceiling on the
whole manoeuvre was about 1% of the block and was not computed before the
work was done. Reproduce with `--norm-residual`.

## All-layer streamed, 62 layers

| Repetitive corpus, gate/up only | `strassen_gemma3_27b_streamed_v5e.jsonl` | `0ab3407560adbbaa60bfe78601ac095bcc7c84780c1e55ed074897065e1ac48e` |
| 6-layer smoke | `strassen_gemma3_27b_streamed_smoke.jsonl` | `eff0cfdaceee95e1a2ed19534e22d050c2079d527dfbc16ac4ac2a2ffc0e3008` |

Top-1 agreement `0.99890` on the repetitive corpus, gate passes.

### Failing gates, retained

Both files below record `passes: false`, and both were measured **without
the BOS token** -- the same defect the lm-eval rows above quantify, in the
regime where it does most damage. Native top-1 on this corpus is `0.3952`,
a near-flat distribution in which agreement metrics are maximally sensitive
to any perturbation. These are published as failures, and they are not
evidence that Gemma fails quality: the measurement that replaced them, under
the reference harness with BOS, shows zero delta. They are pending re-run.

| Gate | Result | Artifact | SHA-256 |
|---|---|---|---|
| WikiText-2 streamed, fused q/k | `0.96215` top-1, fails at `0.97` | `strassen_gemma3_27b_streamed_v5e_fused_wikitext_bk768.jsonl` | `13220fcb91b8548f817dddf61571558f48938f828c780b18ef895cc604ef5846` |
| WikiText-2, task records only | `0.95910` top-1, fails | `strassen_gemma3_27b_streamed_v5e_fused_wikitext_taskonly.jsonl` | `b4d56f085c4701a3af9fa9aad8f20b0c32026eb50bd996e31fa611de3389fa9c` |
| HellaSwag/LAMBADA agreement | `0.9875` / `0.9625`, fails at `0.97` | `strassen_gemma3_27b_downstream_v5e_fused.jsonl` | `21d07098ab58f4fdfa7d712c6bcbf3bbacfe87387f1961ddf9a897dfa355143c` |
