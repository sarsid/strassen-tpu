# Gemma 3 experiment

The generalisation arm. Qwen3 establishes the result; this directory asks
whether it is a property of the method or of Qwen3's particular block.

Gemma 3 27B was chosen because it is awkward for this kernel in four
specific ways: four norms per block instead of two, `(1 + w)` RMSNorm
rather than `w`, GeGLU rather than SwiGLU, and a 5:1 sliding/global
attention pattern carrying two RoPE bases (`10000` and `1000000` with a
linear factor of 8). It also ties `embed_tokens` as the output head and
scales embeddings by `sqrt(hidden_size)`.

From the repository root:

```bash
python experiments/gemma3/run.py --stage block --device v5e --fused-qk
python experiments/gemma3/run.py --stage streamed --corpus wikitext2
python experiments/gemma3/run.py --stage downstream
```

Add `--dry-run` to validate stage, tile and output selection without
initializing JAX or allocating a TPU.

The harnesses import the Qwen3 modules by flat name, because the checkpoint
reader and the model geometry table are shared; `run.py` puts
`experiments/qwen3` on the path rather than duplicating them. Running a
harness directly, outside `run.py`, needs that path set by hand.

## What the stages establish

- `benchmark_gemma3_block.py` — layer-0 block against real weights. The
  result is `1.1577x` on v5e with fused q/k and `1.0781x` without, so the
  `q_norm`+RoPE fusion is worth the same 8 points here as on Qwen3.
- `benchmark_gemma3_streamed.py` — all 62 layers, weights streamed, with a
  task gate. Per-layer RoPE base is selected by `index % 6 == 5`, verified
  against `transformers`' own `layer_types` for all 62 layers.
- `benchmark_gemma3_downstream.py` — HellaSwag and LAMBADA agreement.
  Resumable in layer chunks via `GEMMA_LAYER_START` / `GEMMA_LAYER_STOP` and
  a checkpoint guarded by a token digest, because a 54 GB stream across 62
  layers does not reliably survive a single session.

## Two results worth reading as negative

`--norm-residual` routes `o` and `down` through the `norm_residual_add`
epilogue. In Qwen3 those sites carry a fusable residual and win; in Gemma a
norm sits between each projection and its residual add, so we built an
epilogue for norm+residual+add. **It loses 7.3 points** (`1.0841x` against
`1.1568x`): its reduction spans the output width, which forces `bn == n`
and gives up more tile freedom than the saved traffic is worth. The ceiling
on the whole manoeuvre was about 1% of the block, and was not computed
before the work was done. The flag exists so the refutation reproduces.

The streamed WikiText-2 gate and the downstream gate are both published
**failing**, and both were measured before the BOS-token defect was found.
See `evidence/gemma3/README.md`; they are pending re-run and are not
evidence that Gemma fails quality. The measurement that replaced them runs
under lm-evaluation-harness in `experiments/lm_eval/` and shows zero delta.

## Fidelity

`tools/reference_fidelity.py` runs one real decoder layer through
`transformers` and through our formulation on CPU in FP32 and compares
them. Both Gemma and Qwen reach l2 `0.0`, max_abs `0.0` -- bit-exact. Run
it before trusting any speedup from a block you have just edited; the tool
itself was wrong first, reporting a divergence that came from passing
`attention_mask=None` and getting eager bidirectional attention.
