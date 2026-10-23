# Adopt the reproduced parent S1 implementation

The default matched Qwen3-32B v6e path now calls the frozen parent's original
`make_product_layer('gated_strassen')` function. This is adoption of an existing
implementation, not a newly measured optimization. The measured 832.429 ms
current-stack result remains in the original replication report; this code
integration has no new TPU timing claim.

Offline validation passed in
`runs/20261002T235434Z-parent-s1-adoption-v001-b21b20`: all three full-shape JAX
graphs match the parent exactly, reduced Native execution is bitwise equal,
and unsupported/mismatched exact profiles are rejected. This validation used
CPU JAX 0.7.2; it does not establish a new current-stack TPU timing. The default
configuration is `configs/qwen3_32b_v6e_parent_adopted_v001.json`; the compact
evidence is in `results/integration/parent_s1_adoption_20261002_v001/`.

## Choices

* Preserve the parent's DEFAULT fused Q/K normalization precision, BF16 stores,
  product scheduling, weight layouts, Q/K and gate/up full contraction,
  O/down routing and tiles. Do not approximate it by changing only our norm flag.
* Use the parent v6e kernel budgets (120/124 MiB) independently of the 48 MiB
  Native compiler setting. Validate the model, batch, sequence, tiles, policy,
  imported module locations and all frozen source hashes before exact dispatch.
* Default to the exact parent Native/cubic/S1 trio and its original parameter
  tree. There is no arithmetic wrapper around those model functions. Preparation
  remains the original stream loader, outside resident timing.
* Expose our previously tuned equally fused cubic as `--cubic-control matched`.
  Parent cubic has fewer fusion opportunities; its comparison alone does not
  isolate the benefit of Strassen arithmetic. The matched-control runner uses
  an explicit prepared parameter bundle and records that difference.
* Retain S2 and FP32 as explicitly labelled extensions with their historical
  tuning choices and HIGHEST normalization. This adoption does not claim these
  extensions were retuned to the parent's precision. Exact-parent S2/FP32
  requests are rejected. Existing studies and runners remain unchanged.
* Treat the adopted S1 as a frozen whole-layer candidate in subsequent tuning.
  Candidate builders expose `prepare_weights`, metadata and compiler settings
  to the same interface as our custom adapter. A new tuner may compare an
  extension against it, but must retain parent S1 unless a separately confirmed
  speed/error tradeoff justifies replacement.

## Entry points

`src/strassen_mm/parent_fused_adapter_v002.py` supplies the default exact callable
and explicit extension dispatch. `tools/adopt_parent_profile_v001.py` generates
the adopted profile from the preserved down-budget profile without modifying it.

On the qualified current TPU stack (JAX/jaxlib 0.11.2, libtpu 0.0.48), run:

```sh
PYTHONPATH=src python tools/compare_parent_fused_v002.py --dry-run
PYTHONPATH=src python tools/compare_parent_fused_v002.py \
  --stage streamed --output /content/results/adopted-stream --cache /content/checkpoint-cache
PYTHONPATH=src python tools/compare_parent_fused_v002.py \
  --stage quality --output /content/results/adopted-quality --cache /content/checkpoint-cache
```

Use `--cubic-control matched` for the equally fused cubic comparison. Extensions
are selected with `--family s2` or `--dtype float32 --cubic-control matched`.
Each invocation must use a fresh process and a new output directory. The runner
does not allocate a TPU, schedule work or change prior archives. The published
resident timing scope and token/quality gates remain unchanged.

## Gemma evidence audit

At public-preview commit `95be1fb088656a89813b04492e1d77c66b36ccf9`, the parent
layer module registers `google/gemma-3-27b-it` at revision
`005ad3404e59d6023443cb575daa05336842228a`. Its comments describe GeGLU, four
normalizations per block and model width 5376, which need a different fusion
and tiling policy from Qwen. That registry entry alone is not a runnable,
qualified Gemma experiment: the public runner offers only Qwen 8B/14B/32B.

The complete public-preview tree and the published result documentation contain
no identified Gemma timing/quality artifact. Therefore no Gemma speedup or
accuracy result is asserted here. There may have been unpublished work; this
audit only establishes what is available in the specified public branch.

Sources: [parent model registry](https://github.com/sarsid/strassen-tpu/blob/95be1fb088656a89813b04492e1d77c66b36ccf9/experiments/qwen3/benchmark_qwen3_32b_layer.py#L62),
[public results](https://github.com/sarsid/strassen-tpu/blob/95be1fb088656a89813b04492e1d77c66b36ccf9/docs/RESULTS.md),
[runner](https://github.com/sarsid/strassen-tpu/blob/95be1fb088656a89813b04492e1d77c66b36ccf9/experiments/qwen3/run.py).
