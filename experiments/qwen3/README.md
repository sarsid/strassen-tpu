# Qwen3 experiment

This directory contains the complete promoted experiment. One runner selects
the model, stage and platform profile.  The profile carries the promoted
tiles, budgets and scoped-vmem flag, because those differ between v5e and
v6e; `--device v6e` reproduces the Trillium result and `--tile` overrides
the gate/up tile for a one-off.

From the repository root:

```bash
python experiments/qwen3/run.py --model 8b --stage synthetic
python experiments/qwen3/run.py --model 8b --stage layer
python experiments/qwen3/run.py --model 8b --stage streamed
python experiments/qwen3/run.py --model 8b --stage quality
```

Add `--dry-run` to validate model, stage, tile, and output selection without
initializing JAX or allocating a TPU.

Valid models are `8b`, `14b`, and `32b`. Use `--stage tune` only when
repeating the equal-budget tile search. New JSONL files go to `runs/` locally
or `/content/runs/` on Colab; use `--output-dir` to override that location.

Colab CLI cannot pass arguments to `exec -f`. For that path, copy
`job.example.json` to `job.json`, select one model and stage, upload it beside
`run.py`, and execute `run.py`. Restart the remote Python kernel when changing
models so imported model constants cannot remain cached.

The modules retain their original research filenames so published artifacts
can be traced directly to the code that emitted them:

- `benchmark_qwen3_scaling_tile_tune.py` — equal-budget tile search.
- `benchmark_qwen3_downstream_tasks.py` — zero-shot HellaSwag and
  LAMBADA agreement; generates the downstream artifact in `evidence/`.
- `benchmark_qwen3_32b_product_aware_inference.py` — isolated gate/up plus
  SwiGLU.
- `benchmark_qwen3_32b_full_layer_product_inference.py` — real layer 0.
- `benchmark_qwen3_32b_streamed_inference.py` — all-layer resident compute.
- `benchmark_qwen3_streamed_natural_gate.py` — pinned WikiText-2 quality.
- `benchmark_qwen3_32b_layer.py` — checkpoint and model definitions.
- `benchmark_common.py` and `benchmark_cubic_control.py` — shared measurement
  and matched Pallas control code.
