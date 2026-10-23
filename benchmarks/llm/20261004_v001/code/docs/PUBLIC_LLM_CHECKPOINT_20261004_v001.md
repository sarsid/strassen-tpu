# v6e LLM experiments — October 4 checkpoint

This is an **in-progress snapshot: 75 of 200 active BF16 comparisons**, comprising
15 complete workloads with five algorithms each. Qwen3-8B has all 50 comparisons,
Qwen3-32B has 20 (batch 1, 512/1024/2048/4096 tokens), and Qwen3-14B has an earlier
five-comparison workload. Gemma4-31B has no real-checkpoint measurements yet.
The running Qwen32 workload is excluded from this immutable snapshot.

The queue order is Qwen3-8B, Qwen3-32B, Gemma4-31B, Qwen3-14B. Both Gemma3 models
and both Mistral models are deferred. Earlier FP32 data is retained separately
and does not count toward active completion. This publication does not alter
the completed 168-shape MM study in the parent `benchmarks` folder.

## Results and scope

- [Complete Qwen3-8B report](results/qwen8_complete_v001/README.md).
- [First three Qwen3-32B workloads](results/qwen32_interim_3_workloads_v001/README.md).
- [Latest Qwen32 B1/S4096 summary](results/qwen32_b1s4096_summary.json).
- [Active raw result ledger](results/completed_entries.jsonl), including timing
  samples, selected profiles, tiles, per-layer errors and predictive metrics.
- [All historical ledger entries](results/completed_entries_all_history.jsonl).
- [Machine-readable publication state](checkpoint.json).
- [Earlier parent replication and follow-ups](parent_study/parent_llm_study_20261002_v001/README.md).

The five algorithms are Default Native, Tuned Native, Tuned Cubic, S1 and S2.
BF16 output uses DEFAULT arithmetic and FP32 accumulation where required.
Profiles freeze before held-out evaluation and 15 matched confirmation rounds,
with three warmups after each policy activation. Full streamed forward includes
serialized recurring weight transfers, embeddings and every token's vocabulary
logits; compilation and host-layout preparation are excluded. The separate
resident pass sums layer medians and excludes transfers, embeddings and head.
It is not a full-model resident latency claim.

All 75 completed active comparisons passed the configured finite-output, NLL,
KL and token-agreement gates. Quality uses fixed held-out WikiText2 windows,
not downstream task accuracy. Passing gates does not imply identical activations.

At Qwen32 B1/S4096, S1 takes 5951.0 ms versus Tuned Native's 6013.2 ms, a 1.03%
reduction; separate resident time falls 5.20%. Its paired pointwise 95% speedup
interval is 1.0028–1.0333x. S2's nominal full-forward reduction is 0.68%, with an
interval including a tie. These intervals are not adjusted for multiple
comparisons. The completed Qwen8 suite does not show a conclusive S1/S2
full-forward win over Tuned Native. Larger Qwen32 batches remain pending.

## Code and evidence

`code/` is the complete committed research-source snapshot identified by
`checkpoint.json`, including versioned fused kernels, tuning, streaming, cache,
runtime control, tests and design notes. Original measurement bundles are
preserved separately in the evidence index. Current source is not retroactively
attributed to measurements made with an older frozen bundle.

Gemma4 code has passed local synthetic tests for its architecture, official
loader, fusion, streaming, caching, tuner and reference-input export. It remains
behind official-checkpoint and TPU/runtime qualification gates; those local
checks are not Gemma4 speed or real-model accuracy results. See the `GEMMA4_*`
design/qualification documents under `code/docs/`.

`evidence/index.json` maps every exported canonical workload artifact and
qualification run to lossless gzip blobs. Identical bytes are stored once.
This includes tuning menus/events, profiles, official reference activations,
timings, errors, source archives and qualification outcomes. Original filenames
and SHA256 hashes are retained. Duplicate archive containers, private model
weights/caches/credentials, environments and transient controller files are not
published. Earlier failed qualification versions remain in source and evidence.

From this directory, verify with Python 3.11+:

```sh
python verify.py
python restore.py
python restore.py --group qwen3_32b-b1s4096 --output /tmp/qwen32-b1s4096
```

These tools use no accelerator or network. Archived absolute paths are provenance,
not portable runtime settings. This snapshot is complete for its stated scope;
the 200-comparison campaign is still running.
