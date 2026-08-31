# Evidence index

Every Qwen3 result indexed below is raw JSONL emitted by its benchmark. Timing
samples, runtime metadata, accuracy metrics, and verdicts remain in-band.
These files are the evidence for the current Qwen3 public claim.

## Tile selection

| Model | Artifact | SHA-256 |
|---|---|---|
| Qwen3-8B | `strassen_qwen3_8b_scaling_tiles.jsonl` | `966ceadb44199684e9076ae2aef2abaebe6744604a0cf526db8a67a47d98cb9c` |
| Qwen3-14B | `strassen_qwen3_14b_scaling_tiles.jsonl` | `a1630d503599e5bdf3672ecf73ef45d84dea834200a81ddf65ad2e0099faee97` |
| Qwen3-32B | `strassen_qwen3_32b_scaling_tiles.jsonl` | `3b9bd1c73d36bd7471b7dbe210b3ff92771cd06fe9945e8b711464a7b949609e` |

## Isolated gate/up plus SwiGLU

| Model | Artifact | SHA-256 |
|---|---|---|
| Qwen3-8B | `strassen_qwen3_8b_product_aware_inference.jsonl` | `2b93e8e5da17b10b6d025fcd153bc0ffa545837fadaeac8810fec1e2f2326473` |
| Qwen3-14B | `strassen_qwen3_14b_product_aware_inference.jsonl` | `362607a8e55b97f3582d11c8b9c93446e6a682210d7fffdf2948593ade0737f5` |
| Qwen3-32B, promoted | `strassen_qwen3_32b_product_aware_inference.jsonl` | `e65c54ea23e7a6e52a19ad12aa78e41d5bcab1a11636c969bcd36f4f660e268d` |
| Qwen3-32B, campaign repeat | `strassen_qwen3_32b_product_aware_inference_campaign.jsonl` | `94f85ecfb0dc53ec42992b114e5436768a266d6c02b1394f56f14743c2ad4534` |

## Real layer 0

| Model | Artifact | SHA-256 |
|---|---|---|
| Qwen3-8B | `strassen_qwen3_8b_full_layer_product_inference.jsonl` | `dc870b114cd052d5a597213380154763050217daa29b0b134b20cdb6e00b6a02` |
| Qwen3-14B | `strassen_qwen3_14b_full_layer_product_inference.jsonl` | `49f5c240d1a76a5778da51780b1d72903ddf6187afea1d5d83c3dde5f6158767` |
| Qwen3-32B, promoted | `strassen_qwen3_32b_full_layer_product_inference.jsonl` | `2fbc1fb37761e01c866381c1f09ae8b59d0039fed3ee0f3be46f6f9634a9e7b3` |
| Qwen3-32B, campaign repeat | `strassen_qwen3_32b_full_layer_product_inference_campaign.jsonl` | `7b64fa74147558c585adc1a836754cb48c3db3ef7f1df37744795dcfaa003717` |

## All-layer resident compute

| Model | Artifact | SHA-256 |
|---|---|---|
| Qwen3-8B | `strassen_qwen3_8b_streamed_product_inference.jsonl` | `26766b75833feee11f944e2a6ac3ec9a64f61102017a1dbe4ed55bf3c703f86a` |
| Qwen3-14B | `strassen_qwen3_14b_streamed_product_inference.jsonl` | `07b914762830528e59e62dc9ee6573233a30c4c71ca7338a529e800d1c793fe1` |
| Qwen3-32B, promoted | `strassen_qwen3_32b_streamed_product_inference.jsonl` | `2cdf084f69b8a957bcef1f2240cf11ddea7ba718b6bbb9637b8d9a8590579757` |
| Qwen3-32B, campaign repeat | `strassen_qwen3_32b_streamed_product_inference_campaign.jsonl` | `0a1a27b39926f601cc3e5b36c0dcf6002ce42b46b960d9af273978e2638e9174` |

## Natural-text quality

| Model | Artifact | SHA-256 |
|---|---|---|
| Qwen3-8B | `strassen_qwen3_8b_natural_task_gate.jsonl` | `88146cd3da96db3bf3847ed095523deb402b6b4c1453f333b7f3c1127e3e1c19` |
| Qwen3-14B | `strassen_qwen3_14b_natural_task_gate.jsonl` | `fff281752e3f991268a15bfdcebb1802bb68cd23824b382786e8f99d2ea6bb48` |
| Qwen3-32B | `strassen_qwen3_32b_natural_task_gate.jsonl` | `3cd5807eb09825444bcd3115f084fd3701d17221af42857b52a4616ab11d3ac5` |

The WikiText dataset revision is
`b08601e04326c79dfdd32d625aee71d232d685c3`. The token tensor SHA-256 recorded
in each natural-text artifact is
`4aa8e3628d16c9a26876aedb8c99525c6a15ee601c88d93deadae15855a84d74`.
The immutable runs predate the explicit `dataset_revision` field now emitted
by the harness; the pinned revision was independently checked to reproduce
that exact token hash.

## Earlier evidence

The remaining JSONL and NPY files are the kernel, VMEM, numerical-stability,
and Mistral evidence retained from the first public snapshot. They are useful
historical context but are not required for the current Qwen3 claim. OOM
records are feasibility evidence, not successful performance measurements.
