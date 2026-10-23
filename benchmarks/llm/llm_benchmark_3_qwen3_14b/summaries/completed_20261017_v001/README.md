# LLM Benchmark 3: Qwen3-14B — completed six-batch report

All six batches completed on one v6e and passed the registered quality gates. Tuned Strassen S1 was the fastest of the four arms in the summed resident-layer measurement at every batch. It used **5.86% less time than Tuned Native at B1 and 23.08–29.83% less at B4–B64**. The maximum natural-text perplexity increase relative to Native was **0.1191%**.

The remote worker ran from 17:53:18 to 22:03:15 UTC on October 17, 2026 (about 4 hours 10 minutes, including installation, tuning and checkpoint loading). All 30 stages succeeded on their first attempt. All archives were verified and the TPU was released; its absence and the exit of the supervisor/keepalive were independently checked.

## Matched setup

Qwen/Qwen3-14B at revision `40c069824f4251a91eefaf281ebe4c544efd3e18`; 40 layers, hidden width 5120, FFN width 17408. Sequence length is 1024; batches 1, 4, 8, 16, 32 and 64 were executed in order 8, 16, 32, 64, 4, 1. Every batch used JAX/jaxlib 0.11.2 and libtpu 0.0.48, BF16 inputs/output, parent FP32 accumulation and DEFAULT precision.

Fresh tuning was performed per batch and frozen before confirmation or held-out quality. Native VMEM and each custom arm’s gate/up and O/down tiles were selected independently. Q/K used the parent published tile, with BM clipped at B1; V remained native. The parent kernels and measurement contracts are unchanged. See [protocol](../../PROTOCOL.md) and [frozen code](../../code/initial_20261017_v001/source).

## Resident decoder-layer computation

Times below are **milliseconds summed over 40 separately warmed layer means**, with two warmups and five timed samples per layer. They exclude weight preparation/transfers, embedding, final norm/head and compilation. They are not uninterrupted whole-model latency. Each arm propagates its own activations; arm order rotates/reverses. Resident projections and block0 additionally have three warmups and 20 samples in the raw events.

| Batch | Native ms | Tuned Native ms | Tuned Cubic ms | Strassen S1 ms | Less time vs Tuned Native | Speedup vs Tuned Native | Less time vs Native |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 67.569 | 65.387 | 67.185 | 61.554 | 5.86% | 1.062× | 8.90% |
| 4 | 255.442 | 245.837 | 251.498 | 189.095 | 23.08% | 1.300× | 25.97% |
| 8 | 541.038 | 544.966 | 497.004 | 382.640 | 29.79% | 1.424× | 29.28% |
| 16 | 1209.441 | 1085.581 | 1055.552 | 761.729 | 29.83% | 1.425× | 37.02% |
| 32 | 2443.264 | 2162.537 | 2107.444 | 1560.980 | 27.82% | 1.385× | 36.11% |
| 64 | 5048.988 | 4328.811 | 4539.532 | 3159.687 | 27.01% | 1.370× | 37.42% |

Tuned Native at B8 was slightly slower than default Native during confirmation. The preselected profile is retained without selecting again on confirmation results. Stored layer-bootstrap intervals describe variation among measured layers and are not confidence intervals from independent whole-model runs.

## Cached-disk-offloaded prefill

Means below are **seconds for three measured complete passes per arm**, after one warmup. These include cached weight reads, packing, H2D transfers, embedding, all layers and last-position logits. The common loader performs Strassen preparation for every arm; this is not a separately optimized native serving baseline. No decode, initial network downloads or compilation is timed.

| Batch | Native s | Tuned Native s | Tuned Cubic s | Strassen S1 s | Less time vs Tuned Native | Less time vs Native |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 59.090 | 59.302 | 59.023 | 59.011 | 0.49% | 0.13% |
| 4 | 59.167 | 59.413 | 59.172 | 59.102 | 0.52% | 0.11% |
| 8 | 59.898 | 59.586 | 59.511 | 59.428 | 0.27% | 0.78% |
| 16 | 60.239 | 60.252 | 59.041 | 59.684 | 0.94% | 0.92% |
| 32 | 61.032 | 61.032 | 60.382 | 60.102 | 1.52% | 1.52% |
| 64 | 64.339 | 62.986 | 63.353 | 62.092 | 1.42% | 3.49% |

Strassen’s mean offloaded time is lower than both native baselines at every batch, but most differences are small relative to the observed pass variation. Cubic has the lowest offloaded mean at B16. At B64, all three Strassen passes are below all three passes of each native baseline; the mean saving is 0.894 seconds (1.42%) versus Tuned Native and 2.247 seconds (3.49%) versus Native. Three passes on one allocation do not establish a general deployment gain.

The large resident savings therefore carry only modestly into this offloaded harness. Recurring loading/preparation/transfers dominate its wall time. These are distinct measurement scopes, so their difference should not be reported as a direct measurement of transfer time.

All individual offloaded pass times (seconds):

| Batch | Native | Tuned Native | Tuned Cubic | Strassen S1 |
|---:|---|---|---|---|
| 1 | 59.261, 58.983, 59.026 | 59.293, 58.951, 59.663 | 59.109, 59.068, 58.893 | 58.965, 58.941, 59.128 |
| 4 | 59.315, 58.893, 59.291 | 59.480, 59.422, 59.335 | 59.584, 58.879, 59.053 | 59.729, 58.323, 59.252 |
| 8 | 59.828, 60.124, 59.743 | 58.714, 59.635, 60.410 | 58.811, 58.974, 60.747 | 58.847, 58.664, 60.774 |
| 16 | 60.676, 59.870, 60.169 | 59.326, 60.604, 60.826 | 59.380, 58.676, 59.066 | 59.302, 59.897, 59.853 |
| 32 | 60.704, 61.072, 61.318 | 61.390, 60.862, 60.845 | 60.507, 60.864, 59.775 | 59.487, 60.045, 60.774 |
| 64 | 64.005, 63.868, 65.143 | 62.663, 62.632, 63.662 | 62.899, 63.614, 63.546 | 61.970, 62.281, 62.025 |

## Numerical cost

All four arms passed repetitive-prompt and natural-text gates at every batch. Natural evaluation uses four contiguous WikiText2 test segments; token hashes are recorded. Larger batches cover different corpus spans, so absolute perplexity across batches is not a quality scaling result. Top-1 agreement below is agreement with Native predictions, not ground-truth accuracy.

| Batch | Native PPL | S1 PPL | S1 relative PPL change | S1 top-1 agreement | S1 logit relative L2 | S1 KL (nats) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.955858 | 9.955107 | -0.0075% | 98.705% | 2.180% | 0.0008200 |
| 4 | 8.651647 | 8.654687 | +0.0351% | 98.638% | 2.534% | 0.0009097 |
| 8 | 9.547934 | 9.553060 | +0.0537% | 98.671% | 2.160% | 0.0008961 |
| 16 | 9.763172 | 9.774799 | +0.1191% | 98.751% | 2.069% | 0.0008716 |
| 32 | 9.363616 | 9.367503 | +0.0415% | 98.730% | 2.212% | 0.0008734 |
| 64 | 9.664840 | 9.669219 | +0.0453% | 98.742% | 2.147% | 0.0008825 |

Passing the predictive gates does not mean identical logits. Natural-text S1 logit relative L2 is about 2.07–2.53%; repetitive-prompt logit differences are larger. The complete NLL, PPL, KL, agreement, finite checks, relative L2 and maximum errors for every arm are retained in [summary.json](summary.json), with projection/block RMSE and other diagnostics in the raw events.

Thresholds are unchanged: repetitive absolute NLL difference ≤0.01, KL ≤0.005 and agreement ≥0.99; natural absolute NLL difference ≤0.01, KL ≤0.02 and agreement ≥0.97. Native capture reconstruction relative L2 must be ≤0.01. This follows the parent Qwen computation; it does not add an independent cross-framework model-equivalence claim.

## Frozen tuning decisions

Tiles below are `(BM, BN, BK)` as recorded by the parent implementation. The model-specific geometry was used throughout; Qwen32 tiles were not copied as tuning results. All candidate measurements, filtering/compilation events and profile identities remain in each batch’s raw archive.

| Batch | Native VMEM MiB | S1 gate/up | S1 O | S1 down | Cubic gate/up | Cubic O | Cubic down | Fixed Q/K |
|---:|---:|---|---|---|---|---|---|---|
| 1 | 96 | 1024×1024×5120 | 1024×2560×1024 | 1024×2560×1024 | 1024×2048×5120 | 1024×1024×5120 | 1024×5120×1024 | 1024×1024×5120 |
| 4 | 112 | 2048×1024×5120 | 4096×512×5120 | 2048×2560×1024 | 2048×1024×5120 | 4096×512×5120 | 2048×5120×512 | 2048×1024×5120 |
| 8 | 112 | 2048×1024×5120 | 4096×512×5120 | 2048×2560×1024 | 2048×1024×5120 | 2048×1024×5120 | 4096×2560×1024 | 2048×1024×5120 |
| 16 | 112 | 2048×1024×5120 | 1024×2560×5120 | 2048×2560×1024 | 2048×1024×5120 | 1024×2560×5120 | 2048×5120×1024 | 2048×1024×5120 |
| 32 | 112 | 2048×1024×5120 | 2048×1024×5120 | 2048×2560×1024 | 2048×1024×5120 | 2048×1024×5120 | 4096×2560×1024 | 2048×1024×5120 |
| 64 | 112 | 2048×1024×5120 | 2048×1024×5120 | 2048×2560×1024 | 2048×1024×5120 | 2048×1024×5120 | 2048×5120×1024 | 2048×1024×5120 |

## Preservation and interpretation

This is evidence that the parent fused Strassen implementation can improve LLM decoder computation using real checkpoint weights and propagated activations. It does not isolate arithmetic savings from fusion/scheduling, or measure future hardware benefits. All four controls remain available.

The first allocation request received HTTP 502 and succeeded on retry. A preliminary local validation environment lacked `requests`; the unchanged checks passed in the existing validation environment. Neither issue caused a TPU measurement-stage retry or a change to scientific settings.

- [Immutable batch results](../../results/20261017_v001): all raw timing/error events, profiles and per-batch tar files.
- [Completion and release evidence](../../operations/completed_20261017_v001): full remote archive, environment freeze, installation report and independent release/process checks.
- [Initial index snapshot](../../operations/initial_index_20261017_v001): the original launch README and manifest remain preserved.
- `build_summary.py` regenerates this report and JSON using only published raw summaries and refuses to replace differing output.

Source archive SHA-256: `f3ff049d57634fd5a7d4638120cfc8316772e2b64a59c33d4f2d69ea37cc62e1`. Full results archive SHA-256: `5acc78792d443dceb850edfd67fdcb7c4a91354e4a5e246c0df71966daa04524`.

The code, protocol, historical partial summaries and prior Qwen32/Gemma results remain unchanged. No other model, architecture or new experiment is queued.
