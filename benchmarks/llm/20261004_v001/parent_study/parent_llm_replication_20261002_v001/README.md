# Parent Qwen3-32B reproduction on v6e

Qwen3-32B B8 S1024; resident compute excludes downloads, packing, transfer, embedding lookup and final scoring.

| Environment | Native block ms | S1 block ms | Block speedup | Native 64-layer ms | S1 64-layer ms | Streamed speedup |
|---|---:|---:|---:|---:|---:|---:|
| JAX 0.7.2, libtpu 0.0.21.1 | 17.648 | 14.594 | 1.2093x | 1130.351 | 872.050 | 1.2962x |
| JAX 0.11.2, libtpu 0.0.48 | 17.646 | 14.021 | 1.2585x | 1129.029 | 832.429 | 1.3563x |

The published targets are 1.2093x for the block and 1.2950x for streamed resident compute.
The single-block and streamed parent scripts use different down-projection tiles; both are preserved as written.

The current-stack block fails the parent's **additional scheduling gate**:
product-aware early finalization takes 14.021391 ms versus 14.003977 ms for
standard fused Strassen. Its measured difference is +0.017413 ms, with the
parent's 95% interval [+0.000823, +0.034004] ms. Both variants beat Native and
cubic. The original-stack block passes all three gates; both stacks pass the
64-layer speed/task gates and WikiText quality gate. “Completed” execution
therefore must not be read as every optimization hypothesis passing.

| Environment | WikiText Native PPL | S1 PPL | Absolute NLL delta | KL | Top-1 agreement | Quality gate |
|---|---:|---:|---:|---:|---:|---|
| parent | 8.457450 | 8.457997 | 0.00006466 | 0.00148836 | 98.3107% | Pass |
| current | 8.457450 | 8.456219 | 0.00014554 | 0.00151429 | 98.2527% | Pass |

All 64 original-stack per-layer error records and repetitive-text task metrics match the published artifact exactly.

These are parent-code replications. A newer-stack improvement is a software-environment effect, not evidence of an improvement in our kernels.
WikiText covers 32,736 scored positions; repetitive text covers 8,184. Agreement with the parent Native implementation is not independent official-model qualification.

Detailed timing samples, error metrics, token hashes and metadata are in the six JSONL files. Source, setup and release evidence are in the linked source run in summary.json.
