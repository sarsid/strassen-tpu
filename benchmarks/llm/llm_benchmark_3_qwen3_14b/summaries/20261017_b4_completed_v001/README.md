# Qwen3-14B completed B4 checkpoint

B4/S1024 is complete; all four arms pass repetitive and natural-text gates. Five batches are now complete. B1 is tuning automatically.

| Algorithm | Resident layer-sum ms | Cached-disk-offloaded prefill seconds |
|---|---:|---:|
| Native | 255.442 | 59.166536 |
| Tuned Native | 245.837 | 59.412642 |
| Tuned Cubic | 251.498 | 59.171876 |
| Tuned Strassen | 189.095 | 59.101520 |

Resident time saved by Strassen is 25.97% versus Native and 23.08% versus Tuned Native. These are sums of 40 independently warmed decoder-layer means and exclude weight preparation/transfers, embedding and output head; they are not uninterrupted model latency.

Offloaded prefill time saved by Strassen is +0.11% versus Native and +0.52% versus Tuned Native. Positive savings means lower time; negative means slower. Each mean uses three full measured passes after one warmup, with the common cached-read/preparation/transfer pipeline. Interpret small mean differences with the raw sample variation; they are not a deployment guarantee.

Natural perplexity is 8.651647 for Native and 8.654687 for Strassen (+0.03513%). Top1 agreement is 98.6376%, KL 0.00090972 nats, logit relative L2 2.5338%. Agreement compares against Native predictions, not labels. Nonzero numerical drift remains despite passing all gates. Corpus spans differ by batch size.

The full verified archive preserves raw candidate timings, confirmation samples, profiles, errors, token/input hashes and stage logs. No weights, raw activations or credentials are published.
