# Qwen3-14B completed B16 checkpoint

B16/S1024 is complete. All four arms pass repetitive and natural-text gates. B8 is also complete; B32 is tuning, and B64/B4/B1 remain pending.

| Arm | Resident layer-sum ms | Cached-disk-offloaded prefill seconds |
|---|---:|---:|
| native | 1209.441 | 60.238695 |
| native_tuned | 1085.581 | 60.252225 |
| cubic_tuned | 1055.552 | 59.040688 |
| strassen_tuned | 761.729 | 59.684077 |

Resident Strassen saves 37.02% versus Native and 29.83% versus Tuned Native. This sums 40 individually warmed layers, excluding preparation/transfers, embedding and output head; it is not uninterrupted model latency.

Offloaded prefill averages 0.92% less time for Strassen versus Native and 0.94% less versus Tuned Native. Only three measured passes per arm are available and their ranges overlap, so the small mean differences are not conclusive. Cubic has the lowest offloaded mean despite Strassen having the lowest resident mean; the serialized pipeline is dominated by weight reads/preparation/transfers.

Natural perplexity is 9.763172 for Native and 9.774799 for Strassen (+0.11909%); prediction agreement is 98.7506%, KL 0.00087162 nats, and logit relative L2 2.0691%. Agreement is against Native predictions, not ground-truth labels. Numerical differences remain nonzero despite passing all task gates.

Raw per-candidate and per-invocation samples, metrics, frozen tuning, tokens/input hashes and logs are preserved in the complete batch archive. No weights, raw activations or credentials are published.
