# Qwen3-14B B8: resident confirmation and quality checkpoint

Completed resident timing and quality stages; offloaded prefill is still running.
This is a partial workload checkpoint, not a completed six-batch study.

| Algorithm | Resident layer-sum ms |
|---|---:|
| native | 541.038 |
| native_tuned | 544.966 |
| cubic_tuned | 497.004 |
| strassen_tuned | 382.640 |

Strassen saves 29.28% versus Native and 29.79% versus Tuned Native. These sums cover 40 individually warmed decoder layers and exclude preparation/transfers, embeddings, final norm and output head. They are not uninterrupted whole-model latency. The frozen Tuned Native choice is slightly slower than default Native in this confirmation; both baselines are retained.

All arms passed repetitive and natural-text gates. Strassen natural perplexity is 9.553060 versus 9.547934 Native (+0.05368%); top1 agreement 98.6712%, logit relative L2 2.1603%, KL 0.00089607 nats. Agreement is against Native, not ground-truth accuracy. There is nonzero numerical drift despite passing task gates. Raw per-invocation samples will accompany the full immutable batch archive.
