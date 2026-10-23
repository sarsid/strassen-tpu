# Qwen3-14B completed B64 checkpoint

B64/S1024 is complete and every arm passes repetitive and natural-text gates. B8/B16/B32 are also complete; B4 is tuning and B1 remains pending. There were no B64 stage retries, workload reductions, profile reuse from other batches or kernel changes.

| Algorithm | Resident layer-sum ms | Cached-disk-offloaded prefill seconds |
|---|---:|---:|
| Native | 5048.988 | 64.338657 |
| Tuned Native | 4328.811 | 62.985756 |
| Tuned Cubic | 4539.532 | 63.353146 |
| Tuned Strassen | 3159.687 | 62.091969 |

Strassen reduces resident layer-sum time by 37.42% versus Native and 27.01% versus Tuned Native. This sums 40 separately warmed decoder layers. Preparation/transfers, embedding and output head are excluded; the sum is not uninterrupted model latency.

Transfer-inclusive offloaded prefill averages 3.49% less time than Native and 1.42% less than Tuned Native. All three Strassen samples are below all three samples of either Native baseline. This remains a three-pass result from one runtime, not a deployment-wide guarantee. The common serialized loading/preparation pipeline dominates total time.

Natural perplexity is 9.664840 for Native and 9.669219 for Strassen (+0.04531%). Top1 agreement is 98.7418%, KL 0.00088246 nats, and logit relative L2 2.1466%. The natural test scored 261888 positions. Agreement is against Native predictions, not ground-truth labels; passing task gates does not mean identical outputs.

All raw candidate and confirmation samples, profiles, metric outputs, hashes and stage logs are preserved in the complete verified batch archive. No weights, raw activations or credentials are published. Compare quality within each batch; different batches cover different corpus spans.
