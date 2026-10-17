# Qwen3-14B completed B32 checkpoint

B32/S1024 is complete and all four arms pass repetitive and natural-text gates. B8/B16 are also complete. B64 is tuning automatically; B4/B1 remain pending.

| Algorithm | Resident layer-sum ms | Cached-disk-offloaded prefill seconds |
|---|---:|---:|
| Native | 2443.264 | 61.031520 |
| Tuned Native | 2162.537 | 61.032236 |
| Tuned Cubic | 2107.444 | 60.381888 |
| Tuned Strassen | 1560.980 | 60.101875 |

Strassen reduces resident layer-sum time by 36.11% against Native and 27.82% against Tuned Native. This is the sum of 40 separately warmed layers; preparation/transfers, embedding and output head are excluded. It is not uninterrupted model latency.

Offloaded prefill includes the common cached weight reads, preparation, transfers, embedding, decoder and last-position head. Its Strassen mean is 1.52% lower than Tuned Native. All three Strassen samples are below the three Tuned Native samples, but this is a small three-pass observation, not a deployment-level performance guarantee. Native and Strassen ranges overlap. The pipeline remains dominated by preparation and transfers.

Natural perplexity is 9.363616 for Native and 9.367503 for Strassen (+0.04152%). Top1 agreement is 98.7300%, KL 0.00087338 nats, logit relative L2 2.2122%. Agreement is against Native predictions, not ground-truth labels. All gates pass with nonzero numerical differences. Different batches cover different corpus spans; do not rank their absolute perplexities as a batch effect.

The full hash-verified archive preserves raw candidate and confirmation samples, exact frozen profiles, input/token hashes and stage logs. No checkpoint weights, activation tensors or credentials are published.
