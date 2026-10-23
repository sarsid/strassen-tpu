# Qwen3-14B completed B8 checkpoint

B8/S1024 is complete and all four arms pass repetitive and natural-text gates. The other five batches are still pending; B16 is tuning.

| Arm | Resident layer-sum ms | Cached-disk-offloaded prefill seconds |
|---|---:|---:|
| native | 541.038 | 59.898201 |
| native_tuned | 544.966 | 59.586445 |
| cubic_tuned | 497.004 | 59.510871 |
| strassen_tuned | 382.640 | 59.428389 |

Resident Strassen saves 29.28% versus Native and 29.79% versus Tuned Native. This sums 40 independently warmed layers and excludes preparation/transfers, embeddings and the head; it is not uninterrupted whole-model latency.

Offloaded prefill includes the common cached weight reads, packing and transfers plus embedding, decoder and last-position head. Strassen averages 0.78% lower time than Native and 0.27% lower than Tuned Native. There are only three measured passes per arm and their ranges overlap substantially, so these small differences do not establish a reliable end-to-end advantage. The whole pipeline remains transfer/preparation dominated.

All task gates pass. Natural perplexity: Native9.547934 versus Strassen9.553060 (+0.05368%); top1 agreement98.6712%, KL0.00089607 nats, logit relative L2 2.1603%. Agreement compares predictions with Native, not labels; numerical differences are nonzero.

The full archive includes every stage attempt, candidate timing, raw confirmation sample, profile, input/token hashes and metric. The archive SHA-256 is recorded in summary.json. No raw weights or activations are published.
