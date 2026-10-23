# Qwen3-14B B16 confirmation and quality

Resident layer sums: Native1209.441 ms, Tuned Native1085.581 ms, Tuned Cubic1055.552 ms, Strassen761.729 ms. Strassen saves37.02% versus Native and29.83% versus Tuned Native across40 separately warmed layers, excluding preparation/transfers, embedding and head. This is not uninterrupted whole-model latency.

All repetitive and natural-text task gates pass. Natural perplexity changes from9.763172 Native to9.774799 Strassen (+0.11909%); top1 agreement98.7506%, KL0.00087162 nats, logit relative L2 2.0691%. Agreement compares to Native predictions, not labels.

Offloaded prefill is still running. This checkpoint preserves final frozen tuning, qualification, resident confirmation and quality summaries; it is not yet a completed batch archive. B8 is complete; the other batches remain unfinished.
