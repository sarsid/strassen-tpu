# B64 natural-text quality checkpoint

All four algorithms passed unchanged natural-text gates on 261,888 next-token
positions (four independent batches of 64 sequences, each length 1,024 with BOS).
Native perplexity is 5.7990985226; Strassen S1 is 5.7993672578, a relative change
of +0.004634%. S1 absolute NLL delta is 0.00004634, mean KL is 0.00185747 nats,
next-token agreement with Native is 98.3905%, and relative logit L2 is 4.4380%.
Agreement is against Native, not ground-truth accuracy. Tuned Native and Tuned
Cubic also differ numerically from Native and pass all gates; see summary.json.

Resident timing and repetitive quality were already preserved in the separate
20261017_b64_residency_confirm_v001 checkpoint. Serialized offloaded prefill is
now running. This checkpoint does not claim a completed full batch or a total
inference speedup. The full retry archive will follow when execution finishes.
