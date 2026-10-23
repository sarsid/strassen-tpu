# B64 resident confirmation checkpoint

The residency retry passed qualification and completed all 60 decoder layers.
Summed warmed resident-layer times are Native 9923.863 ms, Tuned Native
8349.988 ms, Tuned Cubic 8979.613 ms and Strassen S1 7540.176 ms. S1 saves
2383.687 ms (24.0%) versus Native and 809.812 ms (9.7%) versus Tuned Native.
These sums exclude embeddings, output head, weight preparation and transfers;
they are not full offloaded inference latency. Per-layer bootstrap intervals in
the summary are not confidence intervals from independent whole-model runs.

The repetitive-input task gates passed for all four arms. S1 has 99.9878%
next-token agreement, absolute NLL delta 0.00009075 and KL 0.00008279 nats.
Its relative logit L2 is 9.31%; passing task gates does not mean identical logits.
Natural-text quality is still running (layer 14 in the recorded observation),
and offloaded prefill remains pending. This is not a complete B64 batch result.

Sources, repair validation and saved-profile identity are preserved in
../../operations/b64_residency_repair_v001/. The full retry log archive follows
when execution ends; no incomplete historical timing attempts are pooled here.
