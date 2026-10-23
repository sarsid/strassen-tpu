# Fresh Qwen3-32B admission after runtime replacement

The retained replacement TPU passed actual-engine training checks at sequence prefixes 64, 129 and 1024, followed by fresh test-process admission. Native cached/full outputs were identical. At 129 and 1024, Strassen cached/full outputs also matched its own full recomputation; all outputs were finite. These are implementation checks, not Native-versus-Strassen downstream accuracy results.

The regenerated execution contract exactly matches the original digest `e416613ce19ecc7c55d4b5f1ed8367a534716f5dd40c88d4f33f82042467d1b9`. All 21 restored batches (15 MMLU, four perplexity, two GSM8K) match the original seed hashes byte-for-byte. The preflight source hashes match the unchanged frozen scientific bundle.

The original five completed task evaluations and sixteen paired GSM8K examples remain preserved. Qwen3-32B GSM8K batch 3 of 4 resumed at 19:05:19 UTC on October 20; the event snapshot includes resumed decoding. No new completed downstream task or preservation conclusion is reported in this snapshot. The original scientific settings, strict answer extraction and numerical margins remain unchanged.

The local laptop paused monitoring during part of the run, but fresh remote observations confirmed continued foreground execution. The retained session is v009 and the recovery run is v010. This additive snapshot preserves admission, contract, checkpoint verification and raw resume-event evidence; prior publications are unchanged.
