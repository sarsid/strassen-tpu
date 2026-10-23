# Declared timing repeat and v6e down-cap audit

This follow-up reuses the same owned v6e, current software stack and checkpoint cache after all eight main evaluation stages. Main-study results remain unchanged.

Exactly one unchanged S1/BF16 streamed repeat tests the isolated 197.153625 ms sample in the first pass. No sample is deleted and neither pass is substituted for the other.

The down-only audit raises the (2048,2560,1024), two-buffer kernel cap from 112 MiB to the parent allowances (120 MiB Strassen, 124 MiB cubic). It retains current settings unless the interleaved score improves by more than 1%. Other sites and norm precision stay fixed.

| Stage | Native ms | Cubic ms | S1/S2 ms | Speedup vs Native | Performance gate |
|---|---:|---:|---:|---:|---|
| s1-bfloat16-repeat | 1126.743 | 931.805 | 887.782 | 1.2692x | Pass |
| s1-bfloat16-streamed | 1127.106 | 931.773 | 884.971 | 1.2736x | Pass |

Changed profile groups: s1/bfloat16.
Changed profiles receive separate 64-layer and WikiText evaluations. Unchanged profiles retain the main-study quality evidence; they are not counted as new quality measurements.

The revised S1/BF16 policy passes the WikiText gate: Native perplexity 8.457450,
candidate perplexity 8.455338, absolute NLL delta 0.00024982, KL 0.00139212 nats,
and top-1 agreement 98.3168%. All layer outputs are finite. Its runtime is
1.0529x the equally fused cubic control. See the
[combined report](../parent_llm_study_20261002_v001/README.md) for the parent
comparison and precision/scope qualifications.

Timing remains the sum of resident layer compute, excluding checkpoint I/O, compilation, weight preparation/transfer, embedding lookup and final scoring. Both output-store variants feed a BF16 model.
Raw samples, gates, profile changes, compilation failures and numerical errors are preserved in the JSONL files and summary.
