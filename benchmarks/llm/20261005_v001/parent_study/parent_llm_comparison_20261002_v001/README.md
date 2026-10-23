# Our fused kernels in the parent Qwen3-32B experiment

Qwen3-32B B8 S1024; resident 64-layer compute sum, not end-to-end serving latency.

Final profiles use the separately archived interleaved v005 refinement. The v004 screen remains intact; its normalized scores are not reported runtimes.

| Custom store | Family | Native 64-layer ms | Cubic ms | Candidate ms | Speedup vs Native | Speedup vs cubic |
|---|---|---:|---:|---:|---:|---:|
| bfloat16 | S1 | 1126.742 | 931.907 | 924.635 | 1.2186x | 1.0079x |
| bfloat16 | S2 | 1126.508 | 931.328 | 931.269 | 1.2096x | 1.0001x |
| float32 | S1 | 1127.164 | 990.611 | 959.115 | 1.1752x | 1.0328x |
| float32 | S2 | 1127.846 | 990.562 | 990.124 | 1.1391x | 1.0004x |

The initial S1/BF16 timing includes one 197.153625 ms sample and fails the
speed-versus-cubic gate. S2/BF16 also fails that gate because it is tied with
cubic. Both FP32 policies pass, although S2's advantage is only 0.044%.
The [separate follow-up](../parent_llm_followup_20261002_v001/README.md) preserves
an unchanged-profile repeat and a revised S1/BF16 down tile; it does not replace
these original measurements. The [combined report](../parent_llm_study_20261002_v001/README.md)
explains the complete comparison with the parent.

| Custom store | Family | WikiText Native PPL | Candidate PPL | Absolute NLL delta | KL | Top-1 agreement | Candidate quality gate |
|---|---|---:|---:|---:|---:|---:|---|
| bfloat16 | S1 | 8.457450 | 8.454608 | 0.00033615 | 0.00140318 | 98.3565% | Pass |
| bfloat16 | S2 | 8.457450 | 8.454008 | 0.00040712 | 0.00190903 | 98.0022% | Pass |
| float32 | S1 | 8.457450 | 8.455130 | 0.00027438 | 0.00126517 | 98.3016% | Pass |
| float32 | S2 | 8.457450 | 8.458034 | 0.00006903 | 0.00184837 | 98.0602% | Pass |

Native fallback is eligible at each site; family names describe tuned policies, not a claim that every projection uses Strassen.
FP32/BF16 label custom kernel stores. Both feed the BF16 parent model, with consumer conversion included in timing. This is not an all-FP32 model comparison.
Our Q/K normalization retains HIGHEST dot precision, while the unchanged parent uses DEFAULT. Parent-vs-ours differences therefore include this precision choice, packing/layout and tile policy, not only the multiplication algorithm.
Downloads, compilation, weight packing/transfer, embedding lookup and final scoring are excluded from resident compute timing.
Quality is teacher-forced agreement against the parent Native implementation, over 32,736 WikiText positions and 8,184 repetitive-text positions. It is not an independent official-model qualification.
All execution outcomes, quality/speed verdicts, raw samples, per-layer errors and profile details remain in summary.json and the JSONL files. Execution completion does not imply every benchmark gate passed.
