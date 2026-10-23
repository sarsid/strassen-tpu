# B64 recovery: diagnostic and quality buffer residency

Launched at 02:38 UTC October 17, 2026 on the same retained v6e. Five original
batches (1, 4, 8, 16, 32) are fully complete and published. This operation does
not claim a complete B64 result. Existing CPU FP32 evidence verification has
passed; block qualification is active in the recorded observation.

Stage v006 parks inactive diagnostic and natural-quality states on the host and
releases projection outputs between calls. All model calls retain full B64 ×
1,024 dimensions, original BF16 states, saved tuning, kernels, timing boundaries,
warmups and quality limits. It does not microbatch. Projection host transfers
occur outside projection timers; natural quality has no performance claim.
Resident layer timing and transfer-inclusive prefill are unchanged.

Twenty-three CPU checks passed, including exact BF16 state/output preservation,
projection enumeration/order and sample counts, bounded error metrics, admission
and archive integrity. The initial v005 validation caught dictionary reordering
by JAX device_get; its failure is preserved. Stage v006 saves original site order
before transfer. See the source protocol and both validation receipts.

Run: `20261017-gemma4-b64-residency-retry-v001`.
Source archive SHA-256:
`5b2418c0c0e662806eafb1096f039cf88f83619374edb1565909d77ae2ef7474`.
Worker v003 and observer v002 never allocate or release the TPU. The observer
preserves checkpoints and complete archives; errors leave the TPU retained.
No activation tensors, weights or credentials are included.
