# Lost launch acknowledgment, September 27 2026

At 03:12:43 UTC (September 26, 11:12:43 PM Eastern), the Colab client returned `Connection was lost.` while waiting for the arch-027-confirm launcher. No HTTP status or more specific disconnect cause was supplied. The controller correctly avoided a duplicate launch but stopped the full campaign because it could not reconcile this uncertainty.

The detached worker had started on the same allocation and completed at 03:13:06 UTC. Its archive was ready at 03:13:07 UTC. All 63 confirmation cases were `ok` and numerically eligible. The shape was M=32768, K=512, N=64; both FP32 and BF16 outputs were measured. No kernel/OOM/numerical failure caused this interruption.

The downloaded archive matches the remote SHA-256, and every manifest artifact, original source hash, run identity and planned case count was verified. The original failed controller receipt remains intact. Twenty-seven earlier shapes and this recovered shape give 28 completed geometries (56 output contracts).

Transport v006 polls the existing run after a lost launch response and never repeats the launch. Tests cover successful recovery, an ordinary acknowledged launch, a missing worker and a wrong-source archive. A new continuation preserves every measurement source byte and resumes the other 140 shapes on the same allocation. The underlying reason for the connection dropping remains unknown.
