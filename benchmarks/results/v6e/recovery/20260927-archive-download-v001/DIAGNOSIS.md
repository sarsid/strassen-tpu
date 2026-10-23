# Archive download failure

The M=4096, K=16384, N=512 screening phase finished remotely at 16:50:34 UTC on September 27. All 760 candidate outcomes were accounted for: 653 successful cases and 107 compiler/device memory exclusions. The latter were handled tuning outcomes and did not stop the worker.

At 16:51:06 UTC the archive download returned TransportError without an HTTP status. The local controller stopped after that single download failure. The TPU allocation was absent when checked at 21:21 UTC; the service supplied no termination reason. Sleep started after the controller stopped, so the Mac power log does not establish the original download failure's cause.

The remote archive could no longer be retrieved, so local partial evidence is preserved here and will not be used as a complete comparison. Sixty completed shapes remain available. The new supervisor resumes the incomplete comparison on a replacement runtime and then continues automatically through the remaining v6e shapes and the historical v5e output studies.

Automatic recovery now covers transient transfers, existing-worker reconciliation, runtime replacement after verified absence, immutable checkpoints, and the architecture transition. A periodic backup task checks for an unexpected supervisor crash. The kernels, tuning protocol and accuracy gates are unchanged.
