# Controlled allocation diagnosis — October 19 UTC

The user confirmed sufficient Colab credits. Thirteen attempts with one
persistent notebook UUID accepted the allocation handshake quickly, followed
by repeated HTTP 502 at approximately 240 seconds on the allocation POST.
This establishes failed provisioning requests, not a functioning queue, and
does not identify capacity or account limits as the cause.

Pause the existing retry supervisor and recovery heartbeat, let its one
in-flight request finish, then independently check allocation absence. The
v002 diagnostic reconciles the old notebook identity before trying one recorded
fresh identity with the same V6E1 request, authentication and software. It
records status, timing and a narrow allowlist of error-body classifications;
it never records raw bodies, tokens or arbitrary response headers. Reconcile
both identities after a lost response; never make competing POST requests.

If either identity has an allocation, retain it, save it under the existing
study session and start its normal keepalive. Resume the existing supervisor
to adopt that allocation and run the unchanged scientific source. If neither
identity allocates, preserve the diagnostic and restore the current retry
supervisor and heartbeat. A successful fresh identity supports a request-path
problem but cannot establish the precise provider-side cause.

No model, subset, precision, algorithm, hardware generation or statistical
setting changes. No diagnostic benchmark and no release of a useful TPU.
