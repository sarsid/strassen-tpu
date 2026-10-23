# Gemma v003 qualification checkpoint

The new owned v6e was allocated at 14:46:47 UTC October 16. Qualification passed
by checking the approved CPU FP32 evidence against the unchanged model and
checkpoint source hashes. This is explicitly reused evidence, not a new FP32
execution. The original cross-backend BF16 failures remain preserved.

At the recorded observation, B8 local/global tuning was in progress. There
are no completed batch timing or quality results in this checkpoint. The same
TPU BF16 algorithm-quality gates must still pass. The supervisor and provider
keepalive were independently observed running, and Colab listed the owned
endpoint. No release or duplicate worker was requested.
