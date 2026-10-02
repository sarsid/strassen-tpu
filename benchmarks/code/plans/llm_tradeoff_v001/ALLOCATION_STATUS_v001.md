# Allocation diagnosis — 2026-09-22

The new LLM tradeoff experiment has completed CPU validation, but has run
zero new TPU measurements. The previous Qwen and partial Mistral experiments
remain separate evidence.

## v5e: usage-time quota exceeded

The allocation POST returned HTTP 503 with this sanitized service response:

```json
{"accelerator":"","endpoint":"","machineShape":0,"outcome":2,"sub":0,"subTier":0,"variant":0}
```

Google's official Colab client defines `QUOTA_EXCEEDED_USAGE_TIME = 2`.
The response body therefore identifies a usage-time quota rejection, despite
the generic HTTP 503 status. The initially reported possibility of temporary
capacity exhaustion is superseded by this more specific evidence.

Source, pinned to the last inspected v1 client implementation:
[Outcome enum](https://github.com/googlecolab/colab-vscode/blob/ece98d78bf2a0762d050e2ff28d2e1cf10ca3f0f/src/colab/client/v1/api.ts#L51).
The same client's assignment handling maps that outcome to
[InsufficientQuotaError](https://github.com/googlecolab/colab-vscode/blob/ece98d78bf2a0762d050e2ff28d2e1cf10ca3f0f/src/colab/client/v1/index.ts#L246).
Both files were retrieved and inspected during this diagnosis.

This identifies the allocation rejection; it does not establish that the
paid compute-unit balance is zero, specify when access resets, or prove why
the previous Mistral runtime disappeared.

## v6e: exact reason unavailable

The allocation POST returned HTTP 400 with a generic HTML Bad Request page
and no structured outcome. The precise cause is unconfirmed. Do not infer
that the v5e quota diagnosis necessarily explains the v6e response.

## Account details and next step

Read-only account/subscription queries returned HTTP 403 SERVICE_DISABLED
for this client's API project. That diagnostic API restriction is distinct
from the v5e allocation response. It prevented retrieving the account's
balance, accelerator eligibility or quota refill time. No billing, API
settings, subscription or quota changes were made.

Further allocation attempts require restored Colab quota or another available
TPU runtime/project. Repeating requests with unchanged quota is not a useful
recovery action. No new allocation is currently owned by this study.

## Prepared and validated

The versioned protocol covers Qwen3-8B, Mistral-7B-v0.3, Gemma 3-12B and
Qwen3-14B if memory permits, comparing Native/cubic/one-level/two-level
Strassen and a full-JIT Native control. It measures held-out next-token
accuracy, perplexity, distribution drift and numerical error, plus resident
prompt-forward and scoring latency.

CPU validation passed all 15 checks, including an independent numerical
oracle, a tiny official Qwen forward comparison, and a check that resident
forward execution cannot read checkpoint files after loading:
`runs/20260922T122344Z-llm-tradeoff-cpu-check-v002-832a03/artifacts/summary.json`.
These are correctness/integration checks, not TPU performance results.
