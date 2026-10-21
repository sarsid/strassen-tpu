# Resume Gemma GSM8K after confirmed runtime loss

The v011 runtime was last readable at 11:08:29 UTC on October 21. Its runtime
APIs then returned HTTP404 while allocation listing and keepalive still worked.
At11:19UTC the observer and keepalive independently recorded absence. A fresh
11:36UTC allocation listing was also empty. No release was requested; the exact
provider cause is unknown. Do not infer a credit or subscription failure.

Runv012 restores both verified Qwen model archives and all21 saved paired Gemma
batches:114 MMLU,32 perplexity windows and16 GSM8K examples. It resumes only
GSM8K batches3 and4, after the same fresh train-only preflight and admission.
The scientific archive, operational evaluation worker, numerical commands,
software pins, execution contract and scoring rules are unchanged.

The new bootstrap/seed builder validate each saved batch against its immutable
verified report and record its execution origin. Thirteen MMLU batches are
fromv010, whose original numeric admission receipt remains unavailable. Eight
other batches (last2MMLU,4perplexity,2GSM8K) are fromv011 with actual passed
preflight/admission receipts. Preserve those old receipts under provenance;
freshv012 receipts must not replace or impersonate them. Exact regenerated
contract equality is required by the unchanged scientific driver before reuse.

The versioned supervisor uses the new seed validator and bootstrap but preserves
serialized allocation requests and foreground launch reconciliation. The sleep
guard only changes the expected supervisor identity fromv007 tov008; bounded
assertion behavior stays unchanged. No scientific source is edited. Recovery
checks cover exact restoration, preserved origin/hash checks, complete pairs,
missing-batch rejection, failed admission rejection, safe extraction, unchanged
scientific commands and failed-preflight retention. Keep the same account,
retry automatically, and do not create duplicate allocations or workers.

Retain allocated TPUs on errors. Release only after9/9 tasks and three model
archives plus the final archive are locally hash-verified. Publish scientific
code/results additively, preserving all earlier snapshots and evidence gaps.
Raw local machine/process diagnostics remain local.
