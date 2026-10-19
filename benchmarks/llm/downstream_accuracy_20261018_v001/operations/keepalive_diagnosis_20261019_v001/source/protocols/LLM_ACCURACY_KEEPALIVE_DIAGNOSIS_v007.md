# Capture keepalive evidence after repeated allocation loss

The v002 worker launched at 04:51:38 and last responded at 05:12:24 UTC.
The v003 worker launched at 06:09:50 and last responded at 06:30:32 UTC.
Both disappeared approximately 21 minutes after launch while decoding the
first GSM8K batch. v003 restored the original contract/checkpoints and reached
GSM8K without recomputing MMLU or perplexity. Neither run saved a complete
GSM8K batch. Independent assignment checks found no retained TPU.

The repeated timing suggests an allocation lifecycle issue worth investigating;
it does not prove idle timeout, a quota issue or memory failure. The original
keepalive process sent its stdout/stderr to DEVNULL, losing error evidence.
The installed client also suppresses request ReadTimeout and documents an
assumption that activity was recorded. That assumption is not a response
acknowledgement. A request to the old absent endpoint returned 404, not a scope
error; the saved credential scopes include Colab and Drive access.

Controller v004 preserves sanitized keepalive outcomes in the run directory.
It observes status and elapsed time, recording exceptions even if the installed
SDK suppresses them. Only HTTP 2xx is marked acknowledged. It never logs headers,
request/response bodies, credential values or query strings. No method, endpoint,
cadence, scopes, allocation policy or runtime computation is changed.

Recovery v004 uses the same verified checkpoint seed and byte-identical scientific
source bundle as v003. The new host starter/allocator use logged keepalive v004.
Once a TPU is obtained, inspect actual keepalive responses as well as fresh
remote progress. Retain on errors. If responses reveal a specific fault, fix
that fault on the owned endpoint before further relaunches. Continue persistent
allocation attempts as the user requested, without duplicate allocations.

All study settings, snapshots, preservation criteria and release requirements
remain unchanged. This diagnostic does not bypass any service access limit.
