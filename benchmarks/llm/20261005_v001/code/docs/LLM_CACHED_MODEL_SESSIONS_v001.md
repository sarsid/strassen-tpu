# Cached model sessions

The user approved all three optimizations on October 3: cache prepared weights,
add parent-style resident measurements while preserving full-forward timing,
and reuse each model's runtime/checkpoint. The current legacy Qwen3-14B workload
must finish unchanged. The queue order remains Qwen8, Qwen32, Gemma12, Gemma27,
Mistral24, Qwen14 and Mistral7. No experiment identities or quality thresholds
change, and all 15 randomized full-forward confirmation rounds remain.

New files are versioned; existing executed sources remain immutable. Deployment
requires archived local correctness/recovery checks. The first workload of every
new model session additionally compares the cached and original full Native
model's hidden states, logits and quality metrics bit-for-bit in both stores.
This check must pass before that precision's timed confirmation. Cache timing,
hit/miss/eviction/bypass counts, disk bytes and peak process RSS are recorded.
It is a qualification pilot within an already authorized workload, not an extra
algorithm comparison or permission to change numerical gates.

The cache stores exact canonical transposes and packed layouts privately,
keyed by pinned checkpoint file hashes, configuration and packing source.
Layout keys omit algorithm/depth/output dtype only when the existing packer
uses exactly the same layout. All projection-specific padding, interleave tile
width and rotary head width remain in the key. Common head and norm weights
are shared. Files have SHA256 metadata, atomic publication and read-only mmap
views. A newly opened cache verifies content before reuse. Only one algorithm's
full prepared host references are active; inactive pages are reclaimable by the
OS. Active arrays are prefaulted before the unchanged three warmups and timed
forward. The disk budget defaults to the smaller of 180 GiB and 65% of initial
free disk. Unreferenced derived layouts can be evicted; a full cache falls back
to a temporary one-arm buffer rather than changing arithmetic or exceeding the
cache limit. Cache performance therefore remains an empirical result.

The resident pass propagates each algorithm's own hidden states through every
layer, rotating algorithm order deterministically per layer. Each algorithm's
layer parameters remain resident for three warmups and 15 repeats. It reports
per-layer samples and the sum of layer medians, excluding embedding, weight
transfers and the head. It is stored separately as `resident_pass`; it never
replaces `elapsed_median_ms`, full-forward confidence intervals or held-out
quality. The full forward still includes recurring serialized H2D, embeddings,
normalization and every token's logits.

A session holds one v6e for the remaining queued workloads of one pinned model.
Environment and generic kernel/semantic checks run once. Official model inputs,
full corpus tokens and the 64-token oracle are hashed and reused; each workload
gets its own exact token slice and provenance. Native checkpoint qualification
is reused only when checkpoint, oracle, code, device/software and gate identity
match. Each workload still tunes its own shape, freezes profiles and evaluates
its held-out window. Workloads use fresh Python processes to release JAX buffers
and failed compilation state; prepared files persist. We do not yet reuse JAX
executables across those processes.

The local supervisor tracks a session separately from its 70 workload states.
Only one controller/endpoint is active. Five-arm precision checkpoints download
and merge independently, and each finished/failed case also has an archive.
Unexpected runtime loss requeues only unfinished groups; partial rounds are
never combined across runtimes. Unstarted cases do not consume retry budgets.
Shared setup/cache failures stop the queue after release. No new workload starts
after 20 hours; remote work has a 22-hour session limit and local control a
23-hour limit. Provider termination remains possible; completed local groups
survive it. The owned TPU is released at the end of the model session, or on
controller failure. Model changes use a new allocation.

Source changes, frozen launch bundles, input/weight cache qualification and
recovery evidence must be committed before deployment. Raw checkpoint bytes,
credentials and weight/layout caches are outside all result archives.
