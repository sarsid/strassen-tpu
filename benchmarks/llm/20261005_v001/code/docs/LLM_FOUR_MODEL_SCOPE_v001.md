# Four-model v6e LLM campaign

The October 3 user request replaces the seven-model queue with this order:

1. Qwen3-8B: continue the active session and finish its remaining workloads.
2. Qwen3-32B.
3. Gemma 4-31B (`google/gemma-4-31B`).
4. Qwen3-14B: retain its completed BF16 B1/S512 group.

Each model has ten prefill workloads and five algorithms, for 200 active BF16
comparisons. Workloads, tuning budgets, DEFAULT arithmetic, warmups, 15 paired
confirmation rounds and quality gates remain unchanged. BF16 output does not
change FP32 accumulation where the kernels use it. No additional FP32 runs,
v5e experiments or depths 3/4 are queued.

Both Gemma3 models and both Mistral models are deferred. Their job records and
attempts move to `deferred_jobs` in the queue state; original plans, executed
sources and results remain intact. The new active plan preserves the existing
Qwen entry identities. Historical FP32 results do not count toward the 200.

## Gemma4 qualification

The official checkpoint revision is
`5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89`. Its official configuration and digest
are preserved in the campaign's `inputs_gemma4_v001` folder. Only public metadata
was downloaded for this scheduling change, not the checkpoint weights.

[Official model](https://huggingface.co/google/gemma-4-31B) and
[configuration](https://huggingface.co/google/gemma-4-31B/blob/5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89/config.json).
Its text decoder has 60 layers, width 5376 and MLP width 21504. Local attention
has 256-wide heads, while global attention has 512-wide heads, a distinct K/V
head count, shared global K/V and proportional partial RoPE. The current pinned
Transformers 4.56.2 and Gemma3 implementation do not support this architecture.

The ten Gemma4 workloads are planned but explicitly `awaiting_qualification`.
Their projection geometry is left unset until a correct adapter is qualified;
we do not apply the Gemma3 geometry helper to them. Before measurement, add and
validate a versioned adapter and compatible official-model oracle, qualify its
outputs against the official checkpoint, and validate the actual source bundle.
The existing five algorithm and error requirements apply to Gemma4 as well.

Queue v008 stops before allocating for an unqualified model. It also stops
before moving on to Qwen3-14B, preserving the requested order. The watchdog
recognizes this explicit gate and does not repeatedly restart or allocate.
The active Qwen8 controller and its frozen measurement code remain unchanged;
only local scheduling and dashboard processes are replaced for this revision.
