# Actual-model companion to the main v5e matrix study

Frozen before execution. Synthetic large-model shapes and actual-model execution
are separate evidence. The actual branch uses original N9 checkpoints:
Qwen3-0.6B, Mistral7B-v0.3, and Gemma3-1B. It does not claim actual execution of
Qwen3-8B or Gemma3-12B. Pinned official revisions are in the acquisition helper.

1. Download each checkpoint from the official repository using existing account
   access. Record every file hash. Keep checkpoint payloads and authentication private
   outside git; corpus token IDs and reference outputs are archived as evidence.
   An isolated CPU environment generates an independent Transformers4.56.2,
   PyTorch2.8.0 BF16 eager reference for the first64tokens. Global JAX remains
   untouched. Preparation failures are recorded; no substitute checkpoints.
2. Qualify Native actual execution against all63next-token positions and the
   official final hidden state. Preserve N9 logit gates (L2 .03, KL .01, absolute
   meanNLLdifference .05, top1agreement .90). Gemma additionally retains its
   existing final-hidden L2 .02 gate. A failure blocks full-model quality and
   speedup claims; it does not erase first-layer real-operand MM observations.
3. Obtain actual gate/up and down operands from Native layer0 execution on the
   first four1024-token WikiText2 windows. Window0 screens; windows1/2/3 confirm.
   All methods receive identical exact BF16 operands. Native has four compiler
   VMEM settings; each custom family receives the same16candidate attempt
   budget as the main campaign, selected independently at the actual geometry.
   This is a bounded search; Native's tile dimensions remain compiler-managed.
4. Freeze the screen winner per family/projection before confirmation. Confirm
   top3 and candidates within5percent, capped at4/family. Report omissions,
   failures, all-K FP64 sampled reference errors, complete-output finiteness,
   hierarchical paired latency intervals and discrete near-best tile sets.
   A numerical failure is never a valid win. No confirmation-based reselection.
5. All five policies use the same composed layer stages: Native attention,
   separately compiled gate/up, activation, separately compiled down, final
   norm/residual. This preserves independently tuned projection compiler
   options. Resident full-layer timing includes Python dispatch and uses the
   same incoming Native state. A separate full-JIT default Native resident
   and streamed full-model control exposes the overhead of composition; do not use a deliberately
   fragmented Native baseline as evidence of an application-level win.
6. For independently qualified models, propagate each policy's own hidden
   states through every layer on32fixed1024-token windows (32736scored positions).
   Record perplexity/NLLdifference, KL, top1agreement, output L2 and maximum
   error. The quality windows overlap the small performance-tuning subset:
   these are descriptive in-distribution results, not held-out quality claims.
   Preserve N9 quality thresholds (absoluteNLLdifference .01, KL .02,
   top1agreement .97). Report metrics whether they pass or fail.
7. Three streamed full-model repeats include checkpoint reads, host layout,
   H2D transfers, every layer and the full vocabulary projection. Downloads,
   tokenization and compilation are excluded. This is teacher-forced prefill,
   not token generation or production serving. Three repeats are descriptive.

Single v5e only; serial device execution. Later v6e requires a separate cohort.
Each model stage has a5400second wall budget and independent source/results
archive. A blocked qualification is an explicit scientific outcome while the
remaining models and168shape study can continue. No threshold is relaxed.
