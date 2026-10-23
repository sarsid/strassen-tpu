# Qwen3-32B B64 on the repaired compiler stack

The user's October 15 repair-and-run authorization continues. The v003 collector
successfully compiled and ran all four full-size capture steps, but both fresh
processes then crashed compiling the unchanged Native full block on JAX 0.7.2.
Evidence is preserved under results/v6e/qwen32_b64_retry_20261015_v003; that owned
TPU was released. Capture segmentation alone does not fix the broader compiler
failure, and v003 supplies no confirmed performance result.

Use JAX/jaxlib 0.11.2 and libtpu 0.0.48, the previously used current stack.
JAX issue 40935 reports a related v6e attention compiler stack overflow beginning
in 0.7.2 and resolved in 0.11.2. This motivates testing the update; it is not
proof that it fixes this workload. Parent kernel source, model revision,
BF16/DEFAULT semantics and the B64/S1024 workload remain unchanged.

Treat v004 as a separate software-stack extension. Retune Native VMEM, Cubic,
and Strassen using the original parent-based candidate menus and timing
procedure. Do not deploy the old-stack selected profiles as newly tuned results.
Freeze and hash the new profile before qualification and confirmation. The
default Native arm retains empty compiler options; all four arms share the
same software stack and device. No B32 repeat or other model is queued.

Run tune, qualify, confirm, quality, and prefill in that order, with at most
one fresh-process retry per failing stage. Qualification uses the validated
v003 segmented full-batch capture and <=.01 relative-L2 reconstructed Native
check. Timing loops, sample counts, original predictive gates, real model
weights, repetitive fixtures and four held-out WikiText segments remain as
specified in QWEN32_PARENT_BATCH_v001.md. Corpus capacity was verified locally:
298939 tokens available, 262144 needed, with token hashes matching the completed
B32 run. Compile/host staging remain outside resident measurement.

Store all evidence under results/v6e/qwen32_b64_current_stack_20261015_v004.
Label hardware/software on every report. Within-B64 four-arm comparisons are
matched; comparisons with old-stack B1-B32 confound batch and compiler changes.
Preserve failed attempts, commit sources/results, update the live journal and
release the owned TPU after archival. Do not modify historical source bundles.
