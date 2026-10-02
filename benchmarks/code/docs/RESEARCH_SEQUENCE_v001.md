# Sequence from matrix experiments to LLM applications

Written September 26, 2026. This records the proposed research sequence and the
current archive work; it does not launch or claim completion of new TPU studies.
The six-shape v6e joint-tuning pilot is complete. Its observations guide the next
search; they do not establish a rule for unseen dimensions.

## 1. Preserve the historical v5e reference

Export the September 21 **168-shape MAIN study** to
[`results/v5e/168_shapes_fp32_20260921_v001`](../results/v5e/168_shapes_fp32_20260921_v001/README.md).
Keep the original `runs/` evidence intact. Include all 12 screening phases, all
12 confirmation phases, the main smoke phase, raw timings, failed candidates,
frozen selections, uncertainty, numerical errors, configurations and source and
environment provenance. Verify checksums, coverage, and the original selection
and statistical calculations before committing the bundle.

This historical dataset uses BF16 inputs and FP32 accumulation/output. It is
separate from prior v5e LLM studies, v6e studies, and future matched reruns.
Archive it as historical evidence, not as a completed BF16-output or updated
joint-tuner experiment.

## 2. Freeze the shared contract for two matched 168-shape studies

Use the same 168 distinct M,K,N tuples and preserve their original IDs and
sampling provenance. Include small, skinny and boundary cases, even when custom
kernels lose or fail. Use identical operand generation, seed policy, reference
calculation, error gates, timing scope and confirmation rules on both devices.

| Method | FP32 output | BF16 output |
|---|---|---|
| Default Native | Measure | Measure |
| Tuned Native | Measure | Measure |
| Independently tuned cubic | Measure | Measure |
| Tuned Strassen, one recursive level | Measure | Measure |
| Tuned Strassen, two recursive levels | Measure | Measure |

Inputs are BF16 and accumulation is FP32 within each comparison. Record
Strassen's pre-addition precision explicitly. Include final output conversion
and any required padding/cropping in complete-call time and error. Keep host
transfer, compilation and tuning time separate. Default and tuned Native may
share an execution when the selected configuration is identical; do not invent
independent samples.

Pin hardware and software identity. Prefer a common supported JAX/jaxlib/libtpu
stack for both architectures after qualification. If that is impossible,
document the version difference and avoid attributing the entire performance
gap to hardware. Use architecture-appropriate memory limits and candidate
geometries under the same declared search principles. Equal search policy does
not require identical tiles or physical memory limits.

Before committing to the full cost, count offered candidates and estimate the
compilation budget from a representative preflight. The 1,680 shape/method/output
entries per architecture are reporting cells, not the number of tuning trials.

## 3. Qualify the general search and its evidence records

Search output tiles BM,BN; K-panel length BK including full K and registered
shorter panels; one/two buffers; and applicable accumulator/reconstruction
strategies. Keep recursion depths 1 and 2 independently tuned. Tune cubic
independently as a real baseline and retain selected tile-matched cubic checks
where useful for separating tiling from arithmetic effects.

The completed pilot covers six aligned shapes. Verify applicability across all
168 shapes before broad execution: alignment, dimensions smaller than tiles,
padding/cropping, memory requirements and precision. Retain established general
kernel controls where applicable. Record unsupported or pruned candidates;
do not silently remove shapes or revive the discarded experimental padding
families without a new design decision.

Separate screening inputs from fresh confirmation inputs and freeze winners
before confirmation. Preserve error failures and allocation failures. Record
the practical speed margin, numerical requirements and statistical criterion
before seeing new results. Use Native when a custom choice fails these gates.

The per-configuration record should include:

- Shape, input/accumulation/output dtype, layout, architecture, software and
  compiler identity, kernel version, allocation, seed and input fingerprint.
- Algorithm and depth, BM/BN/BK, padding ratio, storage/reconstruction strategy,
  buffer count, compiler options, memory limits and feasibility outcome. Native
  tiling is compiler-managed; do not fabricate a tile size for it.
- Raw timing rounds, warmups, ordering, compile time, available memory estimates,
  confirmation intervals and comparisons to default Native, tuned Native and cubic.
- Relative L2, maximum absolute error, RMSE, MAE, median/p99 absolute error,
  normwise error, finiteness, reference precision and sample scope/indices.
- Candidate exclusions, screening rank, frozen selection, tested alternatives,
  and the reason for accepting or rejecting the recommendation.

Dimensions alone cannot guarantee error bounds for arbitrary values. Gaussian
geometry tests and any separately registered numerical stress tests must retain
their own labels. Neither substitutes for later model-quality validation.

## 4. Run and commit the updated 168-shape v6e experiment

Apply the qualified search to both output contracts. Archive and verify every
screen/confirmation batch as it finishes. Generate results, configuration
tables, errors, exclusions and per-choice explanations. Commit a dedicated
v6e results bundle and verify runtime release before declaring it complete.

This new broad run extends the joint-tuner pilot. Keep it separate from the
earlier FP32-only v6e 168-shape study and from the six-shape pilot.

## 5. Run and commit the matched updated v5e experiment

Use the same registered experimental contract and algorithm capabilities,
with v5e-specific qualification and tuning. Measure **both** FP32 and BF16
output. The historical archive remains a reference/control; its timings do
not become samples of the new study.

New measurements are needed to close the current gaps: BF16 output across
all five families, the new joint contraction/storage/buffering search, and
software comparability. Those are missing experimental conditions, not missing
files that can be recovered from the old allocation.

Commit the complete v5e results separately, including losses and failures.
Keep paired comparisons within an allocation; do not pool interrupted phases
with fresh-runtime timings. This step and step 4 may exchange execution order
for availability, but neither is complete based on the historical dataset.

## 6. Compare architectures and inspect representative mechanisms

For matching shapes and precision, compare S1/S2 against each device's own
default Native, tuned Native and tuned cubic. Report absolute latency as well
as relative speedup. Compare selected geometry, BK, storage, buffers, errors,
memory failures and S2-versus-S1 transitions.

Profile representative wins, losses and reversals to investigate matrix-unit
(MXU) utilization, vector/reconstruction work, memory traffic and overlap where
the available tooling can measure them. Separate observations from hypotheses.
MXU count or peak throughput alone does not establish the cause of a change;
software and tuning differences remain potential confounders.

Preserve the wide/tall and FP32/BF16 reversals from the pilot as useful probe
cases. Its full-contraction comparisons change complete configurations, so
they are not an isolated causal experiment on BK alone.

## 7. Produce a configuration table; keep a general predictor optional

First build an exact lookup table for measured operation contracts. The key
includes M,K,N, architecture, precision, kernel/software version and relevant
layout/fusion context. Store the selected algorithm and parameters plus its
evidence. An error requirement may restrict empirically eligible entries, but
is not a guarantee for arbitrary data. Use a conservative Native fallback for
unmeasured contracts.

A learned or analytical predictor for unseen dimensions requires separate
held-out shapes and explicit validation. The 168 development shapes cannot
also be its untouched test set. **Such a predictor is not a prerequisite for
the LLM experiments.**

## 8. Tune and evaluate fixed LLM workload profiles with fused kernels

Choose a model/checkpoint, device/sharding, dtype, batch size, prompt-length
policy and workload mode. Inventory the actual unique operations and their
layouts, including the fused operations we intend to benchmark. Capture real
activations for kernel checks and use disjoint inputs for final quality tests.

For a dense projection, flattening B sequences of S tokens usually gives
`[B*S, K] @ [K, N]` during prefill. K and N are normally fixed by that projection
and its sharding; M depends on the workload. For ordinary one-token decoding,
M is approximately the active batch size. Attention also depends on the current
KV-cache length. Chunked prefill, continuous batching and routing can change
the effective shapes further. Padding or bucketing can intentionally make a
profile static; account for its cost.

For a fixed profile, tune each distinct operation once, store a versioned
configuration and specialize/compile the kernel with those parameters. Reuse
it for repeated calls and layers with the same operation contract. This is
more appropriate than rerunning a shape selector during every multiplication.
Use a small cached table if several batch/sequence buckets or modes are needed.
One LLM still has multiple projection and attention shapes; one tile is not a
universal configuration for the whole model. Retune fused operations in their
actual context because fusion changes live memory and scheduling.

JAX supports ahead-of-time compilation for a specified input contract; its
compiled execution is specialized to the traced shapes/types. See the
[JAX compilation documentation](https://docs.jax.dev/en/latest/aot.html).
The growing attention cache is described in the
[Transformers cache documentation](https://huggingface.co/docs/transformers/en/cache_explanation).

Compare equally fused Native/cubic/Strassen implementations, plus the normal
application baseline. Report kernel, resident-layer and complete-model timing;
separate prefill, decode and teacher-forced scoring; account for transfer and
memory. Measure logits, layer/output errors, distribution changes, perplexity
and task accuracy where suitable. Freeze acceptable quality changes and the
evaluation corpus before final measurement. A fast standalone MM is useful
only to the extent that it improves the measured application workload.

## 9. Audit the evidence and write the findings

Before synthesis, commit both architecture datasets, the configuration tables,
fused-kernel sources, model evaluation inputs/revisions, raw measurements and
quality results. Record all negative and inconclusive outcomes. Explain which
claims are empirical, which generalize to held-out cases, and which remain
hardware hypotheses. Then prepare the figures and written results around the
demonstrated findings, without conditioning the archive on positive outcomes.

## Runtime discipline

The present controller and Colab keep-alive run on the Mac. Keep it powered,
online and awake with the lid open during these runs; the existing idle-sleep
guard did not prevent lid-close sleep. For unattended operation, move control
and durable checkpointing to an always-on host before relying on laptop
disconnection. Changing where credentials are stored requires its own concrete
access decision. Verify release after result retrieval.
