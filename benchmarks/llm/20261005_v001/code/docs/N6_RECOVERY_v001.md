# N6 device-trace recovery

The first N6 execution, `20260919T063959Z-N6-v5e-v004-d93a61`, completed its 18 ordinary timing/numerical result rows, but **none of its 18 profiler captures contained an identifiable TPU device process**. The strict trace analyzer correctly recorded `diagnostic_failure`. These artifacts provide host traces and compiler diagnostics; they do not provide the requested device-duration evidence.

## Configuration mistake

The N6 protocol author mistakenly froze `tpu_trace_mode: "TRACE_ONLY"`. This was not the setting used in the project's previously qualified profiler. The retained September 11 v5e profiling plan used `TRACE_COMPUTE_AND_SYNC` with the same JAX/JAXlib 0.7.2 and libtpu 0.0.21.1 versions.

The [official JAX profiling documentation](https://docs.jax.dev/en/latest/profiling.html) lists `TRACE_ONLY_HOST`, `TRACE_ONLY_XLA`, `TRACE_COMPUTE`, and `TRACE_COMPUTE_AND_SYNC`; `TRACE_ONLY` is not a supported documented value. The documentation also specifies that device tracer level 1 enables device tracing. Checked September 19, 2026.

The archived Perfetto data contains `/host:CPU`, host dispatch events and eight user step annotations per capture; it has no device process for the analyzer to select. This is a collection failure, not evidence that the TPU did no work. The unsupported option is the leading explanation, supported by the configuration mismatch and the earlier successful protocol. Its causal role will be checked by the separate recovery run; successful recovery is not assumed in this document. TensorFlow Python-hook import warnings also appear in the log, but the evidence does not establish that installing TensorFlow would repair device collection. No package installation or runtime upgrade is part of this recovery.

## Bounded recovery implementation

New module: `src/strassen_mm/benchmark_n6_recovery_v001.py`. It imports the unchanged `benchmark_n6_n7_v002.py` runner and applies two explicit profiling settings:

- `experiments.N6.profiler.tpu_trace_mode`: `TRACE_ONLY` → `TRACE_COMPUTE_AND_SYNC`.
- `ProfileOptions.device_tracer_level`: explicitly 1, instead of relying on the runtime default.

The original campaign file and its SHA256 remain unchanged. The wrapper uses a deep copy for the effective profiling mode, writes an exclusive `profile_recovery_override.json` before profiling, and includes that artifact in the normal result seal. It briefly wraps the public options constructor to set device tracing on the real JAX options object, then restores the constructor even if profiling raises an exception. This is limited to a fresh serialized N6 process.

The wrapper requires `--original-profile` pointing to the canonical failed N6 run/artifacts. It validates the original seal and the recorded 18 attempted/zero successful captures; the exact original campaign; the same selection and confirmation hashes; and unchanged benchmark, selector and kernel dependency hashes. Before executing a group it requires identical machine identity, representatives and planned cases. Consequently, shapes, selected kernels, tiles, variants, inputs, numerical gates, ordinary timing rounds and allocation remain fixed. The wrapper accepts only `--phase N6`; it cannot run N7 or retune a choice.

Each of the original three representatives retains three arms and two captures per arm, with eight synchronized calls per capture and reversed arm order in the second block. Ordinary timing is retained before and after profiling. The existing analyzer still requires one actual TPU process, one XLA Modules track, exactly eight positive-duration module events and no overlapping module intervals. No host-event fallback is introduced. If any of the first group's six captures fails coverage, the recovery ends as a failed execution before attempting further groups. All collected evidence remains archived.

The recovery's ordinary timing is a diagnostic repeat on the same cohort, not an additional independent replication of N5 confirmation. Profiler wall times and device durations remain separate from headline complete-call latency. Static memory/cost/roofline diagnostics are not hardware utilization counters.

## Execution interface and status

Use the existing versioned launcher with module `strassen_mm.benchmark_n6_recovery_v001`, phase `N6`, the original frozen campaign, original N5 selections and confirmation, the same expected identity/allocation, and the additional `--original-profile` argument. The parent execution manager owns snapshots, execution, sealing and commits. All original v001/v002 source and failed N6 evidence remain intact.

Four new offline tests cover the exact profiling-only copy, preservation of the real options type with explicit device tracing, rejection of N7 dispatch, and constructor restoration on capture failure. The existing v002 analyzer tests already cover host-only rejection and valid device-module extraction. No tests or TPU recovery were executed by the protocol agent. This document records the proposed correction, not a successful recovery result.

Local evidence: [original N6 summary](../runs/20260919T063959Z-N6-v5e-v004-d93a61/artifacts/summary.json), [original N6 journal](../runs/20260919T063959Z-N6-v5e-v004-d93a61/artifacts/results.jsonl), and the existing project's `strassen-tpu/evidence/validation/2026-09-11-v5e-profiling/plan.json` and `profile_qwen3_finalization.py`.
