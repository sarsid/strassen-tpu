# Historical v5e results: 168 matrix shapes, FP32 output

This directory preserves the **September 21, 2026 MAIN matrix study** as a
separate v5e dataset. It contains actual copied evidence, not links to live
campaign state. Original files under `runs/20260921-mlsys-main-v5e-v001` remain
unchanged. No new TPU measurement was made during this export.

Start with [SUMMARY.md](SUMMARY.md) for timings and comparison counts,
[summary.json](summary.json) for selected configurations and numerical errors,
and [WHAT_IS_MISSING.md](WHAT_IS_MISSING.md) for the additional measurements
needed by the proposed matched studies.

## Experimental contract

- 168 distinct M,K,N shapes in their original order, with original labels and
  sampling provenance in [config/shapes.json](config/shapes.json).
- Default Native, tuned Native, independently tuned cubic, one-level Strassen
  and two-level Strassen. Native may share samples between its default/tuned
  labels when the selected configuration is identical.
- **BF16 inputs; FP32 accumulation and output.** This is not a BF16-output study.
- One qualified v5e allocation. JAX/jaxlib 0.7.2, libtpu 0.0.21.1. Exact runtime
  identity and flags remain in each phase's environment record.
- Complete resident call includes device padding/cropping, excludes compilation
  and host transfers. Native tile sizes are compiler-managed and reported as
  null where unobserved.
- Screening offers four Native settings and sixteen tiles per custom family;
  each family freezes its screening choice. Confirmation uses three fresh
  inputs and thirty paired rounds per input. Retained alternatives and tile
  uncertainty are preserved as well as the headline choices.
- Numerical metrics compare the same BF16-quantized operands to an all-K FP64
  reference. Reference scope may be full or sampled; finiteness covers the
  complete output. Error eligibility is not an arbitrary-input safety bound
  or evidence of acceptable LLM quality.

The exact original contract is copied in [PROTOCOL.md](PROTOCOL.md) and
[config/campaign.json](config/campaign.json). This prior implementation/search
predates the six-shape joint contraction/storage/buffering tuner.

## Files and verification

| Location | Contents |
|---|---|
| `phases/MAIN-smoke.tar.gz` | Original main smoke evidence |
| `phases/MAIN-01-screen.tar.gz` through `MAIN-12-screen.tar.gz` | All screening artifacts, raw timing/error events, failed candidates and frozen tuning selections |
| `phases/MAIN-01-confirm.tar.gz` through `MAIN-12-confirm.tar.gz` | All confirmation artifacts, raw rounds, per-input errors, selected configurations and uncertainty |
| `audits/` | Fresh replay audits of all 25 phases; all 12 confirmation audits replay their saved statistical intervals |
| `summary.json` | 168 shape records, each with five method entries, selected configuration, metadata, timings, paired intervals and worst observed error metrics |
| `provenance/phase-index.json` | Original run/receipt identities, bundle checksums and every uncompressed member checksum |
| `provenance/source.tar.gz` | Complete original frozen cohort source archive, compressed without changing its uncompressed bytes |
| `provenance/frozen.json`, `cohort.json`, `plan.json` | Original source commit, source hashes and experiment plan |
| `EXPORT_MANIFEST.json` | Export completion flag and hashes of the exported files |

Each phase bundle contains its original `artifacts/` directory, including
executed source/configuration snapshots, seeds, reference indices, raw events,
statistics and environment identity. It also preserves the original receipt,
completion, execution record and benchmark log. The original full-run manifest
is named `original-artifact-manifest.json`; it refers to the larger original
run, including files intentionally not duplicated here. The portable bundle's
own member inventory is in `provenance/phase-index.json`.

The exporter verifies original artifact hashes, replays the frozen audit code,
and reads each compressed member back to check byte identity. Audits check
recorded metric consistency; they do not recompute the original large matrix
products. Treat the export as complete only when `EXPORT_MANIFEST.json` reports
`completed: true` and its checksums validate.

## Separation and interpretation

Only the MAIN 168-shape study and its smoke phase are included. The separate
18-shape preliminary LLM geometry study, actual-model runs, earlier v5e studies,
v6e runs and interrupted retries are not mixed into these results. The original
cohort's broader controller state includes unrelated work and is not the
completion authority for this exported MAIN dataset; the 25 verified phase
receipts and audits establish its coverage.

The 168 shapes are development data. A selector evaluated on them has not been
validated on unseen shapes. Preserve these historical results as their own
version; future FP32/BF16 and updated-tuner measurements need new directories.

See [the research sequence](../../../docs/RESEARCH_SEQUENCE_v001.md) for the
planned matched architecture experiments and fixed-workload LLM specialization.
