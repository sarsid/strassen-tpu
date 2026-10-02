# TPU matrix multiplication benchmarks — September 29, 2026 checkpoint

The experiment is **paused after Colab credits ran out**. This is a reproducible
checkpoint, not a completed 168-shape v6e study. Account reauthentication and
further accelerator allocation are deferred until the owner resumes the work.

| Study | Saved measurements | Status |
|---|---|---|
| v6e joint tuner, FP32 and BF16 output | 113/168 geometries; 226 separate geometry/output comparisons | 55 geometries remain |
| Historical v5e study, FP32 output | 168/168 geometries | Complete, preserved as a separate older protocol |
| Planned v5e historical tuner + BF16 output | Code, configuration and protocol | No new measurements yet |

## Read the results

- [v6e summary and numerical error](results/v6e/report/SUMMARY.md)
- [v6e timings, selected tiles and tuning decisions](results/v6e/report/RESULTS.md)
- [Tuner design choices](results/v6e/report/TUNER_DESIGN.md) and [saved recommendation table](results/v6e/report/tuner_policy.json)
- [Historical v5e results](results/v5e/168_shapes_fp32_20260921_v001/README.md)
- [Pending v5e protocol](results/v5e/168_shapes_legacy_fp32_bf16_20260927_v001/PROTOCOL.md)
- [Machine-readable checkpoint](checkpoint.json) and [resume notes](RESUME.md)

For **each output dtype separately**, the v6e study compares default Native,
tuned Native, independently tuned cubic, Strassen one level and Strassen two
levels. Inputs and recursive pre-adds are BF16; accumulation/reconstruction is
FP32. The final output conversion/store is included in time and error. Current
experiments exclude recursion depths three and four; older source modules are
preserved only for provenance.

The v6e compiler is pinned to JAX/jaxlib 0.11.2 and libtpu 0.0.48. Its tuner
searches tile geometry, K-panel length (including full contraction), accumulator
strategy and one/two buffers, with a v6e-specific memory allowance. Native and
cubic are tuned independently. The pending v5e study keeps the original v5e
kernels, search menu and JAX/jaxlib 0.7.2 / libtpu 0.0.21.1 stack, adding a timed
BF16 output conversion. These are intentionally different architecture policies.

Screening choices are frozen before three fresh Gaussian inputs × 30 paired
timing rounds. The report retains raw samples, failures, every numerical metric,
reference scope, tile and compiler settings, and pointwise paired 95% intervals.
It does not reselect winners from confirmation. Error metrics compare exact
BF16 operands against a full or sampled all-K FP64 reference; full-output
finiteness is checked. These are matrix benchmarks, not LLM prediction-accuracy
measurements. Shapes are development data and execution order is not random;
partial win counts cannot estimate the final grid win rate. Historical v5e
results and the new v6e results are not a matched-protocol causal comparison.

## Source and evidence

`code/` is the committed research source at the commit recorded in
`checkpoint.json`: kernels, tuners, benchmark drivers, architecture-specific
configurations, tests, reports, runtime control, documentation and live-log
implementation. Some historical source documentation references local runs
outside this checkpoint; this README defines the included scope.

`results/v6e/phases/` holds one canonical artifact bundle for each completed
screen/confirm phase and each saved qualification phase. The index records
SHA-256 checksums for the bundle and every uncompressed member. The original
manifest is retained inside each bundle; it also names redundant files from the
larger original execution folder that are deliberately not repeated here.
Every original artifact in the canonical `artifacts/` subtree is included,
including compiler HLO/cost metadata, seeds, raw timings, failures and errors.
Byte-identical remote copies and per-phase nested source archives are omitted;
the exact frozen source archive for every cohort is retained once under
`results/v6e/provenance/`. Failed/incomplete attempts are under `recovery/` and
are not pooled into completed comparisons.

`results/v6e/report/results.json.gz` and `candidate_decisions.json.gz` are
lossless gzip encodings of the full report and candidate ledger. For example:

```python
import gzip, json
with gzip.open('results/v6e/report/results.json.gz', 'rt') as f:
    report = json.load(f)
```

`provenance/` contains the pause receipt, supervisor history and the append-only
progress journal. Runtime identities and original local paths are provenance;
they are not portable executable configuration. Credentials, session-token
stores, virtual environments and model weights are not part of this checkpoint.

## Verify and reproduce the report

From this `benchmarks/` directory, Python 3.11+ is sufficient for:

```sh
python verify.py
```

To recompute the v6e statistics from the exported raw samples, use an environment
with NumPy and run:

```sh
python replay.py --output /tmp/strassen-checkpoint-replay.json
```

This is CPU-only, makes no network/accelerator calls, and takes several minutes
with several gigabytes of temporary disk space. It checks all scientific report
fields, paired intervals, the full candidate ledger and the frozen policy.
Temporary path-dependent receipt hashes are the only ignored report fields.
The original sources, report bytes and scientific data are preserved unchanged.
`MANIFEST.json` establishes export completeness; it explicitly does not claim
that the full v6e experiment has finished.

The parent repository's existing files are unchanged by this additive export.
