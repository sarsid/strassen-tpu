# TPU matrix multiplication benchmarks — final v6e results

The **168-shape v6e study is complete**, with separate FP32 and BF16 output
comparisons. The owned TPU was released, the supervisor finished, and the
recovery monitor was deleted. No subsequent experiments are queued.

| Study | Saved measurements | Status |
|---|---|---|
| v6e joint tuner, FP32 and BF16 output | 168/168 geometries; 336 geometry/output comparisons | Complete |
| Historical v5e study, FP32 output | 168/168 geometries | Complete; separate older protocol |
| Proposed v5e historical tuner + BF16 output | Code, configuration and protocol | Deferred; no new measurements |

## Read the results

- [v6e speed and numerical-error summary](results/v6e/report/SUMMARY.md)
- [All method timings, selected tiles and decisions](results/v6e/report/RESULTS.md)
- [Tuner design choices](results/v6e/report/TUNER_DESIGN.md)
- [Saved research recommendation table](results/v6e/report/tuner_policy.json)
- [Historical v5e results](results/v5e/168_shapes_fp32_20260921_v001/README.md)
- [Machine-readable completion state](checkpoint.json)

For each output dtype separately, the v6e study compares default Native,
tuned Native, independently tuned cubic, Strassen one level and Strassen two
levels. Inputs and recursive pre-adds are BF16; accumulation and reconstruction
are FP32. Final output conversion and storage are included in time and error.
Historical depth-three/four modules are retained only as source provenance.

The v6e compiler is pinned to JAX/jaxlib 0.11.2 and libtpu 0.0.48. The tuner
searches tile geometry, K-panel length (including full contraction), accumulator
strategy and one/two buffers with v6e-specific memory allowances. Native and
cubic are tuned independently. Historical v5e uses its original kernels,
search menu and JAX/jaxlib 0.7.2 / libtpu 0.0.21.1 stack. These studies are not
a matched-protocol causal comparison of the architectures.

Screening choices are frozen before three fresh Gaussian inputs × 30 paired
timing rounds. The report retains samples, failures, numerical metrics,
reference scope, tile and compiler settings, and pointwise paired 95% intervals.
It does not reselect winners from confirmation. There is no multiple-comparison
adjustment. Shapes and input distributions are development data.

Numerical errors compare exact BF16 operands against a full or sampled all-K
FP64 reference; finiteness is checked over the full output. These matrix
benchmarks do not measure LLM prediction accuracy. The saved recommendation
table is a research artifact, not an installed general-purpose selector.

## Source and evidence

`code/` contains committed research source at the revision in `checkpoint.json`:
kernels, tuning drivers, architecture configurations, report tools, runtime
control, tests and research documentation. Historical source notes may reference
local runs outside this export; this README defines the current included scope.

`results/v6e/phases/` contains one canonical artifact bundle for each completed
screen/confirmation phase and saved qualification phase. `phase-index.json`
records SHA-256 checksums for each bundle and every uncompressed member.
Every original file in each canonical `artifacts/` subtree is included, including
compiled HLO/cost metadata, seeds, timings, failures and numerical errors.
Byte-identical remote duplicates are omitted; each cohort's exact frozen source
archive is retained once in `results/v6e/provenance/`. Whole comparisons share
one device identity. Retry timings are never pooled. Recovery evidence remains
separate, with explicit provenance for the recovered confirmation phase.

`results/v6e/report/results.json.gz` and `candidate_decisions.json.gz` are
lossless gzip encodings of the full report and candidate ledger:

```python
import gzip, json
with gzip.open('results/v6e/report/results.json.gz', 'rt') as f:
    report = json.load(f)
```

`provenance/` retains lifecycle history, release/completion evidence and the
append-only live log. Original local paths and runtime identities are provenance,
not portable execution settings. Credentials, session-token stores, virtual
environments and model checkpoints are excluded.

## Verify and reproduce

From `benchmarks/`, use Python 3.11+ for integrity verification:

```sh
python verify.py
```

To recompute the report from raw samples, install NumPy and run:

```sh
python replay.py --output /tmp/strassen-final-replay.json
```

Replay uses CPU only and makes no network or accelerator calls. It takes several
minutes and temporary disk space. It checks all scientific report fields and
paired intervals, the full candidate ledger and the frozen policy. Only
temporary path-dependent receipt hashes are excluded from the comparison.
Original report bytes and scientific data remain unchanged. `MANIFEST.json`
records export completeness; `results/v6e/COMPLETE.json` records study completion
and the original uncompressed results checksum.

The previous 113-shape publication remains in Git history. Existing parent
repository content outside `benchmarks/` is unchanged by this update.
