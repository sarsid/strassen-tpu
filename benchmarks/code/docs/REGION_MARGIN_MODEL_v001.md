# Direct margin model for a separate region cohort

`tools/fit_region_margin_v001.py` learns the relative runtime advantage directly:

\[
g_c(M,N,K) = \log\left(T_{\mathrm{native\ default}}/T_c\right).
\]

There is one regularized linear model per declared candidate, including native
compiler presets, cubic tiles and Strassen tiles. Positive margin means the
candidate is faster than compiler-default native. This is a new experiment;
neither the historical power-grid observations nor the existing formula are
changed or pooled into its training data.

## Input and provenance

The tool accepts the new campaign configuration, its sampled-shape manifest, one
completed `GRID-screen` journal, and that run's `environment.json`. Both manifest
and configuration must declare the requested `cohort_id`. The journal must have
exactly one `run_start`, matching allocation identity and matching source-snapshot
hashes for the supplied configuration and manifest. Concatenated journals, mixed
phases, conflicting cohort/allocation IDs and legacy artifacts lacking the new
cohort declaration are rejected. Output directories and JSON files are created
exclusively; use a new output directory for every analysis.

Every requested geometry must have exactly one eligible complete-call result for
every declared candidate, with matching algorithm, tile, compiler options, input
seed, Gaussian distribution and repeat count. A failed, duplicate or missing
candidate stops the analysis with an explicit error. Candidates are not silently
dropped, and a partial catalog is not presented as a complete oracle.

Existing screen batches do **not** pair every custom candidate with the same
native reference. Consequently, training uses **unpaired ratios of arithmetic
screen means**. These ratios are exploratory point estimates; this tool does not
invent paired confidence intervals or treat screening wins as confirmation.

The serialized model and evaluation preserve cohort ID, allocation ID, full
environment identity and hashes of all four inputs. Another machine requires a
separate cohort and analysis. Repeated anchor geometries can later support a
comparison of cohorts, with the allocation labels preserved; they do not authorize
pooling latency measurements across machines.

## Features, baseline and validation

Thirteen features use only M,N,K and each candidate's declared tile: logarithmic
volume; two log aspect ratios and their squares; axis spread; volume hinges at
2^36 and 2^39 elements; logical output-tile count; sequential K-panel count;
padding volume ratio; and hinges for fewer than eight output tiles or four
K panels. Native tiles remain compiler-managed and receive no invented tile
counts. Features describe potential geometry effects; the coefficients are
empirical, not a hardware proof.

The default ridge penalty is 1.0 on standardized features; the intercept is
unpenalized. Standardization, model fitting and fixed-native-preset selection
use training geometries only. In each validation fold, the fixed native preset is
the one with the best geometric-mean speed relative to default native on that
fold's training shapes. This is the practical native-only baseline, selected
without looking at the evaluation shapes.

Five geometry-block folds are the default. Dimensions are rounded to the nearest
power-of-two exponent, then the three exponents are sorted into a block key.
Thus nearby 8191/8193 boundary shapes and axis permutations stay in the same fold.
Blocks are assigned deterministically using geometry and block size, never
runtime labels. This reduces optimistic local interpolation; it does not prove
out-of-domain generalization. There is no search over hyperparameters inside this
tool. Changing its defaults after examining evaluation results is exploratory
model development and still needs a separately frozen holdout test.

The policy selects the highest predicted Strassen margin only when its predicted
speedup over the fold-frozen native preset exceeds 3%; otherwise it dispatches
that native preset. The 3% threshold is a decision rule, **not a confidence
bound**. Cubic models participate in the measured finite-catalog oracle comparison;
the dispatch policy specifically compares Strassen against native.

The report includes geometric-mean speedup over the fixed native policy, regret
against the observed screen catalog, worst slowdown, Strassen selection mistakes
and predicted Strassen tile shortlists. The best measured result inside a
predicted shortlist is labeled a **retrospective shortlist oracle**, which would
require additional tuning. It must not be reported as the speed of the dispatched
policy. Broad, focused and repeated-anchor metrics remain separate. The aggregate
describes the chosen design mix, not a uniform-grid win fraction; even the broad
screen win count remains exploratory until an independent confirmation analysis.

## Run after collecting a new screen

From the `Strassen_MM_Focus` repository root, using a Python environment with NumPy:

```sh
PYTHONPATH=src python tools/fit_region_margin_v001.py \
  --manifest configs/generated_region_grid_v001/sampled_shapes.json \
  --config configs/generated_region_grid_v001/campaign_region_grid_v001.json \
  --screen NEW_SCREEN_ARTIFACTS/results.jsonl \
  --environment NEW_SCREEN_ARTIFACTS/environment.json \
  --cohort-id region_grid_v001_20260920 \
  --allocation-id ACTUAL_NEW_ALLOCATION_ID \
  --out-dir NEW_COHORT_ANALYSIS/margin_v001
```

This creates `margin_model.json` and `margin_evaluation.json`. The model can be
loaded into Python and passed to `select_shape(model, m, n, k)` for prospective
inference without timing those shapes. The result includes actual tile tuples,
the native fallback, memory preflight, distance to training data and cohort
identity. Predictions on a new allocation remain unvalidated.

Optional `--confirm PATH/results.jsonl --confirmation-environment
PATH/environment.json` evaluates the frozen out-of-fold actions where the
selected action and fold-native baseline both occur in the **same timed group**
with matching seed, distribution and repeat count. Confirmation environment
identity must match the screen. Different groups are never combined into a
comparison. Missing coverage remains explicit, and identical native fallback
actions have ratio 1 by definition. No confirmation result enters fitting or
native selection. This selective coverage is not a full-domain performance test.

All new holdout IDs, original protected holdout IDs, and protected geometries
under renamed IDs are rejected. This version does not unlock any holdout or
consume the twelve original reserved shapes. A later independent holdout test
must use the frozen model rather than refitting this screen tool on holdouts.

## CPU validation

```sh
PYTHONPATH=src .venv-plots-v001/bin/python -m unittest discover \
  -s tests -p test_region_margin_v001.py -v
```

Tests use synthetic observations and cover allocation/cohort rejection, source
hashes, failures and duplicate data, protected geometry aliases, boundary blocks,
training-only normalization/native selection, native fallback, honest shortlist
metrics, same-group confirmation coverage and exclusive output creation. They do
not run a TPU benchmark or produce new scientific timing evidence.
