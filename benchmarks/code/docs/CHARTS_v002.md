# N1-N9 experiment graphs

The [interactive gallery](../runs/20260919T144414Z-N1-N9-chartbook-v003-008b6c/artifacts/index.html)
contains 144 comparisons, including all measured matrix shapes. Its
[47-page PDF](../runs/20260919T144414Z-N1-N9-chartbook-v003-008b6c/artifacts/N1_N9_key_comparisons.pdf)
collects the overview figures in N1-N9 order. No new TPU measurements were run.

Start with N1 and **Overview**. Choose **Relative** when different matrix sizes
make absolute times hard to compare. Relative timing bars show time divided by
the stated reference time: below 1 is faster. Use **All details** for complete
tile searches, variant comparisons, preparation and model-layer results.

Blue means Native, orange means Cubic, and green means Strassen in algorithm
comparisons. The extra quadrant-cubic control is retained where measured.
N3 uses a separate variant legend because each plot compares optimizations
within one algorithm. Matrix dimensions are M x K x N for (M x K) times (K x N).

| Experiment | What to look at |
|---|---|
| N1 | Fixed-tile baseline bars for all 44 shapes; includes the quadrant control. |
| N2 | All tile/algorithm combinations, including memory failures; per-tile bars in Details. Native was not measured. |
| N3 | Plain, product order, output accumulator and both changes; matched Cubic/Strassen in Details. Native was not measured. |
| N4 | Numerical error by distribution and shape, all three seeds; cancellation failures remain visible. No latency measurements. |
| N5 | Fresh three-family confirmation; complete tuning screens in Details. |
| N6 | Ordinary call timings and separately profiled device durations; original representatives and post-hoc large shapes distinguished. |
| N7 | Held-out evaluation and fresh-v5e replication, with the frozen rule as a fourth bar; within-cohort speedup comparison in Details. |
| N8 | Synthetic application results; selected family comparisons, all layout/fusion variants, early finalization and preparation. |
| N9 | Real-model resident layers, streamed full forwards and quality; individual layers/windows in Details. Gemma was access-blocked. |

## Reading the evidence

- Complete-call, prepared-kernel, profiled-device, resident-layer and streamed
  forward measurements have separate charts. Machine allocations remain separate.
- Ordinary latency whiskers are one sample standard deviation, not confidence
  intervals. Relative timing views use archived paired 95% intervals when
  available; otherwise they show descriptive ratios without inferred intervals.
- Accuracy bars summarize three seeds with median, range and individual dots.
  Hatched bars failed the unchanged numerical gate.
- N5/N7 screening and N8 selection views use the selection data themselves and
  are labeled exploratory. They do not replace independent confirmation.
- Profile captures and the three streamed-forward repeats show descriptive
  ranges. Different model layers are not treated as independent replicates.
- Memory failures and unmeasured values are never converted into zero timings.
  Native's zero model-quality error is a self-comparison; its independent
  implementation qualification has a separate chart.

Every figure offers PNG, SVG and its plotted JSON data. Points carry source
journal references. `sources.json` records the run identities and hashes;
`coverage.json` accounts for all 4,751 case-result rows from 15 canonical runs,
including the explicitly disclosed blocked Gemma row.

## Reproducibility and versions

The plotting environment is isolated in `.venv-plots-v001`; its archived package
inventory is under `runs/20260919T141916Z-plot-environment-setup-v001-17a1cc/`.
The v001 rendering is preserved under
`runs/20260919T143628Z-N1-N9-chartbook-v001-0244fe/`.
Version 002 groups the PDF in experiment order and gives near-equal ratios an
extra significant digit. Delivered version 003 also displays all-OOM panels
without a numeric latency scale, shortens application labels, rotates dense
category labels and reserves room for heatmap colorbar text. Earlier renderings
are preserved. All benchmark evidence is unchanged.

To serve the delivered gallery locally:

```sh
python3 -m http.server 8766 --bind 127.0.0.1 --directory runs/20260919T144414Z-N1-N9-chartbook-v003-008b6c/artifacts
```

Then open `http://127.0.0.1:8766/`. The gallery also works directly from its
`index.html` file; all data and figures are local.
