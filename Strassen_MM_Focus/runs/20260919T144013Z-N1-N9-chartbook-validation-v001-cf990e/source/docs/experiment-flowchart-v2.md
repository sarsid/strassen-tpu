# Proposed experiment flowchart

Status: proposed experiments, not findings. No new experiments have been launched.

The matrix multiplication study is N1–N7, including the new N6a. N8 studies
matrix multiplication with subsequent operations. N9 tests LLM applications.
The original numbers are retained to keep earlier references meaningful.

The shared shape suite includes squares, rectangles with independently varied
M/K/N, actual LLM projection dimensions at varied token counts, and dimensions
requiring partial tiles or padding.

```mermaid
flowchart TD
    N1["N1. What happens with basic implementations?<br/>Plain cubic, one-level Strassen, native XLA.<br/>Check correctness and timing across the shape suite."]
    N1 --> N2["N2. How much does tile choice matter?<br/>Sweep matching tiles across shapes.<br/>Identify promising regions and poor fits."]
    N2 --> N3["N3. Which MM optimizations actually help?<br/>Change one thing at a time, then combine.<br/>Separate individual gains from interactions."]
    N2 --> N4["N4. Where does numerical accuracy deteriorate?<br/>Test random, difficult and real matrix values.<br/>Identify accuracy limits and necessary fallbacks."]
    N3 --> N5["N5. Who wins after fair tuning and preparation?<br/>Retune Strassen and strong full-tile cubic fairly; compare with native XLA.<br/>Time both kernel-only and complete MM calls, including padding and layout changes.<br/>Separate per-call work from reusable preparation."]
    N4 --> N5
    N5 --> N6["N6. Why do some shapes win and others lose?<br/>Profile representative wins, ties and losses.<br/>Explain compute, memory and scheduling costs."]
    N6 --> N6a["N6a. Can we predict when Strassen will help?<br/>Build a rule that selects cubic or Strassen and optionally tiles.<br/>Freeze the rule before testing on new shapes."]
    N6a --> N7["N7. Does the frozen rule generalize?<br/>Test shapes not used to build or tune the rule.<br/>Compare its choices with measured alternatives.<br/>Repeat in fresh v5e/v6e runs; report wins and losses."]
    N7 --> N8["N8. Does subsequent computation add a gain?<br/>Compare MM plus activation or residual work.<br/>Rerun fusion and early-finalization experiments."]
    N8 --> N9["N9. Do these gains help actual LLMs?<br/>Qwen plus Mistral and Gemma.<br/>Measure layer/model performance and model quality."]
```

## Timing and prediction details

- N5 retains independent tuning under comparable budgets and fresh confirmation
  measurements. Apply the same requested precision and stated accuracy gates.
- Report kernel latency and the complete device-side MM call separately. Include
  method-specific padding, packing/layout changes and output conversion in the
  complete call wherever required. These are MM timings, not end-to-end inference.
- Report preparation that is needed for each call separately from preparation
  that can be reused with a fixed operand. State the reuse assumptions and show
  amortized costs when claiming a benefit from reuse.
- Reserve the N7 evaluation shapes before constructing the N6a rule. Do not use
  their timings to tune the rule or a predicted tile configuration. After freezing
  its choices, independently benchmark alternatives on those shapes to evaluate
  the selection quality and resulting latency. Predeclare any device-specific
  calibration and distinguish it from transfer without retuning.

## Optional later study

Leave two-level Strassen as a future note, outside the current experiment sequence.
If pursued later, compare zero levels (cubic), one level, and two levels on a small
selection of shapes. This note does not authorize implementing or running that
extension now.
