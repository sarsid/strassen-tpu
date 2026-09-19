# Explained N1-N9 chartbook

The [revised PDF](../runs/20260919T150447Z-N1-N9-explained-chartbook-v002-65ebef/artifacts/N1_N9_explained_comparisons.pdf)
has 49 pages: a completed-study flowchart, a reading guide, and all 47 original
overview comparisons with an observation and interpretation beneath every graph.
The original PDF and chart gallery remain available and unchanged.

Short black horizontal segments mark the lowest eligible measured mean within
each timing group, with a diamond on the corresponding bar. These are descriptive
markers, not declarations of statistical significance. In N4 the markers refer
to median numerical error; the red acceptance thresholds remain separate.
N9 model-quality panels retain tolerance gates without treating Native's
self-comparison zero as a performance win.

Interpretations compare Strassen with Native and Cubic where those controls
exist. They distinguish N7's independently tuned comparison alternatives from
its unchanged selection rule, N8 selection measurements from fresh confirmation,
and resident model layers from streamed full forwards. The flowchart records
completed v5e work; v6e and two-level Strassen remain future work.

## Evidence and validation

- Source chartbook: `20260919T144414Z-N1-N9-chartbook-v003-008b6c`.
- Explained chartbook: `20260919T150447Z-N1-N9-explained-chartbook-v002-65ebef`.
- Audit: `20260919T150527Z-N1-N9-explained-validation-v001-e1e4e1`.
- All 47 graphs retain exactly the original plotted points, categories, series,
  source references, timing scopes and cohort labels. No TPU experiment ran.
- All 215 output files pass their recorded hash/size checks. The final PDF has
  49 pages, all experiment labels in the flowchart, and both explanation fields
  on all 47 graph pages. Every PDF page was rasterized for visual inspection.
- Parallel reviewers checked N1-N9 claims against the final reports. All contact
  sheets and representative full pages were visually checked; no unresolved
  clipping, overlap or material narrative error was found.

The first assembly attempt, `20260919T150242Z-N1-N9-explained-chartbook-v001-efde07`,
failed because ReportLab's HexColor received the color name `white` instead of a
hexadecimal value. Its code and partial output are preserved. Version 002 fixes
that display issue and clarifies the N7 comparator label, without changing data.

Executed source versions are recorded in `protocols/code_freezes_v001.jsonl`.
The final PDF keeps vector plots and flowchart elements, searchable text,
page numbers, and experiment bookmarks.
