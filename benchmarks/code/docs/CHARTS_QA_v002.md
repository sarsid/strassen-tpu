# Final chartbook validation

Delivered rendering: `20260919T144414Z-N1-N9-chartbook-v003-008b6c`.
Successful audit: `20260919T144511Z-N1-N9-chartbook-validation-v003-1c5e81`.

All 580 sealed output files match their plot and outer archive manifests.
There are 144 logical comparisons, 215 absolute/relative image variants (PNG
and SVG), 7,305 plotted points with journal references, and 47 PDF overview
pages. Coverage accounts for all 4,751 case-result rows from 15 canonical
experiment executions, including the explicit unmeasured Gemma disclosure.

All final PDF pages were rasterized and visually checked on contact sheets.
Independent agents reviewed the preceding version's complete figure contact
sheets and flagged two all-OOM N2 panels, crowded eight-shape N8 labels, and
the N8 heatmap colorbar margin. Version 003 corrects these issues; full-size
checks of the corrected figures confirm visible failure markers, no numerical
latency axis for all-failed groups, readable category labels and an unclipped
colorbar caption. Titles, legends and footnotes remain separated.

The gallery's schema, filter/navigation code, relative fallback, 430 image
links, 144 chart-data links and PDF/provenance links passed static review.
The final gallery is served on localhost port 8766. Browser interaction was
not automated; generated resource existence is checked by the audit.

The optional-pdftotext failure and successful recovery documented in
`CHARTS_QA_v001.md` remain preserved. The final audit uses the already validated
`audit_chartbook_v002.py`. No chart value, source measurement, uncertainty
interval, selection rule or timing scope changed during the display repairs.

Code and all output versions are committed locally. Entry point:
`docs/CHARTS_v002.md`, linked from the repository README.
