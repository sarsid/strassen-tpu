# Chartbook validation

Delivered rendering: `20260919T143842Z-N1-N9-chartbook-v002-471048`.
Successful automated validation: `20260919T144117Z-N1-N9-chartbook-validation-v002-4f43ea`.

The audit verified all 580 sealed output files against both the plot manifest
and the archive's outer seal. All 144 chart data files match the gallery manifest;
all 7,305 plotted points have source-journal references. Coverage accounts for
all 4,751 case-result rows from the 15 canonical experiment executions. There
are 215 absolute/relative image variants, each provided as PNG and SVG.

The PDF has 47 overview pages in N1-N9 order. Every page was rasterized with
Poppler and reviewed on contact sheets. Full-size representative plots were
also inspected, including N1, N2, N3, N4 cancellation, N5, N8 contrasts and
N5 anchors, and N9 resident timing/qualification. No clipping or overlapping
titles, legends or captions was observed in these checks. Scientific PDF/SVG
exports retain vector text and graphics.

The gallery's schema, navigation/filter code, relative-view fallback, 430
image links, 144 data links and PDF/provenance links passed independent static
review. The served gallery returned HTTP 200 at `http://127.0.0.1:8766/`.
Browser interaction itself was not automated in this validation.

The first validation execution
`20260919T144013Z-N1-N9-chartbook-validation-v001-cf990e` failed because optional
`pdftotext` was absent after its structure/hash checks and `pdfinfo` succeeded.
Its code and output are preserved. Version 002 removes that optional extraction
step and successfully performs all structure checks and full PDF rasterization.
No benchmark or chart values changed during this repair.
