# v6e final results: 168/168 shapes

Start with [the summary](report/SUMMARY.md), [timings and decisions](report/RESULTS.md), and [tuner design](report/TUNER_DESIGN.md). Both output contracts have 168 completed geometries.

The final report is complete, the TPU was released, and no experiments are queued. See [checkpoint.json](../../checkpoint.json). `ORIGINAL_README.md` is historical.

`phases/` retains canonical raw evidence with member hashes in `phase-index.json`. `report/results.json.gz` and `report/candidate_decisions.json.gz` decompress to the original JSON bytes. Recovery evidence is excluded from statistics unless its whole comparison was verified. Only one copy of each phase's artifacts is kept; cohort source archives remain in `provenance/`.
