# Final publication verification

All checks passed on October 2, 2026. Validation used CPU only.

- 168 complete screen/confirmation pairs and 336 separate output comparisons.
- 1,423 original export files and 195,645 canonical v6e archive members verified.
- Rebuilt scientific fields and paired confidence intervals exactly match the final report.
- Candidate ledger and frozen tuning policy are byte-identical after replay.
- Targeted credential-pattern scan found no matches across 207,498 payloads, including nested archives (6,716,374,223 uncompressed bytes). This is not a general proof that arbitrary data is secret-free.

Replay used NumPy 2.3.5. Run `python replay.py` from the benchmarks root to reproduce it.

`checked-content-MANIFEST.json` is the exact export manifest checked by this validation. `validation.json` records its SHA-256. The current root manifest additionally includes these receipts and this note; all scientific files are unchanged. Source paths and commits in the validation manifest identify the archived local validation execution.
