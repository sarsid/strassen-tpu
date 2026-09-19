# Strassen MM Focus — first completed v5e study

Completed on 2026-09-19. This is a new, independent Git repository; the historical project was preserved.

**N1–N4 are complete on one v5e allocation. The evidence audit passed.**

[Read the experiment summary](runs/20260919T052240Z-campaign-evidence-audit-v001-4e0aac/artifacts/summary.md) · [Detailed JSON evidence](runs/20260919T052240Z-campaign-evidence-audit-v001-4e0aac/artifacts/summary.json) · [Live local status](http://127.0.0.1:8765)

| Experiment | Coverage | Recorded outcome | Archive commit |
|---|---|---|---|
| N1: basic implementations | 44 square, rectangular, partial-tile and LLM-derived shapes; four algorithms | 352/352 arm/scope records passed | `51d955e` |
| N2: tile screening | 12 shapes, six tile choices, three custom algorithms | 372 passed; 60 configured-memory-limit failures retained | `5615ad7` |
| N3: optimization comparisons | Six shapes, two algorithms, four variants at a fixed tile | 96/96 arm/scope records passed | `ad65aaf` |
| N4: synthetic numerical stress | Eight shapes, five input distributions, three seeds, four algorithms | 456 passed; 24 numerical failures retained | `15dcfdb` |
| N4: actual-weight supplement | Two verified Qwen weight tensors, two token counts, three seeds, four algorithms | 48/48 accuracy cases passed | `dac9e27` |

N1–N3 record full-call and prepared-kernel scopes separately, so their row counts include both. N4 has no timing measurements.

## What these experiments tell us

- At N1's fixed 256×256×256 tile, Strassen has no clear timing wins over full-tile cubic or native XLA; it beats the matched eight-product quadrant-cubic control on 27 of 44 shapes. Baseline choice matters.
- N2 finds different promising tiles across shapes. These are screening results and require fresh confirmation; they are not final best-performance claims. The largest tile exceeds the fixed 48 MiB compiler VMEM budget in many cases.
- N3 shows that some changes improve both cubic and Strassen. The report separates each change and their combination.
- All 24 N4 cancellation failures belong to basic Strassen. Its median relative L2 error there is about 0.527; the three cubic/native controls pass. Strassen passes the other tested distributions and the actual-weight supplement under the frozen numerical gate.

Actual-weight testing uses checkpoint BF16 weights and **synthetic Gaussian activations**. It is not an LLM inference or model-quality evaluation. N5's independently tuned comparison, N6–N7, v6e replication and two-level Strassen have not been run.

## Where everything lives

- `src/strassen_mm/`: versioned kernels and runners, with no imports from the historical project.
- `configs/` and `protocols/`: frozen shapes, seeds, tile budgets, precision and numerical gates.
- `runs/`: unique execution directories containing source snapshots, manifests, raw results, failures, logs and completion records.
- `data/qwen3-0.6b-real-v001/`: verified public tensor bytes, fixed checkpoint revision, hashes and license.
- `runtime/` and `tools/`: allocation controls, sequential execution, archival commits and evidence audit.
- `status/`: local status panel with agent states and measured results. Start it with `python3 status/server_v003.py` if needed.

The old flowchart in `docs/experiment-flowchart-v2.*` is retained as the proposal, not as a current execution-status document.

## Verification and preservation

Seventeen local tests and 24 hardware-smoke records passed. The final audit checks result counts, planned cases, raw timing summaries, numerical gates, artifact hashes and matching allocation/host/boot/device/software identity across all five study runs. Its archive commit is `8caad43`.

Every completed execution has a source snapshot and an archival commit. Executed source revisions and sealed results are preserved; later implementations have new versioned filenames. Git commits are local; nothing was pushed to a remote repository.

See [the initial archive note](docs/ARCHIVE_NOTES_v001.md) for the first local test's ignored-bytecode manifest limitation, and [evidence interpretation notes](docs/EVIDENCE_NOTES_v001.md) for numerical-only result fields.

At this handoff, the v5e session is retained pending the keep/release choice; no experiment is running. Keeping the session preserves the opportunity for continuity, but Colab may reclaim it. The side tab is served locally at the status link above.
