# Accuracy study monitoring recovery — October 19 UTC

The prior worker launch was acknowledged, but the local observer exited immediately because its launcher passed the wrong endpoint option. No downstream results were downloaded. Colab later listed no allocation and returned HTTP 404 for the old owned endpoint. The available evidence does not establish the cause of runtime disappearance or remote evaluation progress. **Zero downstream evaluations from that attempt are verified.**

This additive snapshot preserves that diagnosis and the repaired host orchestration. The remote scientific source, model checkpoints, subsets, algorithms, numerical precision and analysis are unchanged. The repaired starter requires the observer to stay alive, acknowledge the matching PID/endpoint and successfully read remote study status. It then supervises and restarts the observer without duplicating the remote worker.

Four local checks passed: actual launcher-to-observer CLI invocation, matching live-process readiness, complete three-model evaluation driver flow, and resumption from saved paired batches. Driver tests use explicitly fake outputs in temporary directories; they are software validation, **not model accuracy results**. The frozen test sources and validation logs are included.

The recovery is runs/20261019-accuracy-study-v002 in the experiment checkout. It awaits a fresh owned allocation and will save real paired results separately. Earlier published sources and results remain untouched.
