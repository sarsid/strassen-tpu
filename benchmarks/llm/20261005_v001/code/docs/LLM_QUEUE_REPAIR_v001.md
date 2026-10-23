# October 3 queue launch repair

The original Qwen3-8B B1/S512 workload completed all ten comparisons. After that,
the new queue's launcher mistakenly dispatched the old v001 remote worker. It
referenced an input script absent from the frozen bundle; for Gemma it also failed
because credential staging had already created its private directory. No model
measurements ran in those failed attempts. The supervisor treated a common code
error as independent workload failures and repeatedly allocated new runtimes.
The offline tests checked individual components but missed the detached launch
path. This was an orchestration implementation and validation failure.

The queue was parked before repair. Successful results and every failure are
retained. Unfinished jobs are requeued with a fresh retry budget for the repaired
source revision; old attempts remain in their history. No precision, tuning or
quality threshold changes are made. No completed comparison is repeated.

Repair: v003 remote worker launches itself, uses the v002 input and measurement
entrypoints, reads the intended job, and tolerates the pre-created private Gemma
directory. The controller validates the actual archived scripts and CLI options
before allocating a TPU. Tests exercise the real detached launch and complete
worker orchestration with external processes replaced, including a non-default
model, a precision-only resume, and Gemma's existing private directory.

Shared setup/import/CLI/qualification errors stop the whole queue after releasing
the owned TPU. Two repeated unclassified setup failures also stop the sweep.
Per-workload retries are reserved for workload failures; they cannot consume all
jobs on a known shared setup defect. Paused and shared-error states are honored
by both supervisor and watchdog. Recovery evidence must confirm the new run has
passed setup and is performing actual model work before declaring it resumed.
