# Foreground execution recovery

Three detached-worker allocations disappeared approximately 21 minutes after
their last notebook execution, despite acknowledged keepalive traffic on the
third. Internal provider termination cause remains unknown. The observations
suggest notebook execution lifecycle matters.

The new launcher executes the actual original scientific worker synchronously
inside the notebook cell. It does not generate fake activity. The host client
waits for that real workload; an independent contents-API observer continues
saving checkpoints. Eight local restoration/foreground-execution checks passed.
A single replacement request is queued at this snapshot, not yet verified as
executing remotely.

The scientific bundle, checkpoint seed, algorithms, settings, input subsets and
registered analysis remain unchanged. Two complete Qwen14 tasks are preserved;
no completed GSM8K batch result is available from the lost allocations.
