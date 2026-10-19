# Live admission and host sleep interruption

The third runtime started at 06:56:15 UTC. Initial keepalive requests received
HTTP 200, with no sign-in redirect and an empty body from the intended endpoint.
Fresh train-only admission passed at 07:05:58 UTC; Native cached/full-prefix
relative error was zero. Saved MMLU and perplexity tasks were reused. The raw
log snapshot reaches GSM8K generation step 120; 3 of 8 sequences have finished
per arm, but the batch has not finished and no task-accuracy result is claimed.

Host maintenance sleep from 03:08:59 to 03:15:46 local time caused a gap in
keepalive traffic. Temporary AC-only system-sleep prevention was added, tied
to the existing heartbeat process; no persistent power setting was changed.
The earlier allocation losses do not coincide with recorded host sleep, so
those failures remain unexplained. The new runtime was still active beyond
24 minutes at the snapshot check. This does not establish long-term stability.

No scientific code, input data, algorithm settings or analysis changed.
