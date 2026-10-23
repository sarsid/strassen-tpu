# Continue this project in Codex Cloud

Repository: `sarsid/strassen-tpu`.
Branch: `codex/benchmarks-20260929`.

Read the [October 5 handoff](benchmarks/llm/20261005_v001/README.md) for current
code, all 80 completed BF16 comparisons, the timing-scope report, source/evidence
hashes, paused state, limitations and CPU-only verification commands.

In Codex Cloud, create/select an environment for this repository and ensure
that it uses this branch. Ask setup to read that handoff and install only the
analysis dependencies first. Review its verification results and publish the
environment, then start a task with:

> Continue the Faster Strassen TPU project using
> `benchmarks/llm/20261005_v001/README.md`. Read the pause and methodology notes,
> verify the saved evidence and rebuild the offline timing-scope report.
> Summarize the 80/200 experiment state and discuss next steps with me. Keep
> all TPU allocation, checkpoint downloads, queues and monitors paused until
> I explicitly ask to resume. Maintain a live log and preserve all results.

The local conversation, Colab authentication, localhost dashboard, and process
state do not form part of this Git snapshot. Remote TPU access must be
established and tested separately before any authorized resumption. The original
laptop files remain intact.

See [official Codex Cloud setup](https://learn.chatgpt.com/docs/cloud) and
[environment configuration](https://learn.chatgpt.com/docs/environments/cloud-environments).
