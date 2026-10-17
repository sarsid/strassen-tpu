# B64 retry launched on the retained v6e

The predecessor finished at 01:55 UTC October 17, with B1/B4/B8/B16/B32 passed
and B64 confirmation failed in untimed diagnostic memory allocation. Its full
archive was downloaded and hash-verified; the owned TPU was retained.

At 02:04 UTC the prepared B64-only retry started on the same endpoint and
session, reusing the environment, checkpoint cache and exact saved tuning.
The remote launcher verified predecessor completion/profile identity, and the
worker acquired the shared exclusive lock. It verified the saved FP32 formula
evidence and began repeating numerical block qualification. No completed B64
performance or quality result is claimed by this launch record.

The executed remote bundle is unchanged from
[the previously validated repair](../b64_metric_memory_repair_v001/), source SHA
`7665f5adbfa6ffbbbeca7b558a3190a2bbae72453bf00dc132fb27d0886884d1`.
The separately validated local observer refreshes the live dashboard, downloads
verified checkpoints and preserves all results. It cannot allocate, launch
remote computation or release the TPU. Three local observer checks passed.
The original supervisor and keepalive remain active; final release requires
all six batches to pass and their evidence to be preserved.
