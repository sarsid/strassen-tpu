# Retained-TPU recovery through an uploaded bootstrap

A replacement v6e was allocated at 18:25 UTC on October 20. The first recovery restored saved files, then failed while trying to hash a laptop-side `__file__` path on the remote runtime. No model evaluation started in that attempt. Its log and source remain preserved.

The correction uploads the exact existing bootstrap to a real remote file, verifies its SHA256, and dispatches it with `runpy.run_path`. Two tests reproduce notebook execution with a nonexistent host filename and verify path/hash rejection. Source commit: 383595722; validation commit: 5e2e4e3ea.

Run v010 uses the retained v009 session and TPU. No second allocation or release was requested. The scientific source archive, seed, model code, precision, environment pins, admission gates, examples, scoring and analysis are unchanged. The original partial v009 directory is preserved. Exact bundle/seed contents are already archived in the sibling `recovery_20261020_query_checkpoints_v001` snapshot; their unchanged hashes are recorded here.

The repaired foreground launch was acknowledged, pinned environment installation passed, and Qwen3-32B checkpoint loading for training validation was progressing at the saved observation. Five task evaluations and sixteen Qwen3-32B GSM8K pairs remain saved. This snapshot contains no new downstream test result. Fresh admission and exact execution-contract matching remain required before resuming the unfinished examples.

The local dashboard labels the preserved task count explicitly during preflight, whose worker context otherwise temporarily resets the display. This presentation change does not alter scientific results. The read-only observer continues checkpointing; errors retain the TPU, and release requires all nine evaluations and all three locally hash-verified model archives.
