# v6e LLM experiments

[LLM Benchmark 1 (Qwen3-32B)](llm_benchmark_1_qwen3_32b/README.md): **completed October 16, 2026**. Six batches (1, 4, 8, 16, 32, 64), sequence 1,024, four algorithms: Native, Tuned Native, Tuned Cubic, and Tuned Strassen S1. Includes resident and transfer-inclusive timings, all numerical/predictive errors, frozen tuning profiles, raw evidence, exact sources, failure history and offline verification. B64 uses a separately labeled newer compiler stack. TPU released; no next model queued.

[October 5 Codex Cloud handoff](20261005_v001/README.md): **historical 80/200 BF16 checkpoint; that campaign remains paused.** Includes its compute-scope report, completed groups, reproducible offline analysis and historical continuation instructions. This older five-arm campaign is separate from LLM Benchmark 1.

[October 5 validation](validation_20261005_v001/validation.json) passed: snapshot/evidence integrity, targeted credential scan, and exact reconstruction of the saved numerical report from its raw inputs. This validates the portable package locally; a cloud environment and Colab connectivity still need account-side setup.

[October 4 code and results snapshot](20261004_v001/README.md): historical 75/200 checkpoint, taken while the campaign was running. Exact source bundles, timings, tuning choices, numerical errors and local Gemma4 qualification evidence are included.

[Publication validation](validation_20261004_v001/integrity.json) passed; all evidence hashes and restoration of a complete Qwen32 workload were checked. The targeted credential scan found no matches.
