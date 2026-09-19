# N9 input preparation: diagnosed failures and preserved repairs

Two preparation problems occurred before the TPU model-quality experiment. Neither is a Strassen numerical failure. Every failed attempt remains archived, and all replacement code and environments use new versioned names.

## Isolated environment creation

The first model-tools installer failed while `venv` invoked `ensurepip`: run `20260919T065209Z-N9-model-tools-v5e-v004-52799b`. Its partial `model-tools-v001` directory was preserved. The v002 installer creates a fresh environment with `venv --without-pip` and uses the existing global pip's explicit `--python` target to install into it. It does not upgrade global pip or the TPU's JAX packages. Process-group cleanup and environment-isolation checks were added and locally validated; an initial macOS test-fixture path problem and its new test version are retained in [local validation incidents](LOCAL_VALIDATION_INCIDENTS_v001.md).

The replacement completed in `20260919T074551Z-N9-model-tools-v002-v5e-v004-34afbd`, archive commit `c2c4c8e`. Qwen tokenization and its independent official CPU reference then completed:

| Step | Sealed run |
|---|---|
| Qwen tokens | `20260919T074649Z-N9-tokenize-qwen-v002-v5e-v004-1fdbd7` |
| Qwen official reference | `20260919T074805Z-N9-reference-qwen-v002-v5e-v004-6928ce` |

## Mistral tokenizer dependency

Mistral tokenization failed with `ImportError` in `20260919T074921Z-N9-tokenize-mistral-v002-v5e-v004-ab489e`. The original helper retained the exception type but not its detailed traceback. The workflow stopped with a sealed partial input bundle; Qwen's successful artifacts were preserved.

A new read-only diagnostic loaded only the exact cached Mistral tokenizer, checked its small-file hashes and recorded the full sanitized traceback. Run `20260919T075743Z-N9-tokenizer-diagnostic-v5e-v004-3be96d` showed that the official `LlamaTokenizerFast` path, using `legacy=False`, invokes SentencePiece's generated protobuf module. That import failed because `google.protobuf` was absent. The diagnostic read no checkpoint weight shards and changed no installed distribution. Diagnostic completion means that the error was captured; its tokenizer-loaded outcome was false.

Decision D018 authorized a fresh `model-tools-v003` environment with only `protobuf==6.32.1` added. This release has Python 3.13 support and compatible wheels in its [official PyPI record](https://pypi.org/project/protobuf/6.32.1/). No other optional tokenizer package was added. All earlier explicit pins were retained, and a full inventory comparison rejects changed or removed packages or any additional package beyond that exact protobuf version.

| Repair evidence | Run | Outcome |
|---|---|---|
| Offline repair checks | `20260919T080144Z-model-tools-protobuf-tests-v001-e0c45a` | Three tests passed; commit `5f536d7` |
| Fresh isolated installation | `20260919T080202Z-N9-model-tools-v003-v5e-v004-ee21ab` | Only protobuf 6.32.1 added; global core unchanged; commit `4ecc631` |
| Repaired tokenizer diagnostic | `20260919T080327Z-N9-tokenizer-diagnostic-v003-v5e-v004-0775cd` | `LlamaTokenizerFast` loaded; no exception; cleanup proven; commit `fd73239` |
| Mistral corpus preparation | `20260919T080427Z-N9-tokenize-mistral-v003-v5e-v004-1b9bce` | Completed; commit `7d24ad8` |

The repaired Mistral token array has shape 32×1024, contains the first 32,768 tokens of the registered corpus, and supplies 32,736 next-token scoring positions. Its SHA256 is `6c07f71c6dc007516db6a8554a03778997773cedaa732e00af8d3e78bf657d33`. The full corpus yielded 330,303 tokens. The official tokenizer was used with `add_special_tokens=False`, as originally registered.

A read-only process observation (`20260919T080847Z-N9-input-worker-observation-v001-349fbf`) found no active preparation worker because tokenization had already finished. It is operational evidence, not a performance measurement.

## Preserved scientific contract

Qwen's existing token and reference artifacts are reused unchanged. Mistral retains the same official checkpoint revision, corpus revision, tokenizer settings, token positions and reference implementation. The CPU reference uses the first 64 fixed tokens, official Transformers 4.56.2, BF16, eager attention, evaluation mode and no KV cache. The isolated tools have no JAX installation, while every preparation adapter verifies the original allocation, host, boot and global core package versions.

The repairs change neither N8-selected application policies nor the fixed native-qualification and candidate-quality thresholds. They provide no TPU model-quality or speedup result by themselves. Gemma's original official-access denial remains a separate explicit limitation. Model-quality and performance conclusions must come from the subsequent sealed N9 benchmark and its independent audit.
