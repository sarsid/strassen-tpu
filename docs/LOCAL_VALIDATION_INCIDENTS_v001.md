# Local validation incidents

These failures concern local validation infrastructure or fixtures. They are separate from TPU compilation, numerical accuracy, and model-quality outcomes. Every attempt remains archived with its original source and completion status.

| Attempt | Finding | Preserved recovery |
|---|---|---|
| `20260919T064609Z-n6-recovery-tests-v001-436d05` | The selected local interpreter had no `pytest` module; the tests never started. | `20260919T064639Z-n6-recovery-unittest-v001-2d5f2d` ran the same four tests using their standard-library unittest entry point; all passed. No dependency installation or experimental-code change was needed. |
| `20260919T071044Z-model-tools-recovery-tests-v002-876d2a` | Two of five checks failed because macOS temporary paths used `/var`, while the path guards resolved that symlink to `/private/var`. The fixtures had substituted unresolved roots. | New `tests/test_model_tools_v003.py` resolves its fixture roots and checks precise rejection messages. It retains the unchanged v002 production implementations. All five checks passed in `20260919T071300Z-model-tools-recovery-tests-v003-d44a46`. |

The original v002 negative path tests could also pass at the wrong guard. The v003 assertions now require the intended rejection reason, so those passing checks are meaningful. The production continuation derives its root from a resolved source-file path; the installer targets the canonical Linux `/content/Strassen_MM_Focus` directory recorded by the host preflight.

The successful local tests establish guard behavior, not successful remote installation. The remote model-tools repair and reference preparation still require their own archived executions.

## N9 input finalization layout mismatch

`20260919T081201Z-N9-input-finalization-v003-73bb4e` failed locally before producing a combined input bundle (commit `d1204ce4`). The v003 validator incorrectly required an additional `artifacts/artifact_manifest.json` in every preparation run. The frozen `application_prep_v001` actually emits a child `preparation-00/artifact_manifest.json`; the orchestrator's outer `artifact-manifest.json` seals that child manifest and every preparation artifact. The newer installer and tokenizer diagnostic also emit the additional canonical manifest. No original evidence was missing from its applicable archive layout, and no checkpoint, token or reference file was changed.

The failed v003 code and new partial plan directory remain preserved. Recovery uses a new v004 validator that recognizes the existing preparation layout, verifies every artifact against the outer seal and every child artifact against its child seal, and continues to require the canonical seal for the newer installer and diagnostic formats. The agents' initial static review missed this schema difference; the repair review explicitly checks the actual completed preparation fixtures.

The recovery completed in `20260919T081552Z-N9-input-finalization-v004-a8f5bc` (commit `30bd41ce`). It validated all eleven source runs and created the new sealed plan `plans/n9_inputs_final_20260919_v004`, preserving the original Qwen/Gemma entries and all earlier failed steps. The subsequent configuration-generation execution `20260919T081613Z-N9-campaign-generation-v001-7c24bd` completed (commit `52c331de`) before N9 was launched.
