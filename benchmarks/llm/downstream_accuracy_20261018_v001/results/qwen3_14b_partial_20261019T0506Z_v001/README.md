# Qwen3-14B — first 80 MMLU pairs (partial)

The alternate account obtained a v6e and the unchanged study passed runtime admission. This immutable snapshot preserves the first 80 of 114 registered MMLU questions, with identical frozen examples and checkpoint revision for Native and plain Strassen S1.

| Partial metric | Native | Strassen S1 |
| --- | ---: | ---: |
| Correct answers | 67 / 80 | 67 / 80 |
| Observed accuracy | 83.75% | 83.75% |

The two arms selected identical answers on all 80 questions. This is an incomplete prefix of the stratified subset, not the final MMLU score, and no accuracy-preservation verdict is assigned. GSM8K and perplexity results are not included. Later complete results must be added separately rather than replacing this snapshot.

Verification checked the execution-contract hash, pinned source hashes, unchanged frozen example manifest, exact paired checkpoint identities/order, and passed train-only runtime admission. The last archived checkpoint was independently read from the live TPU and matched byte for byte. File hashes are in artifact-manifest.json; verification.json states the partial scope explicitly. Hardware is the owned Colab V6E1 assignment, and the driver additionally enforces single-device v6e before evaluation.
