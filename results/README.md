# Result index

Every JSONL file contains raw benchmark records, including compilation status,
accuracy metrics, timing samples, sentinels, and runtime metadata. The concise
interpretation is in `../EXPERIMENTS.md`; detailed contemporaneous reports and
recorded SHA-256 hashes are in `../docs/archive/`.

## Canonical evidence

| Question | Files |
|---|---|
| Native BF16 and FP32 baselines | `baseline_v5e_bf16_square_8k.jsonl`, `baseline_v5e_fp32_anchor_square.jsonl` |
| Classical versus Winograd accuracy | `strassen_v5e_classical_comparison.jsonl`, `strassen_v5e_multiseed_accuracy.jsonl` |
| Large-square classical performance | `strassen_v5e_classical_large.jsonl`, `strassen_v5e_best_square_4k_8k.jsonl`, `strassen_v5e_best_square_12k_16k.jsonl` |
| VMEM deployment frontier | `strassen_vmem_profile_compact16.jsonl`, `strassen_vmem_profile_spatial34.jsonl`, `strassen_vmem_profile_max48.jsonl` |
| Cleaned BF16 dispatcher release gate | `strassen_bf16_dispatch.jsonl` (SHA-256 `c829f9d4aceb870c3766c7daad81ca3347fde1223a0ad0dd4f7bc8227f4e2236`) |
| Dependency-spaced product order | `strassen_product_order.jsonl`, `strassen_product_order_confirm.jsonl` (SHA-256 `e5bffb635b5fdc8b349a79418b162e737d3a7b8ec2d424dc7539ccd27be10d4c`, `07f924452a9422a9ad6ab5279fe19565412281a9b274e1d658bce43ecdfda02e`) |
| Controlled Transformer-block integration | `strassen_model_block.jsonl` (SHA-256 `27576f055735628cbb6303b0f463d7ed9ba4a6b18a6146e3930ba3d877d60372`) |
| Real-checkpoint resident layer | `strassen_checkpoint_layer.jsonl` (SHA-256 `4ef55cec132662d060c16d3fbc85b0602e97b3528741aa2c930f2ddf8da5c9b6`) |
| Full-checkpoint all-layer quality | `strassen_checkpoint_model.jsonl` (SHA-256 `71c5bc28c2958beabcb6a6647e5648369b78e177038fdde4c0179171cbb53169`) |
| Sparse and terminal checkpoint schedules | `strassen_checkpoint_schedules.jsonl`, `strassen_checkpoint_terminal.jsonl` (SHA-256 `5383e00c84c8df813dec4f7f3e096bf60970a99adcace97e8ddd890decc62f83`, `cc78c842b275460c36b715d610444bd889024c748d61f182d08045ec53daaf07`) |
| Frozen disjoint-text holdout | `strassen_checkpoint_holdout.jsonl` (SHA-256 `5211ca10f5708d93e2458fc9c6235095f2d04db96493835237629f1222222058`) |
| Equivalent classical block permutations | `strassen_checkpoint_permutations.jsonl` (SHA-256 `755842564c869ad66a4edabc0562b80acfff7c3aa60f7766a799aa063f1cff00`) |
| Power-of-two real-layer equilibration | `strassen_checkpoint_equilibration.jsonl` (SHA-256 `28ff92ca93c386cb1f35b38983090072208ba6e8ec799c5465376c52c9b1fecd`) |
| Dual rank-7 formula screen | `strassen_checkpoint_dual.jsonl` (SHA-256 `c1e258b24555c131e0bfb1ab057b0bb8b7421c867dbc9017ab5f13518a99673b`) |
| Equivalent-formula error correlation | `strassen_checkpoint_error_correlation.jsonl` (SHA-256 `7f0253a490c942a664741a18c49989be0f3ff7647a7cf423a69ec89d6a595340`) |
| All-layer equivalent-formula sequences | `strassen_checkpoint_variant_sequences.jsonl` (SHA-256 `4bd03cfc2dcf9b6c166b8a32ee1e48b305dcc0a3572ff897a7368c04cfc0cda4`) |
| FP32 output-retention screen | `strassen_checkpoint_wide_outputs.jsonl` (SHA-256 `2d397235400f21aee661b3569a6ce29989d6a726056d1efdb62bea6b504f6d57`) |
| Lower-growth rank-7 VMEM feasibility | `strassen_checkpoint_powers_fp32preadd_oom.jsonl`, `strassen_checkpoint_powers_bf16preadd_oom.jsonl`, `strassen_checkpoint_powers_epilogue_oom.jsonl` (SHA-256 `9492576efa684494bbbc7308519de6c892c38ddcbb206ca45f01e396363c708c`, `e54639041f326270b298ed427826339ca7b1889f89b58d484ac1afc197121427`, `d1fa5ca42fe3ebda4d0c783f69c5e2c7e535b6eedd7b3cae9df88f4cb0f5fb93`) |
| Datatype/precision screen | `strassen_datatype_screen.jsonl` |
| Rectangular projections and decode | `strassen_rectangular_workloads.jsonl` |
| Cubic Pallas attribution control | `strassen_cubic_control.jsonl` (SHA-256 `9c29b4f2e8e2a6a6615226f71465283d453fdfd6e0b7b360988d10250a6b1e63`); 47 MiB OOM records in `strassen_cubic_control_oom.jsonl` (`7d42a6da14c436d230986bf7b8f8008a3921385897250a213aa34945fec2deb3`) |
| Native sensitivity to the scoped-VMEM flag | `native_vmem_default.jsonl`, `native_vmem_scoped_48m.jsonl` (SHA-256 `ff65b61235531d84bc2ff618e896b3f0d5a0b31fad990236991d4349cdb51fc0`, `d28be65f8d961990611d2a7621533757171c8d55c5d158543485d53551d31c82`) |
| Outlier-panel hybrid, split-K realizations | `strassen_checkpoint_outlier_panels.jsonl` (SHA-256 `d4ec7f1ed2a618d75a27489582ad0aa62426afd927e1b8876fc63a98b86abfe6`) |
| Outlier-panel hybrid, in-kernel at 96 MiB | `strassen_checkpoint_panel_kernel.jsonl` (SHA-256 `721f3a1a3a3cb54e9bc2222f83a416e114754157ea86272a532a066adc0a20b7`) |

## Historical search evidence

- `strassen_v5e_four_acc_bf16_sums.jsonl` and
  `strassen_v5e_fp32_tile_sweep.jsonl`: early accumulator/sum precision search.
- `strassen_candidate1_*.jsonl`: seven-product accumulator schedule.
- `strassen_candidate3_*.jsonl`: hybrid cubic/Strassen panels.
- `strassen_candidate4.jsonl`: power-of-two equilibration.
- `strassen_candidate5_*.jsonl`: two-level Strassen.
- `strassen_scoped_vmem_32m.jsonl` and
  `strassen_single_buffer_16m.jsonl`: VMEM feasibility and buffering.
- `strassen_vmem_profile_balanced42.jsonl`: measured but rejected intermediate
  VMEM deployment point.
- `strassen_sparse_output_basis.jsonl` and
  `strassen_sparse_output_basis_confirm.jsonl`: exact 10-update output-basis
  schedule and its 60-run long-K rejection confirmation (SHA-256
  `df824f13271db05c64bc088d0a86df14db725df486cc449bb3eeda13defd89c4` and
  `edbfe49fc0440c0916b3307fdd90a5d9b6c9f179b3ea41d45ac4a8bbe2555ca7`).
- `strassen_vmem64_bk1024.jsonl`: feasible but rejected 64 MiB
  `(2048,2048,1024)` profile (SHA-256
  `3b7f24a9ac6c9833711cd2ac89369e5e46fe99f84e5a9056cac7dc23fafb66f7`).
- `strassen_expanded_tile_search.jsonl`, `strassen_extreme_tile_search.jsonl`,
  and `strassen_large_tile_candidates.jsonl`: tile frontier search.

OOM records are retained because they identify compiler-reported VMEM demand;
they are not successful performance measurements.

The corresponding removed harnesses are preserved in
`../snapshots/experimental-code-2026-08-15.tar.gz`.
