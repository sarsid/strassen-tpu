# Gemma4 tuner design and local qualification

The Gemma4 tuner uses the existing bounded v6e search policy with geometry
computed separately for every representative attention type. This avoids
applying the 256-wide local heads to 512-wide global attention, and avoids
tuning a nonexistent global V matrix multiplication. No active Qwen source or
queue deployment was changed.

New sources `llm_campaign_gemma4_v001.py` and `tuner_llm_gemma4_v001.py` were
committed before testing in `ef9cd4be7`. Three local tests passed in 2.784 seconds:
`runs/20261004T015811Z-gemma4-tuner-qualification-v001-1fce44` (archive commit
`8b9e3e9ca`). The tests are CPU validation, not TPU-selected profiles.

## Decisions that the tuner records

1. Propagate Native activations through the first layer of each attention type.
   For the official 31B layout those representatives are local layer 0 and global
   layer 5. Freeze each type's real M/K/N geometry and Gemma4 epilogue metadata
   in the profile. Held-out quality data never chooses a tile.
2. Compare six Native whole-layer compiler settings: default and scoped VMEM
   limits of 32, 48, 64, 96 and 112 MiB. Choose by synchronized median layer time.
   Record compilation failures and numerical rejection, rather than treating an
   unmeasured setting as fast. BF16 output is required.
3. Offer v6e cubic/full-contraction cubic, S1 and S2 candidates. Preserve both
   accumulator strategies and one/two buffers. The existing tile menu uses BM
   256/512/1024/2048, BN 512/1024/2048/2560 where permitted, and BK
   512/1024/2048 plus the aligned full contraction. Whole-row RMS requires an
   output tile covering the row; rotary pairs require compatible head alignment.
4. Record illegal layout/alignment, padding work above 2x, and rough VMEM
   estimates above 112 MiB as explicit pruning reasons. The VMEM estimate is a
   search heuristic, not proof of hardware infeasibility. Prefer 256-aligned
   v6e leaves while retaining legal alternatives. This is the v6e policy, not
   a transplanted v5e tile policy.
5. Shortlist at most 12 candidates per site/family with deterministic coverage
   across implementation, accumulator, buffer count and full/blocked
   contraction categories. All offered candidates retain a selected or
   outside-budget disposition. Candidate order is reproducibly randomized.
6. Measure each fused projection with the BF16 consumer cast, three warmups and
   eight screening repeats. Candidates must be finite and within 10% relative
   L2 of the Native calibration projection. Choose the fastest eligible median;
   if none qualifies, keep Native for that site and record the fallback.
7. Compile and check each assembled whole-layer policy. This Gemma4 version
   also requires the composed layer to be finite and within 10% relative L2,
   applying the existing projection eligibility ceiling to the composed result.
   The old Qwen tuner's composed-layer check is finite-only; it is unchanged.
   Any failed composed-layer gate stops qualification without silently replacing
   the policy. Tuned Native also receives the finite/10% calibration check.

Those calibration limits do not replace or relax held-out predictive gates.
Official-checkpoint checks, finite outputs, absolute delta NLL <=0.01, KL <=0.02
and top-1 agreement >=97% remain required. Fifteen independent confirmation
rounds and the separate resident-layer pass remain the campaign protocol.

## Evidence and remaining limits

The pinned 31B geometry audit covered 33 custom site/family menus containing
6,144 candidates, of which 5,323 were offered before the bounded shortlist.
It checked deterministic selection, category coverage, legal rotary widths,
correct kernel metadata, explicit pruning and exclusion of FP32/S3/S4.
No candidate in this audit was device measured.

A tiny synthetic two-layer checkpoint exercised the actual tuning function on
CPU with propagated activations, all five profile families, 33 persisted menus
and six composed-layer gates. Custom tiles were deliberately pruned by the
unchanged padding rule, exercising explicit Native fallbacks. CPU supports the
default Native compiler setting, so two representative Native screens succeeded;
the unsupported TPU-specific settings were recorded as failures. The generated
profiles are marked validation evidence and must not be deployed to the TPU.

Dense custom kernel semantics and host packing were qualified in the earlier
whole-layer and cache tests. Actual custom candidate compilation and performance
still require v6e validation with official checkpoint activations. Next steps
are the independent official input/oracle integration, modern remote runtime,
resident/cached pilot integration and archived launch/recovery validation before
the Gemma4 qualification gate can be cleared.
