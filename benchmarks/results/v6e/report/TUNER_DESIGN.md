# Tuner design choices

1. Treat hardware/software identity, input/accumulation/output precision and supported input family as part of the tuning contract. Keep FP32 and BF16 output independent.
2. Search two accumulation strategies: seven retained outer products; or immediate outer-quadrant updates (direct FP32 output, four FP32 scratch quadrants for BF16). S2 always reconstructs its inner S1 within each K panel. This varies outer accumulator lifetime without changing the seven-product formula or BF16 pre-add contract.
3. Tune blocked cubic and full-tile cubic independently. Search output geometry, K-panel length, accumulation strategy and one/two input buffers together. Include past winning geometries and panels so a narrower new menu cannot silently discard known candidates.
4. Bound compile cost explicitly: a registered output-tile menu and selected K panels including standard tile-multiple boundary padding. Record every omitted divisor and every heuristic memory prune. A rough footprint above 160 MiB prunes candidates against a 112 MiB custom allowance; this is intentionally loose and is not proof a compiler would reject the candidate. Historical controls bypass that estimate.
5. Compile randomized batches of at most twelve custom candidates, each with Native_default. Normalize screen latency by that batch anchor to reduce sensitivity to drift. Retain all raw timings, compilation failures and numerical exclusions.
6. Choose only complete measurements passing the frozen numerical gate. Choose the minimum normalized latency; break exact ties by stable candidate ID. Freeze overall, Native, S1/S2 and accumulator/contraction-stratum choices before confirmation.
7. Confirm on three fresh Gaussian inputs and thirty paired rounds per input. Rerun historical controls, each selected depth at the other accumulator/buffer setting, and tile/buffer-matched blocked cubic. These measurements explain tradeoffs; they do not trigger reselection.
8. Require a 1% mean improvement and a paired 95% lower bound above one for the frozen overall custom winner, plus all confirmation error checks. Otherwise recommend tuned Native. The 1% threshold is a declared practical margin, not a hardware law.
9. Persist configuration, identity, rule, numerical scope and measured evidence. Use Native outside the validated contract. This study does not train or validate an unseen-shape prediction model, install a general dispatcher, or certify LLM quality.
10. Explain choices using measured alternatives. Memory layout may suggest a mechanism, but proving stalls, overlap or spills requires compiled/profiler evidence beyond a simple footprint estimate.

The per-shape choices and measured counterfactuals are in RESULTS.md; full records are in candidate_decisions.json and results.json.
