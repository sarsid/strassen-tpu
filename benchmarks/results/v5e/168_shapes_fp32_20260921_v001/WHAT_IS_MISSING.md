# What more do we need from v5e?

**No new TPU access is needed to preserve or inspect the completed historical
168-shape FP32-output study.** Its screening/confirmation records, selected
tiles, raw samples, errors, seeds, source and environment identity are local
and included in this bundle. The export audit separately verifies all phases.

For the proposed updated architecture comparison, additional **measurements**
are needed:

| Question | Already available here | Additional work |
|---|---|---|
| How did the historical five families perform on the 168 shapes? | Complete FP32-output screening, frozen selections and fresh confirmation | No historical rerun required |
| How do they perform with BF16 output? | BF16 **input**, not BF16 output | Run all five families under the BF16-output contract, including the final conversion |
| How does the new joint tuner perform on v5e? | Older fixed tile menus and kernel variants; BK is recorded but was not searched under the new joint policy | Qualify and run the updated tile/BK/storage/buffer search, with both output types |
| Is a difference from v6e caused by architecture? | Device/runtime identity and source versions are preserved | Use comparable implementations, supported software and search rules; the old v5e stack is JAX 0.7.2/libtpu 0.0.21.1, whereas the recent v6e work used 0.11.2/0.0.48 |
| Why does a configuration win or lose? | Timings, errors, compile events, available compiler memory estimates and source | Profile representative cases for device execution, memory traffic, overlap and matrix/vector work where measurable; this MAIN bundle contains no dedicated device-trace/HLO collection |
| Does a rule generalize? | Development shapes and fresh timing/input confirmation | Reserve new shapes for selector generalization and independent inputs for numerical stress tests |
| Will the gain survive in a real LLM? | Synthetic matrix geometry evidence | Tune the actual fused workload and measure full-model speed, memory and quality on real weights/activations |

Some historical settings are fixed in the source rather than exposed as tuning
axes. The archive retains those source snapshots; it does not invent buffer
counts, hardware counters or compiler-managed Native tile sizes where they
were not explicitly recorded.

Independent allocation repeats of selected positive, negative and borderline
cases would strengthen robustness claims. They must remain separate cohorts;
the within-allocation paired intervals do not measure cross-allocation variation.

The historical data can therefore serve as a valuable reference and control,
but cannot replace the updated matched v5e experiment. Native fallback and
per-workload LLM specialization also remain useful without first developing
a general predictor for unseen dimensions.
