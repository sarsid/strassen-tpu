# Separate FP32 formulation control

The matched-operation diagnostic shows late propagated divergence on JAX CPU as
well as TPU. Before attributing that divergence to backend arithmetic, compare
the existing Native formula with official Transformers in FP32 on CPU at real
layers 0, 5 and 53. These cover local attention, global shared K/V, and a late
global layer near the failed hidden-state gate. Use the official BF16 inputs
captured by the first diagnostic, lifted exactly to FP32, and the same checkpoint
weights lifted to FP32. No new tokenization or checkpoint selection is introduced.

The JAX control executes the existing formulation with only its BF16 storage
boundaries temporarily lifted to FP32 inside an isolated CPU process. Restore
those functions after each comparison. Official Transformers also executes in
FP32. Report relative L2, maximum absolute difference and finiteness. This is a
formula diagnostic, not production BF16 execution, and never clears or relaxes
the existing benchmark gates. Its summary retains all_pass=null.

Run only after the all-layer diagnostic finishes, on that same retained machine,
using the existing pinned Python environment and private checkpoint cache. Keep
all historical sources and attempts. Do not release or allocate a TPU. Local
validation compares both attention types beyond the local sliding-window length
and checks boundary restoration after success and an exception.
