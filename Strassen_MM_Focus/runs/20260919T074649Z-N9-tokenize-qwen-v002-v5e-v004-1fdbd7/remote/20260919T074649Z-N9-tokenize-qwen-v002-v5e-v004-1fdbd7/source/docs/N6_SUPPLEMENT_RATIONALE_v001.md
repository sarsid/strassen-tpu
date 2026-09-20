# Post-hoc N6 supplement: largest confirmed joint wins

The original preregistered N6 study selected median-volume examples from the N5 win, inconclusive and loss classes. Its valid recovery traces exposed differences between profiled device-module rankings and ordinary complete-call rankings, including native being fastest on-device for all three representatives. Those results remain part of the paper evidence and will not be replaced.

The original representatives did not include either large headline N5 result. A bounded supplementary diagnostic will ask whether the largest complete-call wins against both controls also have device-module advantages. This is explicitly **post-hoc diagnostic selection**, motivated by the observed N6 scope differences. It is not another independent confirmation, a replacement for the original N6 representatives, or evidence of a general device mechanism.

## Frozen selection rule and scope

Use only the sealed N5 confirmation and its exact preexisting screening selections. Require all three complete-call arms—native XLA, selected cubic, selected Strassen—to pass their fixed numerical gate and have 30 samples. Require Strassen's valid paired CI95 lower endpoint to exceed 1 against both native and selected cubic. Among eligible shapes, select exactly two by descending M*K*N, breaking volume ties by ascending shape ID. If fewer than two qualify, fail explicitly instead of substituting an unqualified shape.

For the existing N5 confirmation this rule selects the Qwen-shaped (8192,5120,51200) product first and the (8192,8192,8192) square second. No N7 results, real-model quality outcomes or later performance observations enter the choice. The shapes, selected tiles/variants, Gaussian seed, BF16/FP32 contract and numerical gates are inherited unchanged from N5. No selector is fitted or updated.

Each shape retains native, selected cubic and selected Strassen. Measure ordinary complete calls before and after profiling with five warmups and 30 paired rounds. Collect two capture blocks per arm, reversing arm order in the second block, with eight synchronized invocations per capture. This gives 12 expected device captures, 96 module invocations, 12 ordinary result rows and 360 ordinary timing samples. The same qualified v5e allocation and software identity are required.

Device tracing uses the already validated `TRACE_COMPUTE_AND_SYNC` mode and explicit device tracer level 1. The original campaign bytes stay fixed; an exclusive `supplementary_design.json` records the effective override, selection rule, selected representatives, source identity and hashes linking the exact N5 confirmation, choices and prior successful N6 recovery. Journal events carry `supplement_id: n6_large_joint_wins_v001` and `post_hoc_diagnostic: true`.

## Implementation and execution ownership

New module: `strassen_mm.benchmark_n6_supplement_v001`. It imports the untouched v002 N6 runner and the existing qualified recovery capture method. It changes only representative selection, explicit supplementary provenance and the documented profiling options. The strict analyzer still requires exactly eight positive nonoverlapping module durations on one actual TPU XLA Modules track. Missing coverage causes failure before further groups. Existing source files and all original results remain unchanged.

Launch with phase `N6`, the original campaign, exact N5 selections and confirmation, the same expected identity/allocation, and required `--prior-profile` pointing to the successful three-representative N6 recovery (`20260919T064743Z-N6-v5e-v004-e7bc5e`). The wrapper checks the prior seal, 18 successful captures, unchanged core dependencies and matching identity/selection/confirmation provenance. The parent execution manager must freeze, run, archive and commit it sequentially after N7 screening, with no simultaneous TPU timing.

Four new offline tests cover joint-win eligibility, volume/tie ordering, numerical and sample-count exclusions, rejection of invalid intervals/N7/duplicate input, and restoration of wrapper hooks on failure. No test or TPU execution was performed by the protocol agent during implementation.

## Interpretation limits

Report the supplement alongside the original three representatives, including the ranking reversals. Keep ordinary complete-call timing, instrumented wall time and actual device-module duration separate. Profiling can perturb execution; a difference between these scopes cannot be assigned solely to dispatch overhead. These two selected favorable cases do not estimate the prevalence of wins or establish held-out generalization. Their repeated calls share one allocation and are not independent replications. No measured utilization, vector/MXU overlap, achieved bandwidth or model-quality claim follows from these traces or static cost estimates. The prior warm-trace HBM-bound contradiction remains disclosed.
