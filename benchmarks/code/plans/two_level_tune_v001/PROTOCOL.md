# Small tile-tuning probe for two-level Strassen

Question: can tile tuning make the existing two-level kernel competitive with
one level on a small representative subset? No new kernel schedule is introduced.

Use two synthetic BF16 shapes: square (M,K,N)=(12288,12288,12288), and Qwen
gate/up (16384,4096,24576). They cover square and wide rectangular products.
Do not add Mistral, Gemma, Native, cubic, or AlphaTensor to this bounded probe.

Both depths receive the same eight (BM,BN,BK) candidates:

- (1024,1024,512)
- (1024,2048,512)
- (2048,1024,512)
- (1024,2048,1024)
- (2048,1024,1024)
- (2048,2048,512)
- (2048,2048,1024)
- (2048,2048,2048)

Smaller output tiles may reduce the scratch-memory problem in the previous
probe. Larger K panels enlarge the leaf products. Include both previous tiles
as anchors and retain any repeated failures. This is a local search, not a claim
that this grid contains the global optimum. Do not change the frozen grid after
observing timing or compilation results.

Screening: 16 tile groups, 32 outcomes, one Gaussian seed per shape, three
warmups and ten timed rounds per executable arm. Shuffle tile order per shape
using a frozen seed. Alternate arm order within each pair. All choices for a
shape use identical quantized input operands. Select the fastest numerically
eligible arithmetic mean separately for each shape and depth; exact ties use
lexicographic (BM,BN,BK) order. Require ten samples. Failures cannot win.

Confirmation: save selections before this separate phase. Rerun each selected
pair on three fresh Gaussian seeds with five warmups and 30 alternating rounds.
This gives two groups, six seed pairs, 12 outcomes and up to 360 timing samples.
Compile each selected arm once per shape and reuse it for the three inputs.
Do not reselect using confirmation data. Report every seed's comparison and
the ratio of pooled arithmetic means. Pointwise paired bootstrap intervals
are conditional on this run; any aggregate interval must preserve seed groups.

Baseline is preserved `kernels_v002` one-level
`interleaved_output_accumulator`. Two levels uses the previously validated
`kernels_two_level_v001` scratch implementation, unchanged. Both use BF16
inputs and pre-additions, FP32 accumulation/output, DEFAULT leaf-dot precision,
and complete-call timing including device padding/crop but excluding compilation
and host transfer. The 48 MiB VMEM cap and 8 GiB known executable memory check
remain unchanged. Do not claim memory tuning has changed the schedule.

Use the exact quantized operands for the host FP32 reference, every K term,
128 x 128 sampled positions including edges, and whole-output finiteness.
Eligibility gates remain relative L2 <= 0.02 and maximum absolute error <=
0.001 + 0.05 times maximum absolute reference. Report increased error explicitly;
passing the gate does not establish equal accuracy or unchanged model quality.

Use one new v5e allocation for both phases, pinned JAX/jaxlib 0.7.2 and
libtpu 0.0.21.1 with the existing qualified Mosaic shim. Verify setup identity
in each phase. Old machine timings may be context only, not the new baseline.
No concurrent device work. Each phase has a 30-minute device deadline. Retrieve
and verify screening artifacts before confirmation. Release the allocation
after both phases are retrieved and verified. No AlphaTensor run follows
automatically.

Freeze and commit sources/configuration before execution. Preserve each phase
and all failed candidates; use scoped commits and the append-only progress
journal. Validate candidate selection, seed separation, and plan construction
locally before provisioning. The kernel algebra is unchanged from the archived
CPU/TPU checks; numerical validation still runs for every candidate on the TPU.

Decision guide: a confirmation improvement beyond 3% on either shape, passing
all fresh-seed gates, supports further exploration; wins on both give stronger
evidence. A near tie may justify a targeted memory/scheduling change. Consistent
losses make broad two-level sweeps lower priority. These are practical exploration
criteria, not publication or statistical significance thresholds.
