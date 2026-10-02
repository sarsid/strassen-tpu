# AlphaTensor: four-shape proof of concept

Requested scope: inspect and run the official released code on a few shapes.
This is a bounded feasibility comparison, not a new algorithm search, a tuning
sweep, an original-paper reproduction, or a claim about all AlphaTensor methods.

## Source

Official repository: https://github.com/google-deepmind/alphatensor
Pinned commit: `1949163da3bef7e3eb268a3ac015fd1c2dbfc767`.
Selected algorithm: `get_4x4x4_alphatensor_tpu`, a rank-49 real-arithmetic
factorization found for BF16 8192-square MM on TPU v2. This is not the modulo-2
rank-47 algorithm. Preserve the original license, documentation and source bytes.

## Frozen scope and decisions

Four shapes, M x K x N:

- Square: 8192 x 8192 x 8192, the original hardware search target size.
- Qwen gate/up: 8192 x 4096 x 24576.
- Mistral down: 8192 x 14336 x 4096.
- Gemma gate/up: 8192 x 3840 x 30720.

Every dimension is divisible by four, so the released four-by-four block algebra
applies without external padding. Our Pallas kernels retain their normal padding.
One newly allocated v5e is used for every comparison. Old and new timings are
never pooled. Pin the same JAX 0.7.2/jaxlib 0.7.2/libtpu 0.0.21.1 environment.
Check the exact tensor identity and two tiny rectangular numerical fixtures
locally before allocating; these are correctness checks, never performance data.

Six arms per shape:

1. Native XLA, BF16 inputs and FP32 output.
2. Our optimized cubic, frozen prior tile.
3. Our one-level Strassen, frozen prior tile.
4. Released TPU AlphaTensor algebra, BF16 subproduct results/recombination/output.
5. Explicitly adapted AlphaTensor: same coefficients, order and BF16 pre-adds,
   but request FP32 dot output and consequently FP32 recombination/output.
6. Native BF16 output, a precision-context baseline for the released code.

All custom tiles are 2048 x 2048 x 512 (BM x BN x BK), except Mistral cubic
uses its previously selected 1024 x 2048 x 512. The square tile is a declared
fixed candidate. No arm is claimed newly tuned or globally optimal. The two
AlphaTensor versions use XLA for their 49 subproducts, not our Pallas tiling.
Different recursion depth and implementations limit algorithm-only conclusions.

Load three exact upstream function ASTs to avoid importing the unrelated
dm-tree timing helpers. Do not execute GPU clock-setting commands. The adapted
mode replaces precisely the one `left @ right` expression with
`jnp.dot(left, right, precision='DEFAULT', preferred_element_type=jnp.float32)`.
The original generator and factors remain archived unchanged. Both modes wrap
splitting and concatenation inside a single JIT accepting/returning full arrays.
This measures usable complete calls; the upstream benchmark instead supplied
pre-split block arrays. Its repeated A @ C timing loop is also unsuitable for
general rectangles, so use our existing synchronized independent-call harness.

## Measurements and safeguards

One Gaussian input seed per shape (2026092111 + shape index), shared by all arms;
the existing host generator quantizes once to BF16. Five warmups, 30 rounds with
seeded rotating arm order, serial timings and full-output synchronization.
Exclude compilation and host transfers; include device layout/padding/crop and
AlphaTensor split/assembly. Retain raw samples, compilation durations, executable
memory estimates, failures, and exact input/source/environment hashes.

Reference: existing sampled 128 x 128 output positions using every K term and
the exact BF16 input values cast to host FP32. Whole-output finite check. Keep
the existing relative-L2 <= 0.02 and max-absolute gate unchanged; report all
errors. Failed-gate arms may retain diagnostic timings but cannot support an
eligible speedup claim. Shared gates do not imply equal error or model quality.
Pointwise paired bootstrap intervals are exploratory, not broad generalization.

Compiled executable argument + output + temporary memory must be <= 8 GiB when
the compiler supplies those values. Observe failures instead of silently changing
shape, dtype, numerical threshold or machine. Outer runtime limit: 40 minutes.
Every execution receives a new archive and scoped commit. Release the allocation
after verified evidence retrieval; preserve it for recovery if remote state is
uncertain. Show actual progress in the append-only work journal.
