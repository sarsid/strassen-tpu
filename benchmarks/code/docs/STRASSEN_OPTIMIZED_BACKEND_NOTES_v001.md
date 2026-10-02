# Strassen optimized backend notes, version 001

This is a source/API review for the new `strassen_optimized` implementation, not a performance result. The frozen kernels remain unchanged. The installed backend inspected was **JAX 0.7.2** in `../.venv-reconcile/lib/python3.12/site-packages/jax`. No TPU allocation, compilation or timing was performed for this review.

## Boundary semantics verified in the pinned backend

`BlockSpec` can describe a final partial block. Its logical reference still has the full block shape: out-of-bounds input values are unspecified and output elements outside the original array are discarded. Every block must intersect the array. A ceil-divided grid satisfies this condition for positive dimensions. The public [BlockSpec guide](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html) describes this contract; the installed `jax/_src/pallas/hlo_interpreter.py:405` pads interpreted floating inputs with uninitialized values and `:489` crops the final output to its declared array shape.

The pinned TPU block-mapping check, `jax/_src/pallas/mosaic/lowering.py:645`, requires the last two **block** dimensions to be multiples of 8 and 128, respectively, or equal to the complete array dimensions. It does not require the global matrix dimensions to divide the block dimensions. Retain the kernel's stricter Strassen contract BM multiple of 16 and BN/BK multiples of 256 so each half-tile dot and slice remains aligned.

Do not implement this path using generic `pl.load(..., mask=...)`: the 0.7.2 TPU lowering explicitly raises `NotImplementedError` when a load mask is supplied (`mosaic/lowering.py:1564`). BF16 masked swaps are also unsupported there (`:1733`). Some newer examples and APIs do not describe this pinned backend.

## Minimal-padding masked-edge design

Use the original `(M,K)` and `(K,N)` inputs, an output declared as `(M,N)`, the same aligned block sizes, and grid `(ceil(M/BM), ceil(N/BN), ceil(K/BK))`. Read the block references normally, then form BF16 local values:

```text
Ablock = where((i*BM + rows < M) & (s*BK + columns < K), a_ref[...], BF16(0))
Bblock = where((s*BK + rows < K) & (j*BN + columns < N), b_ref[...], BF16(0))
```

Apply these selections **before** splitting quadrants or performing any Strassen input additions/subtractions. Mask M and N as well as K: Strassen mixes rows and columns across quadrants, and unspecified edge values must not enter those combinations. Use selection, not multiplication by a zero mask, because NaN times zero remains NaN. `jnp.where`'s select operation has a TPU lowering in `mosaic/lowering.py:3039`.

This removes explicit whole-array input padding and output cropping from the source path. It can reduce preparation and edge memory movement. The grid still executes full tile-panel products, including arithmetic corresponding to masked zeros; do not report reduced dot FLOPs without a separate change that actually removes those products. CPU interpretation validates algebra and mask placement but does not establish TPU compilation or speed.

## Local accumulation and first-panel initialization

The frozen interleaved order is p4, p6, p5, p2, p7, p3, p1. First contributions are p4 for C00/C10, p6 for C11, and p5 for C01. Keeping four FP32 quadrant values locally, preserving each quadrant's subsequent addition/subtraction sequence, and writing each quadrant once per panel is a testable source-level optimization. Explicit BF16 rounding of all Strassen operand combinations must remain unchanged.

Guard following-panel output reads with `lax.cond(step == 0, first_panel, following_panel)`. Place each `out_ref[...]` read **inside** `following_panel`; computing those reads before the conditional or passing their values as operands reads uninitialized output on the first panel. A `where` is a value selection and is not an equivalent guard for those reference reads. The pinned lowering implements `lax.cond` as an `scf.IfOp` with separate branch regions (`mosaic/lowering.py:3255`). Both branches are traced, but their reference operations belong in their respective runtime regions.

Fewer source reference writes do not establish an equal reduction in HBM traffic. The compiler already retains consecutive output blocks during reduction. The conditional and larger live local arrays may alter scheduling or register/VMEM pressure, so isolate this variant from the edge-handling variant when testing.

## Traversal and existing pipelining

Keep K panels consecutive for each output tile. Comparing logical M,N,K with N,M,K traversal is a bounded experiment; update both index maps and mask coordinates when exchanging the outer grid axes. Moving K outward changes accumulator lifetime and is not an equivalent traversal-only edit.

When there is one K panel, M,N,K order presents the same A block across adjacent N tiles; N,M,K order presents the same B block across adjacent M tiles. With multiple panels, the last panel of one output tile and the first panel of the next generally use different input blocks, so this immediate reuse argument no longer applies. Any benefit for those shapes needs evidence.

The existing `pallas_call` with a grid and block specifications already overlaps communication and computation. The [software-pipelining guide](https://docs.jax.dev/en/latest/pallas/pipelining.html) also describes eliding copies for consecutive identical slices. Adding an explicit DMA pipeline is therefore a different implementation choice, not automatically a new optimization. In pinned 0.7.2, ordinary `pallas_call` lowering permits only one or two buffers and rejects lookahead (`mosaic/lowering.py:845`); do not import broader promises from newer pipelining documentation.

## Native peeled tails: separate experiment

A hybrid can compute the aligned top-left core using Strassen, add its native K-tail product, compute the top-right strip natively over all K, and compute the entire bottom strip natively over all K. Handle zero-sized core dimensions explicitly. This can avoid mostly-empty tile products but adds native calls, core accumulation and output assembly; input slices may also require materialization. It changes floating-point operation ordering and the fraction of work using Strassen. Identify it as a hybrid and time the complete call. It is not a clean first ablation for the masked-edge path.

## Required discriminating checks

- Odd and half-tile boundaries in each of M,N,K, including values just below and above BM/2, BN/2 and BK/2.
- K below BK, equal to BK, just above BK, and above two BK panels; first-panel seeding alone cannot test following-panel reads.
- Multiple M/N tiles under both traversal orders; use non-square dimensions to expose swapped index maps.
- Fully divisible shapes, tiny shapes, explicit edge masks with NaN-filled interpreter padding, and the existing numerical contract against an FP32 reference on the same BF16 inputs.
- Separate baseline, masked-edge, local-accumulator and combined variants. Preserve shape, tile, product order, precision, compiler settings and timing scope for each ablation.

Record these as optimization proposals and correctness-qualified variants until a fresh, controlled TPU comparison establishes their performance.
