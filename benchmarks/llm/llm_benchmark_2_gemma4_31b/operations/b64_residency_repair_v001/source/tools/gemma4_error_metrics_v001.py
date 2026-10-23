"""Bounded-memory diagnostics; never used inside measured model execution."""
import math

CHUNK_ELEMENTS = 1024 * 1024


def chunked_error(reference, candidate, *, jax, jnp, chunk_elements=CHUNK_ELEMENTS):
    """Same L2/max-error definitions without full-size FP32 copies.

    Each tile reduces in FP32 on the device. Only five scalars per tile are
    copied to the host; math.fsum combines squared norms in host FP64. This
    changes reduction grouping, not the error definitions or quality limits.
    Both original arrays remain untouched. Tail tiles include every element.
    """
    if reference.shape != candidate.shape or not reference.size:
        raise ValueError('Expected nonempty matching arrays')
    if not isinstance(chunk_elements, int) or chunk_elements <= 0:
        raise ValueError('Expected positive integer chunk size')

    def stats(r, v):
        r = r.astype(jnp.float32)
        v = v.astype(jnp.float32)
        d = v - r
        return (jnp.all(jnp.isfinite(v)), jnp.sum(d * d),
                jnp.sum(r * r), jnp.max(jnp.abs(d)), jnp.max(jnp.abs(r)))

    # One executable for full tiles and at most one for the final short tile.
    reduce_tile = jax.jit(stats)
    r = reference.reshape(-1)
    v = candidate.reshape(-1)
    finite = True
    differences = []
    references = []
    max_difference = 0.0
    max_reference = 0.0
    for start in range(0, reference.size, chunk_elements):
        stop = min(start + chunk_elements, reference.size)
        ok, dd, rr, dm, rm = jax.device_get(reduce_tile(r[start:stop], v[start:stop]))
        finite = finite and bool(ok)
        differences.append(float(dd))
        references.append(float(rr))
        # Propagate NaNs as the original global reductions do.
        max_difference = max(max_difference, float(dm)) if not math.isnan(float(dm)) else math.nan
        max_reference = max(max_reference, float(rm)) if not math.isnan(float(rm)) else math.nan
    denominator = max(math.sqrt(math.fsum(references)), 1e-30)
    return dict(finite=finite,
                relative_l2=math.sqrt(math.fsum(differences)) / denominator,
                max_abs=max_difference,
                maxnorm_relative=max_difference / max(max_reference, 1e-30))
