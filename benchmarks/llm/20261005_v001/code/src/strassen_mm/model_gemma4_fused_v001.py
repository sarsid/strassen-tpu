"""Gemma4 whole-prefill layers with versioned v6e fused projections.

BF16 output only. Global K/V shares exactly one raw K matrix multiplication;
its K normalization/rotation and V normalization follow outside that kernel.
Local V normalization also follows its projection. All such scopes are recorded.
Not integrated into the running campaign or qualified on official weights yet.
"""
import jax
import jax.numpy as jnp
from .gemma4_contract_v001 import validate_config, layer_geometry
from . import model_gemma4_v001 as native
from .fusion_v004 import Epilogue, reference, store
from .kernels_fused_v005 import make_projection

VERSION = "model_gemma4_fused_v001"


def native_policy(config, layer_index=0, output_dtype="bfloat16"):
    if output_dtype != "bfloat16":
        raise ValueError("Gemma4 campaign supports BF16 output only")
    sites = layer_geometry(config, layer_index)["projection_shapes_mkn"]
    return {site: dict(architecture="v6e", implementation="native", depth=0,
                      accumulator="products", tile=None, buffers=2,
                      vmem_limit_bytes=112*1024**2, output_dtype=output_dtype,
                      compiler_options={}) for site in sites}


def geometry(config, batch, sequence, layer_index=0, *, rounding="accumulator"):
    c = validate_config(config)
    g = layer_geometry(c, layer_index, batch, sequence)
    kinds = dict(q="qk_norm_rope", k="none" if g["shared_raw_kv"] else "qk_norm_rope",
                 v="none", o="norm_residual_add", gateup="geglu", down="norm_residual_add")
    specs = {site: Epilogue(kinds[site], rounding=rounding, norm="gemma4",
                            head_dim=g["head_dim"], eps=c["rms_norm_eps"])
             for site in g["projection_shapes_mkn"]}
    return c, g, specs


def build_layer(config, sequence_length, policy=None, *, batch_size=1, layer_index=0,
                output_dtype="bfloat16", rounding="model", fused_sites=None,
                early_sites=(), compiler_options=None, interpret=False,
                attention_backend="xla", capture=False):
    c, g, base_specs = geometry(config, batch_size, sequence_length, layer_index, rounding=rounding)
    if attention_backend not in ("explicit", "xla"):
        raise ValueError("Unknown attention backend")
    shapes = g["projection_shapes_mkn"]
    sites = tuple(shapes)
    chosen = native_policy(c, layer_index, output_dtype)
    if policy is not None:
        if not set(policy) <= set(sites):
            raise ValueError("Policy includes a nonexistent Gemma4 projection")
        chosen.update(policy)
    fused_sites = sites if fused_sites is None else tuple(fused_sites)
    if not set(fused_sites) <= set(sites) or not set(early_sites) <= set(fused_sites):
        raise ValueError("Unknown or unfused projection requested")
    options = dict(compiler_options or {})
    specs, projections = {}, {}
    for site in sites:
        arm = chosen[site]
        if arm["output_dtype"] != output_dtype:
            raise ValueError("One BF16 output contract per layer")
        if arm.get("compiler_options") and arm["compiler_options"] != options:
            raise ValueError("Projection compiler options differ from the complete layer")
        spec = base_specs[site]
        spec = Epilogue(spec.kind, "model" if arm["implementation"] == "native" else rounding,
                        "gemma4", spec.head_dim, spec.eps, site in early_sites)
        specs[site] = spec
        if site not in fused_sites:
            # This is an internal diagnostic intermediate, not an FP32-output
            # campaign arm; the consumer still stores the requested BF16 value.
            arm, spec = dict(arm, output_dtype="float32"), Epilogue()
        projections[site] = make_projection(arm, shapes[site], spec, interpret=interpret)
    bsz, length, h = batch_size, sequence_length, c["hidden_size"]
    m, heads, kv, dim = bsz*length, g["query_heads"], g["kv_heads"], g["head_dim"]
    eps = c["rms_norm_eps"]
    cosine, sine = native.rotary_tables(c, layer_index, jnp.arange(length)[None])
    position = {"cos": jnp.tile(cosine[0, :, :dim//2], (bsz, 1)),
                "sin": jnp.tile(sine[0, :, :dim//2], (bsz, 1))}
    mask = native.attention_mask(length, g["sliding_window"])
    allowed = mask == 0
    extras = {"norm1": h, "norm2": h, "norm3": h, "norm4": h,
              "qnorm": dim, "knorm": dim, "layer_scalar": 1}

    def prepare_weights(weights):
        if set(weights) != set(sites) | set(extras):
            raise ValueError("Layer state differs from Gemma4 checkpoint contract")
        for name, width in extras.items():
            permitted = (jnp.bfloat16, jnp.float32) if name == "layer_scalar" else (jnp.bfloat16,)
            if weights[name].shape != (width,) or weights[name].dtype not in permitted:
                raise ValueError("Incorrect Gemma4 normalization/scalar state: " + name)
        return {name: projections[name].prepare_weights(value) if name in sites else value
                for name, value in weights.items()}

    def call(x, weights):
        squeeze = x.ndim == 2
        shape = (length, h) if squeeze and bsz == 1 else (bsz, length, h)
        if x.shape != shape or x.dtype != jnp.bfloat16:
            raise ValueError("Expected fixed unpadded BF16 prefill input")
        x = x.reshape(m, h)
        traces = {}

        def project(site, value, **aux):
            fn = projections[site]
            y = fn.prepared(value, weights[site], **aux) if site in fused_sites else reference(
                fn.prepared(value, weights[site]), specs[site], jnp.bfloat16, **aux)
            if capture:
                traces[site] = dict(input=value, aux=aux, output=y)
            return store(y, jnp.bfloat16)

        normalized = native.rms(x, weights["norm1"], eps)
        q = project("q", normalized, **position, scale=weights["qnorm"]).reshape(bsz, length, heads, dim)
        if g["shared_raw_kv"]:
            raw_kv = project("k", normalized).reshape(bsz, length, kv, dim)
            k = native.apply_rotary(native.rms(raw_kv, weights["knorm"], eps), cosine, sine)
            v = native.rms(raw_kv, None, eps)
        else:
            k = project("k", normalized, **position, scale=weights["knorm"]).reshape(bsz, length, kv, dim)
            v = native.rms(project("v", normalized).reshape(bsz, length, kv, dim), None, eps)
        if attention_backend == "xla":
            value = jax.nn.dot_product_attention(q, k, v, mask=allowed[None, None], scale=1.0, implementation="xla")
            value = native.rounded(value)
        else:
            k, v = jnp.repeat(k, heads//kv, 2), jnp.repeat(v, heads//kv, 2)
            scores = native.rounded(jnp.einsum("bthd,bshd->bhts", q, k,
                precision=jax.lax.Precision.DEFAULT, preferred_element_type=jnp.float32))
            scores = native.rounded(scores.astype(jnp.float32) + mask[None, None].astype(jnp.float32))
            probabilities = native.rounded(jax.nn.softmax(scores.astype(jnp.float32), axis=-1))
            value = native.rounded(jnp.einsum("bhts,bshd->bthd", probabilities, v,
                precision=jax.lax.Precision.DEFAULT, preferred_element_type=jnp.float32))
        residual = project("o", value.reshape(m, heads*dim), scale=weights["norm2"], residual=x)
        normalized = native.rms(residual, weights["norm3"], eps)
        middle = project("gateup", normalized)
        y = project("down", middle, scale=weights["norm4"], residual=residual)
        y = native.rounded(y.astype(jnp.float32) * weights["layer_scalar"].astype(jnp.float32)).reshape(bsz, length, h)
        result = y[0] if squeeze else y
        return (result, traces) if capture else result

    jit_options = {"compiler_options": options} if options else {}
    complete = jax.jit(lambda x, w: call(x, prepare_weights(w)), **jit_options)
    complete.prepare_weights = prepare_weights
    complete.prepared = jax.jit(call, **jit_options)
    complete.metadata = dict(version=VERSION, model_type=c["model_type"], batch_size=bsz,
        sequence_length=length, layer_index=layer_index, attention_type=g["layer_type"],
        output_dtype=output_dtype, rounding=rounding, attention_backend=attention_backend,
        capture=capture, fused_sites=list(fused_sites), early_sites=list(early_sites),
        compiler_options=options, projections={site: fn.metadata for site, fn in projections.items()},
        policy=chosen, shared_raw_kv=g["shared_raw_kv"], matrix_product_sites=list(sites),
        postprocessing_outside_projection=["value_rms", "layer_scalar"] + (["global_key_rms_rope"] if g["shared_raw_kv"] else []),
        attention_scope="whole unpadded prefill; no KV-cache updates", consumer_dtype="bfloat16",
        rounding_enforcement="reduce_precision_8_7_in_XLA",
        scope="whole layer including layout restoration, postprocessing and consumer casts; prepared excludes weight packing")
    return complete
