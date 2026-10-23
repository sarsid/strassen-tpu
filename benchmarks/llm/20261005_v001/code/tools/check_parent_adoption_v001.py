"""Offline qualification of exact dispatch, full-shape tracing and provenance.

No checkpoint download, TPU allocation or performance claim. Abstract tracing
uses the registered production dimensions without allocating model weights.
"""
import importlib
import json
import os
from pathlib import Path
import hashlib
import parent_stage_v002 as setup
from adopt_parent_profile_v001 import adopt
from compare_parent_fused_v002 import resolve, parser
from strassen_mm import parent_fused_adapter_v002 as adapter


def rejects(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('Invalid configuration was accepted')


def main():
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ['STRASSEN_EXECUTION_DIR']) / 'artifacts'
    out.mkdir()
    source = root / 'results/v6e/parent_llm_followup_20261002_v001/budget-profile.json'
    profile = adopt(source)
    (out / 'adopted-profile.json').write_text(json.dumps(profile, indent=2) + '\n')
    args = parser().parse_args([])
    assert (args.family, args.dtype, args.cubic_control) == ('s1', 'bfloat16', 'parent')
    specs = resolve(profile, args.dtype, args.family, args.cubic_control)
    assert all(v['backend'] == 'parent_exact' for v in specs.values())
    assert resolve(profile, 'bfloat16', 's1', 'matched')['gated_cubic']['backend'] == 'extension'
    assert resolve(profile, 'bfloat16', 's2', 'matched')['gated_strassen']['backend'] == 'extension'
    assert resolve(profile, 'float32', 's1', 'matched')['gated_strassen']['backend'] == 'extension'
    rejects(lambda: resolve(profile, 'float32', 's1', 'parent'))
    # An inherited experimental override must not change the adopted parent.
    os.environ['QWEN3_PRODUCT_PANELS'] = '0,1'
    setup.configure(out)
    stream = importlib.import_module('benchmark_qwen3_32b_streamed_inference')
    layer = importlib.import_module('benchmark_qwen3_32b_layer')
    assert stream.PRODUCT_PANELS == ()
    rejects(lambda: setup.configure(out))
    adapter.validate_parent(layer, stream)
    rejects(lambda: adapter.build(layer, dict(adapter.exact_spec(), output_dtype='float32'), stream=stream))
    rejects(lambda: adapter.build(layer, dict(adapter.exact_spec(), family='s2'), stream=stream))
    rejects(lambda: adapter.build(layer, interpret=True, stream=stream))
    original_tile = stream.QK_TILE
    stream.QK_TILE = (2048, 1024, 512)
    rejects(lambda: adapter.build(layer, stream=stream))
    stream.QK_TILE = original_tile
    import jax
    import jax.numpy as jnp
    import numpy as np

    class AbstractCheckpoint:
        def tensor(self, name):
            shapes = {
                'mlp.gate_proj.weight': (25600, 5120),
                'mlp.up_proj.weight': (25600, 5120),
                'mlp.down_proj.weight': (5120, 25600),
                'input_layernorm.weight': (5120,),
                'post_attention_layernorm.weight': (5120,),
                'self_attn.q_proj.weight': (8192, 5120),
                'self_attn.k_proj.weight': (1024, 5120),
                'self_attn.v_proj.weight': (1024, 5120),
                'self_attn.o_proj.weight': (5120, 8192),
                'self_attn.q_norm.weight': (128,),
                'self_attn.k_norm.weight': (128,),
            }
            return jnp.zeros(shapes[name.removeprefix('model.layers.0.')], jnp.bfloat16)

    params = jax.eval_shape(lambda: stream.load_layer(AbstractCheckpoint(), 0, *layer.rope_values()))
    x = jax.ShapeDtypeStruct((8192, 5120), jnp.bfloat16)
    traces = {}
    for family, arm in adapter.ARMS.items():
        adopted = adapter.build(layer, adapter.exact_spec(family), stream=stream)
        original = stream.make_product_layer(arm)
        assert adopted.__code__ is original.__code__
        assert adopted.__globals__ is original.__globals__
        assert adopted.prepare_weights(params) is params
        old_graph = str(jax.make_jaxpr(original)(x, params))
        new_graph = str(jax.make_jaxpr(adopted)(x, params))
        assert old_graph == new_graph, family
        traces[family] = dict(jaxpr_sha256=hashlib.sha256(new_graph.encode()).hexdigest(),
                              jaxpr_bytes=len(new_graph), exact_graph_equal=True)
    # Execute a small Native layer to check call/parameter compatibility on CPU.
    # Reduced geometry is only a local test fixture, never an adopted profile.
    native = adapter.build(layer, adapter.exact_spec('native'), stream=stream)
    original_native = stream.make_product_layer('regular_xla')
    layer.BATCH = 1; layer.SEQUENCE = 8; layer.TOKENS = 8
    layer.MODEL_DIM = 256; layer.INTERMEDIATE_DIM = 512
    layer.HEADS = 2; layer.KV_HEADS = 1
    rng = np.random.default_rng(20261002)
    def rand(shape):
        return jnp.asarray(rng.normal(size=shape) * .03, jnp.bfloat16)
    cos, sin = layer.rope_values()
    small = layer.Parameters(jnp.ones((256,), jnp.bfloat16), rand((256, 256)),
              jnp.ones((128,), jnp.bfloat16), rand((256, 128)), jnp.ones((128,), jnp.bfloat16),
              rand((256, 128)), rand((256, 256)), jnp.ones((256,), jnp.bfloat16),
              rand((256, 1024)), rand((512, 256)), cos, sin)
    sx = rand((8, 256))
    expected = jax.jit(original_native)(sx, small)
    actual = jax.jit(native)(sx, small)
    np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
    rejects(lambda: adapter.build(layer, stream=stream))
    result = dict(status='passed', default_exact_parent=True, native_cpu_bitwise_equal=True,
                  production_shape_traces=traces, invalid_profiles_rejected=True,
                  parent_source_files_verified=len(adapter.verify_sources()['files']),
                  extension_profiles_preserved=True, jax=jax.__version__, backend=jax.default_backend(),
                  scope='Offline integration checks; no new TPU timing or model quality claim')
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
