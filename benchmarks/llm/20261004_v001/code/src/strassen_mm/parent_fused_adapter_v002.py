"""First-class exact parent S1, with explicitly labelled research extensions.

The exact backend returns the parent's own layer function. It does not route
through our projection factory or change its arithmetic, packing or scheduling.
Use the parent's stream.load_layer to prepare its parameters outside timing.
"""
import hashlib
import json
from copy import deepcopy
from pathlib import Path

COMMIT = '95be1fb088656a89813b04492e1d77c66b36ccf9'
ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / 'third_party/strassen_tpu_public_95be1fb_v001'
COMPILER_OPTIONS = {'xla_tpu_scoped_vmem_limit_kib': 49152}
ARMS = {'native': 'regular_xla', 'cubic': 'gated_cubic', 's1': 'gated_strassen'}
PROFILE = dict(product_tile=[2048, 1024, 5120], cubic_tile=[2048, 2048, 512],
               qk_tile=[2048, 1024, 5120],
               site_tiles={'o': [2048, 512, 8192], 'down': [2048, 2560, 1024]},
               extended_sites=['o', 'down'], strassen_limit_mib=120,
               cubic_limit_mib=124)


def verify_sources():
    provenance = json.loads((PARENT / 'provenance.json').read_text())
    if provenance['commit'] != COMMIT:
        raise ValueError('Unexpected parent revision')
    for name, record in provenance['files'].items():
        data = (PARENT / name).read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Frozen parent source changed: ' + name)
    return provenance


def exact_spec(family='s1'):
    if family not in ARMS:
        raise ValueError('The exact parent provides Native, cubic and S1 only')
    return dict(backend='parent_exact', family=family, output_dtype='bfloat16',
                parent_commit=COMMIT, model='Qwen3-32B', device='v6e',
                batch=8, sequence=1024, tuning=deepcopy(PROFILE))


def validate_parent(layer, stream):
    verify_sources()
    modules = {
        layer: 'experiments/qwen3/benchmark_qwen3_32b_layer.py',
        stream: 'experiments/qwen3/benchmark_qwen3_32b_streamed_inference.py',
        stream.sp: 'strassen_pallas.py',
        stream.cubic: 'experiments/qwen3/benchmark_cubic_control.py',
    }
    for module, name in modules.items():
        if Path(module.__file__).resolve() != (PARENT / name).resolve():
            raise ValueError('Parent module imported from another checkout: ' + name)
    geometry = (layer.MODEL_NAME, layer.BATCH, layer.SEQUENCE, layer.TOKENS,
                layer.MODEL_DIM, layer.INTERMEDIATE_DIM, layer.HEADS,
                layer.KV_HEADS, layer.HEAD_DIM, layer.NUM_LAYERS)
    if geometry != ('32b', 8, 1024, 8192, 5120, 25600, 64, 8, 128, 64):
        raise ValueError('Exact parent profile is qualified only for Qwen3-32B B8 S1024')
    actual = dict(product_tile=list(stream.PRODUCT_TILE), cubic_tile=list(stream.CUBIC_TILE),
                  qk_tile=list(stream.QK_TILE),
                  site_tiles={k: list(v) for k, v in stream.SITE_TILES.items()},
                  extended_sites=list(stream.EXTENDED_SITES),
                  strassen_limit_mib=stream.STRASSEN_LIMIT // 1024**2,
                  cubic_limit_mib=stream.CUBIC_LIMIT // 1024**2)
    if actual != PROFILE or not stream.FUSED_QK or not stream.PRODUCT_AWARE or stream.PRODUCT_PANELS:
        raise ValueError('Parent tile, fusion or panel policy differs from the adopted profile')


def build(layer, spec=None, *, stream, interpret=False):
    """Return a tuner-compatible callable with prepare_weights and metadata.

Omitting spec selects exact parent S1. Extensions must be explicit; FP32 and
S2 are never silently substituted into a profile called parent_exact.
"""
    spec = exact_spec() if spec is None else spec
    if spec.get('backend') == 'extension':
        from .parent_fused_adapter_v001 import build as build_extension
        fn = build_extension(layer, spec['policy'], interpret=interpret)
        fn.metadata = dict(fn.metadata, backend='extension', family=spec['family'],
                           norm_dot_precision='HIGHEST', output_dtype=spec['output_dtype'])
        return fn
    if spec.get('backend') != 'parent_exact' or spec != exact_spec(spec.get('family')):
        raise ValueError('Invalid exact parent specification')
    if interpret:
        raise ValueError('Exact parent execution cannot override its kernel interpret flag')
    validate_parent(layer, stream)
    fn = stream.make_product_layer(ARMS[spec['family']])

    def prepare(params, cache=None):
        if not isinstance(params, stream.FusedQKStreamParameters):
            raise ValueError('Use the frozen parent stream.load_layer for weight/layout preparation')
        return params

    fn.prepare_weights = prepare
    fn.replication_compiler_options = dict(COMPILER_OPTIONS)
    fn.metadata = dict(version='parent_fused_adapter_v002', **spec,
                       norm_dot_precision='DEFAULT', consumer_dtype='bfloat16',
                       preparation='unchanged parent stream.load_layer; excluded from resident timing',
                       computation='original parent function; no arithmetic wrapper',
                       compiler_options=dict(COMPILER_OPTIONS))
    return fn
