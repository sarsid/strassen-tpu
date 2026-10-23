"""Local official Gemma4 oracle export and source-bound input-cache helpers.

No network access, model allocation on a TPU, queue mutation or checkpoint
acquisition. The caller must verify checkpoint bytes before official loading.
An exported oracle is reference data, not a passed JAX qualification gate.
"""
import hashlib
import json
from pathlib import Path

VERSION = 'gemma4_oracle_inputs_v001'
CORPUS_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
OFFICIAL_SOURCES = {
    'models/gemma4/modeling_gemma4.py': '6a86e03348df5ec104703e7161de9a911137cba500a0be0f133e2850ef0bf935',
    'models/gemma4/configuration_gemma4.py': '3eb1d90bffeb0caf9bd51ab5007bac8696e54109a8f43ea8919cdd83319a1401',
    'modeling_rope_utils.py': 'd8c3c0696a10e8041f31037116e35289d66c1629b8d134d5ca9629c3ac115c3c',
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def source_identity():
    import torch
    import transformers
    if torch.__version__.split('+')[0] != '2.8.0' or transformers.__version__ != '5.5.0':
        raise ValueError('Pinned Gemma4 official oracle versions are required')
    root = Path(transformers.__file__).parent
    actual = {name: digest(root/name) for name in OFFICIAL_SOURCES}
    if actual != OFFICIAL_SOURCES:
        raise ValueError('Official Gemma4 source bytes differ from the qualified snapshot')
    return dict(version=VERSION, torch=torch.__version__, transformers=transformers.__version__,
        official_source_sha256=actual, loader_sha256=digest(Path(__file__).with_name('load_gemma4_text_v002.py')),
        input_helper_sha256=digest(__file__), corpus_revision=CORPUS_REVISION)


def token_window(tokens, batch, sequence):
    import numpy as np
    tokens = np.asarray(tokens)
    if type(batch) is not int or batch <= 0 or type(sequence) is not int or sequence < 2:
        raise ValueError('Positive batch and sequence of at least two required')
    if tokens.ndim != 1 or tokens.dtype != np.int32 or np.any(tokens < 0):
        raise ValueError('Expected one-dimensional nonnegative int32 corpus tokens')
    if len(tokens) < batch*sequence:
        raise ValueError('Insufficient tokens; no repetition allowed')
    return tokens[:batch*sequence].reshape(batch, sequence).copy()


def export_oracle(verified_checkpoint, ids, output, emit=lambda row: None):
    """Export all official layer inputs/outputs and complete vocabulary logits.

    IDs must be a single 2--64-token window. This helper intentionally never
    labels synthetic or real inputs as a passed checkpoint qualification.
    """
    import numpy as np
    import torch
    from load_gemma4_text_v002 import load_text
    identity = source_identity()
    ids = np.asarray(ids)
    if ids.dtype != np.int32 or ids.ndim != 2 or ids.shape[0] != 1 or not 2 <= ids.shape[1] <= 64 or np.any(ids < 0):
        raise ValueError('Expected one int32 oracle sequence of 2 to 64 tokens')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(min(8, __import__('os').cpu_count() or 1))
    model = load_text(verified_checkpoint)
    if np.any(ids >= model.config.vocab_size):
        raise ValueError('Oracle token outside vocabulary')
    handles = []
    for index, layer in enumerate(model.model.layers):
        def before(module, args, kwargs, index=index):
            value = args[0] if args else kwargs['hidden_states']
            np.save(output/f'layer-{index:03d}-input.npy', value.detach().float().numpy())
        def after(module, args, value, index=index):
            value = value[0] if isinstance(value, tuple) else value
            np.save(output/f'layer-{index:03d}-output.npy', value.detach().float().numpy())
            emit(dict(kind='official_oracle_layer', layer=index))
        handles.extend((layer.register_forward_pre_hook(before, with_kwargs=True), layer.register_forward_hook(after)))
    try:
        with torch.inference_mode():
            logits = model(torch.tensor(ids, dtype=torch.long), use_cache=False).logits.float().numpy()
    finally:
        for handle in handles:
            handle.remove()
    if logits.shape != (1, ids.shape[1], model.config.vocab_size) or not np.isfinite(logits).all():
        raise ValueError('Invalid official oracle logits')
    np.save(output/'logits.npy', logits)
    expected = {'logits.npy'} | {f'layer-{i:03d}-{side}.npy' for i in range(len(model.model.layers)) for side in ('input','output')}
    if {p.name for p in output.iterdir()} != expected:
        raise ValueError('Incomplete official oracle layer trace')
    for name in expected - {'logits.npy'}:
        value = np.load(output/name, allow_pickle=False)
        if value.shape != (1, ids.shape[1], model.config.hidden_size) or not np.isfinite(value).all():
            raise ValueError('Invalid official layer trace')
    return dict(status='reference_exported', source=identity, tokens=int(ids.size), batch=1,
        layers=len(model.model.layers), token_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        artifacts={name: dict(bytes=(output/name).stat().st_size, sha256=digest(output/name)) for name in sorted(expected)})


def cache_receipt(folder, model, identity):
    """Publish a complete local template receipt, including full corpus arrays."""
    folder = Path(folder).resolve()
    path = folder/'cache-receipt.json'
    if path.exists():
        raise ValueError('Do not overwrite a published input cache')
    required = {'checkpoint-provenance.json','oracle-tokens.npy','summary.json',
                'train-all.npy','test-all.npy','train-tokens.json','test-tokens.json','official/logits.npy'}
    files = {}
    for item in folder.rglob('*'):
        if not item.is_file():
            continue
        if item.is_symlink() or not item.resolve().is_relative_to(folder):
            raise ValueError('Input cache files must not escape the template')
        name = item.relative_to(folder).as_posix()
        files[name] = dict(bytes=item.stat().st_size, sha256=digest(item))
    if not required <= set(files):
        raise ValueError('Incomplete input cache template')
    record = dict(version=VERSION, model=model, source=identity, files=files)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2)+'\n')
    temp.replace(path)
    return record


def verify_cache(folder, model, identity):
    folder = Path(folder).resolve()
    record = json.loads((folder/'cache-receipt.json').read_text())
    if record['version'] != VERSION or record['model'] != model or record['source'] != identity:
        raise ValueError('Model or source identity differs from the input cache')
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name != 'cache-receipt.json'}
    if actual != set(record['files']):
        raise ValueError('Input cache file set changed')
    for name, entry in record['files'].items():
        path = folder/name
        if Path(name).is_absolute() or path.is_symlink() or not path.resolve().is_relative_to(folder):
            raise ValueError('Input cache path escapes the template')
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Input cache checksum mismatch')
    return record


def reuse_inputs(folder, output, model, identity, batch, sequence):
    """Validate frozen reference data and materialize a new exact token window."""
    import shutil
    import numpy as np
    folder, output = Path(folder), Path(output)
    verify_cache(folder, model, identity)
    windows = {split: token_window(np.load(folder/(split+'-all.npy'), allow_pickle=False), batch, sequence)
               for split in ('train','test')}
    output.mkdir(parents=True, exist_ok=False)
    for name in ('checkpoint-provenance.json','oracle-tokens.npy'):
        shutil.copyfile(folder/name, output/name)
    shutil.copytree(folder/'official', output/'official')
    for split, tokens in windows.items():
        np.save(output/(split+'-tokens.npy'), tokens)
        meta = json.loads((folder/(split+'-tokens.json')).read_text())
        meta.update(shape=list(tokens.shape), token_sha256=hashlib.sha256(tokens.tobytes()).hexdigest())
        (output/(split+'-tokens.json')).write_text(json.dumps(meta, indent=2)+'\n')
    summary = json.loads((folder/'summary.json').read_text())
    summary.update(batch=batch, sequence=sequence, reused_model_inputs=True,
        cache_receipt_sha256=digest(folder/'cache-receipt.json'), input_source=identity,
        artifacts={p.relative_to(output).as_posix(): dict(bytes=p.stat().st_size, sha256=digest(p))
                   for p in sorted(output.rglob('*')) if p.is_file()})
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary
