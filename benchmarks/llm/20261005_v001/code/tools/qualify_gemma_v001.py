#!/usr/bin/env python3
"""Independent, offline Gemma 3 adapter qualification; never a timing benchmark.

Official oracle: Transformers 4.56.2 Gemma3ForCausalLM/TextModel, eager,
PyTorch CPU BF16, use_cache=False. No expected attention/MLP is reimplemented.
Sources: https://github.com/huggingface/transformers/blob/v4.56.2/src/transformers/models/gemma3/modeling_gemma3.py

CLI modes: tiny; reference --manifest M --tokens T --token-manifest TM;
compare --manifest M --reference-dir R. Every mode requires --output-dir NEW.
Token manifest: model_id, revision, tokens_sha256, token_count. Text/provenance
fields are retained verbatim. Tokens must contain at least max(529,window+17).
Reference saves full short logits and actual local/global long-prefix layer
inputs/outputs. Compare streams native JAX layers; it makes no speed claim.
Gates below are frozen before execution and are never adjusted to outputs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
DEFAULT_ADAPTER = "strassen_mm.model_gemma_v001"
GATES = {"relative_l2_max": 0.03, "mean_kl_max": 0.01,
         "absolute_nll_delta_max": 0.05, "top1_agreement_min": 0.90}
LAYER_GATES = {"relative_l2_max": 0.02}
PRIMITIVE_GATES = {"relative_l2_max": 0.01, "max_abs_error_max": 0.04}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    with Path(path).open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def save_array(path, data):
    import numpy as np
    with Path(path).open("xb") as f:
        np.save(f, np.asarray(data), allow_pickle=False)


def adapter_module(name=None):
    return importlib.import_module(name or os.environ.get("GEMMA_ADAPTER_MODULE", DEFAULT_ADAPTER))


def official_dependencies():
    import torch
    import transformers
    if transformers.__version__ != "4.56.2":
        raise ValueError("Oracle must be Transformers 4.56.2")
    if torch.__version__.split("+")[0] != "2.8.0":
        raise ValueError("Oracle must be PyTorch 2.8.0 CPU BF16")
    torch.set_num_threads(1)
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    return torch, Gemma3ForCausalLM, Gemma3TextConfig


def environment(adapter=None):
    packages = {}
    for name in ("jax", "jaxlib", "numpy", "ml_dtypes", "torch", "transformers", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"packages": packages, "python": sys.version,
              "harness_sha256": sha256(__file__), "performance_claim": False,
              "gates": GATES, "layer_gates": LAYER_GATES,
              "primitive_gates": PRIMITIVE_GATES}
    if adapter is not None:
        import jax
        result.update(adapter_module=adapter.__name__, adapter_sha256=sha256(adapter.__file__),
                      jax_devices=[str(x) for x in jax.devices()])
    try:
        from transformers.models.gemma3 import modeling_gemma3
        result["official_modeling_source_sha256"] = sha256(modeling_gemma3.__file__)
    except ImportError:
        pass
    return result


def metrics(reference, candidate, *, targets=None, primitive=False, layer=False):
    """Full-array FP64 host metrics; logits use all supplied scored positions."""
    import numpy as np
    if primitive and layer:
        raise ValueError("A metric check must use exactly one tolerance scope")
    r, c = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if r.shape != c.shape:
        return {"passed": False, "reason": "shape_mismatch", "reference_shape": list(r.shape),
                "candidate_shape": list(c.shape)}
    if not np.isfinite(r).all() or not np.isfinite(c).all():
        return {"passed": False, "reason": "nonfinite"}
    delta = c - r
    result = {"finite": True, "shape": list(r.shape),
              "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(r), 1e-30)),
              "max_abs_error": float(np.max(np.abs(delta)))}
    limits = PRIMITIVE_GATES if primitive else LAYER_GATES if layer else GATES
    result["passed"] = result["relative_l2"] <= limits["relative_l2_max"]
    if primitive:
        result["passed"] &= result["max_abs_error"] <= PRIMITIVE_GATES["max_abs_error_max"]
    if targets is not None:
        targets = np.asarray(targets, dtype=np.int64)
        if r.ndim != 2 or targets.shape != (r.shape[0],):
            raise ValueError("Logit targets must match every scored row")
        def log_softmax(x):
            y = x - x.max(axis=-1, keepdims=True)
            return y - np.log(np.exp(y).sum(axis=-1, keepdims=True))
        lp, lq = log_softmax(r), log_softmax(c)
        idx = np.arange(len(targets))
        nll_delta = float(np.mean(lp[idx, targets] - lq[idx, targets]))
        kl = float(np.mean(np.sum(np.exp(lp) * (lp - lq), axis=-1)))
        top1 = float(np.mean(np.argmax(r, axis=-1) == np.argmax(c, axis=-1)))
        result.update(positions=len(targets), mean_kl=kl, nll_delta=nll_delta,
                      top1_agreement=top1)
        result["passed"] &= (kl <= GATES["mean_kl_max"] and
                             abs(nll_delta) <= GATES["absolute_nll_delta_max"] and
                             top1 >= GATES["top1_agreement_min"])
    result["passed"] = bool(result["passed"])
    return result


def manifest_identity(manifest):
    return {key: manifest[key] for key in ("model_id", "revision", "config", "files")}


def verify_manifest(path):
    """Independent authentication before either framework loads weights."""
    data = json.loads(Path(path).read_text())
    revision = data["revision"]
    if len(revision) != 40 or any(x not in "0123456789abcdef" for x in revision):
        raise ValueError("Frozen model revision must be 40 lowercase hex characters")
    if data["config"]["model_type"] != "gemma3_text":
        raise ValueError("Qualification supports text-only Gemma3, not multimodal Gemma3")
    root = Path(data["cache_dir"]).resolve()
    seen = set()
    for row in data["files"]:
        relative = row["path"]
        if relative in seen:
            raise ValueError("Duplicate checkpoint manifest path")
        seen.add(relative)
        file = (root / relative).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError("Manifest path escapes checkpoint or is missing")
        if file.stat().st_size != row["bytes"] or sha256(file) != row["sha256"]:
            raise ValueError("Checkpoint hash/size mismatch: " + relative)
    if "config.json" not in seen or json.loads((root / "config.json").read_text()) != data["config"]:
        raise ValueError("Config missing or different from frozen manifest")
    # Loading may consult an index. Reject unmanifested weight/index files.
    for file in root.glob("*"):
        if file.name.endswith((".safetensors", ".safetensors.index.json")) and file.name not in seen:
            raise ValueError("Unmanifested checkpoint weight/index file: " + file.name)
    if not any(name.endswith(".safetensors") for name in seen):
        raise ValueError("No safetensors checkpoint files")
    return data


def verify_tokens(tokens_path, manifest_path, model_manifest):
    import numpy as np
    info = json.loads(Path(manifest_path).read_text())
    if info["tokens_sha256"] != sha256(tokens_path):
        raise ValueError("Token hash mismatch")
    for key in ("model_id", "revision"):
        if info[key] != model_manifest[key]:
            raise ValueError("Token/model identity mismatch: " + key)
    tokens = np.load(tokens_path, allow_pickle=False)
    if tokens.ndim != 1 or tokens.dtype.kind not in "iu" or len(tokens) != info["token_count"]:
        raise ValueError("Token array must match declared 1-D integer count")
    if not len(tokens) or tokens.min() < 0 or tokens.max() >= model_manifest["config"]["vocab_size"]:
        raise ValueError("Out-of-vocabulary token ID")
    return tokens.astype(np.int64), info


def capture_official(model, tokens, *, selected=None, stop_after=None):
    """Run the actual official forward; optional hook stops a prefix after capture."""
    import torch
    selected = list(range(len(model.model.layers))) if selected is None else list(selected)
    records, handles = {}, []
    def host(x):
        return x.detach().float().cpu().numpy().copy()
    class PrefixComplete(Exception):
        pass
    for index in selected:
        def pre(module, args, index=index):
            records[f"layer_{index:03d}_input"] = host(args[0][0])
        def post(module, args, output, index=index):
            value = output[0] if isinstance(output, tuple) else output
            records[f"layer_{index:03d}_output"] = host(value[0])
            if index == stop_after:
                raise PrefixComplete()
        handles.append(model.model.layers[index].register_forward_pre_hook(pre))
        handles.append(model.model.layers[index].register_forward_hook(post))
    try:
        with torch.no_grad():
            ids = torch.as_tensor(tokens, dtype=torch.long).unsqueeze(0)
            records["embeddings"] = host(model.model.embed_tokens(ids)[0])
            if stop_after is None:
                result = model(ids, use_cache=False, output_hidden_states=True, return_dict=True)
                records["logits"] = host(result.logits[0])
                records["final_hidden"] = host(result.hidden_states[-1][0])
            else:
                try:
                    model.model(ids, use_cache=False, output_hidden_states=False, return_dict=True)
                except PrefixComplete:
                    pass
                else:
                    raise RuntimeError("Official long-prefix stop hook was not reached")
    finally:
        for handle in handles:
            handle.remove()
    for index in selected:
        if f"layer_{index:03d}_output" not in records:
            raise ValueError("Missing official layer capture")
    import numpy as np
    if any(not np.isfinite(value).all() for value in records.values()):
        raise ValueError("Official reference produced nonfinite values")
    return records


def compare_adapter(checkpoint, records, tokens, adapter, *, custom=False):
    import jax
    import jax.numpy as jnp
    import numpy as np
    config = checkpoint.config
    checks = {}
    x = jnp.asarray(checkpoint.embeddings(tokens))
    checks["scaled_embeddings"] = metrics(records["embeddings"], x, primitive=True)
    native = adapter.native_policy()
    for index in range(config["num_hidden_layers"]):
        w = jax.tree_util.tree_map(jnp.asarray, checkpoint.layer(index))
        fn = adapter.build_layer(config, len(tokens), native, layer_index=index)
        expected = records[f"layer_{index:03d}_output"]
        supplied = jnp.asarray(records[f"layer_{index:03d}_input"], dtype=jnp.bfloat16)
        checks[f"layer_{index:03d}_common_input"] = metrics(expected, jax.block_until_ready(fn(supplied, w)), layer=True)
        x = jax.block_until_ready(fn(x, w))
        checks[f"layer_{index:03d}_propagated"] = metrics(expected, x, layer=True)
        if custom and index in (0, 1):
            for algorithm in ("cubic_full", "strassen"):
                policy = {site: {"algorithm": algorithm, "variant": "plain",
                                  "tile": [32, 256, 256], "fused": False,
                                  "early": False, "packed": False} for site in ("gateup", "down")}
                cfn = adapter.build_layer(config, len(tokens), policy, layer_index=index, interpret=True)
                got = jax.block_until_ready(cfn(supplied, w))
                checks[f"layer_{index:03d}_{algorithm}_interpret"] = metrics(expected, got, layer=True)
                del cfn, got
        del w, fn, supplied
        gc.collect()
    norm = jnp.asarray(checkpoint.tensor("model.norm.weight"))
    checks["final_hidden"] = metrics(records["final_hidden"], adapter.rms(x, norm, config["rms_norm_eps"]), layer=True)
    # Vocab chunks bound device memory; independent vocabulary columns do not
    # alter the model. Retain the complete short output, not sampled logits.
    weight = checkpoint.head()
    pieces = []
    for start in range(0, weight.shape[1], 4096):
        block = jnp.asarray(np.array(weight[:, start:start+4096], copy=True))
        pieces.append(np.asarray(jax.block_until_ready(adapter.head_logits(x, norm, block, config)), dtype=np.float32))
        del block
    logits = np.concatenate(pieces, axis=1)
    checks["full_logits"] = metrics(records["logits"], logits)
    checks["next_token_logits"] = metrics(records["logits"][:-1], logits[:-1], targets=tokens[1:])
    return checks, logits


def primitive_checks(adapter):
    """Oracle computations use official modules or PyTorch's official GELU."""
    torch, _, _ = official_dependencies()
    from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm, Gemma3TextScaledWordEmbedding
    import jax.numpy as jnp
    import numpy as np
    rng = np.random.default_rng(20260920)
    x = torch.tensor(rng.normal(size=(19, 32)), dtype=torch.bfloat16)
    norm = Gemma3RMSNorm(32, eps=1e-6).to(dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor(np.linspace(-0.8, 0.9, 32), dtype=torch.bfloat16))
        expected = norm(x).float().numpy()
    checks = {"rms_nonzero_offset_weights": metrics(expected, adapter.rms(jnp.asarray(x.float().numpy(), dtype=jnp.bfloat16),
              jnp.asarray(norm.weight.detach().float().numpy(), dtype=jnp.bfloat16), 1e-6), primitive=True)}
    gx = torch.linspace(-7, 7, 257).to(torch.bfloat16)
    expected = torch.nn.functional.gelu(gx, approximate="tanh").float().numpy()
    checks["gelu_tanh"] = metrics(expected, adapter.gelu(jnp.asarray(gx.float().numpy(), dtype=jnp.bfloat16)), primitive=True)
    emb = Gemma3TextScaledWordEmbedding(7, 1152, None, embed_scale=1152**0.5).to(dtype=torch.bfloat16)
    with torch.no_grad():
        emb.weight.copy_(torch.tensor(rng.normal(size=(7, 1152)), dtype=torch.bfloat16))
        expected = emb(torch.tensor([1, 4, 6])).float().numpy()
    raw = emb.weight.detach().float().numpy()[[1, 4, 6]]
    checks["embedding_scale_1152"] = metrics(expected, adapter.scaled_embeddings(jnp.asarray(raw, dtype=jnp.bfloat16), 1152), primitive=True)
    local = np.asarray(adapter.attention_mask(529, 512))
    full = np.asarray(adapter.attention_mask(529, None))
    mask_pass = (local.shape == (529, 529) and full.shape == (529, 529)
                 and local[511, 0] and not local[512, 0] and local[512, 1]
                 and not local[511, 512] and full[528, 0] and not full[0, 1])
    checks["local_global_boundary_mask"] = {"passed": bool(mask_pass)}
    return checks


def tiny_qualification(adapter_name=None, *, long_window=False, final_softcap=None, custom=False):
    """Deterministic independent whole-model oracle, no remote weights."""
    torch, Model, Config = official_dependencies()
    import numpy as np
    adapter = adapter_module(adapter_name)
    torch.manual_seed(20260920)
    window, length = (512, 529) if long_window else (8, 33)
    types = (["sliding_attention"] * 5 + ["full_attention"] if long_window else
             ["sliding_attention", "full_attention", "sliding_attention", "full_attention"])
    # The custom case crosses multiple M/K panels and has K/N tails at the
    # legal (32,256,256) tile. The long attention case stays narrow on CPU.
    hidden, intermediate = (288, 320) if custom else (32, 64)
    config = Config(vocab_size=97, hidden_size=hidden, intermediate_size=intermediate,
        num_hidden_layers=len(types), num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, hidden_activation="gelu_pytorch_tanh", max_position_embeddings=2048,
        rms_norm_eps=1e-6, query_pre_attn_scalar=12, sliding_window=window,
        layer_types=types, rope_theta=10000.0, rope_local_base_freq=100.0,
        rope_scaling=None,
        final_logit_softcapping=final_softcap, attn_logit_softcapping=None,
        attention_bias=False, attention_dropout=0.0, tie_word_embeddings=True,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False)
    config._attn_implementation = "eager"
    model = Model(config).to(dtype=torch.bfloat16, device="cpu").eval()
    with torch.no_grad():
        # Nonzero offsets are necessary to distinguish (1+w) from w and expose
        # different normalization placement; zero-initialized norms can hide it.
        for name, parameter in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.copy_(torch.linspace(-0.35, 0.40, parameter.numel(), dtype=torch.float32).reshape(parameter.shape).to(torch.bfloat16))
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise AssertionError("Official tiny head is not tied")
    tokens = np.random.default_rng(20260920).integers(3, 97, length, dtype=np.int64)
    tokens[0] = 1
    records = capture_official(model, tokens)
    with tempfile.TemporaryDirectory(prefix="gemma3-qualification-") as directory:
        root = Path(directory)
        model.save_pretrained(root, safe_serialization=True)
        manifest = {"model_id": "local/tiny-gemma3-random", "revision": "0" * 40,
                    "cache_dir": str(root), "config": json.loads((root / "config.json").read_text()),
                    "files": [{"path": f.name, "bytes": f.stat().st_size, "sha256": sha256(f)}
                              for f in sorted(root.iterdir()) if f.is_file()]}
        path = root / "manifest.json"
        write_json(path, manifest)
        verify_manifest(path)
        checkpoint = adapter.Checkpoint(path)
        checks, _ = compare_adapter(checkpoint, records, tokens, adapter, custom=custom)
    return {"scope": "tiny random official whole-model equivalence; not model quality or performance",
            "sequence_length": length, "sliding_window": window, "layer_types": types,
            "hidden_size": hidden, "intermediate_size": intermediate,
            "final_logit_softcapping": final_softcap, "custom_interpret": custom,
            "checks": checks, "passed": all(x["passed"] for x in checks.values()),
            "environment": environment(adapter)}


def reference_run(args, output):
    import numpy as np
    manifest = verify_manifest(args.manifest)
    tokens, token_info = verify_tokens(args.tokens, args.token_manifest, manifest)
    if ("model_manifest_sha256" in token_info and
            token_info["model_manifest_sha256"] != sha256(args.manifest)):
        raise ValueError("Token manifest refers to a different model manifest file")
    torch, Model, _ = official_dependencies()
    model = Model.from_pretrained(manifest["cache_dir"], local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager", use_safetensors=True).to(device="cpu").eval()
    if any(p.dtype != torch.bfloat16 or p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Official model parameters must all be CPU BF16")
    if model.config.model_type != "gemma3_text" or model.config._attn_implementation != "eager":
        raise ValueError("Official model implementation contract changed")
    short_length = 64
    long_length = max(529, int(model.config.sliding_window) + 17)
    if len(tokens) < long_length:
        raise ValueError("Insufficient frozen tokens for long sliding-window qualification")
    layer_types = model.config.layer_types
    global_index = next(i for i, value in enumerate(layer_types) if value == "full_attention")
    local_index = next(i for i in range(global_index - 1, -1, -1) if layer_types[i] == "sliding_attention")
    short = capture_official(model, tokens[:short_length])
    long = capture_official(model, tokens[:long_length], selected=[local_index, global_index], stop_after=global_index)
    for prefix, values in (("short", short), ("long", long)):
        for name, value in values.items():
            save_array(output / f"{prefix}_{name}.npy", value)
    save_array(output / "tokens.npy", tokens)
    write_json(output / "model_manifest.json", manifest)
    write_json(output / "token_manifest.json", token_info)
    report = {"mode": "reference", "passed": True, "environment": environment(),
              "model_identity": manifest_identity(manifest), "model_manifest_sha256": sha256(args.manifest),
              "source_tokens_sha256": sha256(args.tokens), "token_manifest_sha256": sha256(args.token_manifest),
              "short_length": short_length, "long_length": long_length,
              "long_layer_indices": [local_index, global_index],
              "scope": "Official full short-model logits and real long-prefix local/global layer input/output; no long full-model logits or performance claims"}
    return report


def verify_seal(directory):
    directory = Path(directory)
    seal = json.loads((directory / "artifact_manifest.json").read_text())
    for name, digest in seal["sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != digest:
            raise ValueError("Reference artifact seal mismatch: " + name)
    required = {"summary.json", "model_manifest.json", "token_manifest.json", "tokens.npy"}
    if not required.issubset(seal["sha256"]):
        raise ValueError("Incomplete reference artifact seal")
    return seal


def compare_run(args, output):
    import jax
    import jax.numpy as jnp
    import numpy as np
    adapter = adapter_module(args.adapter_module)
    manifest = verify_manifest(args.manifest)
    directory = Path(args.reference_dir)
    seal = verify_seal(directory)
    reference = json.loads((directory / "summary.json").read_text())
    if reference.get("mode") != "reference" or not reference.get("passed"):
        raise ValueError("Reference run was not completed successfully")
    if manifest_identity(manifest) != reference["model_identity"] or reference["environment"]["gates"] != GATES:
        raise ValueError("Frozen reference model or qualification gate mismatch")
    if reference["environment"]["primitive_gates"] != PRIMITIVE_GATES:
        raise ValueError("Frozen primitive qualification gate mismatch")
    if reference["environment"]["layer_gates"] != LAYER_GATES:
        raise ValueError("Frozen layer qualification gate mismatch")
    if manifest_identity(json.loads((directory / "model_manifest.json").read_text())) != reference["model_identity"]:
        raise ValueError("Reference manifest and summary model identities differ")
    checkpoint = adapter.Checkpoint(args.manifest)
    tokens = np.load(directory / "tokens.npy", allow_pickle=False)
    token_info = json.loads((directory / "token_manifest.json").read_text())
    if (tokens.ndim != 1 or tokens.dtype.kind not in "iu" or
            len(tokens) != token_info["token_count"] or len(tokens) < reference["long_length"] or
            tokens.min() < 0 or tokens.max() >= manifest["config"]["vocab_size"] or
            any(token_info[key] != manifest[key] for key in ("model_id", "revision")) or
            token_info["tokens_sha256"] != reference["source_tokens_sha256"]):
        raise ValueError("Reference frozen token identity/count/content contract changed")
    def records(prefix):
        return {p.stem[len(prefix)+1:]: np.load(p, allow_pickle=False)
                for p in sorted(directory.glob(prefix + "_*.npy"))
                if p.name in seal["sha256"]}
    short, long = records("short"), records("long")
    checks, logits = compare_adapter(checkpoint, short, tokens[:reference["short_length"]], adapter)
    for index in reference["long_layer_indices"]:
        x = jnp.asarray(long[f"layer_{index:03d}_input"], dtype=jnp.bfloat16)
        weights = jax.tree_util.tree_map(jnp.asarray, checkpoint.layer(index))
        fn = adapter.build_layer(checkpoint.config, reference["long_length"], adapter.native_policy(), layer_index=index)
        candidate = np.asarray(jax.block_until_ready(fn(x, weights)), dtype=np.float32)
        checks[f"long_layer_{index:03d}"] = metrics(long[f"layer_{index:03d}_output"], candidate, layer=True)
        save_array(output / f"long_layer_{index:03d}_candidate.npy", candidate)
        del weights, fn, x
        gc.collect()
    save_array(output / "short_candidate_logits.npy", logits)
    return {"mode": "compare", "passed": all(x["passed"] for x in checks.values()),
            "checks": checks, "environment": environment(adapter),
            "reference_seal_sha256": sha256(directory / "artifact_manifest.json"),
            "model_identity": manifest_identity(manifest), "current_cache_dir": manifest["cache_dir"],
            "reference_cache_dir": json.loads((directory / "model_manifest.json").read_text())["cache_dir"],
            "scope": reference["scope"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tiny", "reference", "compare"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-module", default=os.environ.get("GEMMA_ADAPTER_MODULE", DEFAULT_ADAPTER))
    parser.add_argument("--manifest")
    parser.add_argument("--tokens")
    parser.add_argument("--token-manifest")
    parser.add_argument("--reference-dir")
    args = parser.parse_args()
    if args.mode == "reference" and not all((args.manifest, args.tokens, args.token_manifest)):
        parser.error("reference requires --manifest, --tokens and --token-manifest")
    if args.mode == "compare" and not all((args.manifest, args.reference_dir)):
        parser.error("compare requires --manifest and --reference-dir")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "arguments.json", vars(args))
    shutil.copyfile(__file__, output / Path(__file__).name)
    try:
        if args.mode == "tiny":
            adapter = adapter_module(args.adapter_module)
            primitives = primitive_checks(adapter)
            cases = [tiny_qualification(args.adapter_module, custom=True),
                     tiny_qualification(args.adapter_module, long_window=True, final_softcap=3.0)]
            result = {"mode": "tiny", "primitives": primitives, "cases": cases,
                      "passed": all(x["passed"] for x in primitives.values()) and all(x["passed"] for x in cases)}
        elif args.mode == "reference":
            result = reference_run(args, output)
        else:
            result = compare_run(args, output)
    except Exception as error:
        result = {"mode": args.mode, "passed": False, "error": str(error), "traceback": traceback.format_exc()}
    write_json(output / "summary.json", result)
    write_json(output / "artifact_manifest.json", {"schema_version": 1, "sha256": {
        p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(json.dumps({"passed": result["passed"], "output_dir": str(output)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
