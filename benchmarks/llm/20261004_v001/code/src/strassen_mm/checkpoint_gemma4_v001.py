"""Strict, read-only Gemma4 text checkpoint reader; no acquisition or device work.

Verify the pinned manifest and all file hashes before memory mapping. Parse all
payload bounds, including unused multimodal tensors. Only explicitly recognized
text prefixes enter the canonical state. Required parameters are BF16; persistent
layer scalars may retain their stored BF16 or FP32 dtype.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import ml_dtypes
import numpy as np
from .gemma4_contract_v001 import validate_config, required_tensor_shapes

VERSION = "checkpoint_gemma4_v001"
WIDTHS = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2,
          "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key: " + key)
        value[key] = item
    return value


def canonical_name(name):
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    if name.startswith("language_model."):
        suffix = name[len("language_model."):]
        return suffix if suffix.startswith("model.") or suffix == "lm_head.weight" else "model." + suffix
    if name.startswith("model.layers.") or name.startswith("model.embed_tokens.") or name.startswith("model.norm.") or name == "lm_head.weight":
        return name
    return None


class Checkpoint:
    def __init__(self, manifest_path):
        if sys.byteorder != "little":
            raise ValueError("BF16 memory mapping requires a little-endian host")
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(), object_pairs_hook=unique_object)
        if not re.fullmatch(r"[0-9a-f]{40}", self.manifest.get("revision", "")):
            raise ValueError("Checkpoint revision must be immutable 40-hex")
        self.root = Path(self.manifest["cache_dir"]).resolve()
        self.raw_config = self.manifest["config"]
        self.config = self.normalized_config = validate_config(self.raw_config)
        self.locations, self.verified = {}, []
        files, config_verified = set(), False
        for item in self.manifest["files"]:
            relative = item["path"]
            path = (self.root / relative).resolve()
            if Path(relative).is_absolute() or not path.is_relative_to(self.root) or path in files:
                raise ValueError("Duplicate or escaping checkpoint file")
            files.add(path)
            if type(item["bytes"]) is not int or item["bytes"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                raise ValueError("Invalid checkpoint file manifest")
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError("Checkpoint file size/hash mismatch")
            if relative == "config.json":
                if json.loads(path.read_text(), object_pairs_hook=unique_object) != self.raw_config:
                    raise ValueError("Manifest config differs from verified config.json")
                config_verified = True
            if path.suffix == ".safetensors":
                self._index(path)
            self.verified.append(dict(item))
        if not config_verified:
            raise ValueError("Verified config.json is required")
        expected = required_tensor_shapes(self.raw_config)
        if "lm_head.weight" in self.locations:
            expected["lm_head.weight"] = expected["model.embed_tokens.weight"]
        if set(expected) != set(self.locations):
            raise ValueError("Missing or unexpected Gemma4 text tensors: " + str(sorted(set(expected) ^ set(self.locations))))
        for name, shape in expected.items():
            _, _, actual_shape, dtype = self.locations[name]
            if shape != actual_shape:
                raise ValueError("Incorrect tensor shape: " + name)
            allowed = ("BF16", "F32") if name.endswith(".layer_scalar") else ("BF16",)
            if dtype not in allowed:
                raise ValueError("Incorrect tensor dtype: " + name)
        if "lm_head.weight" in self.locations:
            embedding, head = self.tensor("model.embed_tokens.weight"), self.tensor("lm_head.weight")
            for start in range(0, len(head), 1024):
                if not np.array_equal(head[start:start+1024].view(np.uint16), embedding[start:start+1024].view(np.uint16)):
                    raise ValueError("Tied output head differs from token embeddings")

    def _index(self, path):
        size = path.stat().st_size
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError("Truncated safetensors length")
            length = struct.unpack("<Q", raw)[0]
            if length > 64 * 1024**2 or 8 + length > size:
                raise ValueError("Invalid safetensors header size")
            header = json.loads(f.read(length), object_pairs_hook=unique_object)
        intervals = []
        for original, item in header.items():
            if original == "__metadata__":
                continue
            shape, dtype, bounds = item["shape"], item["dtype"], item["data_offsets"]
            if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape) or dtype not in WIDTHS:
                raise ValueError("Invalid tensor shape/dtype metadata")
            if not isinstance(bounds, list) or len(bounds) != 2 or any(type(v) is not int for v in bounds):
                raise ValueError("Invalid tensor offsets")
            lo, hi = bounds
            if lo < 0 or hi-lo != math.prod(shape)*WIDTHS[dtype] or 8+length+hi > size:
                raise ValueError("Invalid tensor byte bounds")
            intervals.append((lo, hi))
            name = canonical_name(original)
            if name is not None:
                if name in self.locations:
                    raise ValueError("Duplicate canonical text tensor: " + name)
                self.locations[name] = (path, 8+length+lo, shape, dtype)
        cursor = 0
        for lo, hi in sorted(intervals):
            if lo != cursor:
                raise ValueError("Overlapping or unaccounted safetensors payload")
            cursor = hi
        if 8+length+cursor != size:
            raise ValueError("Unaccounted trailing safetensors bytes")

    def tensor(self, name):
        path, offset, shape, dtype = self.locations[name]
        if dtype == "BF16":
            return np.memmap(path, mode="r", offset=offset, shape=tuple(shape), dtype="<u2").view(ml_dtypes.bfloat16)
        if dtype == "F32":
            return np.memmap(path, mode="r", offset=offset, shape=tuple(shape), dtype="<f4")
        raise ValueError("Unsupported mapped text dtype")

    def layer(self, index):
        if type(index) is not int or not 0 <= index < self.config["num_hidden_layers"]:
            raise ValueError("Layer index out of range")
        prefix = f"model.layers.{index}."
        result = {}
        for site in ("q", "k", "v", "o"):
            name = prefix + f"self_attn.{site}_proj.weight"
            if name in self.locations:
                result[site] = np.ascontiguousarray(self.tensor(name).T)
        result["gateup"] = np.ascontiguousarray(np.concatenate((self.tensor(prefix + "mlp.gate_proj.weight"),
                                                              self.tensor(prefix + "mlp.up_proj.weight")), axis=0).T)
        result["down"] = np.ascontiguousarray(self.tensor(prefix + "mlp.down_proj.weight").T)
        for key, suffix in (("norm1", "input_layernorm.weight"), ("norm2", "post_attention_layernorm.weight"),
                            ("norm3", "pre_feedforward_layernorm.weight"), ("norm4", "post_feedforward_layernorm.weight"),
                            ("qnorm", "self_attn.q_norm.weight"), ("knorm", "self_attn.k_norm.weight"),
                            ("layer_scalar", "layer_scalar")):
            result[key] = np.array(self.tensor(prefix + suffix), copy=True)
        return result

    def embeddings(self, tokens):
        tokens = np.asarray(tokens)
        if tokens.dtype.kind not in "iu" or np.any(tokens < 0) or np.any(tokens >= self.config["vocab_size"]):
            raise ValueError("Invalid token IDs")
        value = np.array(self.tensor("model.embed_tokens.weight")[tokens], copy=True)
        scale = np.asarray(math.sqrt(self.config["hidden_size"]), dtype=ml_dtypes.bfloat16)
        return (value.astype(np.float32) * scale.astype(np.float32)).astype(ml_dtypes.bfloat16)

    def head(self):
        return np.ascontiguousarray(self.tensor("model.embed_tokens.weight").T)
