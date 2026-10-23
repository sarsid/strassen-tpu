"""Independent official Gemma4 5.5.0 numerical tests; synthetic CPU evidence.

No checkpoint download, TPU timing, performance claim, or queue gate mutation.
Layer relative L2 <= 2%; logits <= 3%, plus campaign predictive gates.
"""
import hashlib
import json
import os
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import torch
import transformers
from transformers import Gemma4ForCausalLM, Gemma4TextConfig
from transformers.models.gemma4 import modeling_gemma4 as official
from strassen_mm import model_gemma4_v001 as adapter


def array(x):
    return jnp.asarray(x.detach().float().numpy(), dtype=jnp.bfloat16)


def metric(reference, candidate):
    r, c = np.asarray(reference, np.float64), np.asarray(candidate, np.float64)
    return dict(finite=bool(np.isfinite(c).all()), relative_l2=float(np.linalg.norm(c-r) / max(np.linalg.norm(r), 1e-30)),
                max_abs=float(np.max(np.abs(c-r))))


def layer_weights(layer):
    result = {}
    for site in ("q", "k", "v", "o"):
        projection = getattr(layer.self_attn, site + "_proj")
        if projection is not None:
            result[site] = array(projection.weight.T)
    result["gateup"] = array(torch.cat((layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight)).T)
    result["down"] = array(layer.mlp.down_proj.weight.T)
    for key, name in (("norm1", "input_layernorm"), ("norm2", "post_attention_layernorm"),
                      ("norm3", "pre_feedforward_layernorm"), ("norm4", "post_feedforward_layernorm")):
        result[key] = array(getattr(layer, name).weight)
    result.update(qnorm=array(layer.self_attn.q_norm.weight), knorm=array(layer.self_attn.k_norm.weight),
                  layer_scalar=array(layer.layer_scalar))
    return result


def tiny_config(window):
    c = Gemma4TextConfig(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
        num_global_key_value_heads=1, attention_k_eq_v=True, hidden_size_per_layer_input=0,
        layer_types=["sliding_attention", "full_attention"], sliding_window=window,
        max_position_embeddings=8192, final_logit_softcapping=3.0, dtype="bfloat16")
    c._attn_implementation = "eager"
    return c


class OfficialGemma4Native(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert transformers.__version__ == "5.5.0" and torch.__version__.split("+")[0] == "2.8.0"
        assert hashlib.sha256(Path(official.__file__).read_bytes()).hexdigest() == "6a86e03348df5ec104703e7161de9a911137cba500a0be0f133e2850ef0bf935"
        torch.set_num_threads(1)
        cls.results = {}

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("STRASSEN_EXECUTION_DIR"):
            out = Path(os.environ["STRASSEN_EXECUTION_DIR"]) / "artifacts"
            out.mkdir(exist_ok=True)
            (out / "gemma4-native-metrics.json").write_text(json.dumps(cls.results, indent=2) + "\n")

    def compare(self, name, reference, candidate, tolerance):
        reference = reference.detach().float().numpy() if isinstance(reference, torch.Tensor) else reference
        value = metric(reference, candidate)
        self.results[name] = dict(value, relative_l2_limit=tolerance)
        self.assertTrue(value["finite"], name)
        self.assertLessEqual(value["relative_l2"], tolerance, name + ": " + str(value))

    def test_primitives_and_proportional_rotary_layout(self):
        torch.manual_seed(17)
        config = tiny_config(1024)
        config.head_dim, config.global_head_dim = 256, 512
        positions = torch.tensor([[0, 1, 1023, 1024, 1025, 4095]])
        rotary = official.Gemma4TextRotaryEmbedding(config)
        for index, kind in enumerate(config.layer_types):
            dim = config.head_dim if index == 0 else config.global_head_dim
            x = torch.randn(1, 6, 2, dim, dtype=torch.bfloat16)
            cosine, sine = rotary(x, positions, kind)
            jc, js = adapter.rotary_tables(config.to_dict(), index, jnp.asarray(positions.numpy()))
            self.compare(kind + "_cosine", cosine, jc, .001)
            self.compare(kind + "_sine", sine, js, .001)
            expected = official.apply_rotary_pos_emb(x, cosine, sine, unsqueeze_dim=2)
            actual = jax.jit(adapter.apply_rotary)(array(x), jc, js)
            self.compare(kind + "_rotary", expected, actual, .001)
            if index == 1:
                untouched = list(range(64, 256)) + list(range(320, 512))
                np.testing.assert_array_equal(np.asarray(actual)[..., untouched], np.asarray(array(x))[..., untouched])
            norm = official.Gemma4RMSNorm(dim).to(torch.bfloat16)
            with torch.no_grad():
                norm.weight.copy_(torch.linspace(.5, 1.5, dim))
            self.compare(kind + "_learned_norm", norm(x), jax.jit(adapter.rms, static_argnums=2)(array(x), array(norm.weight), 1e-6), .01)
            norm = official.Gemma4RMSNorm(dim, with_scale=False)
            self.compare(kind + "_value_norm", norm(x), jax.jit(adapter.rms, static_argnums=2)(array(x), None, 1e-6), .01)

    def run_model(self, batch, sequence, window):
        torch.manual_seed(29)
        config = tiny_config(window)
        model = Gemma4ForCausalLM(config).to(torch.bfloat16).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "norm" in name:
                    parameter.copy_(torch.linspace(.75, 1.25, parameter.numel()).reshape(parameter.shape))
            for index, layer in enumerate(model.model.layers):
                layer.layer_scalar.fill_(.625 if index == 0 else 1.375)
        tokens = torch.randint(1, config.vocab_size, (batch, sequence))
        captured = []
        hooks = [layer.register_forward_hook(lambda m, args, output: captured.append((args[0].detach().clone(), output.detach().clone()))) for layer in model.model.layers]
        with torch.no_grad():
            expected_logits = model(tokens, use_cache=False).logits
        for hook in hooks:
            hook.remove()
        key = f"B{batch}S{sequence}W{window}"
        hidden = adapter.scaled_embeddings(array(model.model.embed_tokens.weight[tokens]), config.hidden_size)
        self.compare(key + "_embedding", captured[0][0], hidden, 0)
        for index, layer in enumerate(model.model.layers):
            fn = adapter.build_layer(config.to_dict(), sequence, batch_size=batch, layer_index=index)
            weights = layer_weights(layer)
            self.compare(key + f"_layer{index}_isolated", captured[index][1], fn(array(captured[index][0]), weights), .02)
            hidden = fn(hidden, weights)
            self.compare(key + f"_layer{index}_propagated", captured[index][1], hidden, .02)
        actual_logits = jax.jit(lambda h, n, w: adapter.head_logits(h, n, w, config.to_dict()))(
            hidden, array(model.model.norm.weight), array(model.lm_head.weight.T))
        self.compare(key + "_logits", expected_logits, actual_logits, .03)
        r = expected_logits.detach().float().numpy()[:, :-1].reshape(-1, config.vocab_size).astype(np.float64)
        c = np.asarray(actual_logits)[:, :-1].reshape(-1, config.vocab_size).astype(np.float64)
        def logsoftmax(x):
            x = x-x.max(-1, keepdims=True)
            return x-np.log(np.exp(x).sum(-1, keepdims=True))
        lp, lq = logsoftmax(r), logsoftmax(c)
        target = tokens[:, 1:].numpy().reshape(-1)
        delta = float(np.mean(lp[np.arange(len(target)), target]-lq[np.arange(len(target)), target]))
        kl = float(np.mean(np.sum(np.exp(lp)*(lp-lq), axis=-1)))
        agreement = float(np.mean(r.argmax(-1)==c.argmax(-1)))
        self.results[key + "_predictive"] = dict(delta_nll=delta, mean_kl=kl, top1_agreement=agreement,
            gates=dict(abs_delta_nll_max=.01, mean_kl_max=.02, top1_agreement_min=.97), synthetic=True)
        self.assertLessEqual(abs(delta), .01)
        self.assertLessEqual(kl, .02)
        self.assertGreaterEqual(agreement, .97)

    def test_batched_complete_model_with_nonunit_scalars(self):
        self.run_model(2, 17, 8)

    def test_actual_1024_window_boundary_complete_model(self):
        self.run_model(1, 1029, 1024)


if __name__ == "__main__":
    unittest.main(verbosity=2)
