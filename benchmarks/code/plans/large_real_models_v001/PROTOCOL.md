# Requested larger real-weight models — v001

Authorized by the user: run Qwen3-8B, Mistral-7B-v0.3, and Gemma3-12B now.
Exact official checkpoint revisions are frozen in prepare_large_models_v001.py
and the metadata JSON files here. No smaller checkpoint substitutes.

One new v5e allocation, serial execution; Native default, independently tuned
Native, cubic, one-level Strassen, and two-level Strassen. Preserve BF16 inputs
and pre-additions, FP32 accumulation/output, fixed numerical/model gates.
Sixteen attempted tiles per custom family and four Native compiler settings;
freeze screen winners before three-window confirmation, 30 timing rounds each.
Keep top-three/within-five-percent finalists (cap four), all failures, and
pointwise conditional intervals. No confirmation-based reselection.

Use the first 65536 official-tokenizer WikiText2 raw-test tokens, split into
32 disjoint 2048-token windows. Capture actual first-layer gate/up and down
activations using Native checkpoint execution. For M=2048,8192,16384, concatenate
1,4,8 windows per matrix: first group screens, next three groups confirm.
These M values count token rows across independent contexts, not longer
attention contexts. All weights are exact checkpoint BF16 tensors.

Run independent official Transformers 4.56.2/PyTorch2.8 CPU BF16 qualification
on the first64tokens. Keep prior logit gates; Gemma also retains final-hidden
relative L2 <=.02. A failed qualification blocks full-model claims but retains
real first-layer projection observations. Gemma12B's full multimodal checkpoint
is downloaded and hashed; this experiment executes its text backbone with no
image inputs. Its global linear RoPE factor8 and local unscaled RoPE are tested
against the official implementation before launch; no altered config or model.

Full-model policies use their independently selected M2048 gate/up and down
kernels. Measure all-layer quality on the first16 windows (32752 next-token
positions), seven resident-layer rounds, and three weight-streamed full forwards.
The latter include reads/layout/H2D/layers/full-vocabulary head; downloads and
compilation excluded. Native full-JIT controls expose composition overhead.
Quality is descriptive in-distribution because tuning windows overlap it.
Full-model timing is2048-token prefill, not production serving or generation.

Download budget30GB per model; checkpoint storage/auth stay private. Each
model receives a separate preparation and measurement execution; failures are
preserved and do not silently skip the remaining models. Verify source/input
hashes and runtime identity. Release the owned allocation after retrieval.
No source or results of earlier experiments are overwritten. Local tests and
phase artifacts use scoped commits, excluding unrelated changes.
