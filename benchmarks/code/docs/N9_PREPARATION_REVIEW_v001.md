# N9 preparation review — before acquisition

The archived evidence does **not** establish the current host's RAM or free disk capacity. Mistral's ordinary CPU reference should therefore wait for an archived host-resource preflight. The 8 GiB value in the MM campaign is an estimated **device-memory budget**, not a measurement of CPU RAM.

This review used local source and archived artifacts only. No model preparation, network request, remote probe, test, or commit was performed.

## What the existing evidence establishes

The [current smoke environment](../runs/20260919T055324Z-smoke-v5e-v003-4644df/artifacts/environment.json) identifies one TPU v5 lite device, endpoint `tpu-v5e1-s-kkb-usw1c1-3iv9vtqwssy8q`, hostname `6f3b9b8cc92c`, Python 3.13.15, JAX/jaxlib 0.7.2, libtpu 0.0.21.1 and NumPy 2.1.3. Provisioning, setup, smoke and access-probe records do not contain MemTotal/MemAvailable, cgroup memory limits, CPU quota or free-disk measurements. The NumPy report is insufficient to certify PyTorch CPU BF16 throughput.

The [official access probe](../runs/20260919T055501Z-N9-access-probe-v5e-v004-ebb556/artifacts/summary.json) confirms available public Qwen/Mistral checkpoints and blocked Gemma access. Its repository metadata gives these exact selected weight-file sizes:

| Input | Selected checkpoint bytes | Approximate GiB |
|---|---:|---:|
| Qwen3-0.6B | 1,503,300,328 | 1.40 |
| Mistral-7B-v0.3, three Transformers shards | 14,496,080,928 | 13.50 |
| Combined | 15,999,381,256 | 14.90 |

The downloader correctly avoids Mistral's additional consolidated checkpoint, which would duplicate approximately 14.5 GB. Tokenizer/config files, corpus, isolated PyTorch environment, pip cache, references and retained execution archives require additional disk space. A planning allowance of roughly 22 GiB **new free space** is conservative, not a measurement or guarantee. Explicit disk offload may need more.

## Concrete risks and safeguards

1. **The CPU reference loads a full official model.** `reference_action` uses `AutoModelForCausalLM.from_pretrained(..., torch_dtype=bfloat16, low_cpu_mem_usage=True)` without a memory limit or offload policy. Mistral has approximately 13.5 GiB of BF16 weight storage before loader/runtime overhead. `low_cpu_mem_usage` and possible safetensors memory mapping do not provide a certified RSS bound. If the actual host allowance is 8 GiB, ordinary resident loading is not a safe assumption. In contrast, its 64-token FP32 reference logits are only 8 MiB; reducing the test sequence would not resolve the dominant weight-storage issue. Qwen's corresponding logits are approximately 37 MiB.

2. **Host quota and CPU speed are unknown.** Capture `/proc/meminfo`, visible cgroup memory limit/current and file-cache statistics, CPU affinity/quota and relevant BF16 ISA flags, plus available space on the model, tools and pip-cache filesystems. The current reference deliberately runs one CPU thread. A first forward can be slow on CPUs without efficient BF16 instructions; no reference throughput has yet been measured. The workflow permits a two-hour reference step, so its worst-case total duration exceeds a short overnight window.

3. **A failed attempt leaves immutable paths occupied.** Downloads create the revision cache exclusively; the tools venv is also exclusive. This preserves evidence, but rerunning the full workflow with a fresh local output directory does not resume a partial remote cache or reuse its existing venv. Recovery needs explicitly new cache/venv paths or a versioned verified-reuse mechanism; never delete the original failed attempt. The current top-level workflow does not expose those overrides, although its preparation adapter does.

4. **The current workflow's earlier control-flow bugs are fixed.** It now requires the canonical project root, checks child paths remain under its `runs/` directory, and checks inner preparation status before using a model manifest. A blocked download no longer masquerades as a successfully prepared model. It still downloads both checkpoints before installing/validating the isolated tools environment; checking that environment first would expose Python-wheel/dependency problems before the large transfer.

5. **A partially successful input bundle needs explicit handling.** The N9 campaign generator expects complete model/token/reference bindings for each non-Gemma model. If Mistral preparation fails, passing the whole partial bundle will not automatically run the surviving Qwen study. Preserve the failed model status and use a new versioned configuration/builder path that explicitly marks it unavailable while allowing the prepared model to proceed.

6. **The actual TPU study is streamed.** Its custom JAX runner loads one checkpoint layer at a time and releases it after all windows/policies use it. That substantially reduces model-weight residency compared with the current official CPU-reference loader. It does not prove the CPU reference fits. No preparation or probe should overlap an MM timing phase.

## Recovery without weakening qualification

Run [host_preflight_v001.py](../src/strassen_mm/host_preflight_v001.py) through the existing archive manager during the next idle boundary. It performs no JAX/device work and returns completed diagnostics with explicit feasible/unknown flags. Its 4 GiB Qwen and 20 GiB Mistral available-memory thresholds are conservative planning targets, not loader fit proofs; cgroup file-cache reclamation and hidden ancestor limits remain caveats.

If Mistral falls below the resident budget, retain the same official checkpoint revision, token array, BF16/eager forward semantics and original qualification gates. A new reference version can use official Transformers/Accelerate with an explicit CPU memory budget and a new immutable disk-offload directory, or carefully recorded streaming of official model layers. Record the changed reference execution strategy and required disk headroom. It must still execute the official implementation independently of the custom JAX model; replacing that comparison with self-agreement or relaxing its gate would not qualify the model.
