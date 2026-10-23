# Gemma4 metadata contract validation

October 3 preparation; no active measurement or queue source changed.

`src/strassen_mm/gemma4_contract_v001.py` adds an explicit dense text configuration
contract, per-layer MM shapes, and required canonical checkpoint tensor shapes.
It preserves the supplied configuration and rejects unsupported features before
any checkpoint acquisition or device allocation. It includes the persistent
`layer_scalar`, omits a separate global V projection, and uses the correct local
and global head dimensions and KV head counts. There is no numerical adapter or
checkpoint reader in this module yet.

Three local tests passed in
`runs/20261003T232207Z-gemma4-metadata-contract-v002-e9f8df/`:

- All 60 layers' required parameter and persistent-buffer names and shapes
  match the pinned official decoder constructors instantiated on PyTorch's meta
  device. Per-layer projection dimensions, grouped-query counts and attention
  scaling also match.
- The pinned 31B local/global MM geometry matches the audited dimensions, with
  six local sites and five global sites.
- Unsupported features and invalid dimensions are rejected, while validation
  leaves the official configuration unchanged.

The reference is the SHA-verified Transformers 5.5.0 source at commit
`c1c34249fa27deefbd4a377dfbf883a39baf5c6d`. The test loads only four unchanged class
AST nodes. For this constructor-only check, the gradient-checkpointing base is
`torch.nn.Module` and the optional kernel decorator is an identity shim. No
forward, backward, checkpoint tensor allocation, or numerical oracle comparison
occurs. This is metadata evidence, not full Transformers integration or official
checkpoint qualification.

The initial v001 test attempt failed before running tests because the optional
kernel decorator dependency was not supplied. Its source and log are preserved
in `runs/20261003T232057Z-gemma4-metadata-contract-v001-0634f7/`. Test v002 declares
the constructor-only shims explicitly; the contract module itself is unchanged.
The passing source snapshot is commit `33474d508`.

Next: implement the versioned strict checkpoint reader and native numerical
adapter; create an isolated compatible full Transformers oracle environment;
qualify local/global attention, proportional RoPE, dtype boundaries and scalar
state; then extend fusion, caching and actual launch/recovery bundles. Follow
`GEMMA4_ADAPTER_AUDIT_v001.md` for the complete remaining sequence.

Gemma4's ten workloads remain `awaiting_qualification`. Qwen32 continues on its
existing frozen bundle. No Gemma4 weights were downloaded and no TPU was
allocated for these local checks.
