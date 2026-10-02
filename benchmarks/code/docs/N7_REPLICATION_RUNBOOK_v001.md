# N7 replication on a fresh v5e allocation

This is an execution runbook, not evidence that replication has run. Finish and archive the original-cohort N9 study before releasing its allocation. Preparation, official-reference and model-quality artifacts must already be downloaded, sealed and committed. Do not overlap release or provisioning with an active preparation/benchmark worker. Negative numerical/model-quality findings remain valid outcomes; they are not a reason to rewrite the frozen policy.

The original cohort is session `strassen-mm-focus-v5e-n5n9-20260919`, endpoint `tpu-v5e1-s-kkb-usw1c1-3iv9vtqwssy8q`. Its model checkpoint cache is an external reproducible input: release loses that remote cache. Exact local checkpoint/corpus manifests and all generated results must be preserved first. Credentials remain in the existing SDK store and must never be copied into the repository.

## Versions and frozen inputs

Use **`benchmark_n7_replica_v002`** with the existing `run_phase_v004.py`. V001 is preserved but incompatible with the launcher's identity-file placement: the launcher passes a copied `expected-identity.json`, whereas v001 looked in that file's parent for the complete smoke run. V002 adds required `--smoke-environment`; it checks exact bytes against the launcher copy, then verifies the canonical smoke directory's full seal and all 24 passing result rows. All original policy, source, precision and cohort compatibility checks are unchanged.

| Input | Exact existing source |
|---|---|
| Original evaluation, complete canonical artifacts | `runs/20260919T073017Z-N7-evaluate-v5e-v004-8036b5/artifacts` |
| Independent held-out screen selections | `runs/20260919T065405Z-N7-screen-v5e-v004-6ef489/artifacts/selections.json` |
| Exact selector used by evaluation | Original evaluation's `frozen_selector_input.json` |
| Campaign | `configs/campaign_n5_n9_v1.json` |
| Held-out shape reservation | `configs/shapes_heldout_v1.json` |

Expected SHA256 values:

- Selection: `5376adc461bd38499f0cb5324c59085fe65d6b2ef37b1715b45caaf660c54e37`.
- Selector: `18f0cee20fd8812ba27fdad5f9ceec3bfdcd872cf526195221e55df65bb36a77`.
- Original evaluation journal: `757c77c09ad1016415ee4930305d6d0229af3f128ef802db9433f113cbe88153`.
- Campaign: `535808e9832f4c800c653f5b51eb8707828b2e27c94e705e5bc26c2543da2a5a`.
- Held-out reservation: `e5ed73f42fdba354ef869f60081edb2f63dc43d1ae984ba6446ad5434a6dac0f`.

Do not rerun N7 screening, refit the selector, choose new tiles, change seeds, or rewrite old identity fields. The replica preserves 16 held-out groups, four arms and two scopes: 128 scoped results, with 30 timed rounds per successful result. Its scope remains synthetic Gaussian BF16 inputs and FP32 accumulation/output with DEFAULT dot precision.

## 1. Local preparation and archived offline checks

Run commands from the canonical repository, not a `runs/.../source` snapshot. Replace angle-bracket values below with exact IDs returned by completed executions; they are deliberately not guessed.

```sh
STRASSEN_ROOT='/Users/bagheera/Documents/ChatGPT/Faster Strassen TPU/Strassen_MM_Focus'
STRASSEN_PY='/Users/bagheera/Documents/ChatGPT/Faster Strassen TPU/.venv-colab/bin/python'
STRASSEN_OLD_SESSION='strassen-mm-focus-v5e-n5n9-20260919'
STRASSEN_OLD_ENDPOINT='tpu-v5e1-s-kkb-usw1c1-3iv9vtqwssy8q'
STRASSEN_NEW_SESSION='strassen-mm-focus-v5e-n7replica-20260919-v001'
cd "$STRASSEN_ROOT"
```

If this exact test execution has not already been archived by root:

```sh
"$STRASSEN_PY" tools/archive_v002.py --label n7-replica-tests-v002 -- \
  "$STRASSEN_PY" -m unittest discover -s tests -p test_n7_replica_v002.py -v
```

Proceed only after the archived test result passes. This command freezes source and commits its evidence. Do not rerun it merely to repeat an already completed check.

Stage original artifacts into a **new** data directory so they ride the next frozen source archive. `archive_v002` intentionally excludes `runs/` from uploaded source; passing a local run path directly to a remote runner will not work. Copy the entire evaluation artifact tree, including its existing seal, source/config snapshots and journal. Preserve exact file bytes.

```sh
"$STRASSEN_PY" tools/archive_v002.py --label n7-replica-input-copy-v001 -- \
  "$STRASSEN_PY" -c '
import json, shutil, sys
from pathlib import Path
from strassen_mm import selector_v002 as s
from strassen_mm import benchmark_n7_replica_v002 as r
root=Path(sys.argv[1]).resolve()
eval_run=root/"runs/20260919T073017Z-N7-evaluate-v5e-v004-8036b5"
screen_run=root/"runs/20260919T065405Z-N7-screen-v5e-v004-6ef489"
endpoint="tpu-v5e1-s-kkb-usw1c1-3iv9vtqwssy8q"
def outer_file(run, relative):
    file=run/relative
    entry=s.load(run/"artifact-manifest.json")[relative]
    assert entry=={"bytes":file.stat().st_size,"sha256":s.digest(file)}, relative
    return file
for run in (eval_run,screen_run):
    completion=s.load(outer_file(run,"completion.json"))
    assert completion["status"]=="completed"
    assert completion["allocation_id"]==endpoint
    assert completion["remote_may_still_be_running"] is False
    outer_file(run,"artifacts/artifact_manifest.json")
    s.verify_seal(run/"artifacts")
original,environment,rows,provenance=r.verify_completed_run(eval_run,"N7-evaluate",128)
selection=outer_file(screen_run,"artifacts/selections.json")
selector=outer_file(eval_run,"artifacts/frozen_selector_input.json")
assert s.digest(selection)=="5376adc461bd38499f0cb5324c59085fe65d6b2ef37b1715b45caaf660c54e37"
assert s.digest(selector)=="18f0cee20fd8812ba27fdad5f9ceec3bfdcd872cf526195221e55df65bb36a77"
assert s.digest(original/"results.jsonl")=="757c77c09ad1016415ee4930305d6d0229af3f128ef802db9433f113cbe88153"
r.verify_original_policy(original,environment,s.load(selection),s.load(selector),selection,selector)
r.verify_reused_source(original)
target=root/"data/n7_replication_v001"
target.mkdir(exist_ok=False)
shutil.copytree(original,target/"original_evaluation")
shutil.copyfile(selection,target/"selections.json")
shutil.copyfile(selector,target/"selector.json")
s.verify_seal(target/"original_evaluation")
assert s.digest(target/"selections.json")==s.digest(selection)
assert s.digest(target/"selector.json")==s.digest(selector)
with (target/"provenance.json").open("x") as f:
    json.dump({"original_evaluation":provenance,"selection_source":str(selection),
       "selector_source":str(selector),"policy":"Exact bytes; no refit, retuning or identity rewrite"},f,indent=2)
with (target/"copy_manifest.json").open("x") as f:
    json.dump({str(p.relative_to(target)):{"bytes":p.stat().st_size,"sha256":s.digest(p)}
       for p in sorted(target.rglob("*")) if p.is_file() and p.name!="copy_manifest.json"},f,indent=2)
print(json.dumps({"copied_to":str(target),"original_results":len(rows)}))
' "$STRASSEN_ROOT"
```

The staging command itself runs from a frozen snapshot, writes only the explicitly named new canonical data directory, and commits it on completion. If it fails after creating a partial directory, retain that directory and use a newly versioned destination in a separately recorded retry; do not overwrite it.

## 2. Release the old allocation only after N9 is complete

First record the exact final N9 execution ID whose complete artifacts have been downloaded and committed. Root must have finished all original-cohort N9 work, and no other remote worker may be active. The guard below requires a completed N9 summary; an unfinished/failed orchestration must be resolved while the old allocation still exists.

```sh
STRASSEN_FINAL_N9_RUN='<exact completed original-cohort N9 run ID>'
"$STRASSEN_PY" tools/archive_v002.py --label release-original-v5e-after-n9-v001 -- \
  "$STRASSEN_PY" -c '
import json, sys
from pathlib import Path
sys.path.insert(0,"runtime")
import colab_control_v001 as c
from strassen_mm import selector_v002 as s
root=Path(sys.argv[1]).resolve()
session,endpoint,run_id=sys.argv[2:5]
try:
    run=root/"runs"/run_id
    assert run.resolve().parent==root/"runs"
    seal=s.load(run/"artifact-manifest.json")
    for relative in ("completion.json","artifacts/summary.json","artifacts/artifact_manifest.json"):
        path=run/relative
        assert seal[relative]=={"bytes":path.stat().st_size,"sha256":s.digest(path)}
    completion=s.load(run/"completion.json")
    summary=s.load(run/"artifacts/summary.json")
    assert completion["status"]=="completed" and completion["allocation_id"]==endpoint
    assert completion["remote_may_still_be_running"] is False
    assert summary["phase"]=="N9" and summary["completed"] is True
    s.verify_seal(run/"artifacts")
    store=c.store_at(c.DEFAULT_CONFIG)
    saved=store.get(session)
    assert saved is not None and saved.endpoint==endpoint
    with c.api(c.DEFAULT_TOKEN) as client:
        before=client.list_assignments()
        assert len(before)==1 and before[0].endpoint==endpoint
        assert before[0].accelerator.value=="V5E1"
        client.unassign(endpoint)
        after=client.list_assignments()
        assert not any(a.endpoint==endpoint for a in after)
    store.remove(session)
    c.emit({"kind":"released_after_n9","session":session,"endpoint":endpoint,"n9_run_id":run_id})
except Exception as error:
    c.emit({"kind":"release_failed",**c.safe_error(error)})
    raise SystemExit(1)
' "$STRASSEN_ROOT" "$STRASSEN_OLD_SESSION" "$STRASSEN_OLD_ENDPOINT" "$STRASSEN_FINAL_N9_RUN"
```

`colab_control_v001` has no release subcommand. The archived snippet deliberately uses its sanitized authenticated client and the installed SDK's actual `client.unassign(endpoint)`/`StateStore.remove(name)` APIs. It does not emit credentials. Removing the old named session lets its existing heartbeat stop on its next check; do not kill an unverified PID that could have been reused. If release confirmation is uncertain, inspect an archived sanitized `list` before taking another lifecycle action.

## 3. Create and qualify a new v5e allocation

```sh
"$STRASSEN_PY" tools/archive_v002.py --label n7-replica-allocation-list-v001 -- \
  "$STRASSEN_PY" runtime/colab_control_v001.py list
"$STRASSEN_PY" tools/archive_v002.py --label n7-replica-provision-v5e-v001 -- \
  "$STRASSEN_PY" runtime/colab_control_v001.py create --session "$STRASSEN_NEW_SESSION"
```

The list must be empty before creation. The controller itself refuses a new session when any allocation is present. Require the create result to say `created: true`, `accelerator: V5E1`, and record its exact endpoint. An existing saved name can be reused by the controller, so `created: false` does not qualify this fresh-allocation experiment. Do not pass the old endpoint as `--expect-endpoint` to create. The controller starts the bounded existing heartbeat automatically.

```sh
STRASSEN_NEW_ENDPOINT='<exact endpoint from the successful new create result>'
"$STRASSEN_PY" tools/setup_allocation_v001.py \
  --session "$STRASSEN_NEW_SESSION" --endpoint "$STRASSEN_NEW_ENDPOINT" \
  --controller-python "$STRASSEN_PY"
```

Require the new endpoint to differ from `STRASSEN_OLD_ENDPOINT`. `setup_allocation_v001.py` owns its source/result archives and commits; do not wrap this orchestrator in another snapshot execution. It pins JAX/JAXlib 0.7.2 and libtpu 0.0.21.1, probes a child process for one `TPU v5 lite`, downloads the setup archive, and reports `EXECUTION_DIR`. Record that exact run ID:

```sh
STRASSEN_NEW_SETUP_RUN='<exact successful setup-v5e-v001 run ID>'
STRASSEN_NEW_SETUP_IDENTITY="/content/Strassen_MM_Focus/$STRASSEN_NEW_SETUP_RUN/identity.json"
"$STRASSEN_PY" tools/run_phase_v004.py --phase smoke \
  --session "$STRASSEN_NEW_SESSION" --endpoint "$STRASSEN_NEW_ENDPOINT" \
  --controller-python "$STRASSEN_PY" --expected-identity "$STRASSEN_NEW_SETUP_IDENTITY" \
  --campaign-relative configs/campaign_v1.json \
  --benchmark-module strassen_mm.benchmark_v001 --timeout-seconds 7200
```

The corresponding downloaded setup identity is `runs/<SETUP_RUN>/artifacts/<SETUP_RUN>/identity.json`; do not confuse it with the remote identity path above. Require successful setup status, matching downloaded setup archive/hash evidence, and smoke completion with 24/24 passing results. The complete canonical remote smoke tree stays on this new allocation for the replica:

```sh
STRASSEN_NEW_SMOKE_RUN='<exact successful new smoke run ID>'
STRASSEN_NEW_SMOKE_ENVIRONMENT="/content/Strassen_MM_Focus/runs/$STRASSEN_NEW_SMOKE_RUN/artifacts/environment.json"
```

Setup pins only the three accelerator packages. Replica compatibility also requires NumPy, ml_dtypes, SciPy, requests, runtime image, device metadata, runtime flags and all other recorded identity fields to match the original, apart from the explicitly permitted allocation/host/boot fields. A new provider image or dependency drift can therefore fail compatibility even after smoke passes. Preserve such a failure; do not quietly weaken checks, rewrite the original identity, or repeatedly allocate devices to find favorable timings.

## 4. Run the fixed-policy replica

```sh
"$STRASSEN_PY" tools/run_phase_v004.py --phase N7-replicate \
  --session "$STRASSEN_NEW_SESSION" --endpoint "$STRASSEN_NEW_ENDPOINT" \
  --controller-python "$STRASSEN_PY" --expected-identity "$STRASSEN_NEW_SMOKE_ENVIRONMENT" \
  --campaign-relative configs/campaign_n5_n9_v1.json \
  --benchmark-module strassen_mm.benchmark_n7_replica_v002 --timeout-seconds 21600 \
  --runner-arg=--smoke-environment --runner-arg="$STRASSEN_NEW_SMOKE_ENVIRONMENT" \
  --runner-arg=--setup-identity --runner-arg="$STRASSEN_NEW_SETUP_IDENTITY" \
  --runner-arg=--original-evaluation --runner-arg=data/n7_replication_v001/original_evaluation \
  --runner-arg=--selection --runner-arg=data/n7_replication_v001/selections.json \
  --runner-arg=--selector --runner-arg=data/n7_replication_v001/selector.json
```

Relative runner paths resolve inside the new frozen source directory; all staged data are uploaded in `source.tar`. The two smoke-related flags intentionally start from the same canonical remote path: the launcher copies `--expected-identity` into the new run, while `--smoke-environment` continues to identify the complete canonical smoke evidence. Do not try to override the launcher's reserved `--expected-identity` through `--runner-arg`.

This orchestrator archives source before launch, uses the allocation's execution lock, streams exact bytes, retrieves the complete remote archive, seals it and commits before returning. Retain any compilation, numerical or compatibility failure in its unique execution directory.

## 5. Audit and report cohorts separately

```sh
STRASSEN_REPLICA_RUN='<exact completed or failed N7-replicate run ID>'
"$STRASSEN_PY" tools/archive_v002.py --label n7-replica-evidence-audit-v001 -- \
  "$STRASSEN_PY" tools/audit_n5_n9_v002.py \
  --run "$STRASSEN_ROOT/runs/$STRASSEN_REPLICA_RUN" --output-dir replica_audit
```

The audit output is retained under that audit execution's `source/replica_audit/`. Check `replication_inputs.json`, `cohort_compatibility.json`, exact policy hashes, 16 completed groups, 128 unique scoped results, all failure statuses and paired-round coverage. The five reused implementation modules must still match the original evaluation's source hashes. Preserve fresh setup/smoke links and the `smoke_expected_identity_link` byte-equality proof.

Compare within-cohort paired speedups and report original and replica cohorts separately; do not pool raw latency samples. A distinct logical allocation is observable. Colab does not expose a physical TPU chip serial, so this is a fresh-allocation replication, not proof of a distinct physical chip. v6e replication remains a later experiment.

This runbook and the v002 compatibility fix were prepared without running tests, provisioning, releasing machines, downloading inputs, committing, or executing TPU work.
