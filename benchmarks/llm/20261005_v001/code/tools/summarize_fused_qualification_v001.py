"""Verify saved fusion evidence and emit a scoped, reproducible qualification."""
import hashlib, json, os
from pathlib import Path

root=Path(os.environ['STRASSEN_PROJECT_ROOT'])
out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
names=dict(cpu='20261002T172533Z-fused-integration-cpu-v003-98cd73',
           device='20261002T172539Z-fused-device-qualification-v002-ce2815',
           planner='20261002T172753Z-fused-registry-cpu-v001-ac378b')
inputs={}
def read(role,relative):
    path=root/'runs'/names[role]/relative
    raw=path.read_bytes();inputs[str(path.relative_to(root))]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw)
cpu=read('cpu','artifacts/checks.json')
cpu_status=read('cpu','completion.json')
planner=read('planner','artifacts/summary.json')
planner_status=read('planner','completion.json')
device=read('device','artifacts/summary.json')
device_status=read('device','completion.json')
cases=read('device','artifacts/remote/'+names['device']+'-check/artifacts/cases.json')
assert cpu_status['status']=='failed'  # retained old planner assertion, not hidden
assert len(cpu)==96
for name,count in [('exact_integer_multitile_multipanel',42),('additional_epilogue_contract',32),
                   ('preserved_gaussian_arithmetic_and_final_cast',4),('sparse_cubic_layer_semantics',3),
                   ('gaussian_cubic_fusion_vs_unfused',3),('batch_isolation',3)]:
    assert sum(c['test']==name for c in cpu)==count,(name,count)
assert planner_status['status']==device_status['status']=='completed'
assert planner['completed'] and len(planner['checks'])==4
assert len(cases)==54 and all(c['status']=='passed' for c in cases)
assert device['all_checks_passed'] and device['release_exit_code']==0
release=[]
path=root/'runs'/names['device']/'artifacts/release.log'
inputs[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
for line in path.read_text().splitlines():
    try:release.append(json.loads(line))
    except ValueError:pass
assert any(r.get('verified_absent') and r.get('endpoint')==device['endpoint'] for r in release)
for role in names:
    meta=read(role,'source-manifest.json')
    for name,sha in meta['files'].items():
        assert hashlib.sha256((root/'runs'/names[role]/'source'/name).read_bytes()).hexdigest()==sha
by_dtype={dtype:dict(cases=sum(c['arm']['output_dtype']==dtype for c in cases),
    max_abs_vs_host=max(c['max_abs'] for c in cases if c['arm']['output_dtype']==dtype),
    max_relative_l2_vs_host=max(c['relative_l2'] for c in cases if c['arm']['output_dtype']==dtype))
    for dtype in ('float32','bfloat16')}
summary=dict(completed=True,parent_commit='95be1fb088656a89813b04492e1d77c66b36ccf9',
    latest_api=dict(epilogue='fusion_v003',projection='kernels_fused_v003',layer='model_fused_v003',tuner='tuner_fused_v002'),
    pallas_implementation='kernels_fused_v001',
    source_runs=names,input_sha256=inputs,device_checks=54,device_identity=device['qualification']['identity'],
    device_errors=by_dtype,released_endpoint=device['endpoint'],release_verified_absent=True,
    cpu_checks_before_planner_failure=96,planner_checks_after_fix=4,
    gaussian_fused_vs_unfused_layers=[c for c in cpu if c['test']=='gaussian_cubic_fusion_vs_unfused'],
    limitations=['CPU suite retained its final planner-coverage failure; corrected planner passes separate scoped validation.',
        'Hardware tests qualify fused projection compilation/semantics on small shapes, not optimal tiles or large-model speed.',
        'Whole-layer tests use constructed weights on CPU, not real-checkpoint prediction quality.',
        'Prefill only; KV-cache decoding, image inputs and quantization unsupported.',
        'Older compiled model rounding differs from the explicit-rounding contract; requalify against official model before a quality claim.'])
(out/'qualification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ('completed','latest_api','device_checks','device_errors','release_verified_absent')}),flush=True)
