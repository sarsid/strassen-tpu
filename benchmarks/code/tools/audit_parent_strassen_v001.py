"""Archive a read-only, pinned-source audit; never import or execute parent code."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

REVISION = '95be1fb088656a89813b04492e1d77c66b36ccf9'
PARENT_FILES = ['README.md', 'docs/RESULTS.md', 'docs/COLAB_RUNBOOK.md',
    'strassen_pallas.py', 'mosaic_compat.py', 'tools/verify_evidence.py',
    'experiments/qwen3/benchmark_qwen3_v6e_pure_gemm.py',
    'experiments/qwen3/benchmark_qwen3_scaling_tile_tune.py',
    'experiments/qwen3/benchmark_qwen3_site_tile_tune.py',
    'experiments/qwen3/benchmark_qwen3_32b_full_layer_product_inference.py',
    'experiments/qwen3/benchmark_qwen3_32b_streamed_inference.py',
    'experiments/qwen3/benchmark_common.py', 'experiments/qwen3/benchmark_cubic_control.py',
    'experiments/qwen3/benchmark_qwen3_32b_layer.py', 'evidence/qwen3/README.md']
LOCAL_FILES = ['src/strassen_mm/kernels_v001.py', 'src/strassen_mm/kernels_v002.py',
    'src/strassen_mm/kernels_v6e_v001.py', 'src/strassen_mm/kernels_v6e_v002.py',
    'src/strassen_mm/kernels_v6e_v003.py', 'src/strassen_mm/kernels_two_level_v001.py',
    'src/strassen_mm/kernels_n8_v001.py', 'src/strassen_mm/model_composed_v002.py',
    'src/strassen_mm/benchmark_v6e_suite_v001.py', 'src/strassen_mm/benchmark_v6e_opt_v001.py',
    'src/strassen_mm/benchmark_llm_tradeoff_v001.py', 'src/strassen_mm/benchmark_large_models_v001.py',
    'configs/v6e_suite_v001/campaign.json', 'configs/v6e_suite_v001/shapes.json',
    'docs/N8_FINDINGS_v001.md', 'docs/N9_FINDINGS_v001.md']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(upstream, root, out):
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip() == REVISION
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=upstream,text=True).strip()
    out.mkdir(parents=True,exist_ok=False)
    def copy(p,d):
        d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
    for f in PARENT_FILES:
        copy(upstream/f,out/'parent'/f)
    evidence = upstream/'evidence/qwen3'
    entries = re.findall(r'`(strassen_qwen3_[^`]+\.jsonl)`\s*\|\s*`([0-9a-f]{64})`',
                         (evidence/'README.md').read_text())
    assert len(entries)==len(set(n for n,h in entries))==33
    assert {p.name for p in evidence.glob('*.jsonl')}=={n for n,h in entries}
    for name,expected in entries:
        assert sha(evidence/name)==expected,name
        copy(evidence/name,out/'parent/evidence/qwen3'/name)
    for f in LOCAL_FILES:
        copy(root/f,out/'ours'/f)
    frozen_checks={}
    for cohort, files in [
        ('20260923-v6e-suite-v001',['src/strassen_mm/kernels_v6e_v001.py','src/strassen_mm/kernels_v6e_v002.py',
                                  'src/strassen_mm/kernels_v6e_v003.py','src/strassen_mm/benchmark_v6e_suite_v001.py',
                                  'configs/v6e_suite_v001/campaign.json']),
        ('20260923-mistral-resident-v5e-v001',['src/strassen_mm/model_composed_v002.py',
                                           'src/strassen_mm/benchmark_llm_tradeoff_v001.py'])]:
        for f in files:
            expected=root/'runs'/cohort/'source'/f
            frozen_checks[f]=dict(cohort=cohort,sha256=sha(root/f),matches_frozen=sha(root/f)==sha(expected))
            assert frozen_checks[f]['matches_frozen']
    def records(name):
        return [json.loads(l)for l in (evidence/name).read_text().splitlines()]
    def one(name,kind):
        vals=[r for r in records(name)if r['kind']==kind];assert len(vals)==1;return vals[0]
    pure=one('strassen_qwen3_32b_v6e_pure_gemm.jsonl','confirmation')
    pure_meta=one('strassen_qwen3_32b_v6e_pure_gemm.jsonl','metadata')
    shape=pure['shape'];tile=pure['tiles']['strassen'];m,k,n=shape
    assert shape==[8192,5120,51200] and tile==[2048,1024,5120]
    cfg=json.loads((root/'configs/v6e_suite_v001/campaign.json').read_text())
    def rank(t):
        bm,bn,bk=t;return math.ceil(m/bm)*bm*math.ceil(k/bk)*bk*math.ceil(n/bn)*bn,-bm*bn*bk,tuple(t)
    selected=sorted(cfg['tile_pool'],key=rank)[:6]
    assert tile not in cfg['tile_pool'] and max(t[2]for t in cfg['tile_pool'])==4096
    shape_manifest=json.loads((root/'configs/v6e_suite_v001/shapes.json').read_text())['shapes']
    assert shape not in [[s[d]for d in ('m','k','n')]for s in shape_manifest]
    pairs={}
    for chip in ('v5e','v6e'):
        pairs[chip]={switch:one(f'strassen_qwen3_32b_full_layer_product_inference_{chip}_fusedqk_{switch}.jsonl','performance')
                     for switch in ('on','off')}
    result_path=root/'runs/20260924-v6e-suite-recovery-v001/operations/combined-report/artifacts/report/results.json'
    results=json.loads(result_path.read_text())['results']
    bk_hist={f:dict(Counter(r['methods'][f]['candidate']['tile'][2]for r in results if r['category']=='large_2048_aligned'))
             for f in ('one_level','two_level')}
    summary=dict(upstream_url='https://github.com/sarsid/strassen-tpu',upstream_revision=REVISION,
        indexed_evidence_verified=len(entries),parent_code_executed=False,tpu_experiments_launched=False,
        frozen_source_checks=frozen_checks,parent_pure_gemm=pure,parent_pure_metadata=pure_meta,
        parent_qk_fusion_pairs=pairs,parent_streamed_metadata=one('strassen_qwen3_32b_streamed_product_inference_v6e_fusedqk.jsonl','metadata'),
        ours=dict(candidate_pool=cfg['tile_pool'],offered_tiles_for_parent_shape=selected,
                  parent_winning_tile_present=False,parent_shape_in_168=False,
                  native_candidates=cfg['candidate_families']['native'],large_aligned_selected_bk=bk_hist,
                  results_sha256=sha(result_path)),
        caveats=['Static audit plus existing artifact verification; no new performance claim.',
                 'Parent bare GEMM returns BF16; our grid returns FP32.',
                 'Parent v6e uses JAX 0.7.2/libtpu 0.0.21.1; ours 0.11.2/0.0.48.',
                 'Parent fixes Native scoped VMEM at 48 MiB; ours retains default and tunes per executable.',
                 'Parent product-aware v6e behavior is mixed in raw artifacts, despite categorical README prose.',
                 'Older N8/N9 already implement some fusions; current larger-model composed path does not enable them.'])
    (out/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    copy(root/'docs/PARENT_KERNEL_AUDIT_v001.md',out/'RESULTS.md')
    manifest={str(p.relative_to(out)):sha(p)for p in out.rglob('*')if p.is_file()}
    (out/'audit-files.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(upstream_revision=REVISION,verified_evidence=33,local_frozen_checks=len(frozen_checks),
                         parent_winning_tile=tile,ours_offered=selected,output=str(out)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream-root',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path(os.environ.get('STRASSEN_PROJECT_ROOT','.')))
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args();out=a.output_dir or Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts'
    main(a.upstream_root.resolve(),a.root.resolve(),out.resolve())
