"""Offline structural qualification; never a performance run."""
import json, os
from pathlib import Path
from strassen_mm.model_fused_v004 import validate_config
from strassen_mm.kernels_fused_v004 import make_projection
from strassen_mm.fusion_v003 import Epilogue
from strassen_mm.tuner_joint_v001 import arm
c=json.loads(Path('configs/llm_v6e_v001.json').read_text())
keys={(m['id'],w['batch'],w['sequence'],algorithm,dtype) for m in c['models'] for w in c['prefill'] for algorithm in c['algorithms'] for dtype in c['output_dtypes']}
assert len(keys)==700
assert len(c['models'])==7 and c['precision']['dot']=='DEFAULT'
assert set(c['algorithms'])=={'native_default','native_tuned','cubic_tuned','s1','s2'}
for depth in (1,2):
    p=make_projection(dict(arm((32,512,512),depth=depth),architecture='v6e'),(33,513,768),Epilogue('qk_norm_rope'),interpret=True)
    assert p.metadata['norm_reduction_precision']=='DEFAULT'
    assert p.metadata['kernel_version']=='kernels_fused_v004'
out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
(out/'summary.json').write_text(json.dumps(dict(passed=True,main_entries=len(keys),measured_entries=0,scope='Offline protocol and projection construction checks'),indent=2)+'\n')
print('700 unique planned entries; DEFAULT S1/S2 metadata checked; zero performance entries measured')
