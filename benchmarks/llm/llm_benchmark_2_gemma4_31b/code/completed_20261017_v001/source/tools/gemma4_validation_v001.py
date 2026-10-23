"""Approved model-fidelity evidence reuse and explicit sequence tokenization."""
import hashlib
import json
import math
from pathlib import Path


def natural_sequences(allids,batch,sequence,segments,bos,np):
    if bos is None or min(batch,segments)<1 or sequence<2:
        raise ValueError('BOS and positive dimensions with sequence >=2 are required')
    needed=segments*batch*(sequence-1)
    if len(allids)<needed:
        raise RuntimeError(f'Insufficient unique held-out corpus positions: {len(allids)} < {needed}')
    ids=np.empty((segments,batch,sequence),dtype=np.int32)
    ids[:,:,0]=bos
    ids[:,:,1:]=np.asarray(allids[:needed],np.int32).reshape(segments,batch,sequence-1)
    policy=dict(version='gemma4_natural_bos_v001',bos_token_id=int(bos),
        add_special_tokens=False,sequence_policy='BOS plus sequence_length-1 contiguous corpus tokens',
        corpus_position_order='segment, batch, token; no repeated corpus positions',
        content_tokens=needed,scored_positions=needed)
    encoded=json.dumps(policy,sort_keys=True,separators=(',',':')).encode()
    return ids,dict(tokenization_policy=policy,tokenization_policy_sha256=hashlib.sha256(encoded).hexdigest())


def qualify_evidence(root,plan,model_id,revision,official_path):
    spec=plan['formula_qualification'];path=Path(root)/spec['evidence_file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['evidence_sha256']:
        raise RuntimeError('Scientific FP32 evidence hash mismatch')
    evidence=json.loads(path.read_text())
    if (model_id,revision)!=(evidence['model'],evidence['revision']):
        raise RuntimeError('Scientific FP32 checkpoint identity mismatch')
    for name,digest in evidence['verified_source_files'].items():
        if hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Scientific FP32 source mismatch: '+name)
    if hashlib.sha256(Path(official_path).read_bytes()).hexdigest()!=evidence['official_source_sha256']:
        raise RuntimeError('Scientific official source mismatch')
    rows=evidence['report']['layers']
    if [r['layer'] for r in rows]!=spec['layers']:
        raise RuntimeError('Scientific FP32 evidence has wrong layers')
    passed=all(r['finite'] is True and math.isfinite(r['relative_l2']) and
        0<=r['relative_l2']<=spec['relative_l2_limit'] for r in rows)
    return dict(all_pass=passed,qualification='CPU FP32 formula fidelity',
        reused_source_identical_control=True,control_run=evidence['control_run'],
        evidence_sha256=spec['evidence_sha256'],original_report_sha256=evidence['report_sha256'],
        model=model_id,revision=revision,layers=rows,relative_l2_limit=spec['relative_l2_limit'],
        reference='Official Transformers 5.5.0 CPU FP32; real weights and official layer inputs',
        prior_bf16_gates_status=evidence['prior_bf16_gates_status'],
        algorithm_quality_reference='Native BF16 on the same TPU; all prior algorithm thresholds unchanged')
