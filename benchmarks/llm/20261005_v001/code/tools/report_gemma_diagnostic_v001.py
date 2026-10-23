"""Combine archived TPU localization and BF16-boundary ablation evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import ml_dtypes
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--ablation',type=Path,required=True)
    a=p.parse_args();out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts';out.mkdir()
    provenance={}
    def track(path):
        provenance[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    initial=json.loads(track(a.initial/'diagnosis.json').read_text())
    ablation=json.loads(track(a.ablation/'diagnosis.json').read_text())
    official=np.load(track(a.ablation/'official/official.npz'),allow_pickle=False)
    proof=[]
    for i in (0,5,47):
        co=official[f'rope_{i:03}_cos_input'][0,:,None,:]
        si=official[f'rope_{i:03}_sin_input'][0,:,None,:]
        for kind in ('q','k'):
            before=np.load(track(a.ablation/f'{i}_original_rope_{kind}.npz'),allow_pickle=False)
            after=np.load(track(a.ablation/f'{i}_corrected_rope_{kind}.npz'),allow_pickle=False)
            x=official[f'rope_{i:03}_{kind}_input'][0].transpose(1,0,2)
            rotated=np.concatenate((-x[...,128:],x[...,:128]),-1)
            def bf(v):return v.astype(ml_dtypes.bfloat16).astype(np.float32)
            once=bf(x*co+rotated*si);separate=bf(bf(x*co)+bf(rotated*si))
            proof.append(dict(layer=i,kind=kind,elements=int(x.size),
                old_jit_vs_single_round_unequal=int(np.count_nonzero(before['jit']!=once)),
                reference_vs_separate_round_unequal=int(np.count_nonzero(before['reference']!=separate)),
                corrected_jit_vs_reference_unequal=int(np.count_nonzero(after['jit']!=after['reference']))))
    lines=['# Gemma 3-12B numerical failure diagnosis','',
        'The blocked experiment is a failure of our Native implementation to match the pinned official BF16 reference. It occurs before Strassen quality or resident performance evaluation. Checkpoint download and runtime allocation succeeded.',
        '', '## Scope and controls','',
        'Exact official Gemma 3-12B checkpoint, text backbone, first 64 real corpus tokens, all 48 decoder layers. Official Transformers 4.56.2 eager CPU reference; JAX 0.7.2 on one v6e TPU per diagnostic. This is localization and qualification, not a held-out accuracy estimate or a timing experiment.',
        'Both diagnostic allocations were configured for automatic retrieval and release. Original failed experiments and immutable source versions are preserved.',
        '', '## What was localized','',
        'Embedding outputs match exactly. Independent common-input projections and norms have small differences, while TPU-compiled rotary position encoding (RoPE) shows a concrete BF16 rounding-order mismatch.',
        'The official operation rounds both products before adding: `BF16(BF16(q*cos) + BF16(rotate(q)*sin))`. The original TPU JIT result is compared pointwise below against rounding once after the sum. Explicit same-dtype casts in the original source did not enforce the intended materialization boundaries.',
        'Ordinary CPU JIT replay matched the official rotary operation, so the earlier CPU integration check could not expose this TPU-specific compiled behavior.',
        '', '| Layer | Tensor | Elements | Original TPU differs from single-round replay | Official differs from separate-round replay | Corrected TPU differs from official |',
        '|---:|---|---:|---:|---:|---:|']
    for r in proof:
        lines.append(f"| {r['layer']} | {r['kind']} | {r['elements']} | {r['old_jit_vs_single_round_unequal']} | {r['reference_vs_separate_round_unequal']} | {r['corrected_jit_vs_reference_unequal']} |")
    lines.extend(['','Counts above are unequal elements, not percentages. Small remaining trigonometric differences must be distinguished from product-rounding differences.',
        '', '## Full-model qualification','',
        'Unchanged limits: logit relative L2 ≤3%, final normalized hidden relative L2 ≤2%, mean KL ≤0.01, absolute NLL change ≤0.05, and top-1 agreement ≥90%. All outputs must be finite.',
        '', '| Run / Native execution | Logit relative L2 | Final hidden relative L2 | Mean KL | NLL change | Top-1 agreement | Pass |',
        '|---|---:|---:|---:|---:|---:|---|'])
    for title,data in [('Initial diagnosis',initial),('Rounding ablation',ablation)]:
        for name,r in data['final'].items():
            lines.append(f"| {title}: {name} | {100*r['relative_l2']:.4f}% | {100*r['hidden']['relative_l2']:.4f}% | {r['mean_kl']:.6f} | {r['nll_delta']:+.6f} | {100*r['top1_agreement']:.2f}% | {r['passed']} |")
    lines.extend(['',
        '`original` is the composed Native benchmark. `rope_boundaries` explicitly rounds the rotary products and blocks fusion across those boundaries. `all_boundaries` additionally puts boundaries around normalization, projection and GELU inputs/outputs. No Strassen operations are involved.',
        'The initial experiment has small common-input layer discrepancies but large propagated divergence, especially after layer 24. Its peak raw decoder-output relative L2 is '+f"{100*max(r['propagated']['composed']['relative_l2'] for r in initial['layers']):.2f}%"+'. This distinguishes accumulated numerical sensitivity from a grossly incorrect single layer.',
        '', '## Interpretation',''])
    passed=[name for name,r in ablation['final'].items() if r['passed']]
    if passed:
        lines.append('The following boundary variants pass this unchanged 64-token qualification: '+', '.join(passed)+'. This does not by itself qualify long-context held-out accuracy or establish a Strassen speed benefit. A new speed/quality campaign must use the corrected Native path consistently across all arms.')
    else:
        lines.append('No boundary variant passes the unchanged full-model qualification. The rotary rounding mismatch is a localized implementation issue, but correcting it is not sufficient to resolve the remaining cross-backend full-model discrepancy. Gemma Strassen speed/accuracy claims remain blocked. The remaining differences require additional backend/reference investigation; do not treat the partial correction or looser tolerances as a validated fix.')
    lines.extend(['','This investigation changes neither the Qwen/Mistral measurements nor their conclusion: neither Strassen depth beat the fastest measured Native control at batch 1, sequence 2048 on v6e.',''])
    (out/'DIAGNOSIS.md').write_text('\n'.join(lines))
    (out/'evidence.json').write_text(json.dumps(dict(input_sha256=provenance,pointwise_proof=proof,
        initial_final=initial['final'],ablation_final=ablation['final']),indent=2)+'\n')
    print(json.dumps(dict(completed=True,passing_variants=passed,output=str(out))))


if __name__=='__main__':main()
