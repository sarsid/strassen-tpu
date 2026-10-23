"""Re-render immutable chart data with descriptive markers and build explanations."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


INTERPRETATIONS={
 'N1': 'Saving one block multiplication helps against the matched quadrant-cubic control, but is insufficient against stronger baselines. Across all 44 complete-call shapes, Strassen has 27 supported wins over the quadrant control and none over Native or full-tile Cubic. All custom kernels here use the same small, untuned 256/256/256 tile; this is a starting point, not each algorithm at its best.',
 'N2': 'Tile size is an important performance choice, and the useful choice depends on the matrix shape and algorithm. The largest tile can exceed fast on-chip memory. These seven-repeat screening results identify candidates; they do not prove a globally optimal tile or an independently confirmed advantage. Native XLA was not measured in N2.',
 'N3': 'These changes can improve the implementation, but the effect depends on shape and algorithm. Compare each variant with its own plain version. The fixed tile is 1024/1024/512; there is no Native XLA or full-tile Cubic baseline here. Product reordering alone does not prove that the TPU overlaps vector work and matrix multiplication.',
 'N4': 'Strassen introduces extra additions and subtractions and can have substantially larger numerical error. Passing a fixed tolerance is different from matching the accuracy of Cubic or Native. Performance gains must therefore be restricted to input conditions that have been checked; choosing a shape or tile alone cannot certify accuracy.',
 'N5': 'This is the fairer comparison: Cubic and Strassen each received 20 attempted tile/variant configurations per shape, then separate winners were frozen and measured on fresh data. Across 16 complete-call shapes, Strassen has 6 supported wins, 4 inconclusive comparisons and 6 losses versus Native; versus tuned Cubic it has 6 wins, 8 inconclusive and 2 losses. The benefit is selective, not universal.',
 'N6': 'Host-observed complete-call latency and profiled TPU-module duration answer different questions. Native has the shortest device module for all three original representatives, despite some different call-time rankings. These traces do not establish MXU utilization, vector overlap, or a measured decomposition of dispatch and memory costs.',
 'N7': 'The rule was frozen from N5 before these held-out shapes were measured. It chooses Native on 13 shapes, Strassen on 2 and Cubic on 1. All 3 custom routes beat Native in complete-call timing, and those gains repeat on a fresh v5e allocation. Independent comparator tuning did not update the rule. Duplicate Native/rule bars can differ through timing noise rather than algorithmic improvement.',
 'N8': 'These bars choose an implementation within each family using the same N8 measurements; they are not fresh confirmation. Separate matched-fusion tests show that Strassen is usually faster than custom Cubic, but Native remains essential: fused Strassen loses all 8 complete-call SwiGLU cases and wins 6 of 8 residual cases. Synthetic activations here do not establish real-model quality.',
 'N9': 'Actual Qwen and Mistral checkpoints test whether a fixed custom MLP policy helps a larger computation. Attention and the vocabulary head stay native. Both custom policies pass the measured corpus tolerances, but this is tolerance-based agreement, not identical predictions or downstream-task accuracy. Gemma was not measured because checkpoint access was blocked.'
}


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def label(c,series):return next(s['label'] for s in c['series'] if s['id']==series)

def observation(c):
    pts=c['points']; exp=c['experiment']; scope=c['scope']
    if exp=='N2':
        failures=sum(p['value'] is None for p in pts)
        return (f"Each column is a matrix shape; each row is an algorithm/tile combination. This scope has {len(pts)-failures} measured cells and {failures} memory failures. "
                "Smaller numbers mean less time; the color scale is logarithmic. Compare algorithms within the same column, then compare tile choices. OOM means no valid timing, never zero.")
    if exp=='N4':
        ss=[p for p in pts if 'strassen' in p['series']]
        failed=sum(len(p['row_refs']) for p in ss if p['status']!='ok')
        values=[p['value'] for p in ss]
        return (f"Bars show median relative L2 error across three seeds; dots and whiskers retain the individual seed values. Strassen's medians range from {min(values):.3g} to {max(values):.3g}. "
                f"{failed} Strassen seed cases fail here. The red line is the 0.02 tolerance; the black segment marks the lowest eligible median error. No speed was measured.")
    if exp=='N9' and scope=='quality':
        if 'qualification' in c['id']:
            return 'Each bar is one Native JAX error divided by its allowed limit, checked against official Transformers CPU BF16. Every bar is below 1, so both native implementations pass this short 63-position qualification. These are different checks and tolerances from the full-corpus policy comparison.'
        items=[]
        for p in pts:
            if 'strassen' in p['series']:
                name=p['category'].split('/')[-1]
                items.append(f"{name}: {p['value']:.5g}")
        return ('Strassen values are '+ '; '.join(items)+'. '+
                ('The sign of NLL difference matters; the acceptance gate uses its absolute magnitude.' if 'nll_delta' in c['id'] else 'Lower error or disagreement means closer agreement with Native.')+
                ' Native is zero by self-comparison. Red lines show acceptance gates; these are not timing races.')
    counts=Counter(); ties=0
    for cat in c['categories']:
        group=[p for p in pts if p['category']==cat['id'] and p['value'] is not None and p['status']=='ok']
        if not group:continue
        best=min(p['value'] for p in group)
        winners=[p for p in group if math.isclose(p['value'],best,rel_tol=1e-12,abs_tol=1e-15)]
        if len(winners)>1:ties+=1
        for p in winners:counts[p['series']]+=1
    result='The black segments mark the lowest observed mean within each group. '
    result+='Lowest-mean counts on this page: '+', '.join(f"{label(c,s)} {counts[s]}" for s in counts)+'. '
    if ties:result+=f'{ties} groups have exactly tied displayed-source means. '
    natives={p['category']:p for p in pts if 'native' in p['series']}
    strata=[p for p in pts if 'strassen' in p['series'] and p['status']=='ok' and p['value'] is not None]
    ranks=Counter()
    for p in strata:
        native=natives.get(p['category'])
        if not native:continue
        ref=native.get('extra',{}).get('arm_id',native['series'])
        pair=next((v for v in p.get('comparisons',[]) if v['reference_arm']==ref and v.get('valid_numerical_comparison')),None)
        if pair:
            lo,hi=pair['speedup_ci95']
            ranks['faster' if lo>1 else 'slower' if hi<1 else 'inconclusive']+=1
    if ranks:
        result+=f"Archived paired intervals for Strassen versus Native: {ranks['faster']} faster, {ranks['slower']} slower, {ranks['inconclusive']} inconclusive."
    elif len(c['categories'])==1 and strata and natives:
        sp=strata[0]; np=natives[sp['category']]
        result+=f"Native time / Strassen time = {np['value']/sp['value']:.4f}x; above 1 favors Strassen. This ratio is descriptive."
    else:
        result+='A lowest mean alone is not proof of a clear win.'
    return result


def interpretation(c):
    exp=c['experiment']; ident=c['id']; scope=c['scope']
    text=INTERPRETATIONS[exp]
    if exp=='N4' and 'cancellation' in ident:
        text='All 24 Strassen cancellation cases fail, while the other algorithms pass. The median checked Strassen error is about 0.527, far above the 0.02 limit. This is a real input-sensitivity limit of the tested numerical contract; the speed plots cannot justify replacing Native or Cubic for arbitrary input values.'
    elif exp=='N4' and 'n4-real' in ident:
        text='All 48 cases using actual Qwen projection weights pass. However, the activations are synthetic Gaussian values. This checks compatibility with those weights, not real-model quality or arbitrary-input safety; the cancellation failures in the previous graph still apply.'
    elif exp=='N5' and scope=='prepared_kernel':
        text='Prepared inputs change the practical comparison: across all 16 shapes, Strassen has 7 wins, 5 inconclusive comparisons and 4 losses versus Native, and 5 wins, 9 inconclusive and 2 losses versus tuned Cubic. The chosen configurations were selected by complete-call timing. Prepared performance assumes reusable prepared buffers and should not replace complete-call results.'
    elif exp=='N6' and 'supplement' in ident:
        text='The large Qwen and 8192-square cases also show a Strassen advantage inside the profiled device module: 1.061x and 1.129x versus Native. Before/after ordinary timings retain the same favorable direction. These shapes were chosen afterward from N5 winners, so this is a diagnostic of favorable cases, not an independent estimate of how often Strassen wins.'
    elif exp=='N7' and 'replica' in ident:
        text='No retuning or rule update occurred on this fresh v5e allocation. The same three custom complete-call gains repeat: deep-K Cubic 1.080x, tall Strassen 1.115x and square Strassen 1.087x versus Native. A nominal extra Native-versus-Native win is timing variability, not a new algorithmic gain. This is fresh-v5e replication; v6e remains future work.'
    elif exp=='N8' and 'residual' in ident:
        text='Residual addition is the more favorable application pattern. In the separate matched-fused comparison, Strassen beats Native on 6 of 8 shapes and loses on the two M=512 cases. The selected-family bars here use the same selection data, and may choose different variants; they should not be mistaken for independent confirmation or an isolated arithmetic effect.'
    elif exp=='N8' and scope=='prepared_kernel':
        text='Reusing prepared operands can materially improve SwiGLU comparisons. The separate matched-fused test gives Strassen 5 wins, 1 inconclusive comparison and 2 losses versus Native in this scope, despite losing all 8 complete-call cases. These selected-family bars retain choices made by complete-call time. A deployment benefit requires an explicit assumption that preparation can be reused.'
    elif exp=='N9' and scope=='resident':
        text=('Strassen is about 2.81% slower than Native on Qwen, although slightly faster than the tested Cubic policy. This is a mean across different resident transformer layers; there is no pooled confidence interval and no whole-model speedup claim.' if 'qwen' in ident else
              'Strassen gives a descriptive 1.106x speedup over Native and 1.044x over the tested Cubic policy on resident Mistral layers. This is a useful application result, but the layers are different workloads rather than independent replicates. It does not automatically translate into transfer-inclusive model speedup.')
    elif exp=='N9' and scope=='streamed':
        text='The three policies are near parity in the transfer-inclusive full forward. Compared with Native, Strassen changes the Qwen mean by +0.591% and the Mistral mean by -0.028%. Only three repeats were recorded per policy. The lowest-mean line makes the descriptive ordering visible; it is not evidence of a reliable serving-speedup gain.'
    elif exp=='N9' and scope=='quality':
        text='Both custom policies pass all measured aggregate corpus gates over 32,736 positions per model. Strassen top-1 agreement is 97.73% on Qwen and 98.94% on Mistral, with NLL differences below 0.0003 nats/token. This supports tolerance-based agreement on this corpus, not identical predictions, task accuracy, or universal numerical safety.'
        if 'qualification' in ident:text='This independent check gives a basis for using the Native implementation as a comparison reference. It covers only the first 64 input tokens (63 scored positions), not the complete corpus. The full-corpus custom-policy checks and N4 cancellation tests answer separate questions.'
    return text


def main():
    p=argparse.ArgumentParser();p.add_argument('--chart-run',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--pdf-python',required=True)
    args=p.parse_args();run=args.chart_run.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    outer=json.loads((run/'artifact-manifest.json').read_text());manifest_path=run/'artifacts/manifest.json'
    assert outer['artifacts/manifest.json']=={'bytes':manifest_path.stat().st_size,'sha256':digest(manifest_path)}
    old=json.loads(manifest_path.read_text());charts=copy.deepcopy([c for c in old['charts'] if not c['detail']])
    assert len(charts)==47
    for c in charts:
        c['explanation']={'observation':observation(c),'interpretation':interpretation(c)}
    from plot_experiments_v004 import render
    pages=render(SimpleNamespace(charts=charts),out);assert pages==47
    manifest={'source_chart_run':run.name,'source_manifest_sha256':digest(manifest_path),'charts':charts,
              'sources':old['sources'],'coverage':old['coverage'],'marker_definition':'Per-shape lowest eligible observed mean or median error; not a statistical winner.'}
    (out/'explained_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    subprocess.run([args.pdf_python,str(Path(__file__).with_name('assemble_explained_pdf_v001.py')),
                    '--artifacts',str(out)],check=True)
    files={str(x.relative_to(out)):{'bytes':x.stat().st_size,'sha256':digest(x)} for x in sorted(out.rglob('*')) if x.is_file()}
    (out/'explained_artifact_manifest.json').write_text(json.dumps({'files':files,'charts':47,'pdf_pages':49},indent=2))
    print(json.dumps({'output_dir':str(out),'charts':47,'pdf_pages':49}),flush=True)


if __name__=='__main__':main()
