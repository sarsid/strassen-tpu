"""Retarget the v6e study search menu to an explicit fused-operation contract.

This enumerates candidates and rejection reasons; it does not claim that a
pure-MM winner is also the fused winner. Screen/confirm actual fused calls on
the target TPU and real activations before choosing a per-model profile.
"""
import dataclasses
import math
from .tuner_arch_study_v001 import registry as mm_registry
from .tuner_joint_v001 import arm
from .fusion_v001 import plan


def registry(shape, cfg, spec, *, include_early=False):
    if spec.early:
        raise ValueError('Use include_early to add separately identified early candidates')
    offered, old_trace = mm_registry(shape,cfg)
    # Reconsider ALL old entries: fused scratch differs, so the old heuristic
    # must neither admit nor exclude a candidate on our behalf.
    source = [a for a in offered if a['implementation']=='native']
    source += [r['candidate'] for r in old_trace['candidates']]
    if spec.kind == 'norm_residual_add':
        bn = math.ceil(shape['n']/512)*512
        for bm in sorted({t[0] for t in old_trace['output_geometries']}):
            for bk in old_trace['offered_k_panels']:
                for depth in (0,1,2):
                    for mode in (('outputs',) if depth==0 else ('products','outputs')):
                        for buffers in (1,2):
                            a = dict(arm((bm,bn,bk),depth,mode,buffers,shape['output_dtype']),architecture='v6e')
                            if depth==0:
                                a.update(implementation='cubic',family='cubic',algorithm='cubic',arm_id=f'Cubic_{bm}_{bn}_{bk}_b{buffers}')
                            source.append(a)
                            if depth==0:
                                source.append(dict(a,implementation='cubic_full',arm_id=a['arm_id'].replace('Cubic_','Full_cubic_')))
    candidates, decisions, seen = [], [], set()
    mkn = tuple(shape[x] for x in ('m','k','n'))
    for original in source:
        # Old full-tile cubic used compiler-managed buffering. The fused
        # variant has an explicit pipeline: remeasure both counts.
        buffer_choices = (1,2) if original['implementation']=='cubic_full' else (original.get('buffers'),)
        for buffers in buffer_choices:
            candidate = dict(original,buffers=buffers)
            variants = [spec]
            if include_early and spec.gated and candidate['depth']==1 and candidate['accumulator']=='outputs':
                variants.append(dataclasses.replace(spec,early=True))
            for variant in variants:
                key = (candidate['implementation'],candidate['depth'],tuple(candidate.get('tile') or ()),
                       candidate['accumulator'],buffers,tuple(sorted(candidate.get('compiler_options',{}).items())),variant.early)
                if key in seen:
                    continue
                seen.add(key)
                suffix = f'__{shape["output_dtype"]}__{spec.kind}__{spec.rounding}_{spec.norm}_h{spec.head_dim}_eps{spec.eps:g}'
                name = original['arm_id']+('_b'+str(buffers) if original['implementation']=='cubic_full' else '')+suffix+('_early' if variant.early else '')
                a = dict(candidate,arm_id=name,candidate_id=name,variant=name,epilogue=dataclasses.asdict(variant))
                reasons, meta = [], None
                try:
                    meta = plan(a,mkn,variant)
                    estimate = meta['estimated_vmem_bytes']
                    if estimate is not None and estimate > cfg['search']['estimate_prune_mib']*1024**2:
                        reasons.append('fused_memory_estimate_exceeds_search_cap; not_proven_infeasible')
                except ValueError as exc:
                    reasons.append(str(exc))
                decisions.append(dict(candidate=a,disposition='pruned' if reasons else 'offered',reasons=reasons,metadata=meta))
                if not reasons:
                    candidates.append(a)
    return candidates,dict(shape=shape,epilogue=dataclasses.asdict(spec),include_early=include_early,
        source_menu='tuner_arch_study_v001',candidates=decisions,
        selection_requirement='fresh fused-call screening and independent confirmation per output dtype and whole-layer compiler setting')
