"""Experimental TPU-oriented, one-level tile-local Strassen optimizations.

Public shape order is M,N,K; tile order is BM,BN,BK. Frozen kernels_v001/v002
remain untouched. 'optimized' is a program name, not a measured speedup claim.
Four pure-Strassen variants provide a factorial ablation of implicit masked
boundaries and functional local accumulators. A fifth variant peels native
tails. See docs/STRASSEN_OPTIMIZED_v001.md.
"""
from __future__ import annotations
import argparse
import functools
import json
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from strassen_mm import kernels_v001 as base

VERSION='strassen_optimized_v001'
VARIANTS=('baseline','masked_edges','local_accumulators','optimized','peeled_edges')
TRAVERSALS=('mnk','nmk')
DEFAULT_ORDER=(4,6,5,2,7,3,1)
# Each (zero-based quadrant, sign) is a contribution of product P1 ... P7.
UPDATES={1:((0,1),(3,1)),2:((2,1),(3,-1)),3:((1,1),(3,1)),
         4:((0,1),(2,1)),5:((0,-1),(1,1)),6:((3,1),),7:((0,1),)}


def validate_order(order):
    try:order=tuple(order)
    except TypeError as error:raise ValueError('product_order must be a permutation of integers 1..7') from error
    if len(order)!=7 or any(type(p) is not int for p in order) or sorted(order)!=list(range(1,8)):
        raise ValueError('product_order must be a permutation of integers 1..7')
    return order


def _product(product, a, b):
    a0,a1,a2,a3=a; b0,b1,b2,b3=b
    add=lambda x,y:(x+y).astype(jnp.bfloat16)
    sub=lambda x,y:(x-y).astype(jnp.bfloat16)
    if product==1:return base._dot(add(a0,a3),add(b0,b3))
    if product==2:return base._dot(add(a2,a3),b0)
    if product==3:return base._dot(a0,sub(b1,b3))
    if product==4:return base._dot(a3,sub(b2,b0))
    if product==5:return base._dot(add(a0,a1),b3)
    if product==6:return base._dot(sub(a2,a0),add(b0,b1))
    if product==7:return base._dot(sub(a1,a3),add(b2,b3))
    raise ValueError('Unknown product')


def _kernel(a_ref,b_ref,out_ref,*,shape_mnk,tile,masked,local,order,traversal):
    m,n,k=shape_mnk;bm,bn,bk=tile
    outer0,outer1,step=(pl.program_id(i) for i in range(3))
    i,j=(outer0,outer1) if traversal=='mnk' else (outer1,outer0)
    a,b=a_ref[...],b_ref[...]
    if masked:
        # BlockSpec OOB values are unspecified. Zero BEFORE any pre-addition;
        # garbage/NaN in an invalid quadrant can otherwise poison a valid one.
        # Generic masked pl.load is unsupported by the pinned TPU lowering.
        if m%bm or k%bk:
            a=jnp.where(((i*bm+jnp.arange(bm))[:,None]<m)&
                        ((step*bk+jnp.arange(bk))[None,:]<k),a,jnp.bfloat16(0))
        if k%bk or n%bn:
            b=jnp.where(((step*bk+jnp.arange(bk))[:,None]<k)&
                        ((j*bn+jnp.arange(bn))[None,:]<n),b,jnp.bfloat16(0))
    hm,hn,hk=bm//2,bn//2,bk//2
    aa=(a[:hm,:hk],a[:hm,hk:],a[hm:,:hk],a[hm:,hk:])
    bb=(b[:hk,:hn],b[:hk,hn:],b[hk:,:hn],b[hk:,hn:])
    quadrants=((slice(0,hm),slice(0,hn)),(slice(0,hm),slice(hn,bn)),
               (slice(hm,bm),slice(0,hn)),(slice(hm,bm),slice(hn,bn)))
    if local:
        def panel(first):
            acc=[None]*4 if first else [out_ref[q] for q in quadrants]
            for p_id in order:
                p=_product(p_id,aa,bb)
                for q,sign in UPDATES[p_id]:
                    if acc[q] is None:
                        acc[q]=p if sign==1 else -p
                    else:
                        acc[q]=acc[q]+p if sign==1 else acc[q]-p
            return tuple(acc)
        # Initial panel never reads an uninitialized output. Four first
        # contributions become assignments, retaining each quadrant's order.
        values=jax.lax.cond(step==0,lambda:panel(True),lambda:panel(False))
        for q,value in zip(quadrants,values):out_ref[q]=value
    else:
        @pl.when(step==0)
        def zero():out_ref[...]=jnp.zeros(out_ref.shape,jnp.float32)
        for p_id in order:
            p=_product(p_id,aa,bb)
            for q,sign in UPDATES[p_id]:
                if sign==1:out_ref[quadrants[q]]+=p
                else:out_ref[quadrants[q]]-=p


def describe(shape_mnk,tile,*,variant='optimized',vmem_limit_bytes=48*2**20,
             product_order=DEFAULT_ORDER,traversal='mnk'):
    shape_mnk=base._positive_triple(shape_mnk,'shape (M,N,K)')
    tile=base.check_tile('strassen',tile)
    if variant not in VARIANTS:raise ValueError(f'variant must be one of {VARIANTS}')
    if traversal not in TRAVERSALS:raise ValueError(f'traversal must be one of {TRAVERSALS}')
    if vmem_limit_bytes is not None and (type(vmem_limit_bytes) is not int or vmem_limit_bytes<=0):
        raise ValueError('vmem_limit_bytes must be positive integer or None')
    order=validate_order(product_order);m,n,k=shape_mnk;bm,bn,bk=tile
    mp,np_,kp=((s+b-1)//b*b for s,b in zip(shape_mnk,tile))
    masked=variant in ('masked_edges','optimized');local=variant in ('local_accumulators','optimized')
    im,jn,sk=mp//bm,np_//bn,kp//bk
    full_bytes=2*(mp*kp+kp*np_)+4*mp*np_
    actual_bytes=2*(m*k+k*n)+4*m*n if masked else full_bytes
    additions=im*jn*(12*sk-4 if local else 12*sk)*(bm//2)*(bn//2)
    ids=['SOPT-BASE']
    if masked:ids.append('SOPT-001')
    if local:ids.append('SOPT-002')
    if order!=DEFAULT_ORDER:ids.append('SOPT-003')
    if traversal!='mnk':ids.append('SOPT-004')
    result={'version':VERSION,'algorithm':'strassen','variant':variant,'shape_mnk':list(shape_mnk),
            'shape_mkn':[m,k,n],'tile_bm_bn_bk':list(tile),'logical_padded_shape_mkn':[mp,kp,np_],
            'padded_shape_mkn':[m,k,n] if masked else [mp,kp,np_],
            'grid_mnk':[im,jn,sk],'grid':([im,jn,sk] if traversal=='mnk' else [jn,im,sk]),
            'traversal':traversal,'product_order':list(order),
            'schedule_id':'strassen_schedule_v001_p'+''.join(map(str,order)),
            'optimization_ids':ids,'boundary_policy':'implicit_blocks_explicit_zero_mask' if masked else 'global_zero_pad',
            'padding_required':not masked and (m,n,k)!=(mp,np_,kp),
            'out_of_bounds_masking_required':masked and (m,n,k)!=(mp,np_,kp),
            'arithmetic_padded_volume_ratio':mp*np_*kp/(m*n*k),
            'source_dot_flops':7*mp*np_*kp//4,
            'source_accumulator_element_additions':additions,
            'source_output_quadrant_writes_per_panel':4 if local else 12,
            'source_output_initialization':'first_contribution_seed' if local else 'zero_first_panel',
            'argument_and_output_bytes':actual_bytes,
            'avoided_global_padding_extent_bytes':full_bytes-actual_bytes,
            'memory_estimate_scope':'External input/output extents only; excludes compiler temporaries, pipeline buffers, spills and live input copies. Not measured peak memory.',
            'input_dtype':'bfloat16','strassen_combination_dtype':'bfloat16',
            'accumulation_dtype':'float32','output_dtype':'float32','dot_precision':'DEFAULT',
            'strassen_levels_per_tile':1,'accumulator_storage':'functional_values_then_output_tile' if local else 'output_tile',
            'vmem_limit_bytes':vmem_limit_bytes,'pipeline':'Pallas compiler-managed (existing baseline already pipelines)',
            'complete_call_scope':'device preparation+matmul+finish; excludes host_transfer/compile',
            'kernel_scope':'prepared_device_inputs_to_output',
            'performance_status':'unmeasured_optimization_candidate; no speedup guarantee',
            'numerical_scope':'BF16 combinations retain cancellation risk; first-use seeding can change signed-zero behavior.'}
    if variant=='peeled_edges':
        mc,nc,kc=(s//b*b for s,b in zip(shape_mnk,tile));core=mc*nc*kc
        # No complete tile along an axis means no Strassen core at all.
        if not core:mc=nc=kc=0
        result.update(algorithm='hybrid_strassen_native',optimization_ids=['SOPT-BASE','SOPT-002','SOPT-005']+
                      (['SOPT-003'] if order!=DEFAULT_ORDER else [])+(['SOPT-004'] if traversal!='mnk' else []),
            padded_shape_mkn=[m,k,n],padding_required=False,out_of_bounds_masking_required=False,
            boundary_policy='strassen_full_tile_core_plus_native_exact_tails',
            core_shape_mnk=[mc,nc,kc],strassen_core_volume_fraction=core/(m*n*k),
            native_only_fallback=not bool(core),source_dot_flops=7*core//4+2*(m*n*k-core),
            arithmetic_padded_volume_ratio=1.,argument_and_output_bytes=2*(m*k+k*n)+4*m*n,
            avoided_global_padding_extent_bytes=full_bytes-(2*(m*k+k*n)+4*m*n),
            grid_mnk=[mc//bm,nc//bn,kc//bk],grid=([mc//bm,nc//bn,kc//bk] if traversal=='mnk' else [nc//bn,mc//bm,kc//bk]),
            accumulator_storage='local_Strassen_core_and_native_tail_outputs',
            source_accumulator_element_additions=(mc//bm)*(nc//bn)*(12*(kc//bk)-4)*(bm//2)*(bn//2) if core else 0,
            source_accumulator_count_scope='Strassen core only; native compiler operations unknown',
            source_output_quadrant_writes_per_panel=4 if core else 0,
            source_output_initialization='first_contribution_seed_in_Strassen_core',
            logical_padded_shape_mkn=[m,k,n],
            strassen_levels_per_tile=1 if core else 0,
            numerical_scope='Mixed algorithm: Strassen BF16 preadds in core, native BF16/FP32 tails; same tolerances, different error profile. No accuracy equivalence claim.')
    return result


def make_matmul(shape_mnk,tile,*,variant='optimized',interpret=False,vmem_limit_bytes=48*2**20,
                product_order=DEFAULT_ORDER,traversal='mnk'):
    meta=describe(shape_mnk,tile,variant=variant,vmem_limit_bytes=vmem_limit_bytes,
                  product_order=product_order,traversal=traversal)
    m,n,k=meta['shape_mnk'];bm,bn,bk=meta['tile_bm_bn_bk']
    if variant=='peeled_edges':
        mc,nc,kc=meta['core_shape_mnk']
        core_fn=(make_matmul((mc,nc,kc),tile,variant='local_accumulators',interpret=interpret,
                   vmem_limit_bytes=vmem_limit_bytes,product_order=product_order,traversal=traversal)
                 if mc*nc*kc else None)
        def prepare(a,b):
            base._check_inputs(a,b,(m,k,n));return a,b
        def kernel(a,b):
            base._check_inputs(a,b,(m,k,n))
            if core_fn is None:return base._dot(a,b)
            top_left=core_fn(a[:mc,:kc],b[:kc,:nc])
            if kc<k:top_left=top_left+base._dot(a[:mc,kc:],b[kc:,:nc])
            top=jnp.concatenate((top_left,base._dot(a[:mc,:],b[:,nc:])),axis=1) if nc<n else top_left
            return jnp.concatenate((top,base._dot(a[mc:,:],b)),axis=0) if mc<m else top
        def finish(c):
            if c.shape!=(m,n):raise ValueError(f'Expected output shape {(m,n)}')
            return c
        def complete(a,b):return finish(kernel(*prepare(a,b)))
        complete.prepare,complete.kernel,complete.finish=prepare,kernel,finish
        complete.metadata={**meta,'interpret':bool(interpret)}
        return complete
    mp,kp,np_=meta['padded_shape_mkn'];masked=variant in ('masked_edges','optimized')
    if traversal=='mnk':
        a_index=lambda i,j,s:(i,s);b_index=lambda i,j,s:(s,j);c_index=lambda i,j,s:(i,j)
    else:
        a_index=lambda j,i,s:(i,s);b_index=lambda j,i,s:(s,j);c_index=lambda j,i,s:(i,j)
    body=functools.partial(_kernel,shape_mnk=(m,n,k),tile=(bm,bn,bk),masked=masked,
                           local=variant in ('local_accumulators','optimized'),
                           order=tuple(meta['product_order']),traversal=traversal)
    call=pl.pallas_call(body,grid=tuple(meta['grid']),
            in_specs=[pl.BlockSpec((bm,bk),a_index),pl.BlockSpec((bk,bn),b_index)],
            out_specs=pl.BlockSpec((bm,bn),c_index),
            out_shape=jax.ShapeDtypeStruct((mp,np_),jnp.float32),
            compiler_params=base._TPUCompilerParams(dimension_semantics=('parallel','parallel','arbitrary'),
                                                    vmem_limit_bytes=vmem_limit_bytes),
            interpret=interpret,name=f'{VERSION}_{variant}_{traversal}')
    def prepare(a,b):
        base._check_inputs(a,b,(m,k,n))
        if (mp,kp,np_)==(m,k,n):return a,b
        return jnp.pad(a,((0,mp-m),(0,kp-k))),jnp.pad(b,((0,kp-k),(0,np_-n)))
    def kernel(a,b):
        base._check_inputs(a,b,(mp,kp,np_));return call(a,b)
    def finish(c):
        if c.shape!=(mp,np_):raise ValueError(f'Expected output shape {(mp,np_)}')
        return c if (mp,np_)==(m,n) else c[:m,:n]
    def complete(a,b):return finish(kernel(*prepare(a,b)))
    complete.prepare,complete.kernel,complete.finish=prepare,kernel,finish
    complete.metadata={**meta,'interpret':bool(interpret)}
    return complete


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m',type=int,required=True);parser.add_argument('--n',type=int,required=True);parser.add_argument('--k',type=int,required=True)
    parser.add_argument('--tile',type=int,nargs=3,metavar=('BM','BN','BK'),default=[1024,1024,512])
    parser.add_argument('--variant',choices=VARIANTS,default='optimized')
    parser.add_argument('--traversal',choices=TRAVERSALS,default='mnk')
    parser.add_argument('--order',type=int,nargs=7,default=DEFAULT_ORDER)
    parser.add_argument('--describe-only',action='store_true')
    parser.add_argument('--interpret-correctness',action='store_true')
    args=parser.parse_args();shape=(args.m,args.n,args.k)
    kw=dict(variant=args.variant,product_order=args.order,traversal=args.traversal)
    if args.describe_only:
        print(json.dumps(describe(shape,tuple(args.tile),**kw),indent=2));return
    if args.interpret_correctness:jax.config.update('jax_platforms','cpu')
    devices=jax.devices()
    if not args.interpret_correctness and any(d.platform!='tpu' for d in devices):
        parser.error('TPU required for execution; --interpret-correctness permits CPU algebra checks, not performance results')
    if args.interpret_correctness and math.prod(shape)>50_000_000:
        parser.error('Interpreter example limited to 50 million M*N*K; use small boundary shapes')
    compatibility=base.enable_qualified_mosaic_v7_compat()
    import numpy as np
    rng=np.random.default_rng(20260920)
    a=jnp.asarray(rng.integers(-2,3,size=(args.m,args.k)),jnp.bfloat16)
    b=jnp.asarray(rng.integers(-2,3,size=(args.k,args.n)),jnp.bfloat16)
    fn=make_matmul(shape,tuple(args.tile),interpret=args.interpret_correctness,**kw)
    got=jax.jit(fn)(a,b).block_until_ready();expected=np.asarray(a,dtype=np.float32)@np.asarray(b,dtype=np.float32)
    passed=bool(np.array_equal(np.asarray(got),expected))
    print(json.dumps({'correctness_pass':passed,'validation':'exact small-integer algebra check',
                     'timing_reported':False,'devices':[str(d) for d in devices],
                     'mosaic_compatibility':compatibility,'metadata':fn.metadata},indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
