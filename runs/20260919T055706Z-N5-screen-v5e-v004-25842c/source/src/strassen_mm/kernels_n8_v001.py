"""N8 matched BF16 epilogues with optional Strassen early finalization.

Adapted from historical strassen-tpu/strassen_pallas.py product-aware SwiGLU
and residual schedules, using the new MM study's DEFAULT BF16/FP32 contract.
Every arm rounds each projection to BF16 before the epilogue. SwiGLU rounds
SiLU to BF16 before its BF16 product, matching the application model boundary.
Weight relayout/padding is explicit and may be measured separately from reuse.
"""
from __future__ import annotations
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from . import kernels_v001 as core
from . import kernels_v002 as mm


def activation(gate, up):
    return (jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16)
            * up.astype(jnp.bfloat16)).astype(jnp.bfloat16)


def epilogue(product, residual, kind):
    product = product.astype(jnp.bfloat16)
    if kind == 'swiglu':
        gate, up = jnp.split(product, 2, axis=-1)
        return activation(gate, up)
    return (product.astype(jnp.float32) + residual.astype(jnp.float32)).astype(jnp.bfloat16)


def _body(a, b, residual, output, acc, *, nk, algorithm, kind, early):
    step = pl.program_id(2)
    bm,bk=a.shape; bn=b.shape[1]
    hm,hk,hn=bm//2,bk//2,bn//2

    @pl.when(step == 0)
    def initialize(): acc[...] = jnp.zeros(acc.shape,jnp.float32)

    def store(index):
        if kind == 'swiglu':
            row = slice(0,hm) if index == 0 else slice(hm,bm)
            gate,up = ((acc[0],acc[1]) if index == 0 else (acc[2],acc[3]))
            output[row,:] = activation(gate.astype(jnp.bfloat16),up.astype(jnp.bfloat16))
        else:
            row=slice(0,hm) if index<2 else slice(hm,bm)
            col=slice(0,hn) if index%2==0 else slice(hn,bn)
            output[row,col] = (acc[index].astype(jnp.bfloat16).astype(jnp.float32)
                              + residual[row,col].astype(jnp.float32)).astype(jnp.bfloat16)

    if algorithm == 'cubic_full':
        acc[...] += core._dot(a[...],b[...])
        @pl.when(step == nk-1)
        def finalize_full():
            if kind=='swiglu':
                output[...] = activation(acc[:,:hn].astype(jnp.bfloat16),acc[:,hn:].astype(jnp.bfloat16))
            else:
                output[...] = (acc[...].astype(jnp.bfloat16).astype(jnp.float32)
                               + residual[...].astype(jnp.float32)).astype(jnp.bfloat16)
        return
    a0,a1,a2,a3=a[:hm,:hk],a[:hm,hk:],a[hm:,:hk],a[hm:,hk:]
    b0,b1,b2,b3=b[:hk,:hn],b[:hk,hn:],b[hk:,:hn],b[hk:,hn:]

    def update(i,p,negative=False):
        if negative: acc[i] -= p
        else: acc[i] += p

    if algorithm=='strassen':
        def add(x,y): return (x+y).astype(jnp.bfloat16)
        def sub(x,y): return (x-y).astype(jnp.bfloat16)
        def p1():
            p=core._dot(add(a0,a3),add(b0,b3)); update(0,p); update(3,p)
        def p2():
            p=core._dot(add(a2,a3),b0); update(2,p); update(3,p,True)
        def p3():
            p=core._dot(a0,sub(b1,b3)); update(1,p); update(3,p)
        def p4():
            p=core._dot(a3,sub(b2,b0)); update(0,p); update(2,p)
        def p5():
            p=core._dot(add(a0,a1),b3); update(0,p,True); update(1,p)
        def p6(): update(3,core._dot(sub(a2,a0),add(b0,b1)))
        def p7(): update(0,core._dot(sub(a1,a3),add(b2,b3)))
        def ordinary():
            p4();p6();p5();p2();p7();p3();p1()
        if early:
            @pl.when(step != nk-1)
            def earlier_panel(): ordinary()
            @pl.when(step == nk-1)
            def last_panel():
                if kind=='swiglu':
                    p4();p5();p7();p3();p1();store(0);p6();p2();store(1)
                else:
                    p4();p6();p5();p2();store(2);p7();p3();store(1);p1();store(0);store(3)
            return
        ordinary()
    else:
        products=((0,a0,b0),(0,a1,b2),(1,a0,b1),(1,a1,b3),
                  (2,a2,b0),(2,a3,b2),(3,a2,b1),(3,a3,b3))
        for index in (0,2,4,6,1,3,5,7):
            quadrant,lhs,rhs=products[index]; update(quadrant,core._dot(lhs,rhs))
    @pl.when(step == nk-1)
    def finalize():
        for index in range(2 if kind=='swiglu' else 4): store(index)


def make_epilogue(algorithm, shape, tile, *, kind, fused=False, early=False,
                  variant=None, interpret=False, vmem_limit_bytes=48*1024**2):
    if kind not in ('swiglu','residual'): raise ValueError('Unknown epilogue')
    if early and (not fused or algorithm!='strassen'): raise ValueError('Early mode requires fused Strassen')
    m,k,n=shape; bm,bn,bk=tile
    if kind=='swiglu' and n%2: raise ValueError('SwiGLU requires combined even gate/up width')
    default_variant='interleaved' if algorithm in ('strassen','cubic_quadrant') else 'plain'
    variant=variant or default_variant
    if not fused:
        matrix=mm.make_matmul(algorithm,shape,tile,variant=variant,interpret=interpret,
                              vmem_limit_bytes=vmem_limit_bytes)
        def complete(a,b,residual=None): return epilogue(matrix(a,b),residual,kind)
        complete.prepare=lambda a,b,r=None:(a,b,r)
        complete.prepared=complete
        complete.metadata={**matrix.metadata,'epilogue':kind,'fusion':'outside_custom_mm',
                           'projection_rounding':'BF16 before epilogue','output_dtype':'bfloat16'}
        return complete
    if algorithm=='native': raise ValueError('Native joint graph uses fused=False; XLA owns fusion')
    if variant not in ('plain','interleaved'):
        raise ValueError('Fused BF16-output kernels require scratch accumulators; output-accumulator variant is a separate unfused baseline')
    core.check_tile('strassen' if algorithm!='cubic_full' else algorithm,tile)
    if bm%16 or bn%256: raise ValueError('Epilogue tile alignment requires BM multiple16 and BN multiple256')
    mp=((m+bm-1)//bm)*bm; kp=((k+bk-1)//bk)*bk
    if kind=='swiglu':
        half=bn//2; width=((n//2+half-1)//half)*half; np_=width*2
    else: np_=((n+bn-1)//bn)*bn; width=np_
    body=functools.partial(_body,nk=kp//bk,algorithm=algorithm,kind=kind,early=early)
    if kind=='swiglu':
        def kernel(a,b,o,acc): return body(a,b,None,o,acc)
        in_specs=[pl.BlockSpec((bm,bk),lambda i,j,s:(i,s)),pl.BlockSpec((bk,bn),lambda i,j,s:(s,j))]
    else:
        def kernel(a,b,r,o,acc): return body(a,b,r,o,acc)
        in_specs=[pl.BlockSpec((bm,bk),lambda i,j,s:(i,s)),pl.BlockSpec((bk,bn),lambda i,j,s:(s,j)),
                  pl.BlockSpec((bm,bn),lambda i,j,s:(i,j))]
    output_width=bn//2 if kind=='swiglu' else bn
    call=pl.pallas_call(kernel,grid=(mp//bm,np_//bn,kp//bk),in_specs=in_specs,
         out_specs=pl.BlockSpec((bm,output_width),lambda i,j,s:(i,j)),
         out_shape=jax.ShapeDtypeStruct((mp,width),jnp.bfloat16),
         scratch_shapes=[pltpu.VMEM((bm,bn) if algorithm=='cubic_full' else (4,bm//2,bn//2),jnp.float32)],
         compiler_params=core._TPUCompilerParams(dimension_semantics=('parallel','parallel','arbitrary'),
                                               vmem_limit_bytes=vmem_limit_bytes),
         interpret=interpret,name=f'n8_v001_{algorithm}_{kind}_{"early" if early else "fused"}')

    def prepare(a,b,residual=None):
        core._check_inputs(a,b,shape)
        a=jnp.pad(a,((0,mp-m),(0,kp-k)))
        if kind=='swiglu':
            gate,up=jnp.split(b,2,axis=1)
            gate=jnp.pad(gate,((0,kp-k),(0,width-n//2)))
            up=jnp.pad(up,((0,kp-k),(0,width-n//2)))
            b=jnp.stack((gate.reshape(kp,width//(bn//2),bn//2),
                         up.reshape(kp,width//(bn//2),bn//2)),axis=2).reshape(kp,np_)
            return a,b
        if residual is None or residual.shape!=(m,n): raise ValueError('Residual shape mismatch')
        return a,jnp.pad(b,((0,kp-k),(0,np_-n))),jnp.pad(residual,((0,mp-m),(0,np_-n)))

    def prepared(*operands): return call(*operands)[:m,:(n//2 if kind=='swiglu' else n)]
    def complete(a,b,residual=None): return prepared(*prepare(a,b,residual))
    complete.prepare=prepare; complete.prepared=prepared
    complete.metadata={'kernel_version':'kernels_n8_v001','algorithm':algorithm,'variant':variant,
        'shape_mkn':list(shape),'tile_bm_bn_bk':list(tile),'epilogue':kind,
        'fusion':'early_finalization' if early else 'standard_fused','output_dtype':'bfloat16',
        'projection_rounding':'BF16 before epilogue','accumulation_dtype':'float32',
        'input_dtype':'bfloat16','dot_precision':'DEFAULT','accumulator_storage':'scratch',
        'preparation':'device padding and paired gate/up weight relayout; separately report reuse',
        'early_swiglu_changes_final_panel_accumulation_order':bool(early and kind=='swiglu')}
    return complete
