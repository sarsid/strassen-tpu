"""Aligned panel-length experiments with current and pinned upstream kernels.

No padding or model epilogues. The pinned parent S1 is unmodified. Its blocked
cubic kernel gets an explicit FP32-output wrapper for the FP32 comparison;
the BF16 comparison calls the original parent public entry point verbatim.
"""
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from . import kernels_v001 as base


@functools.lru_cache(maxsize=1)
def parent_modules():
    root=Path(__file__).parent/'_parent_95be1fb'
    provenance=json.loads((root/'provenance.json').read_text())
    for name,info in provenance['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=info['sha256']:
            raise ValueError('Pinned parent source changed: '+name)
        existing=sys.modules.get(name.removesuffix('.py'))
        if existing is not None and Path(existing.__file__).resolve()!=root/name:
            raise RuntimeError('Refuse conflicting upstream module name')
    # Do not allow the parent's deployment defaults to handicap Native.
    os.environ['STRASSEN_MANAGE_CEILING']='0'
    os.environ['STRASSEN_VMEM_PROFILE']='compact16'
    before=os.environ.get('LIBTPU_INIT_ARGS')
    sys.path.insert(0,str(root))
    try:
        sp=importlib.import_module('strassen_pallas')
        cubic=importlib.import_module('benchmark_cubic_control')
    finally:
        sys.path.remove(str(root))
        os.environ['STRASSEN_VMEM_PROFILE']='compact16'
    if os.environ.get('LIBTPU_INIT_ARGS')!=before or sp.MANAGES_PROCESS_CEILING:
        raise RuntimeError('Parent import altered process-wide compiler settings')
    return sp,cubic


def make_matmul(shape, tile, implementation, *, output_dtype='float32', depth=0,
                mode=None, vmem_limit_bytes=112*1024**2, interpret=False):
    shape=base._positive_triple(shape,'shape (M,K,N)')
    m,k,n=shape
    dtype=jnp.dtype(output_dtype)
    if dtype not in (jnp.dtype(jnp.float32),jnp.dtype(jnp.bfloat16)):
        raise ValueError('Only BF16/FP32 output supported')
    if implementation=='native':
        def complete(a,b):
            base._check_inputs(a,b,shape)
            return jnp.matmul(a,b,precision=jax.lax.Precision.DEFAULT,
                              preferred_element_type=jnp.float32).astype(dtype)
        metadata={}
    else:
        tile=base._positive_triple(tile,'tile (BM,BN,BK)');bm,bn,bk=tile
        if any(x%t for x,t in zip(shape,(bm,bk,bn))):raise ValueError('Full-K study prohibits padding')
        if implementation=='current':
            if dtype!=jnp.dtype(jnp.float32):raise ValueError('Current kernel output contract remains FP32')
            from .kernels_v6e_v003 import make_matmul as current
            complete=current(shape,tile,depth,mode=mode or ('deferred' if depth==1 else 'hybrid'),
                buffers=2,vmem_limit_bytes=vmem_limit_bytes,interpret=interpret)
            metadata=dict(complete.metadata)
        elif implementation in ('parent_s1','parent_cubic'):
            sp,cubic=parent_modules()
            if implementation=='parent_s1':
                def complete(a,b):
                    base._check_inputs(a,b,shape)
                    return sp.strassen_matmul(a,b,bm=bm,bn=bn,bk=bk,interleave_products=True,
                        output_dtype=dtype,interpret=interpret,vmem_limit_bytes=vmem_limit_bytes)
            elif dtype==jnp.dtype(jnp.bfloat16):
                def complete(a,b):
                    base._check_inputs(a,b,shape)
                    return cubic.cubic_matmul(a,b,variant='blocked',bm=bm,bn=bn,bk=bk,
                        interpret=interpret,vmem_limit_bytes=vmem_limit_bytes)
            else:
                sp.check_tiling(bm,bn,bk)
                call=pl.pallas_call(functools.partial(cubic._cubic_blocked_kernel,
                    nk=k//bk,dot_precision=jax.lax.Precision.DEFAULT),grid=(m//bm,n//bn,k//bk),
                    in_specs=[pl.BlockSpec((bm,bk),lambda i,j,s:(i,s)),pl.BlockSpec((bk,bn),lambda i,j,s:(s,j))],
                    out_specs=pl.BlockSpec((bm,bn),lambda i,j,s:(i,j)),
                    out_shape=jax.ShapeDtypeStruct((m,n),dtype),
                    scratch_shapes=[pltpu.VMEM((4,bm//2,bn//2),jnp.float32)],
                    compiler_params=base._TPUCompilerParams(dimension_semantics=('parallel','parallel','arbitrary'),vmem_limit_bytes=vmem_limit_bytes),
                    interpret=interpret,name='parent_blocked_cubic_fp32')
                def complete(a,b):
                    base._check_inputs(a,b,shape)
                    return call(a,b)
            metadata=dict(parent_commit='95be1fb088656a89813b04492e1d77c66b36ccf9',
                parent_source_unmodified=True,parent_cubic_fp32_wrapper=implementation=='parent_cubic' and dtype==jnp.dtype(jnp.float32))
        else:raise ValueError('Unknown implementation')
    complete.metadata={**metadata,'kernel_version':'kernels_fullk_v001','implementation':implementation,
        'algorithm':implementation,'variant':mode or implementation,'shape_mkn':list(shape),
        'padded_shape_mkn':list(shape),'padding_required':False,'padded_volume_ratio':1.,
        'tile_bm_bn_bk':list(tile) if tile else None,'input_dtype':'bfloat16','output_dtype':output_dtype,
        'accumulation_dtype':'float32','depth':depth,'strassen_levels_per_tile':depth,
        'vmem_limit_bytes':vmem_limit_bytes if tile else None,
        'k_panels':k//tile[2] if tile else None,'full_contraction':tile is not None and tile[2]==k,
        'complete_call_scope':'aligned device MM including output conversion; excludes transfer/compile'}
    return complete
