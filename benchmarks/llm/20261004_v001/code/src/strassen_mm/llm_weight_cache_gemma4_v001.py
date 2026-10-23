"""Gemma4 private immutable disk layouts; weak mappings and an explicit disk bound.

Raw checkpoint values are never archived. Keys bind to verified checkpoint
bytes and packing implementation. Identical layouts share the same read-only
mapping even when their algorithms/output stores differ.
"""
import hashlib,json,os,shutil,time,weakref
from pathlib import Path
import ml_dtypes
import numpy as np
from .llm_stream_gemma4_v001 import pack_host
from .gemma4_contract_v001 import layer_geometry

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def key(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class LayoutCache:
    def __init__(self,root,identity,max_bytes=None):
        self.root=Path(root)/key(identity);self.root.mkdir(parents=True,exist_ok=True);self.root.chmod(0o700)
        self.identity=identity;self.maps=weakref.WeakValueDictionary();self.verified={}
        budget=self.root/'disk-budget'
        self.max_bytes=int(max_bytes if max_bytes is not None else int(budget.read_text()) if budget.exists() else min(180*1024**3,shutil.disk_usage(self.root).free*.65))
        if not budget.exists():budget.write_text(str(self.max_bytes))
        self.stats=dict(hits=0,misses=0,bytes_written=0,bypasses=0,evictions=0,verified_bytes=0)
        self.used=sum(p.stat().st_size for p in self.root.glob('*.bin'))
    def room(self,size):
        def enough():return self.used+size<=self.max_bytes and shutil.disk_usage(self.root).free>=size+4*1024**3
        if enough():return True
        for p in sorted(self.root.glob('*.json'),key=lambda p:p.stat().st_mtime):
            try:m=json.loads(p.read_text())
            except (ValueError,OSError):continue
            # Canonical weights are shared across every workload. Only evict
            # unreferenced derived layouts; live mappings are never unlinked.
            if m['descriptor']['kind']=='canonical' or p.stem in self.maps:continue
            data=p.with_suffix('.bin');size_old=data.stat().st_size if data.exists() else 0
            data.unlink(missing_ok=True);p.unlink();self.used-=size_old;self.verified.pop(p.stem,None);self.stats['evictions']+=1
            if enough():return True
        return enough()
    def get(self,descriptor,build):
        ident=key(descriptor);meta=self.root/(ident+'.json');data=meta.with_suffix('.bin')
        mapped=self.maps.get(ident)
        if mapped is not None:self.stats['hits']+=1;return mapped
        if meta.exists():
            m=json.loads(meta.read_text())
            if m['descriptor']!=descriptor or not data.exists() or data.stat().st_size!=m['bytes']:raise ValueError('Invalid private cache entry')
            stamp=(data.stat().st_size,data.stat().st_mtime_ns)
            if self.verified.get(ident)!=stamp:
                if digest(data)!=m['sha256']:raise ValueError('Cached weight checksum mismatch')
                self.verified[ident]=stamp;self.stats['verified_bytes']+=m['bytes']
            dtype=ml_dtypes.bfloat16 if m['dtype']=='bfloat16' else np.dtype(m['dtype'])
            mapped=np.memmap(data,dtype=dtype,mode='r',shape=tuple(m['shape']));self.maps[ident]=mapped
            os.utime(meta,None);self.stats['hits']+=1;return mapped
        value=np.ascontiguousarray(build());self.stats['misses']+=1
        if not self.room(value.nbytes):
            self.stats['bypasses']+=1;value.flags.writeable=False;return value
        temporary=data.with_suffix('.bin.tmp');value.tofile(temporary)
        checksum=digest(temporary);temporary.replace(data)
        m=dict(descriptor=descriptor,shape=list(value.shape),dtype=str(value.dtype),bytes=value.nbytes,sha256=checksum)
        temporary=meta.with_suffix('.json.tmp');temporary.write_text(json.dumps(m));temporary.replace(meta)
        self.used+=value.nbytes;self.stats['bytes_written']+=value.nbytes
        self.verified[ident]=(data.stat().st_size,data.stat().st_mtime_ns)
        del value
        return self.get(descriptor,lambda:None)
    @staticmethod
    def prefault(arrays):
        seen=set();total=0
        for a in arrays:
            address=int(a.__array_interface__['data'][0]);identity=(address,a.nbytes)
            if identity in seen:continue
            seen.add(identity)
            # Touch each OS page outside inference timing. The OS may reclaim
            # inactive mappings; the three full warmups are retained as well.
            raw=np.asarray(a).view(np.uint8).reshape(-1)
            int(np.sum(raw[::4096],dtype=np.uint64));total+=a.nbytes
        return total

class CachedCheckpoint:
    def __init__(self,cp,cache):self.base=cp;self.cache=cache;self.config=cp.config
    def __getattr__(self,name):return getattr(self.base,name)
    def layer(self,index):
        # Read/transposes all original tensors only on a canonical-cache miss.
        natural=None
        names=list(layer_geometry(self.config,index)['projection_shapes_mkn'])
        names += ['norm1','norm2','norm3','norm4','qnorm','knorm','layer_scalar']
        def build(name):
            nonlocal natural
            if natural is None:natural=self.base.layer(index)
            return natural[name]
        return {name:self.cache.get(dict(kind='canonical',layer=index,name=name),lambda name=name:build(name)) for name in names}
    def head(self):return self.cache.get(dict(kind='canonical',name='head'),self.base.head)

def layout_descriptor(layer,name,meta):
    if not meta or meta['implementation']=='native':return dict(kind='canonical',layer=layer,name=name)
    _,k,n=meta['shape_mkn'];_,kp,npad=meta['padded_shape_mkn'];spec=meta['epilogue'];kind=spec['kind']
    interleaved=kind in ('rope','qk_norm_rope','swiglu','geglu')
    if not interleaved and (k,n)==(kp,npad):return dict(kind='canonical',layer=layer,name=name)
    return dict(kind='packed',layer=layer,name=name,kn=[k,n],padded_kn=[kp,npad],
                layout=kind if interleaved else 'plain',bn=meta['tile_bm_bn_bk'][1] if interleaved else None,
                head_dim=spec.get('head_dim') if kind in ('rope','qk_norm_rope') else None)

def packed_layer(cp,index,metadata):
    natural=cp.layer(index);out={}
    for name,value in natural.items():
        meta=metadata['projections'].get(name);desc=layout_descriptor(index,name,meta)
        if desc['kind']=='canonical':out[name]=value
        else:out[name]=cp.cache.get(desc,lambda name=name,value=value,meta=meta:pack_host({name:value},{'projections':{name:meta}})[name])
    return out

def cache_identity(cp):
    # Bind all layout/semantic producers, so a future adapter never silently
    # accepts files made under an older geometry or normalization contract.
    sources = ('llm_stream_gemma4_v001.py','llm_weight_cache_gemma4_v001.py',
               'checkpoint_gemma4_v001.py','gemma4_contract_v001.py',
               'model_gemma4_fused_v001.py','model_gemma4_v001.py',
               'fusion_v003.py','fusion_v004.py','kernels_fused_v005.py')
    return dict(revision=cp.manifest['revision'],files=cp.manifest['files'],config=cp.config,
                source_sha256={name:digest(Path(__file__).with_name(name)) for name in sources},
                cache_version='gemma4-v001')
