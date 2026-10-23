"""Reuse immutable layouts; one active model's host references at a time."""
import gc,time
from .llm_stream_v002 import StreamedModel
from .llm_weight_cache_v001 import packed_layer

class ModelPool:
    def __init__(self,cp,batch,sequence,profiles,dtype):
        self.cp=cp;self.batch=batch;self.sequence=sequence;self.profiles=profiles;self.dtype=dtype
        self.models={};self.active=None;self.last_preparation={}
    def activate(self,name):
        started=time.perf_counter();before=dict(self.cp.cache.stats)
        if self.active!=name:
            if self.active is not None:
                old=self.models[self.active];old.host.clear();old.head=None;gc.collect()
            if name not in self.models:
                model=StreamedModel(self.cp,self.batch,self.sequence,self.profiles[name],output_dtype=self.dtype,
                                    prepared_provider=lambda i,m:packed_layer(self.cp,i,m))
                model.host_keys=[k for k,_ in model.host];self.models[name]=model
            else:
                model=self.models[name]
                model.host=[(key,packed_layer(self.cp,i,model.builders[key].metadata)) for i,key in enumerate(model.host_keys)]
                model.head=self.cp.head()
            self.active=name
        model=self.models[name]
        arrays=[a for _,w in model.host for a in w.values()]+[model.head,model.norm]
        touched=self.cp.cache.prefault(arrays)
        elapsed=time.perf_counter()-started
        self.last_preparation=dict(seconds=elapsed,prefault_bytes=touched,cache_delta={k:self.cp.cache.stats[k]-before[k] for k in before})
        return model,elapsed
    def close(self):self.models.clear();self.active=None;gc.collect()
