"""Keep one algorithm's prepared host matrices resident at a time.

Packing and page-in occur before warmup, outside inference timing. Every arm
uses the same serialized layer transfers. Switching arms never pools timings
across runtimes, and compiled functions persist after host buffers are evicted.
"""
import gc,time
import numpy as np
from .llm_stream_v001 import StreamedModel,pack_host

class ModelPool:
    def __init__(self,cp,batch,sequence,profiles,dtype):
        self.cp=cp;self.batch=batch;self.sequence=sequence;self.profiles=profiles;self.dtype=dtype
        self.models={};self.active=None

    def activate(self,name):
        started=time.perf_counter()
        if self.active==name:return self.models[name],0.
        if self.active is not None:
            old=self.models[self.active];old.host.clear();old.head=None
            gc.collect()
        if name not in self.models:
            model=StreamedModel(self.cp,self.batch,self.sequence,self.profiles[name],output_dtype=self.dtype)
            model.host_keys=[key for key,_ in model.host]
            self.models[name]=model
        else:
            model=self.models[name]
            for i,key in enumerate(model.host_keys):model.host.append((key,pack_host(self.cp.layer(i),model.builders[key].metadata)))
            model.head=self.cp.head()
        self.active=name
        return model,time.perf_counter()-started

    def close(self):
        self.models.clear();self.active=None;gc.collect()
