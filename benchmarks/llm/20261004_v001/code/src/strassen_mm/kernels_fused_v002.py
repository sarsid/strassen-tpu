"""v001 Pallas kernels with model-native BF16 XLA epilogues for Native."""
import jax.numpy as jnp
from .kernels_fused_v001 import make_projection as previous
from .fusion_v001 import Epilogue
from .fusion_v002 import reference
from . import kernels_v001 as base


def make_projection(arm,shape,spec=Epilogue(),*,interpret=False):
    original = previous(arm,shape,spec,interpret=interpret)
    if arm['implementation'] != 'native':
        return original
    def prepared(a,b,**aux):
        # Reuse all of the original shape/auxiliary validation on tracing;
        # the unused expression is removed from the compiled graph.
        original.prepared(a,b,**aux)
        return reference(base._dot(a,b),spec,jnp.dtype(arm['output_dtype']),**aux)
    def complete(a,b,**aux):
        return prepared(a,original.prepare_weights(b),**aux)
    complete.prepare_weights = original.prepare_weights
    complete.prepared = prepared
    complete.metadata = dict(original.metadata,native_epilogue_version='fusion_v002')
    return complete
