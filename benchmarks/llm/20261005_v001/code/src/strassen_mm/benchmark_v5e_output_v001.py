"""Historical v5e experiment with only its requested output precision changed.

Candidate inventory, seeds, ordering, statistics, padding, accumulation and
VMEM policy remain in the frozen historical runner. FP32 calls use the exact
original factory; BF16 calls apply one final conversion inside the timed call.
"""
import json
from pathlib import Path
import sys
from . import benchmark_mlsys_shapes_v001 as historical

def main():
    path=Path(sys.argv[sys.argv.index('--campaign')+1]);cfg=json.loads(path.read_text())
    output=cfg['precision']['output_dtype']
    if output not in ('float32','bfloat16'):raise ValueError('Explicit output dtype required')
    if cfg['device']['target']!='v5e':raise ValueError('Historical v5e experiment only')
    if output=='bfloat16':
        original=historical.make_function
        def make_function(arm,shape):
            fn=original(arm,shape)
            import jax.numpy as jnp
            def converted(a,b):return fn(a,b).astype(jnp.bfloat16)
            converted.metadata={**fn.metadata,'output_dtype':'bfloat16',
                'output_adapter':'one final cast of unchanged historical FP32 complete call',
                'complete_call_scope':'historical padding+MM+crop+BF16 conversion; excludes transfer/compile'}
            return converted
        historical.make_function=make_function
    return historical.main()

if __name__=='__main__':raise SystemExit(main())
