"""Architecture diagnostics: modern lowering, controlled scheduling, optional LLO.

Instrumented profiles are diagnostic only. Performance claims use subsequent
uninstrumented screening/confirmation. No periodic-counter claims on v6e.
"""
import os,sys,json,importlib.metadata,inspect,mmap,subprocess,shutil
from pathlib import Path
from . import benchmark_v6e_opt_v002 as routing
from . import benchmark_v6e_opt_v001 as benchmark
original_make=benchmark.make_function

def make_function(arm,shape):
    if arm.get('implementation')=='cubic':
        from .kernels_v002 import make_matmul
        return make_matmul('cubic_full',shape,tuple(arm['tile']),variant='output_accumulator',vmem_limit_bytes=arm['vmem_limit_bytes'])
    if arm.get('implementation')=='pipeline':
        from .kernels_v6e_v003 import make_matmul
        return make_matmul(shape,tuple(arm['tile']),arm['depth'],mode=arm['mode'],
            buffers=arm['buffers'],traversal=arm['traversal'],vmem_limit_bytes=arm['vmem_limit_bytes'])
    return original_make(arm,shape)

original_compile=benchmark.Runner.compile_entries

def compile_entries(self,group,a,b):
    out=self.args.output_dir
    if not (out/'capabilities.json').exists():
        import jax,libtpu
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        caps=dict(jax=jax.__version__,libtpu=importlib.metadata.version('libtpu'),
            profiler=importlib.metadata.version('xprof-nightly'),device=str(jax.devices()[0]),
            compiler_params=str(inspect.signature(pltpu.CompilerParams)),block_spec=str(inspect.signature(pl.BlockSpec)),
            libtpu_init_args=os.environ.get('LIBTPU_INIT_ARGS'),xla_flags=os.environ.get('XLA_FLAGS'),
            hardware_info=repr(pltpu.get_tpu_info()),flags={})
        for p in Path(libtpu.__file__).parent.glob('*libtpu*.so'):
            with p.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as blob:
                for flag in ('xla_xprof_register_llo_debug_info','xla_xprof_enable_custom_call_tracing','tpu_enable_periodic_counter_sampling'):
                    caps['flags'][flag]=blob.find(flag.encode())>=0
        (out/'capabilities.json').write_text(json.dumps(caps,indent=2)+'\n')
    return original_compile(self,group,a,b)

benchmark.make_function=make_function
benchmark.Runner.compile_entries=compile_entries

def main():
    cfg=Path(sys.argv[sys.argv.index('--campaign')+1]);config=json.loads(cfg.read_text())
    instrumented=config.get('llo_instrumentation',False)
    if instrumented:
        os.environ['LIBTPU_INIT_ARGS']='--xla_xprof_enable_custom_call_tracing=true --xla_xprof_register_llo_debug_info=true'
    code=benchmark.main()
    if instrumented and code==0:
        out=Path(sys.argv[sys.argv.index('--output-dir')+1])
        reports=[]
        for profile in sorted((out/'profiles').iterdir()):
            for verb in ('get_kernel_stats','get_llo_analysis','get_llo_debug_string'):
                command=['xprof',verb,str(profile)]
                try:
                    result=subprocess.run(command,text=True,capture_output=True,timeout=180)
                    (profile/(verb+'.stdout.txt')).write_text(result.stdout)
                    (profile/(verb+'.stderr.txt')).write_text(result.stderr)
                    reports.append(dict(command=command,exit_code=result.returncode))
                except Exception as e:reports.append(dict(command=command,error=str(e)))
        (out/'llo-analysis-status.json').write_text(json.dumps(reports,indent=2)+'\n')
    return code

if __name__=='__main__':raise SystemExit(main())
