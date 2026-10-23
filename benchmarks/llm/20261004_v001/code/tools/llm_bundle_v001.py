"""Validate the complete launch bundle before a paid runtime is allocated."""
import ast,hashlib,json,subprocess,sys,tarfile,tempfile
from pathlib import Path
WORKER='runtime/llm_case_remote_v003.py'
PREPARE='tools/prepare_llm_case_v002.py'
MEASURE='tools/run_llm_case_v002.py'
REQUIRED=[WORKER,PREPARE,MEASURE,'tools/load_gemma_text_v001.py','tools/stage_mlsys_auth_v002.py','runtime/prepare_mlsys_private_v001.py','runtime/parent_replication_remote_v001.py','runtime/poll_llm_queue_v001.py','tools/check_fused_tpu_v004.py','tools/check_llm_stream_v002.py','tools/fusion_oracle_v002.py','configs/llm_v6e_v002.json']


def validate(source,job,python=sys.executable):
    source=Path(source)
    for name in REQUIRED:
        if not (source/name).is_file():raise ValueError('Missing launch dependency: '+name)
    tree=ast.parse((source/WORKER).read_text())
    references={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith(('tools/','runtime/')) and n.value.endswith('.py')}
    if not {WORKER,PREPARE,MEASURE}<=references:raise ValueError('Worker does not use the qualified entrypoints')
    for name in references:
        if not (source/name).is_file():raise ValueError('Worker references an absent script: '+name)
    protocol=json.loads((source/'configs/llm_v6e_v002.json').read_text())
    model=next((m for m in protocol['models'] if m['id']==job['model']),None)
    if not model or model['revision']!=job['revision'] or job['architecture']!='v6e':raise ValueError('Unpinned model or architecture')
    if dict(batch=job['batch'],sequence=job['sequence']) not in protocol['prefill']:raise ValueError('Unplanned workload')
    if not job['output_dtypes'] or len(set(job['output_dtypes']))!=len(job['output_dtypes']) or not set(job['output_dtypes'])<=set(protocol['output_dtypes']):raise ValueError('Invalid store subset')
    if json.loads((source/'job.json').read_text())!=job:raise ValueError('Embedded job differs from controller job')
    for script,option in [(PREPARE,'--model'),(MEASURE,'--output-dtypes'),(WORKER,'--worker')]:
        result=subprocess.run([python,str(source/script),'--help'],capture_output=True,text=True,timeout=30)
        if result.returncode or option not in result.stdout:raise ValueError('Bundled CLI contract failed: '+script)
    return dict(status='passed',worker=WORKER,prepare=PREPARE,measure=MEASURE,checked_files={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in REQUIRED})


def validate_archive(archive,job,python=sys.executable):
    with tempfile.TemporaryDirectory(prefix='llm-launch-preflight-') as temporary:
        with tarfile.open(archive) as package:package.extractall(temporary,filter='data')
        return validate(Path(temporary),job,python)
