"""Read-only idle-host resource preflight for official-model preparation.

No JAX import, model download, allocation or device work. Identity verification
follows application_prep_v001: endpoint, hostname, boot ID and installed core
versions must match the sealed cohort. Resource thresholds are conservative
planning estimates, never hardware measurements or proof that a loader fits.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from . import benchmark_v001 as base

GIB=1024**3
CHECKPOINT_BYTES={'Qwen/Qwen3-0.6B':1_503_300_328,'mistralai/Mistral-7B-v0.3':14_496_080_928}


def read_text(path):
    try: return Path(path).read_text().strip()
    except OSError: return None


def integer_or_none(text):
    try: return int(text) if text is not None else None
    except ValueError: return None


def meminfo():
    text=read_text('/proc/meminfo')
    if text is None: return {'available':False}
    fields={}
    for line in text.splitlines():
        key,_,tail=line.partition(':');parts=tail.split()
        if not parts: continue
        value=integer_or_none(parts[0])
        if value is not None:
            fields[key]=value*(1024 if len(parts)>1 and parts[1]=='kB' else 1)
    wanted=('MemTotal','MemAvailable','MemFree','Buffers','Cached','SwapTotal','SwapFree','SReclaimable')
    return {'available':True,'unit':'bytes','fields':{name:fields.get(name) for name in wanted}}


def cgroups():
    membership=read_text('/proc/self/cgroup')
    mounts=read_text('/proc/self/mountinfo')
    records=[]
    if membership is None or mounts is None: return {'available':False,'records':[]}
    members=[]
    for line in membership.splitlines():
        fields=line.split(':',2)
        if len(fields)==3: members.append((fields[1].split(',') if fields[1] else [],fields[2]))
    for line in mounts.splitlines():
        left,separator,right=line.partition(' - ')
        if not separator: continue
        before=left.split();after=right.split()
        if len(before)<5 or len(after)<3 or after[0] not in ('cgroup','cgroup2'): continue
        filesystem=after[0];mount_root=Path(before[3]);mountpoint=Path(before[4])
        controllers=after[2].split(',')
        for member_controllers,member_path in members:
            if filesystem=='cgroup2' and member_controllers: continue
            if filesystem=='cgroup' and not set(member_controllers)&set(controllers): continue
            member=Path(member_path)
            # A namespaced root may hide the original cgroup ancestry. In that
            # case the visible mount root remains useful but is labelled so.
            if member.is_relative_to(mount_root):
                path=mountpoint/member.relative_to(mount_root);resolution='membership_relative_to_mount_root'
            else:
                path=mountpoint;resolution='visible_mount_root_fallback'
            if not path.resolve().is_relative_to(mountpoint.resolve()): continue
            names=('memory.max','memory.high','memory.current','memory.swap.max','memory.swap.current',
                   'cpu.max','cpuset.cpus.effective') if filesystem=='cgroup2' else (
                   'memory.limit_in_bytes','memory.usage_in_bytes','memory.memsw.limit_in_bytes',
                   'memory.memsw.usage_in_bytes','cpu.cfs_quota_us','cpu.cfs_period_us','cpuset.cpus')
            visible=[path]
            while visible[-1]!=mountpoint and visible[-1].parent.is_relative_to(mountpoint):
                visible.append(visible[-1].parent)
            for visible_path in visible:
                record={'version':2 if filesystem=='cgroup2' else 1,'path':str(visible_path),
                        'resolution':resolution,'membership':member_path,'controllers':member_controllers,
                        'visible_ancestor':visible_path!=path}
                record['raw']={name:read_text(visible_path/name) for name in names}
                maximum=record['raw'].get('memory.max') if filesystem=='cgroup2' else record['raw'].get('memory.limit_in_bytes')
                current=record['raw'].get('memory.current') if filesystem=='cgroup2' else record['raw'].get('memory.usage_in_bytes')
                limit=integer_or_none(maximum);used=integer_or_none(current)
                # v1 represents infinity using a large page-aligned integer.
                if limit is not None and limit>=1<<60: limit=None
                record.update(memory_limit_bytes=limit,memory_current_bytes=used,
                              memory_limit_is_unbounded=maximum=='max' or (integer_or_none(maximum) or 0)>=1<<60,
                              conservative_remaining_bytes=max(0,limit-used) if limit is not None and used is not None else None)
                stat=read_text(visible_path/'memory.stat');selected={}
                if stat:
                    for row in stat.splitlines():
                        key,_,value=row.partition(' ')
                        if key in ('anon','file','inactive_file','active_file','cache','rss','total_cache','total_rss'):
                            selected[key]=integer_or_none(value)
                record['memory_stat_bytes']=selected
                records.append(record)
    return {'available':bool(records),'records':records,
            'note':'Remaining=max-current is conservative: current may include reclaimable file cache. Visible container mount roots may hide stricter ancestors.'}


def disk(path):
    requested=Path(path).resolve();existing=requested
    while not existing.exists() and existing!=existing.parent: existing=existing.parent
    try:
        value=os.statvfs(existing)
        return {'requested_path':str(requested),'measured_existing_path':str(existing),'available':True,
                'free_bytes_for_user':value.f_bavail*value.f_frsize,'total_bytes':value.f_blocks*value.f_frsize,
                'free_inodes_for_user':value.f_favail,'filesystem_id':value.f_fsid}
    except OSError as error:
        return {'requested_path':str(requested),'measured_existing_path':str(existing),'available':False,
                'error_type':type(error).__name__}


def cpu():
    text=read_text('/proc/cpuinfo') or ''
    first={}
    for line in text.split('\n\n',1)[0].splitlines():
        key,separator,value=line.partition(':')
        if separator: first[key.strip()]=value.strip()
    flags=set(first.get('flags','').split())
    try: affinity=sorted(os.sched_getaffinity(0))
    except (AttributeError,OSError): affinity=None
    names=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','JAX_PLATFORMS','JAX_ENABLE_X64')
    return {'os_cpu_count':os.cpu_count(),'affinity_cpu_ids':affinity,
            'affinity_cpu_count':len(affinity) if affinity is not None else None,
            'model_name':first.get('model name'),'flags_known':bool(flags),
            'selected_flags':{name:name in flags for name in ('avx2','avx512f','avx512_bf16','amx_tile','amx_bf16')},
            'thread_environment':{name:os.environ.get(name) for name in names},
            'reference_config':'Official CPU reference currently sets torch/OMP/MKL threads to1. CPU quota and BF16 ISA can affect elapsed time; no throughput measured here.'}


def feasibility(resources):
    host_available=resources['host_memory'].get('fields',{}).get('MemAvailable')
    records=resources['cgroup'].get('records',[])
    limits=[r['conservative_remaining_bytes'] for r in records if r.get('conservative_remaining_bytes') is not None]
    values=([host_available] if host_available is not None else [])+limits
    remaining=min(values) if values else None
    cgroup_memory_known=any(r.get('memory_limit_bytes') is not None or r.get('memory_limit_is_unbounded') for r in records)
    memory_complete=host_available is not None and cgroup_memory_known
    total_checkpoints=sum(CHECKPOINT_BYTES.values())
    disk_plan=total_checkpoints+3*GIB+4*GIB
    disk_free=resources['cache_disk'].get('free_bytes_for_user')
    disk_records=[resources[name] for name in ('cache_disk','model_tools_disk','pip_cache_disk')]
    shared_filesystem=all(item.get('available') for item in disk_records) and len({item['filesystem_id'] for item in disk_records})==1
    def memory_check(required):
        meets=None if remaining is None else remaining>=required
        return {'required_available_bytes_estimate':required,'meets_recorded_capacity_estimate':meets,
                'feasible_under_conservative_preflight':meets if memory_complete else None,
                'status':'unknown' if not memory_complete else ('within_conservative_budget' if meets else 'below_conservative_budget')}
    return {'checkpoint_payload_bytes':CHECKPOINT_BYTES,'combined_checkpoint_payload_bytes':total_checkpoints,
            'checkpoint_provenance':'Official access-probe repository manifests, run20260919T055501Z-N9-access-probe-v5e-v004-ebb556; weights only, no duplicate consolidated Mistral file',
            'disk_plan':{'combined_checkpoint_bytes':total_checkpoints,'isolated_tools_and_pip_cache_margin_bytes':3*GIB,
                         'metadata_reference_archive_and_free_space_margin_bytes':4*GIB,'total_new_space_estimate_bytes':disk_plan,
                         'measured_free_bytes':disk_free,'shared_cache_tools_pip_filesystem':shared_filesystem,
                         'cache_filesystem_meets_combined_space_estimate':None if disk_free is None else disk_free>=disk_plan,
                         'feasible_under_conservative_preflight':None if disk_free is None or not shared_filesystem else disk_free>=disk_plan,
                         'separate_filesystem_policy':'If locations differ, inspect each recorded filesystem budget before acquisition; a cache-only pass cannot certify tools/pip capacity.'},
            'host_available_bytes':host_available,'conservative_effective_remaining_bytes':remaining,
            'host_and_visible_cgroup_limits_observed':memory_complete,
            'qwen_cpu_reference':memory_check(4*GIB),'mistral_cpu_reference_resident':memory_check(20*GIB),
            'not_a_fit_proof':'Margins are planning estimates. mmap/file-cache reclamation, hidden ancestor limits, loader transients and concurrent host activity may change actual peak RSS. Swap is not counted as RAM capacity.',
            'recovery_if_mistral_below_budget':'Preserve current outputs. Use a new version of the official Transformers reference with explicit CPU max_memory and immutable disk offload, or recorded official-layer streaming. Retain checkpoint/token hashes, BF16/eager semantics and original numerical gates; do not substitute the custom JAX implementation for its independent reference.'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('campaign','phase','output-dir','expected-identity','allocation-id'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--max-wall-seconds',type=float,default=300)
    parser.add_argument('--cache-root',default='/content/Strassen_MM_Focus/models')
    parser.add_argument('--venv-dir',default='/content/Strassen_MM_Focus/model-tools-v001')
    args=parser.parse_args(argv)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    journal=base.Journal(out,args.phase);started=time.monotonic();completed=False;error=None;resources=None;report=None
    try:
        if 'jax' in sys.modules: raise RuntimeError('Host preflight must not import JAX')
        expected=json.loads(Path(args.expected_identity).read_text());expected=expected.get('identity',expected)
        current={'colab_endpoint':args.allocation_id,'hostname':socket.gethostname(),
                 'boot_id':read_text('/proc/sys/kernel/random/boot_id'),
                 'versions':{name:importlib.metadata.version(name) for name in expected['versions']}}
        base.exclusive_json(out/'environment.json',{'identity':current,'python':sys.version,
                'qualification':'Host/core-package identity only; no JAX import or TPU work'})
        mismatch={name:{'expected':expected.get(name),'actual':current.get(name)}
                  for name in ('colab_endpoint','hostname','boot_id','versions') if expected.get(name)!=current.get(name)}
        if mismatch: raise RuntimeError('Preflight cohort identity mismatch: '+json.dumps(mismatch,sort_keys=True))
        journal.emit('identity_check',status='matched',fields=['colab_endpoint','hostname','boot_id','versions'])
        resources={'captured_utc':base.utc_now(),'host_memory':meminfo(),'cgroup':cgroups(),
                   'cache_disk':disk(args.cache_root),'model_tools_disk':disk(args.venv_dir),
                   'pip_cache_disk':disk(Path.home()/'.cache/pip'),'cpu':cpu()}
        report=feasibility(resources)
        base.exclusive_json(out/'resources.json',resources);base.exclusive_json(out/'feasibility.json',report)
        journal.emit('host_resources',resources=resources);journal.emit('resource_feasibility',**report)
        completed=True
    except BaseException as problem:
        error={'type':type(problem).__name__,'message':str(problem)}
        journal.emit('run_error',**error,traceback=traceback.format_exc())
    finally:
        summary={'phase':args.phase,'status':'completed' if completed else 'failed','completed':completed,
                 'elapsed_seconds':time.monotonic()-started,'diagnostics_only':True,'jax_imported':'jax' in sys.modules,
                 'model_downloaded':False,'feasibility':report,'error':error}
        journal.emit('run_complete',**summary);journal.close();base.exclusive_json(out/'summary.json',summary)
        base.exclusive_json(out/'artifact_manifest.json',{'sha256':{
            str(p.relative_to(out)):base.digest_file(p) for p in sorted(out.rglob('*')) if p.is_file()},'sealed_utc':base.utc_now()})
        print(json.dumps({'phase':args.phase,'status':summary['status'],'diagnostics_only':True,
                          'feasibility':report},sort_keys=True),flush=True)
    return 0 if completed else 1


if __name__=='__main__': raise SystemExit(main())
