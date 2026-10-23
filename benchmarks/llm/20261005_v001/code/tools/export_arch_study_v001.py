"""Incremental verified result bundles; raw phase exports are write-once."""
import argparse,gzip,hashlib,json,shutil,subprocess,tarfile
from pathlib import Path
from run_large_real_v002 import verify_phase,commit,ROOT

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')

def export(cohort,phase,final=False):
    plan=json.loads((cohort/'plan.json').read_text());dest=ROOT/plan['results_relative']
    dest.mkdir(parents=True,exist_ok=True)
    frozen=json.loads((cohort/'frozen.json').read_text())
    prov=dest/'provenance'/cohort.name
    if not prov.exists():
        prov.mkdir(parents=True)
        for name in ('plan.json','cohort.json','frozen.json'):shutil.copyfile(cohort/name,prov/name)
        # This source archive is the one used by every phase, not the mutable tree.
        with (cohort/'source.tar').open('rb') as src,gzip.open(prov/'source.tar.gz','xb') as out:shutil.copyfileobj(src,out)
        save(prov/'manifest.json',dict(source_commit=frozen['baseline_commit'],source_archive_sha256=sha(cohort/'source.tar'),compressed_sha256=sha(prov/'source.tar.gz')))
    if phase:
        run,summary=verify_phase(cohort,phase)
        folder=dest/'phases'/cohort.name;folder.mkdir(parents=True,exist_ok=True)
        archive=folder/(phase+'.tar.gz')
        if archive.exists():raise FileExistsError(archive)
        with tarfile.open(archive,'x:gz') as tar:
            for p in sorted(run.iterdir()):
                if p.name in ('source','source.tar'):continue
                tar.add(p,arcname=p.name)
        with tarfile.open(archive) as tar:
            for p in (run/'artifacts').rglob('*'):
                if p.is_file():
                    data=tar.extractfile(str(p.relative_to(run))).read()
                    if hashlib.sha256(data).hexdigest()!=sha(p):raise ValueError('Export content mismatch')
        save(folder/(phase+'.json'),dict(phase=phase,cohort=cohort.name,run=str(run.relative_to(ROOT)),summary=summary,sha256=sha(archive),bytes=archive.stat().st_size))
    if final:
        report=cohort/'operations/report/artifacts'
        if not json.loads((report/'results.json').read_text())['completed']:raise ValueError('Incomplete report')
        shutil.copytree(report,dest/'report')
        save(dest/'COMPLETE.json',dict(cohort=cohort.name,source_commit=frozen['baseline_commit'],results_sha256=sha(dest/'report/results.json'),note='Measurement/report coverage verified; runtime release is recorded separately by the controller.'))
    revision=commit([dest],('Finalize' if final else 'Checkpoint')+' dedicated architecture study results '+str(phase or 'report'))
    print(json.dumps(dict(exported=True,phase=phase,destination=str(dest),commit=revision)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--phase');p.add_argument('--final',action='store_true');a=p.parse_args();export(a.cohort,a.phase,a.final)
