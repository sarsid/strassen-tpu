"""Resolve public immutable metadata; never print or archive authorization data."""
import argparse, hashlib, json, re, time, os
from pathlib import Path
import requests

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads(a.config.read_text());rows=[]
    for model in c['models']:
        row=dict(model);headers={}
        # Existing Gemma access was authorized; scope it to official Gemma metadata.
        if model['repo_id'].startswith('google/gemma-'):
            try:
                from huggingface_hub import get_token
                token=get_token()
                if token:headers={'Authorization':'Bearer '+token}
            except ImportError:pass
            if not headers:
                token=os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
                for path in [Path(os.environ.get('HF_HOME',str(Path.home()/'.cache/huggingface')))/'token',Path.home()/'.huggingface/token']:
                    if not token and path.is_file():token=path.read_text().strip()
                if token:headers={'Authorization':'Bearer '+token}
        try:
            revision=model['revision']
            if not revision:
                response=requests.get('https://huggingface.co/api/models/'+model['repo_id'],headers=headers,timeout=60)
                response.raise_for_status();revision=response.json()['sha']
            if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Unpinned revision')
            response=requests.get(f'https://huggingface.co/{model["repo_id"]}/resolve/{revision}/config.json',headers=headers,timeout=60)
            response.raise_for_status();config=response.json()
            raw=response.content;name=model['id']+'.config.json';(a.output/name).write_bytes(raw)
            row.update(revision=revision,status='metadata_ready',config_path=name,config_sha256=hashlib.sha256(raw).hexdigest(),config=config)
        except Exception as e:
            row.update(status='metadata_failed',error_type=type(e).__name__,http_status=getattr(getattr(e,'response',None),'status_code',None))
        rows.append(row)
        (a.output/'models.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='config'}),flush=True)
    summary=dict(models=len(rows),ready=sum(x['status']=='metadata_ready' for x in rows),scope='configuration metadata only; checkpoint access not yet proven')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return int(summary['ready']!=len(rows))
if __name__=='__main__':raise SystemExit(main())
