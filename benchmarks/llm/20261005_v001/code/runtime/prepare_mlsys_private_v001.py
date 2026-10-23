"""Create private, nonarchived model input workspace on the allocated Colab."""
import argparse
from pathlib import Path
import json

p=argparse.ArgumentParser();p.add_argument('--private-root',type=Path,required=True)
p.add_argument('--seal-token',action='store_true');a=p.parse_args();root=a.private_root.resolve()
if not root.is_relative_to('/content/Strassen_MM_Focus/.runtime_private'):raise ValueError('Private root required')
if a.seal_token:
    token=root/'hf_token'
    if token.is_file():token.chmod(0o600)
    print(json.dumps({'private_token_staged':token.is_file()}))
else:
    root.mkdir(parents=True,exist_ok=False);root.chmod(0o700)
    print(json.dumps({'private_workspace_created':True}))
