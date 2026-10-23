"""Create a separate local CPU oracle environment without touching Qwen's stack.

Records the full resolved package set and checks the installed official source
against the already audited, pinned upstream files. No model weights are fetched.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import venv


def main():
    root = Path(os.environ["STRASSEN_PROJECT_ROOT"])
    out = Path(os.environ["STRASSEN_EXECUTION_DIR"]) / "artifacts"
    out.mkdir()
    target = root / ".runtime_private/gemma4-oracle-v001"
    if target.exists():
        raise FileExistsError("Oracle environment already exists; preserve it and use a new version")
    requirements = Path("plans/gemma4_oracle_v001/requirements.txt").resolve()
    reference = Path("third_party/transformers_gemma4_v001").resolve()
    venv.EnvBuilder(with_pip=True).create(target)
    python = target / "bin/python"
    subprocess.run([str(python), "-m", "pip", "--isolated", "--disable-pip-version-check",
                    "install", "--no-input", "--index-url", "https://pypi.org/simple",
                    "-r", str(requirements)], check=True)
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    (out / "pip-freeze.txt").write_text(freeze)
    (out / "environment.json").write_text(json.dumps({
        "python": str(python), "platform": platform.platform(), "host_python": sys.version,
        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
        "scope": "Local CPU oracle environment preparation only; no numerical, checkpoint or TPU qualification",
        "existing_environments_modified": False}, indent=2) + "\n")
    check = r'''
import hashlib, json, sys
from pathlib import Path
import jax, torch, transformers
from transformers import Gemma4TextConfig, Gemma4ForCausalLM
import transformers.models.gemma4.modeling_gemma4 as model
import transformers.models.gemma4.configuration_gemma4 as config
import transformers.modeling_rope_utils as rope
reference=Path(sys.argv[1]); out=Path(sys.argv[2])
modules={'modeling_gemma4.py':model,'configuration_gemma4.py':config,'modeling_rope_utils.py':rope}
hashes={}
for name,module in modules.items():
    actual=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    expected=hashlib.sha256((reference/name).read_bytes()).hexdigest()
    hashes[name]={'installed_sha256':actual,'pinned_sha256':expected,'matches':actual==expected}
result={'transformers':transformers.__version__,'torch':torch.__version__,'jax':jax.__version__,
        'devices':[str(d) for d in jax.devices()], 'text_config_imported':Gemma4TextConfig.__name__,
        'text_model_imported':Gemma4ForCausalLM.__name__, 'source_checks':hashes}
(out/'official-source-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
assert transformers.__version__=='5.5.0'
assert all(v['matches'] for v in hashes.values()), 'Installed upstream source differs from audited pin'
assert all(d.platform=='cpu' for d in jax.devices())
'''
    subprocess.run([str(python), "-c", check, str(reference), str(out)], check=True)


if __name__ == "__main__":
    main()
