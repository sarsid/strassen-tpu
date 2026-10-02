"""Check reporting against independently reconciled archived timing controls."""
import json
import os
from pathlib import Path
from report_llm_tradeoff_v002 import report

root=Path(os.environ['STRASSEN_PROJECT_ROOT'])
out=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts'
report(root/'runs/20260922-llm-tradeoff-proplus-v6e-v001',out)
text=(out/'RESULTS.md').read_text()
assert '| one_level | 81.8405 | +10.73%' in text
assert '+26.05% | True |' in text
assert '| one_level | 78.8192 | +7.29%' in text
assert 'Official Native qualification: False.' in text
assert 'gemma' in text
data=json.loads((out/'results.json').read_text())
assert len(data['models'])==4 and len(data['input_sha256'])>=20
print(json.dumps(dict(passed=True,checks=['matched_and_best_native_are_distinct',
    'known_qwen_and_mistral_means_match','failed_gemma_qualification_preserved',
    'all_models_and_input_hashes_recorded'],scope='Archived-data reporting only; no new TPU measurements')))
