"""Validate budgeted menu, Native equivalence, and balanced timing source."""
import ast
from pathlib import Path
import tune_parent_fused_v004 as balanced
import check_parent_adapter_v001

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    source=(root/'tools/tune_parent_fused_v004.py').read_text()
    assert 'statistics.median(' not in source
    assert 'runs=8,warmups=0' in source and 'runs=8,warmups=2' in source
    for name in ('runtime/parent_comparison_remote_v003.py','runtime/stop_comparison_tuning_v002.py','tools/run_parent_comparison_v003.py'):
        ast.parse((root/name).read_text(),filename=name)
    check_parent_adapter_v001.main()
