"""Validate the budgeted menu and unchanged Native adapter before resuming."""
import ast
from pathlib import Path
import tune_parent_fused_v003
import check_parent_adapter_v001

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    for name in ('runtime/parent_comparison_remote_v002.py','runtime/stop_comparison_tuning_v001.py','tools/run_parent_comparison_v002.py'):
        ast.parse((root/name).read_text(),filename=name)
    check_parent_adapter_v001.main()
