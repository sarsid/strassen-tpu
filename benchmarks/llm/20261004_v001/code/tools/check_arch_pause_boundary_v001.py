"""Exercise real process parking: export finishes, next stage cannot start."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from pause_arch_boundary_v001 import park, checked_signal, process, later_started

out = Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts'
out.mkdir()
script = out/'fake_controller.py'
script.write_text('''import subprocess,sys,time
from pathlib import Path
p=Path(sys.argv[1])
c=subprocess.Popen([sys.executable,"-c","import time;from pathlib import Path;time.sleep(1);Path("+repr(str(out/'export-finished'))+").write_text('done')"])
(p/'child-pid').write_text(str(c.pid))
c.wait()
(p/'next-stage').write_text('started')
''')
child = subprocess.Popen([sys.executable, str(script), str(out)])
try:
    until = time.monotonic()+10
    while not (out/'child-pid').exists():
        if time.monotonic()>until: raise TimeoutError('Fake controller did not start')
        time.sleep(.01)
    try:
        checked_signal(child.pid, 'definitely-not-the-controller', signal.SIGSTOP)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Mismatched owner accepted')
    park(child.pid, str(script))
    while not (out/'export-finished').exists():
        if time.monotonic()>until: raise TimeoutError('Export child did not survive parent parking')
        time.sleep(.01)
    assert process(child.pid).startswith('T')
    assert not (out/'next-stage').exists()
    checked_signal(child.pid, str(script), signal.SIGCONT)
    assert child.wait(timeout=5)==0
    assert (out/'next-stage').exists()
finally:
    if child.poll() is None:
        child.kill();child.wait()
cohort=out/'cohort'
(cohort/'operations/next').mkdir(parents=True)
plan={'stages':[{'id':'target'},{'id':'next'}]}
assert later_started(cohort,plan,'target')==[]
(cohort/'operations/next/execution.json').write_text('{}')
assert later_started(cohort,plan,'target')==['next']
checks=['wrong_process_rejected','parent_parked','export_child_finishes_while_parent_parked',
        'next_stage_blocked','explicit_resume_continues','later_stage_detected']
(out/'summary.json').write_text(json.dumps({'passed':True,'checks':checks},indent=2)+'\n')
print(json.dumps({'passed':True,'checks':checks}))
