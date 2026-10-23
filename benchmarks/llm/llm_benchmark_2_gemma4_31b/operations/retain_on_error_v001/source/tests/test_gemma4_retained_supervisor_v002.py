"""Verify a settled allocation is reused and monitoring never releases it."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import start_gemma4_diagnostic_v002 as supervisor


class RetainedSupervisorTest(unittest.TestCase):
    def test_owned_allocation_reused_and_watch_recovers_from_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();run=root/'runs/owned-diagnostic';run.mkdir(parents=True)
            (root/'tools').mkdir()
            controller=root/'tools/run_gemma4_diagnostic_v002.py'
            controller.write_text('# frozen controller\n')
            (run/'source-manifest.json').write_text(json.dumps({'source_sha256':'original-source'}))
            (run/'controller-override-v002.json').write_text(json.dumps({'files':{
                'tools/run_gemma4_diagnostic_v002.py':hashlib.sha256(controller.read_bytes()).hexdigest()}}))
            (run/'allocation-events.jsonl').write_text(json.dumps(dict(
                kind='allocation',created=True,session=run.name,endpoint='same-owned-tpu'))+'\n')
            lists=0;calls=[]
            def fake_run(argv,**kwargs):
                nonlocal lists
                calls.append(argv)
                self.assertFalse(any('allocate_llm' in v or 'release_allocation' in v for v in argv))
                if argv[-1]=='list':
                    lists+=1
                    if lists==1:raise subprocess.TimeoutExpired(argv,90)
                    rows=[dict(endpoint='same-owned-tpu')] if lists==2 else []
                    return subprocess.CompletedProcess(argv,0,stdout=json.dumps(dict(kind='allocations',assignments=rows)))
                return subprocess.CompletedProcess(argv,0)
            def fake_call(argv,**kwargs):
                self.assertIn('same-owned-tpu',argv)
                self.assertEqual(argv[1],str(controller))
                (run/'retained-allocation.json').write_text('{}\n')
                return 1
            args=['starter','--run',str(run),'--controller-python',sys.executable,
                  '--start-authorized','--keep-waiting']
            with patch.object(supervisor,'ROOT',root),patch.object(sys,'argv',args), \
                 patch.object(supervisor.subprocess,'run',side_effect=fake_run), \
                 patch.object(supervisor.subprocess,'call',side_effect=fake_call), \
                 patch.object(supervisor.time,'sleep'):
                self.assertEqual(supervisor.main(),1)
            rows=[json.loads(v) for v in (run/'retention-watch.jsonl').read_text().splitlines()]
            self.assertEqual([v['state'] for v in rows],['unknown','allocated','absent'])
            self.assertTrue(all(v['release_attempted'] is False for v in rows))


if __name__=='__main__':unittest.main(verbosity=2)
