"""Retained-runtime retry admission without remote calls or TPU execution."""
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from gemma4_b64_retry_remote_v002 import preflight
import gemma4_b64_retry_remote_v002 as retry


class RetryChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
        self.spec=json.loads((ROOT/'configs/gemma4_b64_residency_retry_v001.json').read_text())
        self.parent=self.base/'runs'/self.spec['predecessor_run'];self.parent.mkdir(parents=True)
        self.profile=self.parent/'artifacts/b64-tune-a0/profile.json';self.profile.parent.mkdir(parents=True)
        self.profile.write_text(json.dumps(dict(batch=64,sequence=1024,model=self.spec['model'],revision=self.spec['revision'])))
        self.spec['profile_sha256']=hashlib.sha256(self.profile.read_bytes()).hexdigest()
        private=self.base/'.runtime_private'/self.spec['predecessor_run'];(private/'parent/bin').mkdir(parents=True);(private/'parent/bin/python').touch();(private/'checkpoint').mkdir()
        self.state=dict(status='completed_with_failures',batches=[dict(batch=b,status='completed' if b!=64 else 'partial_or_failed',quality_pass=b!=64) for b in (8,16,32,64,4,1)],stages=[dict(name=n,returncode=0) for n in ('b64-tune-a0','b64-qualify-a0')])
        (self.parent/'archive-ready.json').write_text('{}');self.write()
        self.previous=self.base/'runs'/self.spec['previous_retry_run'];self.previous.mkdir(parents=True)
        (self.previous/'status.json').write_text(json.dumps({'status':'failed'}));(self.previous/'archive-ready.json').write_text('{}')
    def write(self):(self.parent/'status.json').write_text(json.dumps(self.state))
    def test_valid(self):self.assertEqual(preflight(self.base,self.spec)[0],self.profile)
    def test_running_predecessor(self):
        self.state['status']='running';self.write()
        with self.assertRaisesRegex(RuntimeError,'not finished'):preflight(self.base,self.spec)
    def test_another_failed_batch(self):
        self.state['batches'][-1]['quality_pass']=False;self.write()
        with self.assertRaisesRegex(RuntimeError,'other failed'):preflight(self.base,self.spec)
    def test_changed_profile(self):
        self.profile.write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'hash changed'):preflight(self.base,self.spec)
    def test_missing_qualification(self):
        self.state['stages'][-1]['returncode']=1;self.write()
        with self.assertRaisesRegex(RuntimeError,'qualification'):preflight(self.base,self.spec)
    def test_missing_archive(self):
        (self.parent/'archive-ready.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'not finished'):preflight(self.base,self.spec)
    def test_previous_retry_active(self):
        (self.previous/'status.json').write_text(json.dumps({'status':'running'}))
        with self.assertRaisesRegex(RuntimeError,'Previous retry'):preflight(self.base,self.spec)
    def test_previous_retry_unarchived(self):
        (self.previous/'archive-ready.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'Previous retry'):preflight(self.base,self.spec)
    def test_exclusive_lock_prevents_worker(self):
        run=self.base/'runs'/'retry';(run/'source/configs').mkdir(parents=True)
        (run/'source/configs/gemma4_b64_residency_retry_v001.json').write_text(json.dumps(self.spec))
        previous=retry.BASE;retry.BASE=self.base
        try:
            with (self.base/'active.lock').open('ab') as lock:
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):retry.worker(run)
                self.assertFalse((run/'artifacts').exists())
        finally:retry.BASE=previous


if __name__=='__main__':unittest.main()
