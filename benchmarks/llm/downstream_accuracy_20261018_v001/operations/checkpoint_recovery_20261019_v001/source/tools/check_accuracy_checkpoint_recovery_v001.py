"""Restore bytes exactly; reject corruption, extra files and duplicate launches."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import resume_accuracy_remote_v001 as remote
from start_accuracy_study_v003 import observer_command


def tar(path,files):
    with tarfile.open(path,'w:gz') as out:
        for name,data in files.items():
            member=tarfile.TarInfo(name);member.size=len(data);out.addfile(member,io.BytesIO(data))


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'source.tar.gz';self.seed=self.root/'seed.tar.gz'
        tar(self.source,{'runtime/accuracy_study_remote_v001.py':b'# unchanged worker\n'})
        self.source_sha=remote.sha(self.source)
        self.files={'artifacts/qwen3_14b/manifest.json':b'{"original":true}\n',
            'artifacts/qwen3_14b/execution-contract.json':b'{"fixed":true}\n',
            'artifacts/qwen3_14b/batches/mmlu-000.json':b'{"results":42}\n'}

    def seed_archive(self,source_sha=None):
        receipt=dict(source_sha256=source_sha or self.source_sha,files={k:hashlib.sha256(v).hexdigest() for k,v in self.files.items()})
        tar(self.seed,dict(self.files,**{'seed-manifest.json':json.dumps(receipt).encode()}))
        return remote.sha(self.seed)

    def test_restore_exact_bytes_and_unchanged_worker(self):
        seed_sha=self.seed_archive()
        with patch.object(remote,'BASE',self.root):
            run,cmd=remote.prepare(self.source,self.source_sha,self.seed,seed_sha,'recovery')
            for name,data in self.files.items():self.assertEqual((run/name).read_bytes(),data)
            self.assertIn('accuracy_study_remote_v001.py',cmd[1])
            self.assertEqual(json.loads((run/'launch.json').read_text())['source_sha256'],self.source_sha)
            with self.assertRaises(FileExistsError):remote.prepare(self.source,self.source_sha,self.seed,seed_sha,'recovery')

    def test_reject_changed_source(self):
        seed_sha=self.seed_archive('wrong-source')
        with self.assertRaises(ValueError):remote.inspect_seed(self.seed,seed_sha,self.source_sha)

    def test_reject_changed_archive(self):
        self.seed_archive()
        with self.assertRaises(ValueError):remote.inspect_seed(self.seed,'wrong-hash',self.source_sha)

    def test_reject_unexpected_files_and_path_traversal(self):
        for name in ('artifacts/qwen3_14b/summary.json','artifacts/../../outside.json','artifacts/unknown/manifest.json'):
            with self.subTest(name=name):
                self.files[name]=b'{}';seed_sha=self.seed_archive()
                with self.assertRaises(ValueError):remote.inspect_seed(self.seed,seed_sha,self.source_sha)
                del self.files[name]

    def test_reject_changed_member(self):
        seed_sha=self.seed_archive()
        with tarfile.open(self.seed) as archive:
            files={m.name:archive.extractfile(m).read() for m in archive.getmembers()}
        files['artifacts/qwen3_14b/batches/mmlu-000.json']=b'changed'
        tar(self.seed,files)
        with self.assertRaises(ValueError):remote.inspect_seed(self.seed,remote.sha(self.seed),self.source_sha)

    def test_new_starter_actual_observer_command(self):
        root=Path(__file__).resolve().parents[1]
        result=subprocess.run(observer_command(root,'recovery','owned')+['--validate-cli'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
