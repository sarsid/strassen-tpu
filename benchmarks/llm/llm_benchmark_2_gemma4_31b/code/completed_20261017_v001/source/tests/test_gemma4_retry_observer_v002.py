"""Retry observation admission and archive integrity, with no remote calls."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from watch_gemma4_retry_v002 import snapshot_from,require_owned,unpack_verified


class Checks(unittest.TestCase):
    def test_snapshot_envelope(self):
        snapshot={'kind':'gemma4_batch_snapshot','status.json':{'status':'running'}}
        self.assertEqual(snapshot_from([{'kind':'remote_stream','text':'ignored\n'+json.dumps(snapshot)+'\n'}]),snapshot)
        self.assertIsNone(snapshot_from([{'kind':'controller_error','message':'timeout'}]))
    def test_owned_assignment(self):
        spec={'session':'original','endpoint':'retained'}
        row={'kind':'allocations','assignments':[{'name':'original','endpoint':'retained','accelerator':'V6E1'}]}
        require_owned([row],spec)
        for records in [[],[{'kind':'allocations','assignments':[]}],[row,row]]:
            with self.assertRaises(RuntimeError):require_owned(records,spec)
    def test_archive_integrity_before_extract(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);package=root/'result.tar.gz';data=b'{"all_pass":true}\n'
            with tarfile.open(package,'w:gz') as handle:
                entry=tarfile.TarInfo('summary.json');entry.size=len(data);handle.addfile(entry,io.BytesIO(data))
            with self.assertRaises(ValueError):unpack_verified(package,'wrong',root/'output')
            self.assertFalse((root/'output').exists())
            unpack_verified(package,hashlib.sha256(package.read_bytes()).hexdigest(),root/'output')
            self.assertEqual((root/'output/summary.json').read_bytes(),data)


if __name__=='__main__':unittest.main()
