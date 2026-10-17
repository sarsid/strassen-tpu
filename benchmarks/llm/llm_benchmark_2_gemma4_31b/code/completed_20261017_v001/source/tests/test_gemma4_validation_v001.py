import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from gemma4_validation_v001 import natural_sequences,qualify_evidence
ROOT=Path(__file__).resolve().parents[1]

class ApprovedValidation(unittest.TestCase):
    def test_bos_sequence_boundaries_and_no_repeated_positions(self):
        ids,meta=natural_sequences(list(range(100,200)),2,5,3,2,np)
        self.assertEqual(ids.shape,(3,2,5))
        self.assertTrue(np.all(ids[:,:,0]==2))
        self.assertEqual(ids[:,:,1:].ravel().tolist(),list(range(100,124)))
        self.assertEqual(meta['tokenization_policy']['scored_positions'],24)
        self.assertEqual(meta,natural_sequences(list(range(100,200)),2,5,3,2,np)[1])
        with self.assertRaises(RuntimeError):natural_sequences(list(range(23)),2,5,3,2,np)
        with self.assertRaises(ValueError):natural_sequences(list(range(100)),2,5,3,None,np)

    def test_qualified_evidence_and_tamper_rejection(self):
        plan=json.loads((ROOT/'configs/gemma4_parent_batch_v003.json').read_text())
        official=ROOT/'third_party/transformers_gemma4_v001/modeling_gemma4.py'
        result=qualify_evidence(ROOT,plan,plan['model'],plan['revision'],official)
        self.assertTrue(result['all_pass']);self.assertTrue(result['reused_source_identical_control'])
        with self.assertRaisesRegex(RuntimeError,'identity'):
            qualify_evidence(ROOT,plan,plan['model'],'different-revision',official)
        bad=copy.deepcopy(plan);bad['formula_qualification']['evidence_sha256']='0'*64
        with self.assertRaisesRegex(RuntimeError,'evidence hash'):
            qualify_evidence(ROOT,bad,plan['model'],plan['revision'],official)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec=copy.deepcopy(plan)
            evidence=json.loads((ROOT/plan['formula_qualification']['evidence_file']).read_text())
            evidence['verified_source_files']={}
            path=root/'evidence.json';spec['formula_qualification']['evidence_file']='evidence.json'
            for key,value in [('relative_l2',.01),('finite',False)]:
                changed=copy.deepcopy(evidence);changed['report']['layers'][0][key]=value
                path.write_text(json.dumps(changed));spec['formula_qualification']['evidence_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertFalse(qualify_evidence(root,spec,plan['model'],plan['revision'],official)['all_pass'])
            evidence['verified_source_files']={'model.py':'0'*64};(root/'model.py').write_text('changed')
            path.write_text(json.dumps(evidence));spec['formula_qualification']['evidence_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError,'source mismatch'):
                qualify_evidence(root,spec,plan['model'],plan['revision'],official)

    def test_timing_tuning_and_algorithm_quality_unchanged(self):
        old=json.loads((ROOT/'configs/gemma4_parent_batch_v002.json').read_text())
        new=json.loads((ROOT/'configs/gemma4_parent_batch_v003.json').read_text())
        for key in old:
            if key not in ('version','result_root'):self.assertEqual(old[key],new[key],key)
        def funcs(version):
            tree=ast.parse((ROOT/f'tools/gemma4_batch_stage_{version}.py').read_text())
            return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
        a,b=funcs('v002'),funcs('v003')
        for name in a.keys()-{'tokens','oracle','main'}:self.assertEqual(a[name],b[name],name)

if __name__=='__main__':unittest.main(verbosity=2)
