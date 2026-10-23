"""Focused numerical and leakage checks for the shape-only cost rule; no TPU."""
import importlib.util
import itertools
import math
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from strassen_mm import power_grid_selector_v001 as rule

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fit_model',ROOT/'tools/fit_power_grid_model_v001.py')
fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit)


class SelectorChecks(unittest.TestCase):
    def setUp(self):
        self.native={'candidate_id':'native','family':'native','tile':None}
        self.custom={'candidate_id':'strassen','family':'strassen','tile':[1024,512,256]}
        self.model={'feature_kind':'affine6','coefficients':{'native':[1,1,0,0,0,0],
                    'strassen':[.2,.5,0,0,0,0]},'candidates':[self.native,self.custom],
                    'memory_budget_bytes':8*2**30,'training_shapes_mnk':[[1024,512,256]],
                    'calibration_contract':{}}

    def test_dimensions_require_positive_integers(self):
        for bad in (0,-1,True,1.5,float('inf'),'256'):
            with self.subTest(bad=bad),self.assertRaises(ValueError):rule.select(self.model,bad,256,256)

    def test_asymmetric_geometry_and_historical_axis_conversion(self):
        g=rule.geometry(1025,513,257,self.custom)
        self.assertEqual([g['padded_m'],g['padded_n'],g['padded_k']],[2048,1024,512])
        self.assertEqual(g['logical_output_tiles'],4);self.assertEqual(g['sequential_k_panels'],2)
        self.assertAlmostEqual(g['padding_volume_ratio'],2048*1024*512/(1025*513*257))

    def test_no_padding_has_no_preparation_charge(self):
        g=rule.geometry(1024,512,256,self.custom)
        self.assertFalse(g['padding_required']);self.assertEqual(g['copy_proxy_bytes'],0)
        n=rule.geometry(1025,513,257,self.native)
        self.assertEqual(n['padding_volume_ratio'],1);self.assertIsNone(n['logical_output_tiles'])

    def test_memory_rejects_all_without_inventing_a_tile(self):
        r=rule.select(self.model,131072,131072,131072)
        self.assertIsNone(r['selection']);self.assertEqual(len(r['exclusions']),2)
        self.assertEqual(r['predicted_near_set'],[])

    def test_native_tile_is_compiler_managed(self):
        self.model['coefficients']['native']=[.01,0,0,0,0,0]
        r=rule.select(self.model,1024,512,256)
        self.assertEqual(r['selection']['family'],'native');self.assertIsNone(r['selection']['tile'])
        self.assertEqual(r['selection']['tile_status'],'compiler_managed')
        with self.assertRaises(ValueError):rule.geometry(128,128,128,{**self.native,'tile':[128,128,128]})

    def test_near_set_is_actual_tuples_not_cartesian_projection(self):
        r=rule.select(self.model,1024,512,256)
        self.assertEqual(r['selection']['tile'],[1024,512,256])
        self.assertEqual([c['tile'] for c in r['per_family_predicted_near_set']['strassen']],[[1024,512,256]])
        self.assertEqual(r['per_family_predicted_near_set']['cubic'],[])

    def test_piecewise_is_continuous_with_independent_nonnegative_slopes(self):
        c=self.native;knot=1000
        x1=rule.features(10,10,10,c,'piecewise7',knot)
        x2=rule.features(10,10,11,c,'piecewise7',knot)
        self.assertEqual(x1[1],x2[1]);self.assertAlmostEqual(x2[2],100/1e12)
        with self.assertRaises(ValueError):rule.features(1,1,1,c,'piecewise7')

    def test_solver_matches_exhaustive_feasible_active_sets(self):
        rng=np.random.default_rng(773)
        for _ in range(20):
            a=rng.normal(size=(12,4));b=rng.normal(size=12)
            best=float('inf')
            for flags in itertools.product((False,True),repeat=4):
                cols=np.array(flags);x=np.zeros(4)
                if cols.any():x[cols]=np.linalg.lstsq(a[:,cols],b,rcond=None)[0]
                if np.all(x>=0):best=min(best,float(np.sum((a@x-b)**2)))
            x,kkt,_=fit.nnls(a,b)
            self.assertAlmostEqual(float(np.sum((a@x-b)**2)),best,places=9)
            self.assertLess(kkt,1e-7)

    def test_zero_and_duplicate_columns_are_legal(self):
        a=np.array([[1,1,0],[2,2,0],[3,3,0]],float);b=np.array([2,4,6],float)
        x,kkt,_=fit.nnls(a,b)
        np.testing.assert_allclose(a@x,b,atol=1e-12);self.assertEqual(x[2],0)

    def test_outer_labels_do_not_affect_inner_tuning_or_fit(self):
        rng=np.random.default_rng(558);x=np.abs(rng.normal(size=(18,3,6)))+.1
        y=np.abs(rng.normal(size=(18,3)))+.2;train=np.arange(12)
        xs={s['name']:x for s in fit.SPECS}
        s1,_=fit.choose_spec(xs,y,train,3,778);b1,_=fit.fit(x,y,train)
        y[12:]*=1e7
        s2,_=fit.choose_spec(xs,y,train,3,778);b2,_=fit.fit(x,y,train)
        self.assertEqual(s1,s2);np.testing.assert_array_equal(b1,b2)

    def test_metric_distinguishes_speedup_from_time_loss(self):
        y=np.array([[1.,1.2],[2.,2.2]])
        m=fit.metric(y,np.array([1,1]),np.array(['native','strassen']))
        self.assertAlmostEqual(m['geomean_regret'],math.sqrt(1.2*1.1))
        self.assertEqual(m['within5_count'],0);self.assertEqual(m['exact_family_count'],0)

    def test_mismatched_preset_cannot_inherit_measured_coefficients(self):
        # Uses the immutable evidence when present; source snapshots omit runs.
        root=ROOT
        while not (root/'runs/20260920T034417Z-GRID-screen-v5e-v004-8046c8/artifacts/results.jsonl').is_file():
            if root.parent==root:self.skipTest('Frozen screening evidence unavailable')
            root=root.parent
        screen=root/'runs/20260920T034417Z-GRID-screen-v5e-v004-8046c8/artifacts/results.jsonl'
        campaign=ROOT/'configs/generated_power_grid_v001/campaign_power_grid_v001.json'
        manifest=ROOT/'configs/generated_power_grid_v001/sampled_shapes.json'
        config=json.loads(campaign.read_text())
        native=config['experiments']['GRID-screen']['candidate_families']['native']
        next(c for c in native if c['candidate_id']=='native_vmem_64m')['compiler_options']['xla_tpu_scoped_vmem_limit_kib']=65535
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mismatched.json';path.write_text(json.dumps(config))
            with self.assertRaises(AssertionError):fit.load(screen,manifest,path)


if __name__=='__main__':unittest.main(verbosity=2)
