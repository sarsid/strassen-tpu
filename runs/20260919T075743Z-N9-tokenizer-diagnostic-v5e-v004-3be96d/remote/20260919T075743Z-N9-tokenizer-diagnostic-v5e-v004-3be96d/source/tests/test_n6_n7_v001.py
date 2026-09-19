"""Offline policy/trace-scope tests; do not produce TPU performance evidence."""
import copy
import tempfile
import unittest
from pathlib import Path

from strassen_mm.selector_v001 import choose
from strassen_mm.benchmark_n6_n7_v001 import analyze_trace, representative_shapes, roofline_estimate, EvidenceJournal


def policy():
    return {"parameters": {"min_m":256,"min_k":512,"min_n":512,"max_log2_distance":1.75,"max_padding_volume_ratio":1.5},
            "allowed_input_scopes":["gaussian_synthetic"],
            "prototypes":[{"shape_id":"training_large","shape_mkn":[2048,2048,2048],"label":"strassen",
                           "choice":{"candidate_id":"s","algorithm":"strassen","variant":"plain","tile":[1024,1024,512]}}]}


class PolicyTests(unittest.TestCase):
    def test_unknown_cancellation_and_real_inputs_do_not_get_custom_route(self):
        for scope in ("unknown","cancellation","real_weight_gaussian_activation","actual_model_activations"):
            self.assertEqual(choose(policy(),[2048,2048,2048],input_scope=scope)["choice"]["algorithm"],"native")
        self.assertEqual(choose(policy(),[2048,2048,2048],input_scope="gaussian_synthetic")["choice"]["algorithm"],"strassen")

    def test_padding_distance_and_small_m_guards(self):
        self.assertEqual(choose(policy(),[257,2048,2048],input_scope="gaussian_synthetic")["label"],"native")
        self.assertEqual(choose(policy(),[16,2048,2048],input_scope="gaussian_synthetic")["reason"],"small_dimension_guard")
        self.assertEqual(choose(policy(),[1281,2048,2048],input_scope="gaussian_synthetic")["reason"],"padding_guard")
        self.assertEqual(choose(policy(),[32768,2048,2048],input_scope="gaussian_synthetic")["reason"],"outside_training_neighborhood")
        self.assertEqual(choose(policy(),[2048,2048,2048],input_scope="gaussian_synthetic",exclude_shape_id="training_large")["reason"],"no_training_prototype")
        self.assertFalse(choose(policy(),[2048,2048,2048],input_scope="gaussian_synthetic")["numerical_accuracy_certified"])

    def test_journal_retains_public_ids_for_regret(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal=EvidenceJournal(Path(temporary),"N7-evaluate")
            journal.groups={"g":{"shape":{"id":"s","m":512,"k":512,"n":512},"arms":[
                {"arm_id":"cubic_basic","public_arm_id":"cubic_selected","family":"cubic","algorithm":"cubic_full","variant":"plain","tile":[256,256,256],"candidate_id":"c"}]}}
            journal.emit("case_result",group_id="g",arm_id="cubic_basic",scope="call",status="ok",timing={},correctness={},comparisons=[])
            self.assertEqual(journal.results[0]["arm_id"],"cubic_selected")
            journal.close()


class ProfileTests(unittest.TestCase):
    def trace(self):
        events=[{"ph":"M","pid":1,"name":"process_name","args":{"name":"/device:TPU:0"}},
                {"ph":"M","pid":1,"tid":2,"name":"thread_name","args":{"name":"XLA Modules"}},
                {"ph":"M","pid":1,"tid":3,"name":"thread_name","args":{"name":"XLA Ops"}}]
        for i in range(8):
            events.extend([{"ph":"X","pid":1,"tid":2,"name":"module","ts":i*2000,"dur":1000},
                           {"ph":"X","pid":1,"tid":3,"name":"nested","ts":i*2000,"dur":900}])
        return {"traceEvents":events}

    def test_actual_device_module_track_excludes_nested_ops(self):
        result=analyze_trace(self.trace(),8)
        self.assertEqual(result["module_mean_ms"],1.0)
        self.assertEqual(len(result["module_samples_ms"]),8)
        with self.assertRaises(ValueError): analyze_trace(self.trace(),9)
        trace=self.trace();trace["traceEvents"][0]["args"]["name"]="host"
        with self.assertRaises(ValueError): analyze_trace(trace,8)

    def test_missing_profile_classes_are_not_fabricated(self):
        pair={"reference_arm":"cubic_selected","valid_numerical_comparison":True,"speedup_ci95":[.8,.9]}
        rows=[{"shape_id":"a","arm_id":"cubic_selected","status":"ok","correctness":{"pass":True,"finite":True}},
              {"shape_id":"a","arm_id":"strassen_selected","status":"ok","correctness":{"pass":True,"finite":True},"comparisons":[pair]}]
        result=representative_shapes(rows,{"by_shape":{"a":{"shape_mkn":[1024,1024,1024]}}})
        self.assertEqual(result["absent_classes"],["win","tie"])
        self.assertEqual(result["representatives"][0]["category"],"loss")

    def test_roofline_is_estimate_with_separate_source_work(self):
        metadata={"shape_mkn":[1024,1024,1024],"padded_shape_mkn":[1024,1024,1024],"algorithm":"strassen","tile_bm_bn_bk":[1024,1024,512]}
        result=roofline_estimate(metadata,{"bf16_peak_flops_per_second":197e12,"hbm_peak_bytes_per_second":800*1024**3,"source":"official-spec"})
        self.assertEqual(result["estimated_mxu_dot_flops"],2*1024**3*7/8)
        self.assertGreater(result["source_level_vector_elementwise_ops_estimate"],0)
        self.assertIn("not measured",result["limitations"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
