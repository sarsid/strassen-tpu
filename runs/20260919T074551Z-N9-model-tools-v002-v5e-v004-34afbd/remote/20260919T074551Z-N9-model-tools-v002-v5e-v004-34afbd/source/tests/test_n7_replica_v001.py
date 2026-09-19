"""Offline cohort/policy invariants; not TPU execution or performance evidence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from strassen_mm.benchmark_n7_replica_v001 import validate_compatibility, validate_setup_smoke, verify_original_policy
from strassen_mm.selector_v002 import digest

PRECISION={"input_dtype":"bfloat16","output_dtype":"float32","accumulator_dtype":"float32","pre_add_dtype":"bfloat16","native_precision":"DEFAULT","reference_dtype":"float32"}


def environment(allocation="old"):
    versions={"jax":"0.7.2","jaxlib":"0.7.2","libtpu":"0.0.21.1","numpy":"2.2.0"}
    devices=[{"kind":"TPU v5 lite","id":0,"platform":"tpu","process_index":0}]
    return {"qualified_single_v5e":True,"backend":"tpu","device_count":1,"local_device_count":1,"process_count":1,
            "identity":{"allocation_id":allocation,"colab_endpoint":allocation,"hostname":allocation+"-host","host_id":allocation+"-host","boot_id":allocation+"-boot",
                        "device_kind":"TPU v5 lite","device_id":0,"process_index":0,"jax_version":"0.7.2","jaxlib_version":"0.7.2","libtpu_version":"0.0.21.1","numpy_version":"2.2.0",
                        "versions":versions,"devices":devices,"runtime_flags":{"JAX_PLATFORMS":"tpu"},"mosaic_compatibility":{"override_enabled":True},"runtime_image":None}}


class CompatibilityTests(unittest.TestCase):
    def test_disclosed_new_allocation_allowed_without_rewriting_old_identity(self):
        old,new=environment(),environment("new")
        old_before=copy.deepcopy(old)
        result=validate_compatibility(old,new,PRECISION)
        self.assertEqual(result["status"],"compatible_new_v5e_cohort")
        self.assertEqual(set(result["observed_identity_changes"]),{"allocation_id","colab_endpoint","hostname","host_id","boot_id"})
        self.assertEqual(old,old_before)
        self.assertEqual(result["original_identity"]["allocation_id"],"old")

    def test_same_allocation_cannot_be_labeled_fresh(self):
        with self.assertRaises(ValueError): validate_compatibility(environment(),environment(),PRECISION)

    def test_precision_software_device_and_flag_changes_rejected(self):
        for field,value in (("jax_version","0.8.0"),("runtime_flags",{"JAX_PLATFORMS":"tpu","XLA_FLAGS":"changed"}),("device_kind","TPU v6e"),("versions",{"jax":"0.8.0"})):
            new=environment("new");new["identity"][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): validate_compatibility(environment(),new,PRECISION)
        precision={**PRECISION,"output_dtype":"bfloat16"}
        with self.assertRaises(ValueError): validate_compatibility(environment(),environment("new"),precision)
        new=environment("new");new["device_count"]=2
        with self.assertRaises(ValueError): validate_compatibility(environment(),new,PRECISION)

    def test_setup_must_match_own_new_smoke(self):
        smoke=environment("new")
        setup={key:smoke["identity"][key] for key in ("colab_endpoint","hostname","boot_id","versions","devices")}
        setup.update(backend="tpu",device_count=1)
        self.assertEqual(validate_setup_smoke(setup,smoke)["status"],"matched")
        setup["boot_id"]="different"
        with self.assertRaises(ValueError): validate_setup_smoke(setup,smoke)


class FrozenPolicyTests(unittest.TestCase):
    def test_policy_bytes_and_original_identity_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);env=environment()
            rule={"source_identity":env["identity"],"prototypes":[],"parameters":{"fixed":1}}
            rule_path=root/'selector.json';rule_path.write_text(json.dumps(rule))
            choices={"phase":"N7-screen","environment_identity":env["identity"],"frozen_selector_sha256":digest(rule_path)}
            choices_path=root/'choices.json';choices_path.write_text(json.dumps(choices))
            (root/'input_provenance.json').write_text(json.dumps({"selection_sha256":digest(choices_path),"frozen_selector_sha256":digest(rule_path)}))
            rule_bytes=rule_path.read_bytes();choices_bytes=choices_path.read_bytes()
            verify_original_policy(root,env,choices,rule,choices_path,rule_path)
            self.assertEqual(rule_path.read_bytes(),rule_bytes)
            self.assertEqual(choices_path.read_bytes(),choices_bytes)
            rule_path.write_text(json.dumps(rule,indent=2))
            with self.assertRaises(ValueError): verify_original_policy(root,env,choices,rule,choices_path,rule_path)


if __name__=='__main__':
    unittest.main(verbosity=2)
