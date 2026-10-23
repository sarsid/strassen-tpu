"""Offline replica v002 cohort/policy and canonical-smoke provenance guards."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from strassen_mm.benchmark_n7_replica_v002 import (
    validate_compatibility, validate_setup_smoke, verify_original_policy, qualify_canonical_smoke, PRECISION_CONTRACT,
)
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
            root=Path(temporary).resolve();env=environment()
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



class CanonicalSmokeEvidenceTests(unittest.TestCase):
    def fixture(self, root):
        smoke = root / "smoke-run" / "artifacts"
        smoke.mkdir(parents=True)
        copied = root / "replica-run" / "expected-identity.json"
        copied.parent.mkdir()
        (smoke / "environment.json").write_text(json.dumps(environment("new"), indent=2))
        copied.write_bytes((smoke / "environment.json").read_bytes())
        (smoke / "summary.json").write_text(json.dumps({
            "phase": "smoke", "completed": True, "status": "completed"}))
        rows = [{"event": "case_result", "group_id": "smoke-" + str(index), "distribution": "gaussian",
                 "seed": 1, "arm_id": "native", "scope": "call", "status": "ok",
                 "correctness": {"pass": True}, "kernel_metadata": dict(PRECISION_CONTRACT)}
                for index in range(24)]
        (smoke / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.seal(smoke)
        return smoke, copied

    def seal(self, smoke):
        value = {path.name: digest(path) for path in smoke.iterdir()
                 if path.is_file() and path.name != "artifact_manifest.json"}
        (smoke / "artifact_manifest.json").write_text(json.dumps({"sha256": value}))

    def test_launcher_copy_in_other_directory_links_to_full_canonical_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            smoke, copied = self.fixture(Path(temporary).resolve())
            directory, env, rows, provenance, link = qualify_canonical_smoke(smoke / "environment.json", copied)
            self.assertEqual(directory, smoke)
            self.assertEqual(len(rows), 24)
            self.assertEqual(env["identity"]["allocation_id"], "new")
            self.assertEqual(provenance["result_count"], 24)
            self.assertEqual(link["status"], "byte_identical")
            self.assertEqual(link["sha256"], digest(copied))
            self.assertNotEqual(copied.parent, smoke)
            self.assertFalse((copied.parent / "results.jsonl").exists())

    def test_semantically_equal_but_different_identity_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            smoke, copied = self.fixture(Path(temporary).resolve())
            copied.write_text(json.dumps(json.loads(copied.read_text())))
            with self.assertRaisesRegex(ValueError, "identity bytes differ"):
                qualify_canonical_smoke(smoke / "environment.json", copied)

    def test_canonical_result_seal_remains_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            smoke, copied = self.fixture(Path(temporary).resolve())
            with (smoke / "results.jsonl").open("a") as stream: stream.write(" ")
            with self.assertRaisesRegex(ValueError, "artifact seal mismatch"):
                qualify_canonical_smoke(smoke / "environment.json", copied)

    def test_numerical_failure_is_rejected_even_with_valid_seal(self):
        with tempfile.TemporaryDirectory() as temporary:
            smoke, copied = self.fixture(Path(temporary).resolve())
            rows = [json.loads(line) for line in (smoke / "results.jsonl").read_text().splitlines()]
            rows[0]["correctness"]["pass"] = False
            (smoke / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.seal(smoke)
            with self.assertRaisesRegex(ValueError, "did not pass every"):
                qualify_canonical_smoke(smoke / "environment.json", copied)

    def test_identity_copy_cannot_stand_in_for_canonical_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, copied = self.fixture(Path(temporary).resolve())
            with self.assertRaisesRegex(ValueError, "canonical smoke environment.json"):
                qualify_canonical_smoke(copied, copied)



if __name__=='__main__':
    unittest.main(verbosity=2)

