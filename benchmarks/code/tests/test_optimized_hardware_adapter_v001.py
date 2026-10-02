"""Provisioning/measurement schema guard; no device access."""
import unittest
from strassen_mm.benchmark_optimized_hardware_v001 import validate_provisioning


class IdentityTests(unittest.TestCase):
    def test_shared_fields_match_and_extraneous_setup_metadata_is_ignored(self):
        identity = dict(colab_endpoint='endpoint', hostname='host', boot_id='boot',
                        versions={'jax':'0.7.2'}, captured_utc='setup-time')
        result = validate_provisioning(identity, 'endpoint', 'host', 'boot', {'jax':'0.7.2'})
        self.assertEqual(result['colab_endpoint'], 'endpoint')

    def test_each_changed_shared_field_fails(self):
        identity = dict(colab_endpoint='endpoint', hostname='host', boot_id='boot', versions={'jax':'0.7.2'})
        args = ['endpoint', 'host', 'boot', {'jax':'0.7.2'}]
        for index in range(len(args)):
            changed = args.copy()
            changed[index] = {'jax':'changed'} if index == 3 else 'changed'
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'mismatch'):
                validate_provisioning(identity, *changed)


if __name__ == '__main__':
    unittest.main()
