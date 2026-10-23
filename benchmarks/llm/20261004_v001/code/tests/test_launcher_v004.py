"""Execution-field protections for the generic frozen remote launcher."""
import argparse
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('launcher',Path(__file__).resolve().parents[1]/'runtime/launch_phase_v004.py')
launcher=importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)

class LauncherContracts(unittest.TestCase):
    def test_only_versioned_package_modules(self):
        self.assertEqual(launcher.checked_module('strassen_mm.benchmark_n5_v001'),'strassen_mm.benchmark_n5_v001')
        for name in ('os','strassen_mm.foo','strassen_mm.x;whoami','strassen_mm../x_v001'):
            with self.assertRaises(argparse.ArgumentTypeError): launcher.checked_module(name)

    def test_extra_arguments_cannot_replace_execution_identity(self):
        for value in ('--allocation-id=x','--output-dir','--campaign=x','--phase=N9','bad\0text'):
            with self.assertRaises(ValueError): launcher.checked_runner_args([value])
        self.assertEqual(launcher.checked_runner_args(['--selection','/content/Strassen_MM_Focus/a.json']),
                         ['--selection','/content/Strassen_MM_Focus/a.json'])

    def test_source_is_syntax_valid(self):
        import ast
        root=Path(__file__).resolve().parents[1]
        for relative in ('runtime/launch_phase_v004.py','tools/run_phase_v004.py',
                         'tools/prepare_model_application_v001.py','src/strassen_mm/application_prep_v001.py'):
            ast.parse((root/relative).read_text(),filename=relative)

if __name__=='__main__': unittest.main()
