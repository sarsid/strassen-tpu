"""Offline guards for the sole protobuf addition; no install or subprocesses."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


old=load('install_model_tools_v002')
new=load('install_model_tools_v003')


class ProtobufRecoveryTests(unittest.TestCase):
    def test_existing_explicit_pins_and_targeted_install_are_preserved(self):
        self.assertEqual(new.PINS,old.PINS+['protobuf==6.32.1'])
        self.assertEqual(new.CORE,old.CORE)
        commands=new.installation_commands('/usr/bin/python3',Path('/fresh/model-tools-v003'),Path('/evidence'))
        for command in commands[1:]:
            self.assertEqual(command[:6],['/usr/bin/python3','-m','pip','--isolated','--python','/fresh/model-tools-v003/bin/python'])
        installs=[command for command in commands if 'install' in command]
        self.assertEqual(sum(command.count('protobuf==6.32.1') for command in installs),1)

    def test_inventory_accepts_only_the_exact_diagnosed_dependency(self):
        before={'all_distributions':[{'name':'NumPy','version':'2.2.6'},{'name':'torch','version':'2.8.0+cpu'}]}
        after={'all_distributions':before['all_distributions']+[{'name':'protobuf','version':'6.32.1'}]}
        value=new.compare_inventory(before,after)
        self.assertTrue(value['only_pinned_protobuf_added'])
        self.assertEqual(value['added'],{'protobuf':'6.32.1'})
        self.assertFalse(value['removed']);self.assertFalse(value['changed'])

    def test_transitive_drift_removal_and_wrong_protobuf_are_rejected(self):
        before={'all_distributions':[{'name':'numpy','version':'2.2.6'}]}
        for rows in (
            [{'name':'numpy','version':'2.3.0'},{'name':'protobuf','version':'6.32.1'}],
            [{'name':'protobuf','version':'6.32.1'}],
            [{'name':'numpy','version':'2.2.6'},{'name':'protobuf','version':'6.32.2'}],
            [{'name':'numpy','version':'2.2.6'},{'name':'protobuf','version':'6.32.1'},{'name':'extra','version':'1'}],
        ):
            with self.subTest(rows=rows):
                self.assertFalse(new.compare_inventory(before,{'all_distributions':rows})['only_pinned_protobuf_added'])


if __name__=='__main__':unittest.main(verbosity=2)
