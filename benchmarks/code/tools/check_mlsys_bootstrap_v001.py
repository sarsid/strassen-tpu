"""Local, offline integration check for bootstrapping a venv without pip.

Installs only a tiny wheel created by this check. No network, TPU, model weights,
or production environment installation. Run through archive_scoped_v001.
"""
from __future__ import annotations
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import venv
import zipfile

import install_mlsys_model_tools_v001 as installer
from strassen_mm import benchmark_mlsys_prepare_v002 as adapter


def make_wheel(path):
    dist='mlsys_bootstrap_fixture-1.0.0.dist-info'
    payload={
        'mlsys_bootstrap_fixture/__init__.py':b"VALUE = 'isolated-bootstrap-ok'\n",
        dist+'/METADATA':b'Metadata-Version: 2.1\nName: mlsys-bootstrap-fixture\nVersion: 1.0.0\n',
        dist+'/WHEEL':b'Wheel-Version: 1.0\nGenerator: local-bootstrap-check\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
    }
    record=io.StringIO();writer=csv.writer(record,lineterminator='\n')
    for name,data in payload.items():
        digest=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
        writer.writerow([name,'sha256='+digest,len(data)])
    writer.writerow([dist+'/RECORD','','']);payload[dist+'/RECORD']=record.getvalue().encode()
    with zipfile.ZipFile(path,'x') as packed:
        for name,data in payload.items():packed.writestr(name,data)


class BootstrapGuards(unittest.TestCase):
    def test_production_commands_bypass_ensurepip_and_target_only_new_venv(self):
        commands=installer.installation_commands('/usr/bin/python3',Path('/private/model/cpu-tools-v002'),Path('/evidence'))
        self.assertEqual(commands[0],['/usr/bin/python3','-m','venv','--without-pip','/private/model/cpu-tools-v002'])
        for command in commands[1:]:
            self.assertEqual(command[:6],['/usr/bin/python3','-m','pip','--isolated','--python','/private/model/cpu-tools-v002/bin/python'])
        self.assertIn('torch==2.8.0',commands[1])
        self.assertIn('https://download.pytorch.org/whl/cpu',commands[1])
        self.assertEqual(commands[2][-len(installer.PINS):],installer.PINS)
        self.assertIn('protobuf==6.32.1',installer.PINS)
        self.assertFalse(any('ensurepip' in str(x) for command in commands for x in command))

    def test_failed_environment_and_paths_remain_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();partial=root/'cpu-tools';partial.mkdir();sentinel=partial/'old.txt';sentinel.write_text('preserve')
            with patch.object(installer,'PRIVATE_BASE',root.parent):
                self.assertEqual(installer.validate_new_target(root/'cpu-tools-v002',root),root/'cpu-tools-v002')
                with self.assertRaises((ValueError,FileExistsError)):installer.validate_new_target(partial,root)
                with self.assertRaises((ValueError,FileExistsError)):installer.validate_new_target(root.parent/'escape-cpu-tools-v002',root)
                link=root/'linked';link.symlink_to(partial,target_is_directory=True)
                with self.assertRaises((ValueError,FileExistsError)):installer.validate_new_target(link,root)
            self.assertEqual(sentinel.read_text(),'preserve')

    def test_input_preparation_targets_v002(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            with patch.object(adapter,'PRIVATE_BASE',root.parent):
                self.assertEqual(adapter.target_paths(root),(root,root/'cpu-tools-v002'))
                with self.assertRaises(ValueError):adapter.target_paths(root,root/'cpu-tools')
                with self.assertRaises(ValueError):adapter.target_paths(root,root.parent/'cpu-tools-v002')
            self.assertNotIn('jax',sys.modules)

    def test_installer_environment_removes_credentials_and_inherited_configuration(self):
        env=installer.install_environment({'PATH':'fixture-path','PIP_INDEX_URL':'fixture-url','PYTHONPATH':'fixture-pythonpath',
            'PYTHONHOME':'fixture-home','VIRTUAL_ENV':'fixture-venv','HF_TOKEN':'test-only-value','HUGGING_FACE_HUB_TOKEN':'test-only-value'})
        for key in ('PIP_INDEX_URL','PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','HF_TOKEN','HUGGING_FACE_HUB_TOKEN'):
            self.assertNotIn(key,env)
        self.assertEqual(env['PIP_CONFIG_FILE'],os.devnull)
        self.assertEqual(env['PATH'],'fixture-path')

    def test_missing_ensurepip_does_not_block_without_pip_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'cpu-tools-v002'
            with patch.object(venv.EnvBuilder,'_setup_pip',side_effect=ModuleNotFoundError('simulated missing ensurepip')) as blocked:
                venv.EnvBuilder(with_pip=False).create(target)
                blocked.assert_not_called()
                with self.assertRaises(ModuleNotFoundError):venv.EnvBuilder(with_pip=True).create(Path(tmp)/'original-style-fails')


def main():
    if os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('Local CPU validation only')
    run=Path(os.environ['STRASSEN_EXECUTION_DIR']);out=run/'artifacts';out.mkdir(exist_ok=False)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(BootstrapGuards)
    tests=unittest.TextTestRunner(verbosity=2).run(suite)
    if not tests.wasSuccessful():return 1
    before=installer.core_versions();env={k:v for k,v in os.environ.items() if not k.startswith('PIP_') and k not in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV')}
    env.update(PIP_CONFIG_FILE=os.devnull,PYTHONDONTWRITEBYTECODE='1',PIP_NO_INDEX='1')
    commands=[]
    def execute(label,command):
        result=subprocess.run(command,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=60)
        row=dict(label=label,argv=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
        with (out/(label+'.json')).open('x') as f:json.dump(row,f,indent=2)
        commands.append(row)
        if result.returncode:raise RuntimeError('Local bootstrap validation failed: '+label)
        return result.stdout
    with tempfile.TemporaryDirectory(prefix='mlsys-bootstrap-local-') as tmp:
        root=Path(tmp).resolve();target=root/'cpu-tools-v002';wheel=root/'mlsys_bootstrap_fixture-1.0.0-py3-none-any.whl'
        make_wheel(wheel)
        planned=installer.installation_commands(sys.executable,target,out)
        execute('create_without_pip',planned[0])
        missing="import importlib.util,json,sys; assert importlib.util.find_spec('pip') is None; print(json.dumps({'pip_missing':True,'prefix':sys.prefix}))"
        execute('verify_target_has_no_pip',[str(target/'bin/python'),'-I','-c',missing])
        pip_prefix=planned[1][:6]
        execute('install_offline_fixture',pip_prefix+['install','--disable-pip-version-check','--no-cache-dir','--no-index','--no-deps',str(wheel)])
        probe="import json,sys,site; from pathlib import Path; import mlsys_bootstrap_fixture as p; target=Path(sys.argv[1]); assert Path(sys.prefix).resolve()==target; assert Path(p.__file__).resolve().is_relative_to(target); assert site.ENABLE_USER_SITE is False; assert p.VALUE=='isolated-bootstrap-ok'; print(json.dumps({'value':p.VALUE,'module_path':p.__file__,'prefix':sys.prefix,'user_site_enabled':site.ENABLE_USER_SITE}))"
        observed=json.loads(execute('verify_isolated_import',[str(target/'bin/python'),'-I','-c',probe,str(target)]))
        execute('external_pip_check',pip_prefix+['check','--disable-pip-version-check'])
    after=installer.core_versions()
    if before!=after:raise RuntimeError('Global package versions changed')
    result=dict(status='passed',guard_tests=tests.testsRun,offline_integration='Tiny local wheel installed/imported in a pip-less target through external pip --python',
        external_interpreter=sys.executable,global_versions_before=before,global_versions_after=after,global_versions_unchanged=True,
        observed_isolation=observed,remote_operations=False,network_used=False,production_dependencies_installed=False,
        limitation='Local bootstrap mechanism and guards validated; full pinned model environment on Colab remains unexecuted.')
    with (out/'summary.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':'passed','guard_tests':tests.testsRun,'remote_operations':False,'network_used':False}))
    return 0

if __name__=='__main__':raise SystemExit(main())
