import ast, importlib.util, io, json, tarfile
from pathlib import Path
root = Path(__file__).resolve().parent
runtime = root.parent
for filename in ('launch_phase_v001.py', 'poll_phase_v001.py'):
    ast.parse((runtime / filename).read_text())
spec = importlib.util.spec_from_file_location('launch_check', runtime / 'launch_phase_v001.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
fixtures = root / 'fixtures'; fixtures.mkdir()
archive = fixtures / 'source.tar'
with tarfile.open(archive, 'w') as package:
    item = tarfile.TarInfo('snapshot/src/strassen_mm/benchmark_v001.py')
    content = b'# Offline fixture only; never executed.\\n'
    item.size = len(content); package.addfile(item, io.BytesIO(content))
destination = fixtures / 'extracted'; destination.mkdir()
assert module.extract_source(archive, destination) == destination / 'snapshot'
for label, name, link in (('traversal', '../escape.py', False),
                          ('absolute', '/tmp/escape.py', False),
                          ('symlink', 'evil.py', True)):
    unsafe = fixtures / (label + '.tar')
    with tarfile.open(unsafe, 'w') as package:
        item = tarfile.TarInfo(name)
        if link: item.type = tarfile.SYMTYPE; item.linkname = '/tmp/target'
        package.addfile(item)
    out = fixtures / label; out.mkdir()
    try: module.extract_source(unsafe, out)
    except ValueError: pass
    else: raise AssertionError(label)
assert module.checked_run_id('smoke-20260919-v001')
try: module.checked_run_id('../bad')
except Exception: pass
else: raise AssertionError('run path escape')
print('PASS: both wrapper scripts parse; source.tar extracts at expected root; traversal, absolute paths, links and invalid run IDs are rejected.')
