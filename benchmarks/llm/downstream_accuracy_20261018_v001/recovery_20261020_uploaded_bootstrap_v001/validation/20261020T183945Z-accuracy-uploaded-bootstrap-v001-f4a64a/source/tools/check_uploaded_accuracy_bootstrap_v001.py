"""Exercise notebook dispatch with a nonexistent host-side __file__."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime"))
import execute_uploaded_accuracy_bootstrap_v001 as launcher


class UploadedBootstrapTests(unittest.TestCase):
    def test_notebook_dispatch_uses_uploaded_file_and_exact_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            script = base / "uploaded.py"
            receipt = base / "receipt.json"
            script.write_text(
                "import hashlib,json,sys\nfrom pathlib import Path\n"
                "Path(sys.argv[1]).write_text(json.dumps(dict("
                "file=__file__,sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"
                "argv=sys.argv[2:])))\n"
            )
            sha = hashlib.sha256(script.read_bytes()).hexdigest()
            wrapper = Path(launcher.__file__).read_text().replace(
                'BASE = Path("/content/Strassen_MM_Focus")', "BASE = Path(" + repr(str(base)) + ")"
            )
            argv = ["nonexistent-host.py", "--bootstrap", str(script),
                    "--bootstrap-sha256", sha, str(receipt), "--run-id", "new-recovery"]
            with patch.object(sys, "argv", argv):
                exec(compile(wrapper, "/nonexistent/laptop/wrapper.py", "exec"),
                     {"__name__": "__main__", "__file__": "/nonexistent/laptop/wrapper.py"})
            result = json.loads(receipt.read_text())
            self.assertEqual(result, dict(file=str(script), sha=sha, argv=["--run-id", "new-recovery"]))

    def test_hash_or_outside_path_never_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            script = base / "uploaded.py"
            script.write_text("raise RuntimeError('must not execute')\n")
            sha = hashlib.sha256(script.read_bytes()).hexdigest()
            with patch.object(launcher, "BASE", base):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    launcher.execute(script, "wrong", [])
            with patch.object(launcher, "BASE", base / "other"):
                with self.assertRaisesRegex(ValueError, "inside"):
                    launcher.execute(script, sha, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
