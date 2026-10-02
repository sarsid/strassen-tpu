"""Run the frozen qualifier inside an immutable scoped execution directory."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest

from qualify_gemma_v001 import main


if __name__ == "__main__":
    run = Path(os.environ["STRASSEN_EXECUTION_DIR"])
    if "--output-dir" in sys.argv:
        raise ValueError("The scoped runner owns the output directory")
    if "tiny" in sys.argv:
        path = Path(__file__).resolve().parents[1] / "tests" / "test_gemma_v001.py"
        spec = importlib.util.spec_from_file_location("gemma_guard_tests", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite = unittest.TestSuite(module.GemmaOfficialQualification(name) for name in (
            "test_metric_gates_reject_large_architectural_error",
            "test_nondefault_rope_is_explicitly_unsupported",
        ))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
    sys.argv.extend(["--output-dir", str(run / "artifacts")])
    raise SystemExit(main())
