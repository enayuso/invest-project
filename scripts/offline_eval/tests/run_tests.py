"""Run only the standalone offline_eval unittest suite."""

from pathlib import Path
import sys
import unittest


TEST_DIR = Path(__file__).resolve().parent
TOOL_DIR = TEST_DIR.parent
sys.path.insert(0, str(TOOL_DIR))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(TEST_DIR), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
