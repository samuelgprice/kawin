"""Install batch safety before collecting package verification tests.

Also puts this directory on ``sys.path`` so ported research tests can import their helpers as
``ported_helpers.<module>`` under every import mode (the acceptance gates use
``--import-mode=importlib``, which does not add test directories to ``sys.path``).
"""
from pathlib import Path
import runpy
import sys

runpy.run_path(str(Path(__file__).parents[1] / "tools/batch_safety.py"))["forbid_input"]()
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
