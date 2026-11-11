"""Install batch safety (and the opt-in independence guards) before collecting package tests.

Also puts this directory on ``sys.path`` so ported research tests can import their helpers as
``ported_helpers.<module>`` under every import mode (the acceptance gates use
``--import-mode=importlib``, which does not add test directories to ``sys.path``).
"""
from pathlib import Path
import runpy
import sys

_SAFETY = runpy.run_path(str(Path(__file__).parents[1] / "tools/batch_safety.py"))
_SAFETY["forbid_input"]()
_SAFETY["guard_independence"]()
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
