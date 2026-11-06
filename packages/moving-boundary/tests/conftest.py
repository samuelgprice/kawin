"""Install batch safety before collecting package verification tests."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).parents[1] / "tools/batch_safety.py"))["forbid_input"]()
