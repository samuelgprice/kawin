"""C25–C29 belong to pytest, with isolated main-versus-installed comparisons."""
from pathlib import Path
import importlib.util
import json
import os

import pytest


@pytest.mark.parametrize("case", ["C25","C26","C27","C28","C29"])
def test_main_vs_package_contract(case):
    """Require immutable main fixtures and the acceptance command's installed target."""
    configured = os.environ.get("MB_ACCEPTANCE_STATE")
    if not configured:
        pytest.skip("Run tools/acceptance.py quick to configure isolated contract comparisons")
    state = json.loads(Path(configured).read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("contract_runner", Path(__file__).parents[1] / "tools/acceptance.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.contract_case(case, state, reference=Path(state["reference"]),
                         tolerance=state["tolerance"], require=state["require"])
