"""Run notebook cells with the documented extraction-root working directory."""

from pathlib import Path
import sys
from types import ModuleType

import pytest

EXAMPLES = Path(__file__).parents[1] / "examples"
PACKAGE = EXAMPLES.parent
REPOSITORY = PACKAGE.parents[1]
NAMES = ["binary_planar_growth", "binary_tlp_bonding", "ternary_interstitial_similarity",
         "surrogate_lifecycle", "three_phase_similarity", "ternary_fecrni_lee1996",
         "surrogate_workflow_fecrni_open"]


def _notebook_namespace(name, cwd, monkeypatch):
    """Execute definitions as cells do, without file metadata or cached companions.

    A registered namespace supports dataclasses. Its name avoids executing the
    scripts' guarded production runs; tests select any solve explicitly.
    """
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path
                                     if Path(entry or cwd).resolve() != EXAMPLES.resolve()])
    for module_name in list(sys.modules):
        if module_name == "advanced" or module_name.startswith("advanced."):
            monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setitem(sys.modules, "ipykernel", ModuleType("ipykernel"))
    module = ModuleType("notebook_" + name)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    script = EXAMPLES / f"{name}.py"
    exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("name", NAMES)
def test_cells_find_companions_from_repository_root_without_file_metadata(name, monkeypatch):
    module = _notebook_namespace(name, REPOSITORY, monkeypatch)
    assert "__file__" not in module.__dict__
    assert Path(module._advanced.__file__).resolve() == (EXAMPLES / "advanced" / f"{name}.py").resolve()
    assert Path.cwd() == REPOSITORY
    if name == "surrogate_workflow_fecrni_open":
        assert module.DATABASE == EXAMPLES / "FeCrNi_MatCalc_open.tdb"
        assert module.DATABASE.is_file()


def test_planar_cells_solve_after_importing_from_repository_root(monkeypatch):
    module = _notebook_namespace("binary_planar_growth", REPOSITORY, monkeypatch)
    result = module.run(module.FAST)
    assert result.model.currentTime == pytest.approx(module.FAST.duration)
    assert result.metrics["inventory_abs"] < module.DEMO_LIMITS["inventory_abs"]
