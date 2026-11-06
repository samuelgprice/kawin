"""Licence-free import and failure contracts for the frozen TC-Python facade.

Fake calculators are used only to force error paths; they do not represent
thermodynamic generation or acceptance data.
"""
import builtins
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_thermocalc_import_is_lazy_and_has_no_research_dependency(tmp_path):
    """Import the installed facade while rejecting TC and debugger/research imports."""
    import moving_boundary
    code = """
import site, sys
sys.path.insert(0, sys.argv[1])
sys.path.extend(site.getsitepackages())
class NoEngineOrResearch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tc_python', 'examples', 'debugpy', 'pdb'}:
            raise AssertionError('Unexpected engine/research/debugger import: ' + fullname)
sys.meta_path.insert(0, NoEngineOrResearch())
from moving_boundary.providers import thermocalc
assert thermocalc.TCPythonThermodynamics
assert not any(name.split('.')[0] in {'tc_python', 'examples', 'debugpy', 'pdb'} for name in sys.modules)
"""
    subprocess.run([sys.executable, '-I', '-S', '-c', code,
                    str(Path(moving_boundary.__file__).parents[1])],
                   cwd=tmp_path, stdin=subprocess.DEVNULL, check=True, timeout=30)


@pytest.fixture
def failure_guard(monkeypatch):
    """Model a notebook failure without allowing prompts or engine/debugger imports."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in {'tc_python', 'examples', 'debugpy', 'pdb'}:
            pytest.fail('Unexpected engine/research/debugger import: ' + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(builtins, 'input', lambda *args: pytest.fail('Unexpected prompt'))
    monkeypatch.setattr(builtins, 'breakpoint', lambda *args: pytest.fail('Unexpected breakpoint'))
    monkeypatch.setitem(sys.modules, 'ipykernel', object())


def _backend(monkeypatch, *, errors, reset_error=None, retry_points=()):
    """Supply only fake calculation plumbing; exercise the real wrapper/retry logic."""
    from moving_boundary.providers import thermocalc as tc
    backend = tc._TCPythonBackend()
    backend._config = tc.ThermoCalcConfig(
        thermodynamic_database='FAKE_THERMODYNAMIC', kinetic_database='FAKE_KINETIC',
        elements=('A', 'B', 'C'), phases=('ALPHA', 'BETA'), reference_element='A',
        cache_dir=None, equilibrium_qthiss_retry_grid_points=retry_points)
    backend._setup = object()
    attempts, grids = [], []
    error_iterator = iter(errors)

    class Calculator:
        def remove_all_conditions(self):
            if reset_error is not None:
                raise reset_error

        def calculate(self):
            error = next(error_iterator)
            attempts.append(error)
            raise error

    def configure(calc, config, kind, **kwargs):
        grids.append(kwargs['grid_points_override'])
        return calc

    monkeypatch.setattr(backend, '_get_calculation', lambda *args: Calculator())
    monkeypatch.setattr(backend, '_get_system', lambda *args:
                        SimpleNamespace(with_single_equilibrium_calculation=Calculator))
    monkeypatch.setattr(backend, '_set_conditions', lambda *args: None)
    monkeypatch.setattr(backend, '_configure_global_minimization', configure)
    return tc, backend, attempts, grids


@pytest.mark.parametrize('failure_at', ['reset', 'calculate'])
def test_calculation_failure_preserves_cause_without_interaction(monkeypatch, failure_guard, capsys, failure_at):
    """Both the former debugger branch and calculate() failures preserve their cause."""
    cause = ValueError('forced fake failure')
    tc, backend, attempts, grids = _backend(monkeypatch, errors=[cause],
        reset_error=cause if failure_at == 'reset' else None)
    with pytest.raises(tc.ThermoCalcCalculationError,
                       match='TC-Python equilibrium calculation failed: forced fake failure') as caught:
        backend._calculate('equilibrium', None, np.array([.2, .1]), 1000.)
    assert caught.value.__cause__ is cause
    assert type(caught.value) is (tc.ThermoCalcCalculationError if failure_at == 'reset' else tc.ThermoCalcSolveError)
    assert backend._last_calculation_kind == 'equilibrium'
    assert len(attempts) == (0 if failure_at == 'reset' else 1)
    assert grids == []
    assert bool(capsys.readouterr().out) == (failure_at == 'reset')


@pytest.mark.parametrize('retry_failure', ['qthiss', 'other'])
def test_qthiss_retry_failure_preserves_cause_without_interaction(monkeypatch, failure_guard, capsys, retry_failure):
    """Exhausted QTHISS retries and a different retry failure retain the last cause."""
    first = RuntimeError('ERROR IN QTHISS: TOO MANY ITERATIONS')
    last = RuntimeError('ERROR IN QTHISS: TOO MANY ITERATIONS' if retry_failure == 'qthiss' else 'forced retry failure')
    errors = [first, first, last] if retry_failure == 'qthiss' else [first, last]
    tc, backend, attempts, grids = _backend(monkeypatch, errors=errors, retry_points=(4000, 8000))
    key = ('equilibrium', None, backend._config.use_default_phases)
    backend._calculations[key] = object()
    with pytest.raises(tc.ThermoCalcCalculationError) as caught:
        backend._calculate('equilibrium', None, np.array([.2, .1]), 1000.)
    assert type(caught.value) is tc.ThermoCalcSolveError
    assert caught.value.__cause__ is last
    assert str(caught.value) == 'TC-Python equilibrium calculation failed: ' + str(last)
    assert attempts == errors
    assert grids == ([4000, 8000] if retry_failure == 'qthiss' else [4000])
    assert backend._retry_settings_dirty
    assert key not in backend._calculations
    assert backend._last_calculation_kind == 'equilibrium'
    assert capsys.readouterr().out.count('Retrying equilibrium') == len(grids)
