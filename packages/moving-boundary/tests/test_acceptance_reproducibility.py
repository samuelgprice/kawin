"""Regression checks for narrow seed scope and true bitwise repeatability."""
import importlib.util
from pathlib import Path
import numpy as np


def runner():
    """Load the verification tool without importing research source packages."""
    spec=importlib.util.spec_from_file_location('repro_runner',Path(__file__).parents[1]/'tools/acceptance.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seeds_are_scoped_to_provider(monkeypatch):
    """Inherited hash/RNG configuration cannot leak into unaffected workers."""
    monkeypatch.setenv('PYTHONHASHSEED','123')
    monkeypatch.setenv('PYTHONPATH','untrusted')
    tool=runner()
    assert tool.worker_environment('O05')['PYTHONHASHSEED']=='0'
    for case in ['C01','C24','C25','LEGACY','F11','O04']:
        env=tool.worker_environment(case)
        assert 'PYTHONHASHSEED' not in env and 'PYTHONPATH' not in env


def test_bitwise_check_distinguishes_signed_zero():
    """Numerically equal signed zeros must not be labelled bit-identical."""
    tool=runner()
    assert not tool.bit_identical(np.array([0.]),np.array([-0.]))
    assert tool.bit_identical(np.array([np.nan]),np.array([np.nan]))


def test_bitwise_check_requires_dtype_and_shape():
    """Equal byte strings with different interpretations are not identical arrays."""
    tool=runner()
    assert not tool.bit_identical(np.array([0.]),np.array([0],dtype=np.int64))
    assert not tool.bit_identical(np.array([0.]),np.array([[0.]]))
