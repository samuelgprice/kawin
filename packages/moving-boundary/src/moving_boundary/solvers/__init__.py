"""Package-owned moving-boundary solvers; host loop and mesh bases still come from kawin.

Each solver family lives in its own subpackage (currently only ``illingworth``),
so other method families can be added beside it without renaming modules. The
public classes are re-exported here lazily; importing this namespace does not
load kawin.
"""
import importlib

__all__ = ['MovingBoundaryIllingworthFD1DModel', 'MovingBoundaryIllingworthTernaryFD1DModel', 'MovingBoundaryIllingworthTernaryThreePhaseFD1DModel']


def __getattr__(name):
    """Load the package-owned solver lazily and report a missing required host."""
    modules = {'MovingBoundaryIllingworthFD1DModel': 'illingworth.binary', 'MovingBoundaryIllingworthTernaryFD1DModel': 'illingworth.ternary', 'MovingBoundaryIllingworthTernaryThreePhaseFD1DModel': 'illingworth.three_phase'}
    if name not in modules:
        raise AttributeError(name)
    try:
        return getattr(importlib.import_module('.' + modules[name], __name__), name)
    except ModuleNotFoundError as exc:
        if exc.name == 'kawin' or (exc.name and exc.name.startswith('kawin.')):
            raise ImportError('This package requires public kawin 0.5.0; see README.') from exc
        raise
