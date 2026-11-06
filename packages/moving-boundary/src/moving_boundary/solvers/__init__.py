"""Relocated Illingworth solvers; host loop and mesh bases still come from kawin."""
import importlib

__all__ = ['MovingBoundaryIllingworthFD1DModel']


def __getattr__(name):
    """Load the package-owned solver lazily and report a missing required host."""
    modules = {'MovingBoundaryIllingworthFD1DModel': 'binary'}
    if name not in modules:
        raise AttributeError(name)
    try:
        return getattr(importlib.import_module('.' + modules[name], __name__), name)
    except ModuleNotFoundError as exc:
        if exc.name == 'kawin' or (exc.name and exc.name.startswith('kawin.')):
            raise ImportError('This checkpoint requires the pinned kawin host 5f4ad574; see README.') from exc
        raise
