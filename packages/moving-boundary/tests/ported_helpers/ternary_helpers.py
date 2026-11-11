"""Ported from research ``kawin/tests/test_illingworth_ternary.py`` at ``7af24f4`` (LF-normalized sha256
``abd346b716ce0c79a118606b9b4bee9dabb0d33863c0bcc2aead886224127025``) by
``tools/port_tests.py``; do not edit by hand. Imports resolve to the package as
``tools/definitions/_bindings.py`` does; the full transform list is in
``docs/test-port-provenance.json``. Excluded tests: none. Selected names (with what they need): _IdentityTernaryThermodynamics, _LinearInterfaceEquilibrium.
"""
# --- kawin/tests/test_illingworth_ternary.py:3-3 (aliases used)
import numpy as np


# --- kawin/tests/test_illingworth_ternary.py:47-57
class _IdentityTernaryThermodynamics:
    def __init__(self, invalid=False):
        self.invalid = invalid

    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        if self.invalid:
            return np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
        return np.eye(2, dtype=np.float64)


# --- kawin/tests/test_illingworth_ternary.py:258-272
class _LinearInterfaceEquilibrium:
    eta_bounds = (0.0, 1.0)

    def __init__(self, mode="valid"):
        self.mode = mode

    def interface_compositions(self, eta):
        eta = float(eta)
        left = np.asarray([0.20 + 0.10 * eta, 0.10], dtype=np.float64)
        right = np.asarray([0.30 + 0.10 * eta, 0.15], dtype=np.float64)
        if self.mode == "nonfinite":
            left[0] = np.nan
        elif self.mode == "degenerate":
            right = left.copy()
        return left, right
