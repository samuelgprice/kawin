# Relocated from samuelgprice/kawin 06dc7b4ab56f67226ceabfde41ebf2690c088841
# Source: examples/Illingworth2005/replicate_illingworth2005_fig4_5_6_planar.py
# (VALIDATION_PARAMS, planar_similarity_integral, solve_planar_zener_j,
# planar_exact_interface, planar_exact_profile, PlanarExactProfile); selected
# definitions only, no MAP-derived code. See NOTICE and docs/LEDGERS.md.
"""
Exact planar Zener growth of a binary interface between fixed compositions.

Phase A occupies ``z < s(t)`` at the uniform composition ``c_a``; phase B has
far-field composition ``c_inf`` and interface composition ``c_b``. With constant
``D_b`` and a semi-infinite phase B, ``s(t) = 2 j sqrt(D_b t)``. The Illingworth
solver therefore starts from this exact state at ``t_init > 0``.
"""

import math

import numpy as np
from scipy import optimize, special

VALIDATION_PARAMS = {
    "c_a": 0.4,
    "c_b": 0.2,
    "c_inf": 0.3,
    "R": 1.0,
    "D_a": 1.0,
    "D_b": 1.0,
    "t_init": 1.0e-4,
    "t_abs_end": 1.0e-2,
    "temperature": 1000.0,
    "tolerance": 1.0e-8,
}


def planar_similarity_integral(eta):
    """Returns the planar Zener similarity integral ``I_1(eta)``."""
    return 0.5 * math.sqrt(math.pi) * special.erfc(eta)


def solve_planar_zener_j(params=None):
    """
    Solves the planar Zener growth constant for the validation composition set.
    """
    p = VALIDATION_PARAMS if params is None else {**VALIDATION_PARAMS, **dict(params)}
    ratio = (p["c_b"] - p["c_inf"]) / (p["c_b"] - p["c_a"])

    def residual(j):
        return 2.0 * j * planar_similarity_integral(j) * math.exp(j * j) - ratio

    return float(optimize.brentq(residual, 1.0e-14, 20.0, xtol=1e-14, rtol=1e-14))


def planar_exact_interface(t_abs, params=None, j=None):
    """Returns the exact planar interface position at absolute time ``t_abs``."""
    p = VALIDATION_PARAMS if params is None else {**VALIDATION_PARAMS, **dict(params)}
    growth_j = solve_planar_zener_j(p) if j is None else float(j)
    t = np.asarray(t_abs, dtype=np.float64)
    return growth_j * np.sqrt(4.0 * p["D_b"] * t)


def planar_exact_profile(z, t_abs, params=None, j=None):
    """
    Returns the analytical planar concentration profile used for initialization.
    """
    p = VALIDATION_PARAMS if params is None else {**VALIDATION_PARAMS, **dict(params)}
    growth_j = solve_planar_zener_j(p) if j is None else float(j)
    z_arr = np.asarray(z, dtype=np.float64)
    eta = z_arr / math.sqrt(4.0 * p["D_b"] * float(t_abs))
    phase_b = p["c_inf"] + (p["c_b"] - p["c_inf"]) * special.erfc(eta) / special.erfc(growth_j)
    interface = planar_exact_interface(t_abs, p, growth_j)
    return np.where(z_arr <= interface, p["c_a"], phase_b)


class PlanarExactProfile:
    """Initial analytical concentration profile sampled by the FD mesh."""

    def __init__(self, t_abs, params=None):
        self.t_abs = float(t_abs)
        self.params = VALIDATION_PARAMS if params is None else {**VALIDATION_PARAMS, **dict(params)}
        self.growth_j = solve_planar_zener_j(self.params)

    def __call__(self, z):
        values = planar_exact_profile(
            np.asarray(z, dtype=np.float64).reshape(-1),
            self.t_abs,
            self.params,
            self.growth_j,
        )
        return values[:, np.newaxis]
