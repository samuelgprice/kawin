from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IllingworthFDState:
    """
    Transformed planar two-phase state for the Illingworth front-fixing scheme.

    The arrays ``p`` and ``q`` are concentrations on the phase-A and phase-B
    Landau coordinates. For planar geometry the physical coordinates are
    ``x = s*u`` in phase A and ``x = s + (R - s)*v`` in phase B.
    """

    p: np.ndarray
    q: np.ndarray
    s: float


def flatten_1d_coordinates(z):
    """Returns a 1D view of a mesh coordinate array."""
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return np.ravel(arr)


def solve_illingworth_tridiagonal(lo, diag, up, rhs):
    """
    Solves the tri-diagonal system using the sign convention in the MAP code.

    The original C++ Thomas sweep stores rows as ``lo, diag, up, rhs`` and
    solves ``lo[i] c[i-1] + diag[i] c[i] + up[i] c[i+1] = rhs[i]`` through a
    recurrence with denominators ``-diag[i] - lo[i]*alpha[i]``. This helper
    preserves that convention so the Python coefficients can be compared
    directly to the authors' planar implementation.
    """
    lo = np.asarray(lo, dtype=np.float64)
    diag = np.asarray(diag, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    n = int(len(diag))
    if not (len(lo) == len(up) == len(rhs) == n):
        raise ValueError("lo, diag, up, and rhs must have the same length.")

    alpha = np.zeros(n + 1, dtype=np.float64)
    beta = np.zeros(n + 1, dtype=np.float64)
    for i in range(n):
        denom = -diag[i] - lo[i] * alpha[i]
        if abs(denom) <= 1e-300:
            raise ZeroDivisionError("Illingworth tri-diagonal solve encountered a zero pivot.")
        alpha[i + 1] = up[i] / denom
        beta[i + 1] = (lo[i] * beta[i] - rhs[i]) / denom

    c = np.zeros(n, dtype=np.float64)
    c[-1] = beta[n]
    for i in range(n - 2, -1, -1):
        c[i] = alpha[i + 1] * c[i + 1] + beta[i + 1]
    return c


def integrate_planar_transformed_profile(p, q, s: float, domain_length: float, u, v):
    """
    Integrates a sharp-interface planar profile in transformed coordinates.

    The integral is ``s int_0^1 p du + (R-s) int_0^1 q dv``. It is the
    conserved solute inventory used by the planar Illingworth discretization.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    left = float(s) * float(np.trapezoid(p, u))
    right = (float(domain_length) - float(s)) * float(np.trapezoid(q, v))
    return left + right


def reconstruct_planar_transformed_profile(z, p, q, s: float, domain_length: float, u, v):
    """
    Maps transformed phase concentrations back onto physical mesh nodes.

    The reconstruction keeps the sharp interface discontinuity by interpolating
    each phase independently on its own side of the interface.
    """
    z = flatten_1d_coordinates(z)
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)

    left_mask = z <= float(s)
    right_mask = ~left_mask
    if np.any(left_mask):
        u_query = np.clip(z[left_mask] / max(float(s), 1e-300), 0.0, 1.0)
        out[left_mask] = np.interp(u_query, u, p)
    if np.any(right_mask):
        span = max(float(domain_length) - float(s), 1e-300)
        v_query = np.clip((z[right_mask] - float(s)) / span, 0.0, 1.0)
        out[right_mask] = np.interp(v_query, v, q)
    return out
