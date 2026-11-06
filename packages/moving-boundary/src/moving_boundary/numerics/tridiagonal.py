"""Scalar Illingworth Thomas sweep, preserving the original operation order.

Source: samuelgprice/kawin, ae9fb2971a9fd49132f71ddc883a5bd5406644b6.
The fork identifies the binary scheme as a port of the authors' MAP C++ code.
See NOTICE and docs/BINARY_SLICE.md; redistribution clearance remains unresolved.

Original: kawin/diffusion/mesh/MovingBoundaryIllingworthFD1D.py,
_solve_illingworth_tridiagonal_python, _solve_illingworth_tridiagonal_numba,
and solve_illingworth_tridiagonal. No pivoting or new validation is added.
"""

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only without optional numba
    njit = None


def _solve_illingworth_tridiagonal_python(lo, diag, up, rhs):
    """
    Pure-Python Thomas sweep for the Illingworth tri-diagonal convention.

    This is kept as the exact fallback and reference implementation for the
    optional compiled path used by ``solve_illingworth_tridiagonal``.
    """
    n = int(len(diag))
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


if njit is not None:
    @njit(cache=True)
    def _solve_illingworth_tridiagonal_numba(lo, diag, up, rhs):
        """Execute the identical non-pivoting Thomas recurrence with Numba."""
        n = int(len(diag))
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
else:
    _solve_illingworth_tridiagonal_numba = None


def solve_illingworth_tridiagonal(lo, diag, up, rhs):
    """
    Solves the tri-diagonal system using the sign convention in the MAP code.

    The original C++ Thomas sweep stores rows as ``lo, diag, up, rhs`` and
    solves ``lo[i] c[i-1] + diag[i] c[i] + up[i] c[i+1] = rhs[i]`` through a
    recurrence with denominators ``-diag[i] - lo[i]*alpha[i]``. This helper
    preserves that convention so the Python coefficients can be compared
    directly to the authors' planar implementation. If ``numba`` is installed,
    the same recurrence is executed through an optional compiled path; otherwise
    the pure-Python reference implementation is used.
    """
    lo = np.asarray(lo, dtype=np.float64)
    diag = np.asarray(diag, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    n = int(len(diag))
    if not (len(lo) == len(up) == len(rhs) == n):
        raise ValueError("lo, diag, up, and rhs must have the same length.")

    if _solve_illingworth_tridiagonal_numba is not None:
        return _solve_illingworth_tridiagonal_numba(lo, diag, up, rhs)
    return _solve_illingworth_tridiagonal_python(lo, diag, up, rhs)
