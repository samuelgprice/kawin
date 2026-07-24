from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TernaryIllingworthFDState:
    """
    Transformed planar two-phase state for ternary Illingworth front fixing.

    ``p`` and ``q`` store the two independent substitutional compositions on
    the left and right Landau grids. The interface position is ``s`` and ``eta``
    is an optional scalar tie-line coordinate used by the interface equilibrium
    closure.
    """

    p: np.ndarray
    q: np.ndarray
    s: float
    eta: float


def flatten_1d_coordinates(z):
    """Returns a 1D view of a mesh coordinate array."""
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return np.ravel(arr)


def validate_ternary_profile(values, name: str) -> np.ndarray:
    """
    Validates and returns a two-component transformed composition profile.

    The ternary Illingworth implementation stores only the two independent
    compositions. The dependent component is reconstructed elsewhere from the
    substitutional sum constraint.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_nodes, 2).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def integrate_planar_transformed_profile_components(p, q, s: float, domain_length: float, u, v):
    """
    Integrates each independent component in planar transformed coordinates.

    The conserved inventory vector is
    ``s int_0^1 p du + (R-s) int_0^1 q dv``. The integration is componentwise
    and uses the supplied Landau grids, which may be nonuniform.
    """
    p = validate_ternary_profile(p, "p")
    q = validate_ternary_profile(q, "q")
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if len(u) != len(p) or len(v) != len(q):
        raise ValueError("Landau grid lengths must match p and q.")
    left = float(s) * np.trapezoid(p, u, axis=0)
    right = (float(domain_length) - float(s)) * np.trapezoid(q, v, axis=0)
    return np.asarray(left + right, dtype=np.float64)


def reconstruct_planar_transformed_profile_components(z, p, q, s: float, domain_length: float, u, v):
    """
    Maps ternary transformed phase profiles back onto physical mesh nodes.

    Each side of the sharp interface is interpolated independently so the
    discontinuity at the interface is retained in the physical profile.
    """
    z = flatten_1d_coordinates(z)
    p = validate_ternary_profile(p, "p")
    q = validate_ternary_profile(q, "q")
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    out = np.empty((len(z), 2), dtype=np.float64)

    left_mask = z <= float(s)
    right_mask = ~left_mask
    if np.any(left_mask):
        u_query = np.clip(z[left_mask] / max(float(s), 1e-300), 0.0, 1.0)
        for component in range(2):
            out[left_mask, component] = np.interp(u_query, u, p[:, component])
    if np.any(right_mask):
        span = max(float(domain_length) - float(s), 1e-300)
        v_query = np.clip((z[right_mask] - float(s)) / span, 0.0, 1.0)
        for component in range(2):
            out[right_mask, component] = np.interp(v_query, v, q[:, component])
    return out


def _solve_2x2_matrix(matrix, values, row_name):
    """Solves one or more 2x2 systems using explicit formulas."""
    matrix = np.asarray(matrix, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if abs(determinant) <= 1e-300:
        raise ZeroDivisionError(f"Block-tridiagonal solve encountered a singular 2x2 pivot at {row_name}.")
    inverse = np.asarray([[matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]]], dtype=np.float64) / determinant
    if values.ndim == 1:
        return _matvec_2x2(inverse, values)
    return _matmul_2x2(inverse, values)


def _matvec_2x2(matrix, vector):
    """Multiplies a 2x2 matrix by a length-2 vector without BLAS dispatch."""
    return np.asarray(
        [
            matrix[0, 0] * vector[0] + matrix[0, 1] * vector[1],
            matrix[1, 0] * vector[0] + matrix[1, 1] * vector[1],
        ],
        dtype=np.float64,
    )


def _matmul_2x2(left, right):
    """Multiplies two 2x2 matrices without BLAS dispatch."""
    return np.asarray(
        [
            [
                left[0, 0] * right[0, 0] + left[0, 1] * right[1, 0],
                left[0, 0] * right[0, 1] + left[0, 1] * right[1, 1],
            ],
            [
                left[1, 0] * right[0, 0] + left[1, 1] * right[1, 0],
                left[1, 0] * right[0, 1] + left[1, 1] * right[1, 1],
            ],
        ],
        dtype=np.float64,
    )


def solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs):
    """
    Solves a block-tridiagonal Illingworth system with dense component blocks.

    Blocks have shape ``(n_nodes, 2, 2)`` and ``rhs`` has shape
    ``(n_nodes, 2)``. A block Thomas sweep preserves full cross-diffusion
    coupling between the two independent ternary components while keeping the
    scalar Thomas solver used by the binary implementation untouched.
    """
    lower = np.asarray(lower, dtype=np.float64)
    diagonal = np.asarray(diagonal, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if diagonal.ndim != 3 or diagonal.shape[1:] != (2, 2):
        raise ValueError("diagonal must have shape (n_nodes, 2, 2).")
    if lower.shape != diagonal.shape or upper.shape != diagonal.shape:
        raise ValueError("lower, diagonal, and upper must have matching shapes.")
    if rhs.shape != diagonal.shape[:2]:
        raise ValueError("rhs must have shape (n_nodes, n_components).")
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(diagonal)) and np.all(np.isfinite(upper))):
        raise ValueError("Block-tridiagonal coefficients must be finite.")
    if not np.all(np.isfinite(rhs)):
        raise ValueError("Block-tridiagonal rhs must be finite.")

    n_nodes = diagonal.shape[0]
    modified_upper = np.zeros_like(upper)
    modified_rhs = np.zeros_like(rhs)
    modified_upper[0] = _solve_2x2_matrix(diagonal[0], upper[0], "row 0")
    modified_rhs[0] = _solve_2x2_matrix(diagonal[0], rhs[0], "row 0")
    for node in range(1, n_nodes):
        pivot = diagonal[node] - _matmul_2x2(lower[node], modified_upper[node - 1])
        row_rhs = rhs[node] - _matvec_2x2(lower[node], modified_rhs[node - 1])
        if node < n_nodes - 1:
            modified_upper[node] = _solve_2x2_matrix(pivot, upper[node], f"row {node}")
        modified_rhs[node] = _solve_2x2_matrix(pivot, row_rhs, f"row {node}")

    solution = np.zeros_like(rhs)
    solution[-1] = modified_rhs[-1]
    for node in range(n_nodes - 2, -1, -1):
        solution[node] = modified_rhs[node] - _matvec_2x2(modified_upper[node], solution[node + 1])
    return solution
