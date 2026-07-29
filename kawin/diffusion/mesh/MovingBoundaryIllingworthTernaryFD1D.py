from dataclasses import dataclass
import math

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


_BLOCK_PIVOT_CONDITION_LIMIT = 1.0e12
_BLOCK_PIVOT_RCOND_LIMIT = 1.0 / _BLOCK_PIVOT_CONDITION_LIMIT
_BLOCK_PIVOT_STATUS_VALID = "valid"
_BLOCK_PIVOT_STATUS_NONFINITE = "nonfinite"
_BLOCK_PIVOT_STATUS_SINGULAR = "singular"


class _BlockThomasPivotError(np.linalg.LinAlgError):
    """Signals that block Thomas elimination needs a full-system fallback."""


def _validate_block_tridiagonal_system(lower, diagonal, upper, rhs):
    """Validates block-tridiagonal coefficient arrays and one or more RHS columns."""
    lower = np.asarray(lower, dtype=np.float64)
    diagonal = np.asarray(diagonal, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if diagonal.ndim != 3 or diagonal.shape[1:] != (2, 2):
        raise ValueError("diagonal must have shape (n_nodes, 2, 2).")
    if diagonal.shape[0] < 1:
        raise ValueError("Block-tridiagonal systems must contain at least one node.")
    if lower.shape != diagonal.shape or upper.shape != diagonal.shape:
        raise ValueError("lower, diagonal, and upper must have matching shapes.")
    if rhs.ndim == 2:
        if rhs.shape != diagonal.shape[:2]:
            raise ValueError("rhs must have shape (n_nodes, 2) or (n_nodes, 2, n_rhs).")
    elif rhs.ndim == 3:
        if rhs.shape[:2] != diagonal.shape[:2]:
            raise ValueError("rhs must have shape (n_nodes, 2) or (n_nodes, 2, n_rhs).")
    else:
        raise ValueError("rhs must have shape (n_nodes, 2) or (n_nodes, 2, n_rhs).")
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(diagonal)) and np.all(np.isfinite(upper))):
        raise ValueError("Block-tridiagonal coefficients must be finite.")
    if not np.all(np.isfinite(rhs)):
        raise ValueError("Block-tridiagonal rhs must be finite.")
    return lower, diagonal, upper, rhs


def _estimate_2x2_rcond(matrix):
    """
    Estimates reciprocal conditioning and status for a real 2x2 block.

    The four entries are read once as Python scalars, checked for finiteness,
    and scaled by their largest absolute value before products are formed. For
    scaled entries ``a, b, c, d``, the estimate is
    ``abs(a*d - b*c) / (max(|a|+|b|, |c|+|d|) * max(|d|+|b|, |c|+|a|))``.
    This is dimensionless, invariant to equation units, and avoids
    overflow/underflow for very small or very large block coefficients.

    Returns ``(rcond, status)`` where ``"nonfinite"`` means at least one matrix
    entry is nonfinite, ``"singular"`` means the finite matrix has no positive
    finite reciprocal-condition estimate, and ``"valid"`` means the returned
    reciprocal condition is positive and finite. Ill-conditioned-but-finite
    pivots remain ``"valid"`` so callers can apply the active
    reciprocal-condition threshold.
    """
    a = float(matrix[0, 0])
    b = float(matrix[0, 1])
    c = float(matrix[1, 0])
    d = float(matrix[1, 1])
    if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c) and math.isfinite(d)):
        return 0.0, _BLOCK_PIVOT_STATUS_NONFINITE
    scale = max(abs(a), abs(b), abs(c), abs(d))
    if scale <= 0.0:
        return 0.0, _BLOCK_PIVOT_STATUS_SINGULAR
    a /= scale
    b /= scale
    c /= scale
    d /= scale
    determinant = a * d - b * c
    matrix_norm = max(abs(a) + abs(b), abs(c) + abs(d))
    inverse_adjugate_norm = max(abs(d) + abs(b), abs(c) + abs(a))
    denominator = matrix_norm * inverse_adjugate_norm
    if denominator <= 0.0 or not math.isfinite(denominator):
        return 0.0, _BLOCK_PIVOT_STATUS_SINGULAR
    rcond = abs(determinant) / denominator
    if not math.isfinite(rcond) or rcond <= 0.0:
        return 0.0, _BLOCK_PIVOT_STATUS_SINGULAR
    return float(rcond), _BLOCK_PIVOT_STATUS_VALID


def _check_block_pivot(matrix, row_name):
    """
    Rejects nonfinite or excessively ill-conditioned 2x2 block pivots.

    The reciprocal-condition threshold is relative and therefore invariant to
    multiplying the assembled equations by a nonzero scalar. Pivots with
    estimated ``rcond(A) < 1 / _BLOCK_PIVOT_CONDITION_LIMIT`` are considered
    unusable for the block Thomas sweep and trigger the exact full-system
    fallback.
    """
    rcond, status = _estimate_2x2_rcond(matrix)
    if status == _BLOCK_PIVOT_STATUS_NONFINITE:
        raise _BlockThomasPivotError(f"Block-tridiagonal solve encountered a nonfinite 2x2 pivot at {row_name}.")
    if status == _BLOCK_PIVOT_STATUS_SINGULAR:
        raise _BlockThomasPivotError(f"Block-tridiagonal solve encountered a singular 2x2 pivot at {row_name}.")
    if rcond < _BLOCK_PIVOT_RCOND_LIMIT:
        raise _BlockThomasPivotError(
            "Block-tridiagonal solve encountered an ill-conditioned 2x2 pivot "
            f"at {row_name}; estimated reciprocal condition={rcond:.3e}, "
            f"minimum={_BLOCK_PIVOT_RCOND_LIMIT:.3e}."
        )


def _solve_checked_2x2_matrix(matrix, values, row_name):
    """Validates one 2x2 pivot once, then solves one or more RHS columns."""
    matrix = np.asarray(matrix, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    _check_block_pivot(matrix, row_name)
    try:
        return np.linalg.solve(matrix, values)
    except np.linalg.LinAlgError as exc:
        raise _BlockThomasPivotError(
            f"Block-tridiagonal solve encountered an unusable 2x2 pivot at {row_name}: {exc}"
        ) from exc


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


def _apply_2x2_block(matrix, values):
    """Applies a 2x2 block to one RHS vector or multiple RHS columns."""
    return np.matmul(matrix, values)


def _assemble_dense_block_tridiagonal(lower, diagonal, upper):
    """Assembles the dense matrix represented exactly by the block arrays."""
    n_nodes = diagonal.shape[0]
    matrix = np.zeros((2 * n_nodes, 2 * n_nodes), dtype=np.float64)
    for node in range(n_nodes):
        rows = slice(2 * node, 2 * node + 2)
        matrix[rows, rows] = diagonal[node]
        if node > 0:
            cols = slice(2 * (node - 1), 2 * (node - 1) + 2)
            matrix[rows, cols] = lower[node]
        if node < n_nodes - 1:
            cols = slice(2 * (node + 1), 2 * (node + 1) + 2)
            matrix[rows, cols] = upper[node]
    return matrix


def _solve_dense_block_tridiagonal(lower, diagonal, upper, rhs, cause):
    """Solves the exact full block-tridiagonal system after block Thomas fails."""
    matrix = _assemble_dense_block_tridiagonal(lower, diagonal, upper)
    condition_number = float(np.linalg.cond(matrix))
    if not np.isfinite(condition_number):
        raise np.linalg.LinAlgError(
            f"Block Thomas fallback failed because the full block-tridiagonal system is singular; original issue: {cause}"
        ) from cause
    if condition_number > _BLOCK_PIVOT_CONDITION_LIMIT:
        raise np.linalg.LinAlgError(
            "Block Thomas fallback rejected the full block-tridiagonal system as numerically unusable; "
            f"condition number={condition_number:.3e}, limit={_BLOCK_PIVOT_CONDITION_LIMIT:.3e}; "
            f"original issue: {cause}"
        ) from cause

    n_nodes = diagonal.shape[0]
    reshaped_rhs = rhs.reshape(2 * n_nodes, *rhs.shape[2:])
    try:
        solution = np.linalg.solve(matrix, reshaped_rhs)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f"Block Thomas fallback failed while solving the full block-tridiagonal system: {exc}; "
            f"original issue: {cause}"
        ) from exc
    if not np.all(np.isfinite(solution)):
        raise np.linalg.LinAlgError(
            f"Block Thomas fallback produced nonfinite values; original issue: {cause}"
        ) from cause
    return solution.reshape(rhs.shape)


def solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs):
    """
    Solves a block-tridiagonal Illingworth system with dense component blocks.

    Blocks have shape ``(n_nodes, 2, 2)`` and ``rhs`` has shape
    ``(n_nodes, 2)`` or ``(n_nodes, 2, n_rhs)``. A block Thomas sweep preserves
    full cross-diffusion coupling between the two independent ternary
    components while keeping the scalar Thomas solver used by the binary
    implementation untouched. Each modified 2x2 pivot is checked once with a
    scale-invariant reciprocal-condition estimate, then ``upper`` and all RHS
    columns are solved together with :func:`numpy.linalg.solve`. If a block
    pivot is unusable, the exact dense block-tridiagonal system is assembled and
    solved directly without regularizing or otherwise changing the equations.
    """
    lower, diagonal, upper, rhs = _validate_block_tridiagonal_system(lower, diagonal, upper, rhs)

    n_nodes = diagonal.shape[0]
    modified_upper = np.zeros_like(upper)
    modified_rhs = np.zeros_like(rhs)
    try:
        first_rhs = np.concatenate((upper[0], np.atleast_2d(rhs[0]).reshape(2, -1)), axis=1)
        first_solved = _solve_checked_2x2_matrix(diagonal[0], first_rhs, "row 0")
        modified_upper[0] = first_solved[:, :2]
        modified_rhs[0] = first_solved[:, 2:].reshape(rhs[0].shape)
        for node in range(1, n_nodes):
            pivot = diagonal[node] - _apply_2x2_block(lower[node], modified_upper[node - 1])
            row_rhs = rhs[node] - _apply_2x2_block(lower[node], modified_rhs[node - 1])
            solve_rhs = np.concatenate((upper[node], np.atleast_2d(row_rhs).reshape(2, -1)), axis=1)
            solved = _solve_checked_2x2_matrix(pivot, solve_rhs, f"row {node}")
            modified_upper[node] = solved[:, :2]
            modified_rhs[node] = solved[:, 2:].reshape(rhs[node].shape)

        solution = np.zeros_like(rhs)
        solution[-1] = modified_rhs[-1]
        for node in range(n_nodes - 2, -1, -1):
            solution[node] = modified_rhs[node] - _apply_2x2_block(modified_upper[node], solution[node + 1])
        return solution
    except _BlockThomasPivotError as exc:
        return _solve_dense_block_tridiagonal(lower, diagonal, upper, rhs, exc)
