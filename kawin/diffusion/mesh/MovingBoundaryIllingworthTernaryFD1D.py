from dataclasses import dataclass
import math

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised when optional numba is unavailable
    njit = None


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


def _validate_transformed_sequence(profiles, interfaces, domain_length, grids):
    """
    Validates a sequential set of ternary transformed phase profiles.

    The intervals are interpreted as ``[0, s0]``, ``[s0, s1]``, ...,
    ``[s_last, domain_length]``. Each profile stores the two independent
    ternary components on its own Landau grid, and each grid must span
    ``[0, 1]`` so that trapezoidal integration remains length-scaled.
    """
    profiles = tuple(validate_ternary_profile(profile, f"profiles[{i}]") for i, profile in enumerate(profiles))
    grids = tuple(np.asarray(grid, dtype=np.float64).reshape(-1) for grid in grids)
    interfaces = np.asarray(interfaces, dtype=np.float64).reshape(-1)
    domain_length = float(domain_length)
    if len(profiles) != len(grids):
        raise ValueError("profiles and grids must contain the same number of phases.")
    if len(profiles) < 1:
        raise ValueError("At least one transformed phase profile is required.")
    if interfaces.size != len(profiles) - 1:
        raise ValueError("interfaces must contain exactly one fewer entry than profiles.")
    if not np.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError("domain_length must be positive and finite.")
    if not np.all(np.isfinite(interfaces)):
        raise ValueError("interfaces must contain only finite values.")
    if interfaces.size and (interfaces[0] <= 0.0 or interfaces[-1] >= domain_length or not np.all(np.diff(interfaces) > 0.0)):
        raise ValueError("interfaces must be strictly ordered inside the domain.")
    for i, (profile, grid) in enumerate(zip(profiles, grids)):
        if len(grid) != len(profile):
            raise ValueError(f"grids[{i}] length must match profiles[{i}].")
        if len(grid) < 3:
            raise ValueError(f"grids[{i}] must contain at least three nodes.")
        if not np.all(np.isfinite(grid)):
            raise ValueError(f"grids[{i}] must contain only finite values.")
        if not np.isclose(grid[0], 0.0, rtol=0.0, atol=1e-14) or not np.isclose(grid[-1], 1.0, rtol=0.0, atol=1e-14):
            raise ValueError(f"grids[{i}] must start at 0 and end at 1.")
        if not np.all(np.diff(grid) > 0.0):
            raise ValueError(f"grids[{i}] must be strictly increasing.")
    return profiles, interfaces, domain_length, grids


def integrate_planar_transformed_profile_sequence(profiles, interfaces, domain_length: float, grids):
    """
    Integrates sequential ternary phase profiles in planar transformed space.

    The conserved inventory is the sum of ``phase_length * int_0^1 c_i dxi``
    over all intervals. This is the three-or-more phase extension of
    :func:`integrate_planar_transformed_profile_components`.
    """
    profiles, interfaces, domain_length, grids = _validate_transformed_sequence(profiles, interfaces, domain_length, grids)
    boundaries = np.concatenate(([0.0], interfaces, [domain_length]))
    inventory = np.zeros(2, dtype=np.float64)
    for profile, grid, left, right in zip(profiles, grids, boundaries[:-1], boundaries[1:]):
        inventory += (float(right) - float(left)) * np.trapezoid(profile, grid, axis=0)
    return inventory


def integrate_planar_transformed_molar_inventories(profiles, interfaces, domain_length: float, grids, phase_molar_volumes):
    """
    Directly integrates all ternary component moles per unit planar area.

    ``phase_molar_volumes`` must supply one positive physical molar volume per
    phase. The dependent fraction is integrated directly as
    ``x0 = 1 - x1 - x2``; it is not obtained by inventory closure.
    """
    profiles, interfaces, domain_length, grids = _validate_transformed_sequence(profiles, interfaces, domain_length, grids)
    molar_volumes = np.asarray(phase_molar_volumes, dtype=np.float64).reshape(-1)
    if molar_volumes.shape != (len(profiles),):
        raise ValueError("phase_molar_volumes must contain one value per phase.")
    if not np.all(np.isfinite(molar_volumes)) or np.any(molar_volumes <= 0.0):
        raise ValueError("phase_molar_volumes must contain positive finite values.")
    boundaries = np.concatenate(([0.0], interfaces, [domain_length]))
    inventory = np.zeros(3, dtype=np.float64)
    for profile, grid, left, right, molar_volume in zip(profiles, grids, boundaries[:-1], boundaries[1:], molar_volumes):
        dependent = 1.0 - np.sum(profile, axis=1)
        full_profile = np.column_stack((dependent, profile))
        inventory += (float(right) - float(left)) * np.trapezoid(full_profile, grid, axis=0) / float(molar_volume)
    return inventory


def reconstruct_planar_transformed_profile_sequence(z, profiles, interfaces, domain_length: float, grids):
    """
    Maps sequential transformed ternary profiles onto physical mesh nodes.

    Nodes are assigned to the leftmost matching interval so discontinuities at
    moving interfaces are preserved deterministically. The dependent ternary
    component is not reconstructed here.
    """
    z = flatten_1d_coordinates(z)
    profiles, interfaces, domain_length, grids = _validate_transformed_sequence(profiles, interfaces, domain_length, grids)
    boundaries = np.concatenate(([0.0], interfaces, [domain_length]))
    out = np.empty((len(z), 2), dtype=np.float64)
    assigned = np.zeros(len(z), dtype=bool)
    for phase_index, (profile, grid, left, right) in enumerate(zip(profiles, grids, boundaries[:-1], boundaries[1:])):
        if phase_index == len(profiles) - 1:
            mask = (z >= left) & (z <= right)
        else:
            mask = (z >= left) & (z <= right) & ~assigned
        if not np.any(mask):
            continue
        query = np.clip((z[mask] - float(left)) / max(float(right) - float(left), 1e-300), 0.0, 1.0)
        for component in range(2):
            out[mask, component] = np.interp(query, grid, profile[:, component])
        assigned[mask] = True
    if not np.all(assigned):
        raise ValueError("Physical coordinates must lie within the transformed profile domain.")
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


if njit is not None:
    @njit(cache=True, fastmath=False)
    def _solve_illingworth_block_tridiagonal_numba(lower, diagonal, upper, rhs):
        """Sweep one two-component RHS with scaled pivot checks and diagonal-block division."""
        n_nodes = diagonal.shape[0]
        modified_upper = np.zeros_like(upper)
        modified_rhs = np.zeros_like(rhs)
        solution = np.zeros_like(rhs)

        for node in range(n_nodes):
            if node == 0:
                a = diagonal[node, 0, 0]
                b = diagonal[node, 0, 1]
                c = diagonal[node, 1, 0]
                d = diagonal[node, 1, 1]
                rhs0 = rhs[node, 0]
                rhs1 = rhs[node, 1]
            else:
                l00 = lower[node, 0, 0]
                l01 = lower[node, 0, 1]
                l10 = lower[node, 1, 0]
                l11 = lower[node, 1, 1]
                u00 = modified_upper[node - 1, 0, 0]
                u01 = modified_upper[node - 1, 0, 1]
                u10 = modified_upper[node - 1, 1, 0]
                u11 = modified_upper[node - 1, 1, 1]
                a = diagonal[node, 0, 0] - (l00 * u00 + l01 * u10)
                b = diagonal[node, 0, 1] - (l00 * u01 + l01 * u11)
                c = diagonal[node, 1, 0] - (l10 * u00 + l11 * u10)
                d = diagonal[node, 1, 1] - (l10 * u01 + l11 * u11)
                rhs0 = rhs[node, 0] - (l00 * modified_rhs[node - 1, 0] + l01 * modified_rhs[node - 1, 1])
                rhs1 = rhs[node, 1] - (l10 * modified_rhs[node - 1, 0] + l11 * modified_rhs[node - 1, 1])

            if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c) and math.isfinite(d)):
                return solution, node, 1, 0.0
            scale = max(abs(a), abs(b), abs(c), abs(d))
            if scale <= 0.0:
                return solution, node, 2, 0.0
            raw_a = a
            raw_b = b
            raw_c = c
            raw_d = d
            a /= scale
            b /= scale
            c /= scale
            d /= scale
            determinant = a * d - b * c
            matrix_norm = max(abs(a) + abs(b), abs(c) + abs(d))
            inverse_adjugate_norm = max(abs(d) + abs(b), abs(c) + abs(a))
            denominator = matrix_norm * inverse_adjugate_norm
            if denominator <= 0.0 or not math.isfinite(denominator):
                return solution, node, 2, 0.0
            rcond = abs(determinant) / denominator
            if not math.isfinite(rcond) or rcond <= 0.0:
                return solution, node, 2, 0.0
            if rcond < _BLOCK_PIVOT_RCOND_LIMIT:
                return solution, node, 3, rcond

            if raw_b == 0.0 and raw_c == 0.0:
                # Direct division preserves the uncoupled reference solve's rounding.
                modified_upper[node, 0, 0] = upper[node, 0, 0] / raw_a
                modified_upper[node, 0, 1] = upper[node, 0, 1] / raw_a
                modified_upper[node, 1, 0] = upper[node, 1, 0] / raw_d
                modified_upper[node, 1, 1] = upper[node, 1, 1] / raw_d
                modified_rhs[node, 0] = rhs0 / raw_a
                modified_rhs[node, 1] = rhs1 / raw_d
            else:
                upper00 = upper[node, 0, 0] / scale
                upper01 = upper[node, 0, 1] / scale
                upper10 = upper[node, 1, 0] / scale
                upper11 = upper[node, 1, 1] / scale
                row_rhs0 = rhs0 / scale
                row_rhs1 = rhs1 / scale
                modified_upper[node, 0, 0] = (d * upper00 - b * upper10) / determinant
                modified_upper[node, 0, 1] = (d * upper01 - b * upper11) / determinant
                modified_upper[node, 1, 0] = (a * upper10 - c * upper00) / determinant
                modified_upper[node, 1, 1] = (a * upper11 - c * upper01) / determinant
                modified_rhs[node, 0] = (d * row_rhs0 - b * row_rhs1) / determinant
                modified_rhs[node, 1] = (a * row_rhs1 - c * row_rhs0) / determinant
            if not (math.isfinite(modified_upper[node, 0, 0])
                    and math.isfinite(modified_upper[node, 0, 1])
                    and math.isfinite(modified_upper[node, 1, 0])
                    and math.isfinite(modified_upper[node, 1, 1])
                    and math.isfinite(modified_rhs[node, 0])
                    and math.isfinite(modified_rhs[node, 1])):
                return solution, node, 4, rcond

        solution[-1] = modified_rhs[-1]
        for node in range(n_nodes - 2, -1, -1):
            solution[node, 0] = modified_rhs[node, 0] - (
                modified_upper[node, 0, 0] * solution[node + 1, 0]
                + modified_upper[node, 0, 1] * solution[node + 1, 1]
            )
            solution[node, 1] = modified_rhs[node, 1] - (
                modified_upper[node, 1, 0] * solution[node + 1, 0]
                + modified_upper[node, 1, 1] * solution[node + 1, 1]
            )
            if not (math.isfinite(solution[node, 0]) and math.isfinite(solution[node, 1])):
                return solution, node, 4, 0.0
        return solution, -1, 0, 0.0
else:
    _solve_illingworth_block_tridiagonal_numba = None


def _solve_illingworth_block_tridiagonal_two_phase(lower, diagonal, upper, rhs):
    """
    Solve one two-component RHS with optional compiled block elimination.

    Input validation and the dense full-system fallback match the public Python
    solver. Numba is used only for a single RHS; other shapes and installations
    without Numba retain the Python implementation.
    """
    if _solve_illingworth_block_tridiagonal_numba is None:
        return solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs)
    lower, diagonal, upper, rhs = _validate_block_tridiagonal_system(lower, diagonal, upper, rhs)
    if rhs.ndim != 2:
        return solve_illingworth_block_tridiagonal(lower, diagonal, upper, rhs)
    solution, row, status, rcond = _solve_illingworth_block_tridiagonal_numba(lower, diagonal, upper, rhs)
    if status == 0:
        return solution
    if status == 1:
        cause = _BlockThomasPivotError(f"Block-tridiagonal solve encountered a nonfinite 2x2 pivot at row {row}.")
    elif status == 2:
        cause = _BlockThomasPivotError(f"Block-tridiagonal solve encountered a singular 2x2 pivot at row {row}.")
    elif status == 3:
        cause = _BlockThomasPivotError(
            "Block-tridiagonal solve encountered an ill-conditioned 2x2 pivot "
            f"at row {row}; estimated reciprocal condition={rcond:.3e}, "
            f"minimum={_BLOCK_PIVOT_RCOND_LIMIT:.3e}."
        )
    else:
        cause = _BlockThomasPivotError(f"Block-tridiagonal solve produced a nonfinite 2x2 result at row {row}.")
    return _solve_dense_block_tridiagonal(lower, diagonal, upper, rhs, cause)


if njit is not None:
    @njit(cache=True, fastmath=False)
    def _fill_ternary_left_planar_interior(lower, diagonal, upper, rhs, p, u, scaled_faces,
                                           phase_uniform, positive_motion, s, future_s):
        """Fill left interior rows with the original front-fixed upwind coefficients.

        ``scaled_faces`` contains dt/future_s-scaled 2x2 face matrices. Boundary
        rows are left untouched; ``phase_uniform`` retains its original diagonal
        arithmetic even though the matrix is broadcast over faces.
        """
        displacement = future_s - s
        for i in range(1, p.shape[0] - 1):
            left_diff = u[i] - u[i - 1]
            right_diff = u[i + 1] - u[i]
            left_sum = u[i] + u[i - 1]
            right_sum = u[i + 1] + u[i]
            cell_width = right_sum - left_sum
            for component in range(2):
                rhs[i, component] = -s * p[i, component] * cell_width / 2.0
                for coupled in range(2):
                    left = scaled_faces[i - 1, component, coupled]
                    right = scaled_faces[i, component, coupled]
                    if phase_uniform:
                        center = -left * (1.0 / left_diff + 1.0 / right_diff)
                    else:
                        center = -left / left_diff - right / right_diff
                    if positive_motion:
                        lower[i, component, coupled] = left / left_diff
                        upper[i, component, coupled] = right / right_diff
                        if component == coupled:
                            center += -(displacement * left_sum / 2.0 + future_s * cell_width / 2.0)
                            upper[i, component, coupled] += displacement * right_sum / 2.0
                    else:
                        lower[i, component, coupled] = left / left_diff
                        upper[i, component, coupled] = right / right_diff
                        if component == coupled:
                            lower[i, component, coupled] -= displacement * left_sum / 2.0
                            center += displacement * right_sum / 2.0 - future_s * cell_width / 2.0
                    diagonal[i, component, coupled] = center


    @njit(cache=True, fastmath=False)
    def _fill_ternary_right_planar_interior(lower, diagonal, upper, rhs, q, v, scaled_faces,
                                            phase_uniform, positive_motion, old_span, span, displacement):
        """Fill right interior rows with the original front-fixed upwind coefficients.

        ``scaled_faces`` contains dt/span-scaled 2x2 face matrices. Boundary
        rows are left untouched; ``phase_uniform`` retains its original diagonal
        arithmetic even though the matrix is broadcast over faces.
        """
        for i in range(1, q.shape[0] - 1):
            left_diff = v[i] - v[i - 1]
            right_diff = v[i + 1] - v[i]
            left_sum = v[i] + v[i - 1]
            right_sum = v[i + 1] + v[i]
            cell_width = right_sum - left_sum
            for component in range(2):
                rhs[i, component] = -old_span * q[i, component] * cell_width / 2.0
                for coupled in range(2):
                    left = scaled_faces[i - 1, component, coupled]
                    right = scaled_faces[i, component, coupled]
                    if phase_uniform:
                        center = -left * (1.0 / right_diff + 1.0 / left_diff)
                    else:
                        center = -right / right_diff - left / left_diff
                    if positive_motion:
                        lower[i, component, coupled] = left / left_diff
                        upper[i, component, coupled] = right / right_diff
                        if component == coupled:
                            center += -(displacement * (1.0 - left_sum / 2.0) + span * cell_width / 2.0)
                            upper[i, component, coupled] += displacement * (1.0 - right_sum / 2.0)
                    else:
                        lower[i, component, coupled] = left / left_diff
                        upper[i, component, coupled] = right / right_diff
                        if component == coupled:
                            lower[i, component, coupled] -= displacement * (1.0 - left_sum / 2.0)
                            center += displacement * (1.0 - right_sum / 2.0) - span * cell_width / 2.0
                    diagonal[i, component, coupled] = center
else:
    _fill_ternary_left_planar_interior = None
    _fill_ternary_right_planar_interior = None
