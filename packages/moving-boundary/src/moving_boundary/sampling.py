"""Composition grids for sampling ternary thermodynamic surrogates."""

import numpy as np


def _project_to_simplex(z):
    """Project rows onto the unit simplex in the Euclidean metric."""
    ordered = -np.sort(-z, axis=1)
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    active = np.count_nonzero(ordered - cumulative / np.arange(1, z.shape[1] + 1) > 0, axis=1)
    shift = cumulative[np.arange(len(z)), active - 1] / active
    return np.maximum(z - shift[:, None], 0.0)


def ternary_simplex_points(spacing, floor=1.0e-5):
    """Return an ordered ``(n, 2)`` grid of independent mole fractions.

    The dependent component is ``1 - x1 - x2``. All three components are
    treated alike by projection onto the simplex buffered by ``floor``.
    ``spacing`` must be the reciprocal of a positive integer. Rows ascend
    first in x1, then x2, and are rounded to ten decimals, preserving the
    example grids and their Delaunay tie ordering. Ascending order refers to
    the unprojected grid; projection can shift edge coordinates slightly.
    A very small positive floor
    can round to zero at this precision.
    """
    spacing, floor = float(spacing), float(floor)
    if not np.isfinite(spacing) or not 0.0 < spacing <= 1.0:
        raise ValueError("spacing must be finite and in (0, 1].")
    count = int(round(1.0 / spacing))
    if not np.isclose(count * spacing, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("spacing must be the reciprocal of a positive integer.")
    if not np.isfinite(floor) or not 0.0 <= floor < 1.0 / 3.0:
        raise ValueError("floor must be finite and in [0, 1/3).")
    grid = np.array([[i, j, count - i - j] for i in range(count + 1)
                     for j in range(count + 1 - i)], dtype=np.float64) / count
    scale = 1.0 - 3.0 * floor
    z = np.maximum((grid - floor) / scale, 0.0)
    z = np.where(z.sum(axis=1, keepdims=True) > 1.0, _project_to_simplex(z), z)
    return np.round(floor + scale * z, 10)[:, :2]
