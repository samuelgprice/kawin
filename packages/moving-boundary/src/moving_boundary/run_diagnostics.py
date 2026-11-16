"""Inventory and sample-coverage checks; neither establishes solution accuracy."""

import numpy as np
from scipy.spatial import Delaunay


def inventory_drift(inventory, *, reduction="max"):
    """Return maximum absolute and component-relative inventory changes.

    Rows are recorded times, columns are independent components, in the same
    inventory units and composition basis throughout. A 1D input is a scalar
    inventory history. Row zero is the reference. ``reduction='max'`` includes
    every recorded excursion; ``'final'`` compares only the final row. A zero
    reference contributes zero relative drift if unchanged, infinity otherwise.
    The two returned maxima can come from different times or components.
    """
    values = np.asarray(inventory, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or 0 in values.shape or not np.all(np.isfinite(values)):
        raise ValueError("inventory must be a nonempty finite history with shape (times, components).")
    if reduction not in ("max", "final"):
        raise ValueError("reduction must be 'max' or 'final'.")
    change = np.abs((values if reduction == "max" else values[-1:]) - values[0])
    relative = np.full_like(change, np.inf)
    np.divide(change, np.abs(values[0]), out=relative, where=values[0] != 0.0)
    relative[(change == 0.0) & (values[0] == 0.0)] = 0.0
    return float(np.max(change)), float(np.max(relative))


def evaluate_bulk_coverage(surrogate, compositions_by_phase):
    """Report fitted-sample hull membership for ``(n, 2)`` compositions per phase.

    Inputs use the surrogate's independent-component basis and order. Each
    phase report contains ``inside`` (a boolean mask), ``count``, ``outside``
    and ``retained_samples``. For full-rank sample sets, membership uses SciPy's
    default Delaunay tolerance, matching the examples. Without a 2D hull only
    retained samples within 32 machine epsilons count as supported. Empty query
    batches are allowed; nonfinite queries count as outside. This checks the
    supplied states only, not earlier states or physical validity of matrices.
    """
    reports = {}
    for phase, compositions in compositions_by_phase.items():
        points, _ = surrogate.get_diffusivity_samples(phase)
        values = np.asarray(compositions, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("Each phase's compositions must have shape (n, 2).")
        inside = np.zeros(len(values), dtype=bool)
        finite = np.all(np.isfinite(values), axis=1)
        if len(points) >= 3 and np.linalg.matrix_rank(points - points[0]) == 2:
            inside[finite] = Delaunay(points).find_simplex(values[finite]) >= 0
        elif len(points):
            for index in np.flatnonzero(finite):
                tolerance = 32 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(values[index]))))
                inside[index] = np.any(np.all(np.abs(points - values[index]) <= tolerance, axis=1))
        reports[phase] = {"inside": inside, "count": len(values),
                          "outside": int(np.count_nonzero(~inside)), "retained_samples": len(points)}
    return reports
