"""Exact solutions and comparison helpers for checking moving-boundary solves.

Three families are provided: planar binary Zener growth, a ternary two-phase
similarity solution, and a ternary three-phase similarity solution. They need only
NumPy and SciPy, so importing this namespace does not load kawin.

Assumptions shared by all of them: planar geometry, constant diffusivities, equal
and constant molar volumes, no bulk flow, and effectively semi-infinite outer
phases. :func:`semi_infinite_validity_time` is a screening estimate of that last
assumption (a wall-tail amplitude, not an error bound); when it matters, repeat
the calculation on a larger domain and compare the results directly.

Compositions are in the solver state basis: mole fractions for substitutional
systems, u-fractions with interstitials. Matrices, far fields and closures passed
to these solutions must all use that basis, and the comparison helpers compare in
it. ``getPhysicalPhaseProfiles`` returns mole fractions, so do not subtract its
values from these exact profiles when interstitials are present.

Only the names in ``__all__`` are supported API; the private modules hold other
helpers that may change.
"""

from ._planar_binary import (
    PlanarExactProfile,
    VALIDATION_PARAMS,
    planar_exact_interface,
    planar_exact_profile,
    solve_planar_zener_j,
)
from ._three_phase import (
    SimilarityInitialProfile,
    TernaryThreePhaseSimilaritySolution,
    compare_three_phase_model_to_similarity,
    solve_three_phase_similarity_roots,
    tie_line_crossing_etas,
)
from ._two_phase import (
    FixedMatrixTernaryDiffusivity,
    TernaryTwoPhaseSimilaritySolution,
    _recorded,
    compare_model_to_similarity,
    semi_infinite_validity_time,
    solve_similarity_roots,
)

__all__ = [
    "FixedMatrixTernaryDiffusivity",
    "PlanarExactProfile",
    "SimilarityInitialProfile",
    "TernaryThreePhaseSimilaritySolution",
    "TernaryTwoPhaseSimilaritySolution",
    "VALIDATION_PARAMS",
    "compare_model_to_similarity",
    "compare_three_phase_model_to_similarity",
    "planar_exact_interface",
    "planar_exact_profile",
    "recorded_history",
    "semi_infinite_validity_time",
    "solve_planar_zener_j",
    "solve_similarity_roots",
    "solve_three_phase_similarity_roots",
    "tie_line_crossing_etas",
]


def recorded_history(history):
    """Return ``(times, values)`` as recorded in a model history such as ``interfaceData``.

    Models have no public way to list recorded times, so this is the single place
    that reads the history's private storage. Use the model's ``get*(time)``
    accessors for values at chosen times.
    """
    return _recorded(history)
