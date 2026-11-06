"""Validity-domain support for ternary interdiffusivity surrogates.

This module deliberately distinguishes evidence about the physical surrogate
domain from the numerical subset that can be used by the Illingworth solver.
"""

from enum import Enum

import numpy as np

try:
    from scipy.spatial import Delaunay, cKDTree
except ImportError:  # pragma: no cover - SciPy is a package dependency.
    Delaunay = None
    cKDTree = None

from ._spectral_validation import (
    TERNARY_DIFFUSIVITY_POSITIVE_EIGENVALUE_TOL,
    TERNARY_DIFFUSIVITY_REAL_SPECTRUM_TOL,
)


class DiffusivityDomainStatus(str, Enum):
    """Classification of a composition relative to labeled diffusivity data."""

    VALID = "VALID"
    KNOWN_INVALID = "KNOWN_INVALID"
    UNKNOWN = "UNKNOWN"
    BOUNDARY_AMBIGUOUS = "BOUNDARY_AMBIGUOUS"
    OUTSIDE_SUPPORT = "OUTSIDE_SUPPORT"
    INSUFFICIENT_FIT_SUPPORT = "INSUFFICIENT_FIT_SUPPORT"


class DiffusivityDomainError(ValueError):
    """Raised when strict diffusivity-domain evaluation cannot return a matrix."""

    def __init__(self, status, reason, *, composition=None, phase=None, context=None):
        self.status = DiffusivityDomainStatus(status)
        self.reason = str(reason)
        self.composition = None if composition is None else np.asarray(composition, dtype=np.float64).copy()
        self.phase = phase
        self.context = context
        location = "" if self.composition is None else f" at composition={self.composition.tolist()}"
        where = "" if phase is None else f" for phase '{phase}'"
        regime = "" if context is None else f" ({context})"
        super().__init__(f"Diffusivity domain {self.status.value}{where}{regime}{location}: {self.reason}")


def classify_source_matrix(matrix):
    """Classify matrix evidence without applying the solver usability cutoff.

    Nonfinite results are unknown evidence: they may be failed calculations,
    not physical-domain evidence. Finite matrices with a genuinely complex or
    nonpositive spectrum establish an invalid diffusivity-surrogate state.
    """
    values = np.asarray(matrix)
    if values.shape != (2, 2) or np.iscomplexobj(values) or not np.all(np.isfinite(values)):
        return DiffusivityDomainStatus.UNKNOWN, False, "nonfinite_or_malformed_source_result"
    values = np.asarray(values, dtype=np.float64)
    scale = float(np.linalg.norm(values, ord=np.inf))
    if not np.isfinite(scale) or scale <= 0.0:
        return DiffusivityDomainStatus.KNOWN_INVALID, False, "zero_matrix_norm"
    eigenvalues = np.linalg.eigvals(values / scale)
    if np.any(np.abs(np.imag(eigenvalues)) > TERNARY_DIFFUSIVITY_REAL_SPECTRUM_TOL):
        return DiffusivityDomainStatus.KNOWN_INVALID, False, "complex_spectrum"
    real = np.real(eigenvalues)
    if np.any(real <= 0.0):
        return DiffusivityDomainStatus.KNOWN_INVALID, False, "nonpositive_spectrum"
    return DiffusivityDomainStatus.VALID, bool(np.all(real > TERNARY_DIFFUSIVITY_POSITIVE_EIGENVALUE_TOL)), (
        "solver_usable" if np.all(real > TERNARY_DIFFUSIVITY_POSITIVE_EIGENVALUE_TOL) else "below_solver_usability_threshold"
    )


def _coerce_status(value):
    return value if isinstance(value, DiffusivityDomainStatus) else DiffusivityDomainStatus(str(value))


def _same_coordinate(left, right):
    """Use one ULP-scale equivalence rule for labels, source rows, and queries."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    return bool(np.all(np.abs(left - right) <= tolerance))


def _same_coordinate_mask(coordinates, point):
    """Vectorized form of :func:`_same_coordinate` for one query point."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    scales = np.maximum(1.0, np.maximum(np.max(np.abs(coordinates), axis=1), np.max(np.abs(point))))
    tolerances = 32.0 * np.finfo(np.float64).eps * scales
    return np.all(np.abs(coordinates - point) <= tolerances[:, None], axis=1)


def _exact_coordinate_indices_many(coordinates, tree, queries):
    """Find first source matches using a conservative spatial search.

    Queries must be finite. The Chebyshev radius bounds every possible match;
    the original per-coordinate tolerance decides which candidates match.
    """
    indices = np.full(len(queries), -1, dtype=np.int64)
    if not len(coordinates) or not len(queries):
        return indices
    source_scale = float(np.max(np.abs(coordinates)))
    query_scales = np.max(np.abs(queries), axis=1)
    radii = np.nextafter(
        32.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.maximum(source_scale, query_scales)), np.inf
    )
    for query_index, candidates in enumerate(tree.query_ball_point(queries, radii, p=np.inf)):
        if candidates:
            candidates = np.asarray(candidates, dtype=np.int64)
            matches = candidates[_same_coordinate_mask(coordinates[candidates], queries[query_index])]
            if len(matches):
                indices[query_index] = np.min(matches)
    return indices


class DiffusivityDomain:
    """Labeled geometric domain and independent solver-usable fit support."""

    def __init__(self, coordinates, statuses, *, reasons=None, fit_usable=None):
        points = np.asarray(coordinates, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.all(np.isfinite(points)):
            raise ValueError("diffusivity validity coordinates must be a nonempty finite (n, 2) array.")
        self.coordinates = points.copy()
        self.statuses = np.asarray([_coerce_status(value) for value in statuses], dtype=object)
        if len(self.statuses) != len(points):
            raise ValueError("diffusivity validity statuses must match validity coordinates.")
        self.reasons = np.asarray(["" for _ in points] if reasons is None else reasons, dtype=object)
        if len(self.reasons) != len(points):
            raise ValueError("diffusivity validity reasons must match validity coordinates.")
        self.fit_usable = np.asarray(
            [status is DiffusivityDomainStatus.VALID for status in self.statuses] if fit_usable is None else fit_usable, dtype=bool
        )
        if len(self.fit_usable) != len(points):
            raise ValueError("diffusivity fit-usable flags must match validity coordinates.")
        valid_mask = np.asarray([status is DiffusivityDomainStatus.VALID for status in self.statuses], dtype=bool)
        self.fit_usable &= valid_mask
        self._canonicalize_duplicates()
        self._exact_tree = cKDTree(self.coordinates)
        self._triangulation = self._make_triangulation(self.coordinates)
        self._fit_triangulation = self._make_triangulation(self.coordinates[self.fit_usable])

    def _canonicalize_duplicates(self):
        """Reject conflicting exact labels and collapse compatible duplicates."""
        # Sorting by the final component makes records that are close in both
        # components adjacent, even when another row shares only x exactly.
        order = np.lexsort((self.coordinates[:, 0], self.coordinates[:, 1]))
        coordinates, statuses = self.coordinates[order], self.statuses[order]
        reasons, usable = self.reasons[order], self.fit_usable[order]
        parents = np.arange(len(coordinates), dtype=np.int64)

        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left, right):
            left, right = find(left), find(right)
            if left != right:
                parents[max(left, right)] = min(left, right)

        if cKDTree is None:  # pragma: no cover - SciPy is a package dependency.
            raise ImportError("cKDTree is required for diffusivity validity canonicalization.")
        global_tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(coordinates))))
        # Euclidean radius sqrt(2)*tol is a conservative candidate search;
        # `_same_coordinate` below retains the exact componentwise semantics.
        candidates = cKDTree(coordinates).query_pairs(np.sqrt(2.0) * global_tolerance, output_type="ndarray")
        for left, right in candidates:
            if _same_coordinate(coordinates[left], coordinates[right]):
                union(int(left), int(right))
        groups = {}
        for index in range(len(coordinates)):
            groups.setdefault(find(index), []).append(index)
        keep, canonical_reasons = [], []
        for matches in groups.values():
            canonical = min(matches)
            point = coordinates[canonical]
            if any(not _same_coordinate(coordinates[index], point) for index in matches):
                raise ValueError(
                    "ambiguous chained near-duplicate diffusivity provenance cluster at "
                    f"canonical coordinate {point.tolist()}."
                )
            labels = {statuses[i] for i in matches}
            if len(labels) != 1:
                raise ValueError(f"conflicting diffusivity validity labels at duplicate coordinate {point.tolist()}.")
            if len({bool(usable[i]) for i in matches}) != 1:
                raise ValueError(f"conflicting diffusivity fit-usability records at duplicate coordinate {point.tolist()}.")
            nonempty_reasons = {str(reasons[i]) for i in matches if str(reasons[i])}
            if len(nonempty_reasons) > 1:
                raise ValueError(f"conflicting diffusivity provenance reasons at duplicate coordinate {point.tolist()}.")
            keep.append(canonical)
            canonical_reasons.append(next(iter(nonempty_reasons), str(reasons[canonical])))
        self.coordinates = coordinates[keep]
        self.statuses = statuses[keep]
        self.reasons = np.asarray(canonical_reasons, dtype=object)
        self.fit_usable = usable[keep]

    @staticmethod
    def _make_triangulation(points):
        if Delaunay is None or len(points) < 3 or np.linalg.matrix_rank(points - points[0]) < 2:
            return None
        return Delaunay(points)

    def _exact_index(self, point):
        found = np.flatnonzero(_same_coordinate_mask(self.coordinates, point))
        return None if not len(found) else int(found[0])

    def _classify_scalar_reference(self, point):
        """Reference one-point classifier retained for semantic fallback checks."""
        point = np.asarray(point, dtype=np.float64).reshape(2)
        exact = self._exact_index(point)
        if exact is not None:
            return self.statuses[exact], str(self.reasons[exact] or "exact_labeled_coordinate")
        if self._triangulation is None:
            return DiffusivityDomainStatus.OUTSIDE_SUPPORT, "labeled support has no two-dimensional simplex"
        simplex = int(self._triangulation.find_simplex(point))
        if simplex < 0:
            return DiffusivityDomainStatus.OUTSIDE_SUPPORT, "outside labeled diffusivity support"
        labels = set(self.statuses[self._triangulation.simplices[simplex]])
        if len(labels) == 1:
            only = labels.pop()
            return only, f"{only.value.lower()} labeled simplex"
        return DiffusivityDomainStatus.BOUNDARY_AMBIGUOUS, "mixed labeled simplex"

    def classify(self, point):
        """Return the domain status/reason with exact labels taking precedence."""
        return self._classify_scalar_reference(point)

    def classify_many(self, points, *, chunk_size=4096):
        """
        Classify a batch while preserving :meth:`classify` label precedence.

        A spatial index narrows finite exact-coordinate candidates before the
        original tolerance check; Delaunay lookup handles remaining queries.
        Nonfinite queries use the scalar reference path so unusual SciPy error
        behavior remains identical to the historical one-point implementation.
        """
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("points must have shape (n, 2).")
        statuses = np.empty(values.shape[0], dtype=object)
        reasons = np.empty(values.shape[0], dtype=object)
        chunk_size = max(1, int(chunk_size))

        for start in range(0, values.shape[0], chunk_size):
            stop = min(values.shape[0], start + chunk_size)
            chunk = values[start:stop]
            finite = np.all(np.isfinite(chunk), axis=1)
            if np.any(~finite):
                for index in np.flatnonzero(~finite):
                    statuses[start + index], reasons[start + index] = self._classify_scalar_reference(chunk[index])

            if not np.any(finite):
                continue
            local_indices = np.flatnonzero(finite)
            valid = chunk[local_indices]
            exact_indices = _exact_coordinate_indices_many(self.coordinates, self._exact_tree, valid)
            exact_mask = exact_indices >= 0
            if np.any(exact_mask):
                exact_indices = exact_indices[exact_mask]
                output_indices = start + local_indices[exact_mask]
                statuses[output_indices] = self.statuses[exact_indices]
                reasons[output_indices] = [str(self.reasons[index] or "exact_labeled_coordinate") for index in exact_indices]

            nonexact_local = local_indices[~exact_mask]
            if not len(nonexact_local):
                continue
            output_indices = start + nonexact_local
            if self._triangulation is None:
                statuses[output_indices] = DiffusivityDomainStatus.OUTSIDE_SUPPORT
                reasons[output_indices] = "labeled support has no two-dimensional simplex"
                continue
            simplex_indices = np.asarray(self._triangulation.find_simplex(chunk[nonexact_local]), dtype=np.int64)
            for output_index, simplex in zip(output_indices, simplex_indices):
                if simplex < 0:
                    statuses[output_index] = DiffusivityDomainStatus.OUTSIDE_SUPPORT
                    reasons[output_index] = "outside labeled diffusivity support"
                    continue
                labels = set(self.statuses[self._triangulation.simplices[simplex]])
                if len(labels) == 1:
                    only = labels.pop()
                    statuses[output_index] = only
                    reasons[output_index] = f"{only.value.lower()} labeled simplex"
                else:
                    statuses[output_index] = DiffusivityDomainStatus.BOUNDARY_AMBIGUOUS
                    reasons[output_index] = "mixed labeled simplex"
        return statuses, reasons

    def has_fit_support(self, point, interpolation):
        """Return whether a valid query has usable support for its evaluator."""
        if not np.any(self.fit_usable):
            return False
        exact = self._exact_index(point)
        if exact is not None:
            return bool(self.fit_usable[exact])
        if interpolation == "nearest":
            if self._triangulation is None:
                return False
            simplex = int(self._triangulation.find_simplex(np.asarray(point, dtype=np.float64).reshape(2)))
            return simplex >= 0 and bool(np.any(self.fit_usable[self._triangulation.simplices[simplex]]))
        if self._fit_triangulation is None:
            return False
        return int(self._fit_triangulation.find_simplex(np.asarray(point, dtype=np.float64).reshape(2))) >= 0
