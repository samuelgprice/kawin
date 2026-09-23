"""Cached-data benchmarks for ternary interdiffusivity surrogates.

This module deliberately does not build or alter any interpolation model.  It
provides a common data set, split, metric, and reporting layer so existing and
future 2x2 interdiffusivity surrogates can be compared on identical samples.
Thermo-Calc is not needed after a :class:`DiffusivityDataset` has been saved.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Protocol
from types import MappingProxyType

import numpy as np


def _readonly(values, dtype=None):
    """Return a copied, read-only array so benchmark inputs cannot be mutated."""
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(value.real), "imaginary": float(value.imag)}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


@dataclass(frozen=True)
class SpectralCriteria:
    """Numerical thresholds used to classify real nonsymmetric 2x2 matrices.

    Eigenvalue tests use the matrix infinity norm as their scale, matching the
    existing moving-boundary surrogate validation convention.  ``positive``
    denotes eigenvalues larger than ``positive_margin``; values between zero
    and that margin are reported as ``near_boundary`` and are excluded from
    fitting. ``near_zero_margin`` is at least ``positive_margin`` and only
    flags a low-margin warning for otherwise spectrally valid predictions.
    ``robust_positive_margin`` defaults to ``near_zero_margin`` and defines
    the stricter safety margin required for robust-positive GT points.
    """

    eigen_imag_tol: float = 1e-12
    positive_margin: float = 1e-14
    near_zero_margin: float = 1e-12
    robust_positive_margin: float | None = None
    condition_limit: float = 1e12
    denominator_floor: float = 1e-300

    def __post_init__(self):
        for name in ("eigen_imag_tol", "positive_margin", "near_zero_margin", "condition_limit", "denominator_floor"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.near_zero_margin < self.positive_margin:
            raise ValueError("near_zero_margin must be greater than or equal to positive_margin.")
        margin = self.near_zero_margin if self.robust_positive_margin is None else float(self.robust_positive_margin)
        if not np.isfinite(margin) or margin < self.positive_margin:
            raise ValueError("robust_positive_margin must be finite and at least positive_margin.")
        object.__setattr__(self, "robust_positive_margin", margin)


@dataclass(frozen=True)
class SpectralRecord:
    """Spectral diagnostics and admissibility classification for one matrix."""

    eigenvalues: np.ndarray
    trace: float
    determinant: float
    discriminant: float
    minimum_real_eigenvalue: float
    condition_number: float
    scale: float
    normalized_minimum_eigenvalue: float
    normalized_discriminant: float
    classification: str
    finite: bool
    real_eigenvalues: bool


def classify_matrix(matrix, criteria: SpectralCriteria = SpectralCriteria()) -> SpectralRecord:
    """Classify a 2x2 matrix without assuming symmetry or positive definiteness."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (2, 2):
        raise ValueError("interdiffusivity matrices must have shape (2, 2).")
    finite = bool(np.all(np.isfinite(values)))
    if not finite:
        return SpectralRecord(_readonly([np.nan + 0j, np.nan + 0j], np.complex128), np.nan, np.nan, np.nan,
                              np.nan, np.nan, np.nan, np.nan, np.nan, "invalid", False, False)
    scale = float(np.linalg.norm(values, ord=np.inf))
    trace = float(np.trace(values))
    determinant = float(np.linalg.det(values))
    discriminant = float(trace * trace - 4.0 * determinant)
    if scale <= 0.0 or not np.isfinite(scale):
        return SpectralRecord(_readonly([0j, 0j], np.complex128), trace, determinant, discriminant,
                              0.0, np.inf, scale, 0.0, np.nan, "invalid", True, True)
    eigenvalues = np.linalg.eigvals(values / scale) * scale
    real = bool(np.all(np.abs(eigenvalues.imag) <= criteria.eigen_imag_tol * scale))
    minimum = float(np.min(eigenvalues.real))
    normalized_minimum = minimum / scale
    normalized_discriminant = discriminant / max(trace * trace, scale * scale)
    try:
        condition = float(np.linalg.cond(values))
    except np.linalg.LinAlgError:
        condition = np.inf
    if not real or minimum <= 0.0:
        classification = "invalid"
    elif normalized_minimum <= criteria.positive_margin:
        classification = "near_boundary"
    else:
        classification = "positive"
    return SpectralRecord(_readonly(eigenvalues, np.complex128), trace, determinant, discriminant, minimum,
                          condition, scale, normalized_minimum, normalized_discriminant, classification, True, real)


@dataclass(frozen=True)
class DiffusivityDataset:
    """Cached ternary compositions and their ground-truth matrices.

    ``compositions`` contains the two independent ternary components.  Invalid
    ground-truth matrices are retained; fitting eligibility is selected by
    :func:`eligible_training_mask`, not by deleting data from this object. Its
    arrays and top-level metadata mapping are immutable; nested metadata values
    are intentionally not deep-frozen.
    """

    compositions: np.ndarray
    matrices: np.ndarray
    phases: Sequence[str] | str = ""
    contexts: Sequence[str] | str = "general"
    temperatures: Sequence[float] | float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        compositions = np.asarray(self.compositions, dtype=np.float64)
        matrices = np.asarray(self.matrices, dtype=np.float64)
        if compositions.ndim != 2 or compositions.shape[1] != 2 or not np.all(np.isfinite(compositions)):
            raise ValueError("compositions must be a finite array with shape (n_samples, 2).")
        if matrices.shape != (len(compositions), 2, 2):
            raise ValueError("matrices must have shape (n_samples, 2, 2).")
        count = len(compositions)

        def labels(values, name):
            if isinstance(values, str):
                values = [values] * count
            values = np.asarray(values, dtype=str)
            if values.shape != (count,):
                raise ValueError(f"{name} must be one label or have one entry per sample.")
            return values

        phases = labels(self.phases, "phases")
        contexts = labels(self.contexts, "contexts")
        if self.temperatures is None:
            temperatures = np.full(count, np.nan, dtype=np.float64)
        elif np.isscalar(self.temperatures):
            temperatures = np.full(count, float(self.temperatures), dtype=np.float64)
        else:
            temperatures = np.asarray(self.temperatures, dtype=np.float64)
            if temperatures.shape != (count,):
                raise ValueError("temperatures must be one value or have one entry per sample.")
        if not np.all(np.isfinite(temperatures) | np.isnan(temperatures)):
            raise ValueError("temperatures must be finite or NaN when not supplied.")
        object.__setattr__(self, "compositions", _readonly(compositions, np.float64))
        object.__setattr__(self, "matrices", _readonly(matrices, np.float64))
        object.__setattr__(self, "phases", tuple(phases.tolist()))
        object.__setattr__(self, "contexts", tuple(contexts.tolist()))
        object.__setattr__(self, "temperatures", _readonly(temperatures, np.float64))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def __len__(self):
        return self.compositions.shape[0]

    def subset(self, indices: Sequence[int]) -> "DiffusivityDataset":
        """Return an immutable row subset while retaining dataset metadata."""
        indices = np.asarray(indices, dtype=np.int64)
        return DiffusivityDataset(self.compositions[indices], self.matrices[indices],
                                  np.asarray(self.phases)[indices], np.asarray(self.contexts)[indices],
                                  self.temperatures[indices], self.metadata)

    def spectral_records(self, criteria: SpectralCriteria = SpectralCriteria()) -> tuple[SpectralRecord, ...]:
        """Compute one spectral record per cached ground-truth matrix."""
        return tuple(classify_matrix(matrix, criteria) for matrix in self.matrices)

    def save(self, path) -> Path:
        """Save arrays and JSON-compatible metadata to a compressed NPZ archive."""
        path = Path(path).with_suffix(".npz")
        np.savez_compressed(path, compositions=self.compositions, matrices=self.matrices,
                            phases=np.asarray(self.phases), contexts=np.asarray(self.contexts),
                            temperatures=self.temperatures,
                            metadata_json=np.asarray(json.dumps(dict(self.metadata), default=_json_default, sort_keys=True)))
        return path

    @classmethod
    def load(cls, path) -> "DiffusivityDataset":
        """Load a dataset saved by :meth:`save` without any thermodynamics queries."""
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls(archive["compositions"], archive["matrices"], archive["phases"], archive["contexts"],
                       archive["temperatures"], json.loads(str(archive["metadata_json"].tolist())))


def eligible_training_mask(dataset: DiffusivityDataset, criteria: SpectralCriteria = SpectralCriteria()) -> np.ndarray:
    """Return only GT-positive samples eligible for fitting.

    Positive means real eigenvalues within tolerance and normalized minimum
    eigenvalue strictly greater than ``positive_margin``. Near-boundary and
    invalid samples remain in the dataset but are not fitting inputs.
    """
    return np.asarray([record.classification == "positive" for record in dataset.spectral_records(criteria)], dtype=bool)


def _state_keys(dataset: DiffusivityDataset) -> tuple[tuple[str, str, float | None], ...]:
    """Return physical-state keys, treating absent temperature metadata as equal."""
    return tuple(
        (str(phase), str(context), None if np.isnan(temperature) else float(temperature))
        for phase, context, temperature in zip(dataset.phases, dataset.contexts, dataset.temperatures)
    )


def _state_indices(dataset: DiffusivityDataset):
    """Group row indices by the phase, query context, and isothermal state."""
    groups = {}
    for index, key in enumerate(_state_keys(dataset)):
        groups.setdefault(key, []).append(index)
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in groups.items()}


def _delaunay_neighbors(points: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Return same-cloud Delaunay adjacency, falling back to no neighbors."""
    count = len(points)
    out = [set() for _ in range(count)]
    if count < 3 or np.linalg.matrix_rank(points - points[0]) < 2:
        return tuple(tuple() for _ in range(count))
    try:
        from scipy.spatial import Delaunay
        simplices = Delaunay(points).simplices
    except Exception:
        return tuple(tuple() for _ in range(count))
    for simplex in simplices:
        for index in simplex:
            out[index].update(int(other) for other in simplex if other != index)
    return tuple(tuple(sorted(values)) for values in out)


def build_neighborhoods(dataset: DiffusivityDataset, *, rule="delaunay", radius=None,
                        adjacency: Sequence[Sequence[int]] | None = None) -> tuple[tuple[int, ...], ...]:
    """Build physical-state-consistent neighborhoods for robust-positive and block splits.

    ``delaunay`` is the default for scattered ternary samples.  ``radius``
    connects samples within a caller-supplied composition-space radius.
    Explicit adjacency uses global dataset row indices and is symmetrized.
    """
    count = len(dataset)
    if adjacency is not None:
        if len(adjacency) != count:
            raise ValueError("explicit adjacency must have one sequence per dataset sample.")
        result = [set() for _ in range(count)]
        for index, neighbors in enumerate(adjacency):
            for neighbor in neighbors:
                neighbor = int(neighbor)
                if neighbor < 0 or neighbor >= count:
                    raise ValueError("explicit adjacency contains an out-of-range row index.")
                if neighbor != index and _state_keys(dataset)[neighbor] == _state_keys(dataset)[index]:
                    result[index].add(neighbor)
                    result[neighbor].add(index)
        return tuple(tuple(sorted(values)) for values in result)
    if rule not in {"delaunay", "radius"}:
        raise ValueError("rule must be 'delaunay' or 'radius'.")
    if rule == "radius" and (radius is None or not np.isfinite(radius) or radius <= 0.0):
        raise ValueError("radius neighborhoods require a positive finite radius.")
    result = [set() for _ in range(count)]
    for indices in _state_indices(dataset).values():
        points = dataset.compositions[indices]
        if rule == "delaunay":
            local = _delaunay_neighbors(points)
            for source, neighbors in enumerate(local):
                result[indices[source]].update(int(indices[target]) for target in neighbors)
        else:
            distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
            for source in range(len(indices)):
                result[indices[source]].update(int(indices[target]) for target in np.flatnonzero((distances[source] <= radius) & (distances[source] > 0.0)))
    return tuple(tuple(sorted(values)) for values in result)


def robust_positive_mask(dataset: DiffusivityDataset, criteria: SpectralCriteria = SpectralCriteria(), *,
                         neighborhoods: Sequence[Sequence[int]] | None = None, neighborhood_rule="delaunay",
                         radius=None, minimum_neighbors=3) -> np.ndarray:
    """Identify positive points with enough positive local evidence.

    A point is robust only when its normalized minimum eigenvalue, and those
    of at least ``minimum_neighbors`` local same-state neighbors, are strictly
    above ``criteria.robust_positive_margin``. This is intentionally stricter
    than fitting eligibility. Empty, degenerate, or failed Delaunay
    neighborhoods therefore never establish robustness.
    """
    minimum_neighbors = int(minimum_neighbors)
    if minimum_neighbors < 1:
        raise ValueError("minimum_neighbors must be at least one.")
    records = dataset.spectral_records(criteria)
    robust_spectral = np.asarray([
        record.classification == "positive" and record.normalized_minimum_eigenvalue > criteria.robust_positive_margin
        for record in records
    ], dtype=bool)
    neighborhoods = build_neighborhoods(dataset, rule=neighborhood_rule, radius=radius, adjacency=neighborhoods)
    return np.asarray([robust_spectral[index] and len(neighbors) >= minimum_neighbors and all(robust_spectral[neighbor] for neighbor in neighbors)
                       for index, neighbors in enumerate(neighborhoods)], dtype=bool)


@dataclass(frozen=True)
class ValidationSplit:
    """Indices for one phase/context/temperature-consistent comparison."""

    name: str
    phase: str
    training_indices: np.ndarray
    validation_indices: np.ndarray
    context: str = "general"
    temperature: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "training_indices", _readonly(self.training_indices, np.int64))
        object.__setattr__(self, "validation_indices", _readonly(self.validation_indices, np.int64))


@dataclass(frozen=True)
class BenchmarkPlan:
    """Reusable paired split plan with optional validation robustness evidence."""

    training_dataset: DiffusivityDataset
    validation_dataset: DiffusivityDataset
    splits: tuple[ValidationSplit, ...]
    validation_robust_positive_mask: np.ndarray | None = None

    def __post_init__(self):
        if self.validation_robust_positive_mask is not None:
            values = np.asarray(self.validation_robust_positive_mask, dtype=bool)
            if values.shape != (len(self.validation_dataset),):
                raise ValueError("validation_robust_positive_mask must have one value per validation sample.")
            object.__setattr__(self, "validation_robust_positive_mask", _readonly(values, bool))


def make_loo_plan(dataset: DiffusivityDataset, criteria: SpectralCriteria = SpectralCriteria(), *,
                  neighborhoods=None, neighborhood_rule="delaunay", radius=None, robust_minimum_neighbors=3) -> BenchmarkPlan:
    """Make deterministic leave-one-out splits for every GT-positive sample."""
    eligible = eligible_training_mask(dataset, criteria)
    states = _state_keys(dataset)
    splits = []
    for held_out in np.flatnonzero(eligible):
        phase, context, temperature = states[held_out]
        training = np.asarray([index for index, state in enumerate(states)
                               if eligible[index] and state == states[held_out] and index != held_out], dtype=np.int64)
        splits.append(ValidationSplit(f"loo:{held_out}", phase, training, [held_out], context, temperature))
    return BenchmarkPlan(dataset, dataset, tuple(splits), robust_positive_mask(
        dataset, criteria, neighborhoods=neighborhoods, neighborhood_rule=neighborhood_rule,
        radius=radius, minimum_neighbors=robust_minimum_neighbors))


def make_blocked_plan(dataset: DiffusivityDataset, criteria: SpectralCriteria = SpectralCriteria(), *,
                      neighborhoods=None, neighborhood_rule="delaunay", radius=None, robust_minimum_neighbors=3) -> BenchmarkPlan:
    """Remove each positive seed and its local neighborhood from phase training data."""
    eligible = eligible_training_mask(dataset, criteria)
    states = _state_keys(dataset)
    neighborhoods = build_neighborhoods(dataset, rule=neighborhood_rule, radius=radius, adjacency=neighborhoods)
    splits = []
    for seed in np.flatnonzero(eligible):
        held_out = np.asarray((seed, *neighborhoods[seed]), dtype=np.int64)
        held_out = held_out[np.asarray([states[index] == states[seed] for index in held_out])]
        training = np.asarray([index for index, state in enumerate(states)
                               if eligible[index] and state == states[seed] and index not in held_out], dtype=np.int64)
        phase, context, temperature = states[seed]
        splits.append(ValidationSplit(f"block:{seed}", phase, training, np.sort(held_out), context, temperature))
    return BenchmarkPlan(dataset, dataset, tuple(splits), robust_positive_mask(dataset, criteria, neighborhoods=neighborhoods,
                                                                                 neighborhood_rule=neighborhood_rule, radius=radius,
                                                                                 minimum_neighbors=robust_minimum_neighbors))


def make_independent_holdout_plan(training_dataset: DiffusivityDataset, validation_dataset: DiffusivityDataset,
                                  criteria: SpectralCriteria = SpectralCriteria()) -> BenchmarkPlan:
    """Use every GT-positive training point and independent off-grid validation rows."""
    eligible = eligible_training_mask(training_dataset, criteria)
    train_states = _state_keys(training_dataset)
    validation_states = _state_keys(validation_dataset)
    splits = []
    unsupported = []
    for state in dict.fromkeys(validation_states):
        training = np.asarray([index for index, candidate in enumerate(train_states) if eligible[index] and candidate == state], dtype=np.int64)
        validation = np.asarray([index for index, candidate in enumerate(validation_states) if candidate == state], dtype=np.int64)
        if not len(training):
            unsupported.append(state)
            continue
        phase, context, temperature = state
        splits.append(ValidationSplit(f"independent:{phase}:{context}:{temperature}", phase, training, validation, context, temperature))
    if unsupported:
        labels = ", ".join(f"phase={phase!r}, context={context!r}, temperature={temperature!r}" for phase, context, temperature in unsupported)
        raise ValueError(f"Independent validation has no eligible training samples for: {labels}.")
    robust = _independent_robust_positive_mask(training_dataset, validation_dataset, criteria)
    return BenchmarkPlan(training_dataset, validation_dataset, tuple(splits), robust)


def _independent_robust_positive_mask(training_dataset, validation_dataset, criteria):
    """Establish off-grid robustness from enclosing eligible training simplices.

    Independent validation points never use accidental adjacency among
    themselves. A point is robust only if its own GT matrix is positive and it
    lies in a Delaunay triangle whose three same-state training vertices are
    positive. Missing, collinear, or outside-hull training clouds yield false.
    """
    training_records = training_dataset.spectral_records(criteria)
    training_robust = np.asarray([
        record.classification == "positive" and record.normalized_minimum_eigenvalue > criteria.robust_positive_margin
        for record in training_records
    ], dtype=bool)
    validation_records = validation_dataset.spectral_records(criteria)
    validation_robust = np.asarray([
        record.classification == "positive" and record.normalized_minimum_eigenvalue > criteria.robust_positive_margin
        for record in validation_records
    ], dtype=bool)
    result = np.zeros(len(validation_dataset), dtype=bool)
    training_states = _state_keys(training_dataset)
    validation_states = _state_keys(validation_dataset)
    try:
        from scipy.spatial import Delaunay
    except ImportError:  # pragma: no cover - SciPy is a dependency.
        return result
    for state in set(validation_states):
        train_indices = np.asarray([index for index, candidate in enumerate(training_states) if candidate == state], dtype=np.int64)
        validation_indices = np.asarray([index for index, candidate in enumerate(validation_states) if candidate == state], dtype=np.int64)
        if len(train_indices) < 3:
            continue
        points = training_dataset.compositions[train_indices]
        if np.linalg.matrix_rank(points - points[0]) < 2:
            continue
        try:
            triangulation = Delaunay(points)
            simplices = triangulation.find_simplex(validation_dataset.compositions[validation_indices])
        except Exception:
            continue
        for local, simplex in enumerate(simplices):
            if simplex >= 0:
                vertices = train_indices[triangulation.simplices[simplex]]
                result[validation_indices[local]] = validation_robust[validation_indices[local]] and bool(np.all(training_robust[vertices]))
    return result


class DiffusivityPredictor(Protocol):
    """Minimal predictor interface used by the benchmark runner."""

    def predict(self, compositions: np.ndarray, *, phase: str, context: str, temperature: float | None) -> np.ndarray: ...


class ThermodynamicsPredictor:
    """Adapter for existing objects exposing ``getInterdiffusivity``."""

    def __init__(self, source, query_context=None):
        self.source = source
        self.query_context = query_context

    def predict(self, compositions, *, phase, context, temperature):
        compositions = np.asarray(compositions, dtype=np.float64)
        predicted = []
        for composition in compositions:
            try:
                value = self.source.getInterdiffusivity(composition, temperature, phase=phase, query_context=context if self.query_context is None else self.query_context)
            except TypeError:
                value = self.source.getInterdiffusivity(composition, temperature, phase=phase)
            predicted.append(value)
        return np.asarray(predicted, dtype=np.float64)


@dataclass(frozen=True)
class SurrogateScheme:
    """Named factory that fits a predictor from one benchmark training subset."""

    name: str
    factory: Callable[[DiffusivityDataset], DiffusivityPredictor]


def pointwise_metrics(prediction, truth, criteria: SpectralCriteria = SpectralCriteria()) -> dict[str, object]:
    """Calculate matrix, operator, and eigenvalue metrics for one prediction.

    Operator error is omitted when the GT matrix is too ill-conditioned.  The
    return mapping preserves ``None`` for undefined metrics instead of making
    poorly conditioned inverses look trustworthy.
    """
    predicted_record = classify_matrix(prediction, criteria)
    truth_record = classify_matrix(truth, criteria)
    result = {"truth_spectral": truth_record, "prediction_spectral": predicted_record,
              "frobenius_relative_error": None, "spectral_relative_error": None,
              "operator_relative_error": None, "operator_error_reliable": False,
              "eigenvalue_signed_error": None, "eigenvalue_absolute_error": None, "eigenvalue_relative_error": None,
              "minimum_eigenvalue_signed_error": None, "minimum_eigenvalue_absolute_error": None,
              "log_eigenvalue_error": None,
              "rms_log_eigenvalue_error": None}
    if not predicted_record.finite or not truth_record.finite:
        return result
    delta = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    frobenius = float(np.linalg.norm(truth, ord="fro"))
    spectral = float(np.linalg.norm(truth, ord=2))
    if frobenius > criteria.denominator_floor:
        result["frobenius_relative_error"] = float(np.linalg.norm(delta, ord="fro") / frobenius)
    if spectral > criteria.denominator_floor:
        result["spectral_relative_error"] = float(np.linalg.norm(delta, ord=2) / spectral)
    if truth_record.condition_number <= criteria.condition_limit:
        result["operator_relative_error"] = float(np.linalg.norm(delta @ np.linalg.inv(truth), ord=2))
        result["operator_error_reliable"] = True
    if truth_record.classification == "positive" and predicted_record.finite and predicted_record.real_eigenvalues:
        truth_eigenvalues = np.sort(truth_record.eigenvalues.real)
        predicted_eigenvalues = np.sort(predicted_record.eigenvalues.real)
        signed = predicted_eigenvalues - truth_eigenvalues
        absolute = np.abs(signed)
        result["eigenvalue_signed_error"] = _readonly(signed, np.float64)
        result["eigenvalue_absolute_error"] = _readonly(absolute, np.float64)
        result["eigenvalue_relative_error"] = _readonly(absolute / np.abs(truth_eigenvalues), np.float64)
        result["minimum_eigenvalue_signed_error"] = float(signed[0])
        result["minimum_eigenvalue_absolute_error"] = float(absolute[0])
        if np.all(predicted_eigenvalues > 0.0):
            log_error = np.log(predicted_eigenvalues / truth_eigenvalues)
            result["log_eigenvalue_error"] = _readonly(log_error, np.float64)
            result["rms_log_eigenvalue_error"] = float(np.sqrt(np.mean(log_error * log_error)))
    return result


def _prediction_failure(record: SpectralRecord, criteria: SpectralCriteria) -> str | None:
    """Return only true spectral-invalidity categories, never low-margin warnings."""
    if not record.finite:
        return "nonfinite"
    eigenvalues = record.eigenvalues
    if np.any(np.abs(eigenvalues.imag) > criteria.eigen_imag_tol * record.scale):
        return "complex"
    if record.minimum_real_eigenvalue <= 0.0:
        return "negative"
    return None


def _low_margin_warning(record: SpectralRecord, criteria: SpectralCriteria) -> bool:
    """Flag valid but low-margin predictions without changing admissibility."""
    return record.classification != "invalid" and record.normalized_minimum_eigenvalue <= criteria.near_zero_margin


def _flatten_record(split, index, dataset, prediction, metrics, robust, criteria):
    truth = dataset.matrices[index]
    predicted_record = metrics["prediction_spectral"]
    truth_record = metrics["truth_spectral"]
    row = {
        "split": split.name, "phase": split.phase, "dataset_index": int(index),
        "composition": dataset.compositions[index].copy(), "truth_matrix": truth.copy(),
        "predicted_matrix": np.asarray(prediction, dtype=np.float64).copy(), "robust_positive": bool(robust[index]),
        "truth_classification": truth_record.classification, "prediction_classification": predicted_record.classification,
        "prediction_failure": _prediction_failure(predicted_record, criteria),
        "prediction_spectrally_invalid": _prediction_failure(predicted_record, criteria) is not None,
        "prediction_unusable": _prediction_failure(predicted_record, criteria) is not None,
        "prediction_low_margin_warning": _low_margin_warning(predicted_record, criteria),
        "context": split.context, "temperature": split.temperature,
        "truth_eigenvalues": truth_record.eigenvalues.copy(), "prediction_eigenvalues": predicted_record.eigenvalues.copy(),
        "truth_trace": truth_record.trace, "truth_determinant": truth_record.determinant,
        "truth_discriminant": truth_record.discriminant, "prediction_trace": predicted_record.trace,
        "prediction_determinant": predicted_record.determinant, "prediction_discriminant": predicted_record.discriminant,
        "frobenius_relative_error": metrics["frobenius_relative_error"],
        "spectral_relative_error": metrics["spectral_relative_error"],
        "operator_relative_error": metrics["operator_relative_error"],
        "operator_error_reliable": metrics["operator_error_reliable"],
        "eigenvalue_signed_error": metrics["eigenvalue_signed_error"],
        "eigenvalue_absolute_error": metrics["eigenvalue_absolute_error"],
        "eigenvalue_relative_error": metrics["eigenvalue_relative_error"],
        "minimum_eigenvalue_signed_error": metrics["minimum_eigenvalue_signed_error"],
        "minimum_eigenvalue_absolute_error": metrics["minimum_eigenvalue_absolute_error"],
        "rms_log_eigenvalue_error": metrics["rms_log_eigenvalue_error"],
        "truth_condition_number": truth_record.condition_number,
        "truth_normalized_minimum_eigenvalue": truth_record.normalized_minimum_eigenvalue,
        "prediction_normalized_minimum_eigenvalue": predicted_record.normalized_minimum_eigenvalue,
        "truth_normalized_discriminant": truth_record.normalized_discriminant,
        "prediction_normalized_discriminant": predicted_record.normalized_discriminant,
    }
    return row


def _metric_summaries(rows):
    """Return distribution summaries for metrics mathematically defined in rows."""
    output = {}
    for name in ("frobenius_relative_error", "spectral_relative_error", "operator_relative_error",
                 "minimum_eigenvalue_absolute_error", "rms_log_eigenvalue_error"):
        finite = np.asarray([row[name] for row in rows if row[name] is not None and np.isfinite(row[name])], dtype=np.float64)
        output[name] = {"count": int(len(finite)), "median": float(np.median(finite)) if len(finite) else np.nan,
                        "p90": float(np.percentile(finite, 90)) if len(finite) else np.nan,
                        "p95": float(np.percentile(finite, 95)) if len(finite) else np.nan,
                        "maximum": float(np.max(finite)) if len(finite) else np.nan}
    return output


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize paired pointwise records without collapsing them into one score."""
    values = list(rows)
    robust = [row for row in values if row["robust_positive"]]
    summary = {"validation_sample_count": len(values), "robust_positive_sample_count": len(robust),
               "robust_positive_spectral_invalid_rate": (sum(row["prediction_spectrally_invalid"] for row in robust) / len(robust)) if robust else np.nan,
               "robust_positive_spectral_invalid_count": sum(row["prediction_spectrally_invalid"] for row in robust),
               "robust_positive_unusable_rate": (sum(row["prediction_unusable"] for row in robust) / len(robust)) if robust else np.nan,
               "robust_positive_unusable_count": sum(row["prediction_unusable"] for row in robust),
               "negative_prediction_count": sum(row["prediction_failure"] == "negative" for row in values),
               "complex_prediction_count": sum(row["prediction_failure"] == "complex" for row in values),
               "nonfinite_prediction_count": sum(row["prediction_failure"] == "nonfinite" for row in values),
               "evaluator_error_count": sum(row["prediction_failure"] == "evaluator_error" for row in values),
               "low_margin_prediction_count": sum(row["prediction_low_margin_warning"] for row in values)}
    for name in ("negative", "complex", "nonfinite", "low_margin"):
        count = summary[f"{name}_prediction_count"]
        summary[f"{name}_prediction_rate"] = count / len(values) if values else np.nan
    summary["robust_positive_invalid_prediction_rate"] = summary["robust_positive_spectral_invalid_rate"]
    summary.update(_metric_summaries(values))
    summary["strata"] = {
        "robust_positive": _metric_summaries(robust),
        "gt_positive": _metric_summaries([row for row in values if row["truth_classification"] == "positive"]),
        "near_boundary": _metric_summaries([row for row in values if row["truth_classification"] == "near_boundary"]),
        "gt_invalid": _metric_summaries([row for row in values if row["truth_classification"] == "invalid"]),
    }
    return summary


def run_paired_benchmark(plan: BenchmarkPlan, schemes: Sequence[SurrogateScheme],
                         criteria: SpectralCriteria = SpectralCriteria(), *, neighborhoods=None,
                         neighborhood_rule="delaunay", radius=None) -> dict[str, dict[str, object]]:
    """Fit every scheme on identical splits and retain every pointwise result.

    A factory is called once per split because LOO and blocked validation change
    training data.  Predictor exceptions are retained as evaluator failures,
    allowing all schemes to finish their paired comparison.
    """
    if not schemes:
        raise ValueError("at least one surrogate scheme is required.")
    robust = (plan.validation_robust_positive_mask if plan.validation_robust_positive_mask is not None
              else robust_positive_mask(plan.validation_dataset, criteria, neighborhoods=neighborhoods,
                                        neighborhood_rule=neighborhood_rule, radius=radius))
    output = {}
    for scheme in schemes:
        rows = []
        failures = []
        for split in plan.splits:
            training = plan.training_dataset.subset(split.training_indices)
            validation = plan.validation_dataset.subset(split.validation_indices)
            try:
                predictor = scheme.factory(training)
                temperatures = validation.temperatures
                if np.all(np.isnan(temperatures)):
                    temperature = None
                elif np.all(np.isfinite(temperatures)) and np.allclose(temperatures, temperatures[0], rtol=0.0, atol=0.0):
                    temperature = float(temperatures[0])
                else:
                    raise ValueError("a Phase 1 predictor split must have one finite temperature or no temperature metadata.")
                try:
                    predicted_values = predictor.predict(validation.compositions, phase=split.phase,
                                                         context=split.context, temperature=temperature)
                except TypeError as context_error:
                    # Phase 1 predictors accepted only phase/temperature. They
                    # remain safe because factories are still invoked per
                    # physical-state split.
                    try:
                        predicted_values = predictor.predict(validation.compositions, phase=split.phase, temperature=temperature)
                    except TypeError:
                        raise context_error
                predictions = np.asarray(predicted_values, dtype=np.float64)
                if predictions.shape != validation.matrices.shape:
                    raise ValueError(f"predictor returned {predictions.shape}, expected {validation.matrices.shape}.")
            except Exception as exc:
                predictions = np.full_like(validation.matrices, np.nan)
                failures.append({"split": split.name, "phase": split.phase, "error": str(exc)})
            for local, dataset_index in enumerate(split.validation_indices):
                metrics = pointwise_metrics(predictions[local], validation.matrices[local], criteria)
                row = _flatten_record(split, int(dataset_index), plan.validation_dataset, predictions[local], metrics, robust, criteria)
                if failures and failures[-1]["split"] == split.name:
                    row["prediction_failure"] = "evaluator_error"
                    row["prediction_spectrally_invalid"] = False
                    row["prediction_unusable"] = True
                rows.append(row)
        summary = summarize_rows(rows)
        sizes = np.asarray([len(split.training_indices) for split in plan.splits], dtype=np.float64)
        summary["gt_sample_count"] = len(plan.training_dataset)
        summary["eligible_training_pool_count"] = int(np.count_nonzero(eligible_training_mask(plan.training_dataset, criteria)))
        summary["per_fit_training_size"] = {"count": int(len(sizes)), "minimum": int(np.min(sizes)) if len(sizes) else 0,
                                             "median": float(np.median(sizes)) if len(sizes) else np.nan,
                                             "mean": float(np.mean(sizes)) if len(sizes) else np.nan,
                                             "maximum": int(np.max(sizes)) if len(sizes) else 0}
        summary["training_sample_count"] = summary["gt_sample_count"]
        output[scheme.name] = {"scheme": scheme.name, "rows": rows, "failures": failures, "summary": summary}
    return output


def result_rows(result: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    """Return DataFrame-friendly rows with the scheme name attached."""
    rows = []
    for scheme, report in result.items():
        for row in report["rows"]:
            rows.append({"scheme": scheme, **dict(row)})
    return rows
