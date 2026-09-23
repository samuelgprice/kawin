"""Optional Phase 2 metrics for ternary diffusivity surrogate benchmarks.

The PDE-facing helpers are deliberately limited to recorded histories from the
two-phase ternary Illingworth solver.  They consume completed runs and do not
alter solver settings or provide three-phase/multi-interface support.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .SurrogateBenchmark import (
    BenchmarkPlan,
    DiffusivityDataset,
    SpectralCriteria,
    SurrogateScheme,
    make_independent_holdout_plan,
    run_paired_benchmark,
)


def _readonly(values, dtype=np.float64):
    """Copy *values* into an immutable NumPy array."""
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _summary(values):
    """Return finite-value percentile statistics suitable for JSON output."""
    finite = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=float,
    )
    if len(finite) == 0:
        return {"count": 0, "median": np.nan, "p90": np.nan, "p95": np.nan, "maximum": np.nan}
    return {
        "count": len(finite),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "maximum": float(np.max(finite)),
    }


def flux_error(prediction, truth, gradients, *, gt_flux_floor=1e-14):
    """Calculate ``J=-Dg`` errors for one or more ternary composition gradients.

    A relative error is intentionally not reported for ground-truth fluxes at
    or below ``gt_flux_floor``.  Such gradients remain in the absolute-error
    statistics and are marked by ``low_signal``.  If the ground truth itself
    is nonfinite, its low-signal status is unavailable (``None``).
    """
    predicted = np.asarray(prediction, dtype=float)
    ground_truth = np.asarray(truth, dtype=float)
    gradient_array = np.asarray(gradients, dtype=float)
    if predicted.shape != (2, 2) or ground_truth.shape != (2, 2):
        raise ValueError("flux_error requires 2x2 matrices.")
    if gradient_array.ndim == 1:
        gradient_array = gradient_array[None, :]
    if gradient_array.ndim != 2 or gradient_array.shape[1] != 2 or not np.all(np.isfinite(gradient_array)):
        raise ValueError("gradients must have shape (n, 2) and be finite.")
    if gt_flux_floor < 0 or not np.isfinite(gt_flux_floor):
        raise ValueError("gt_flux_floor must be finite and nonnegative.")

    if not np.all(np.isfinite(ground_truth)):
        undefined = np.full(len(gradient_array), np.nan)
        return {
            "gt_flux": np.full_like(gradient_array, np.nan),
            "predicted_flux": np.full_like(gradient_array, np.nan),
            "absolute_error": undefined,
            "relative_error": undefined,
            "low_signal": np.full(len(gradient_array), None, dtype=object),
        }

    gt_flux = -(ground_truth @ gradient_array.T).T
    gt_magnitude = np.linalg.norm(gt_flux, axis=1)
    low_signal = gt_magnitude <= gt_flux_floor
    relative_error = np.full(len(gradient_array), np.nan)
    if np.all(np.isfinite(predicted)):
        predicted_flux = -(predicted @ gradient_array.T).T
        absolute_error = np.linalg.norm(predicted_flux - gt_flux, axis=1)
        relative_error[~low_signal] = absolute_error[~low_signal] / gt_magnitude[~low_signal]
    else:
        predicted_flux = np.full_like(gt_flux, np.nan)
        absolute_error = np.full(len(gradient_array), np.nan)
    return {
        "gt_flux": gt_flux,
        "predicted_flux": predicted_flux,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "low_signal": low_signal,
    }


def _flux_summary(rows):
    """Summarize flux rows while retaining low-signal coverage information."""
    absolute_values = []
    relative_values = []
    evaluated_count = 0
    low_signal_count = 0
    for row in rows:
        absolute = row.get("flux_absolute_error")
        relative = row.get("flux_relative_error")
        low_signal = row.get("flux_low_signal")
        if absolute is None or low_signal is None:
            continue
        absolute = np.asarray(absolute, dtype=float).ravel()
        relative = np.asarray(relative, dtype=float).ravel()
        low_signal = np.asarray(low_signal, dtype=object).ravel()
        if len(absolute) != len(relative) or len(absolute) != len(low_signal):
            raise ValueError("flux metric arrays must have matching lengths.")
        evaluated_count += len(absolute)
        low_signal_count += sum(value is True or isinstance(value, np.bool_) and bool(value) for value in low_signal)
        absolute_values.extend(absolute)
        relative_values.extend(relative)
    return {
        "absolute_flux_error": _summary(absolute_values),
        "relative_flux_error": _summary(relative_values),
        "evaluated_gradient_count": evaluated_count,
        "finite_absolute_error_count": int(np.count_nonzero(np.isfinite(absolute_values))),
        "finite_relative_error_count": int(np.count_nonzero(np.isfinite(relative_values))),
        "low_signal_count": low_signal_count,
        "low_signal_fraction": np.nan if evaluated_count == 0 else low_signal_count / evaluated_count,
    }


def attach_flux_metrics(report, gradients, *, gt_flux_floor=1e-14):
    """Attach flux metrics to Phase-1 report rows and aggregate their coverage.

    ``gradients`` may be shared by all rows, a mapping keyed by
    ``(scheme, split, dataset_index)``, or a callable accepting a row.
    """
    results = {}
    for scheme, previous in report.items():
        rows = []
        for previous_row in previous["rows"]:
            if callable(gradients):
                gradient = gradients(previous_row)
            elif isinstance(gradients, Mapping):
                gradient = gradients.get((scheme, previous_row["split"], previous_row["dataset_index"]))
            else:
                gradient = gradients
            row = dict(previous_row)
            if gradient is None:
                row.update(flux_absolute_error=None, flux_relative_error=None, flux_low_signal=None)
            else:
                metric = flux_error(
                    row["predicted_matrix"], row["truth_matrix"], gradient, gt_flux_floor=gt_flux_floor
                )
                row.update(
                    gt_flux=metric["gt_flux"],
                    predicted_flux=metric["predicted_flux"],
                    flux_absolute_error=metric["absolute_error"],
                    flux_relative_error=metric["relative_error"],
                    flux_low_signal=metric["low_signal"],
                )
            rows.append(row)
        strata = {
            "robust_positive": [row for row in rows if row["robust_positive"]],
            "gt_positive": [row for row in rows if row["truth_classification"] == "positive"],
            "near_boundary": [row for row in rows if row["truth_classification"] == "near_boundary"],
            "gt_invalid": [row for row in rows if row["truth_classification"] == "invalid"],
        }
        results[scheme] = {
            **previous,
            "rows": rows,
            "flux_summary": {
                "overall": _flux_summary(rows),
                "strata": {name: _flux_summary(items) for name, items in strata.items()},
            },
        }
    return results


def _dataset_state(dataset):
    """Return the unique phase/context/temperature state of a dataset."""
    states = {
        (dataset.phases[index], dataset.contexts[index], None if np.isnan(dataset.temperatures[index]) else float(dataset.temperatures[index]))
        for index in range(len(dataset))
    }
    if len(states) != 1:
        raise ValueError("refinement studies require one phase/context/temperature state per dataset.")
    return next(iter(states))


def _characteristic_spacing(dataset):
    """Return median nearest-neighbor composition spacing, or NaN for one point."""
    if len(dataset) < 2:
        return np.nan
    distance = np.linalg.norm(dataset.compositions[:, None] - dataset.compositions[None, :], axis=2)
    distance[np.diag_indices_from(distance)] = np.inf
    return float(np.median(np.min(distance, axis=1)))


def run_refinement_study(
    training_dataset: DiffusivityDataset,
    validation_dataset: DiffusivityDataset,
    schemes: Sequence[SurrogateScheme],
    training_subsets: Mapping[str, Sequence[int]],
    criteria: SpectralCriteria = SpectralCriteria(),
    *,
    validation_robust_positive_mask=None,
):
    """Compare supplied training densities against one fixed validation population."""
    state = _dataset_state(training_dataset)
    if _dataset_state(validation_dataset) != state:
        raise ValueError("training and validation refinement states must match.")
    full_plan = make_independent_holdout_plan(training_dataset, validation_dataset, criteria)
    robust_mask = (
        full_plan.validation_robust_positive_mask
        if validation_robust_positive_mask is None
        else np.asarray(validation_robust_positive_mask, dtype=bool)
    )
    if robust_mask.shape != (len(validation_dataset),):
        raise ValueError("validation_robust_positive_mask has wrong length.")
    levels = []
    for label, indices in training_subsets.items():
        subset = training_dataset.subset(indices)
        plan = make_independent_holdout_plan(subset, validation_dataset, criteria)
        plan = BenchmarkPlan(plan.training_dataset, plan.validation_dataset, plan.splits, robust_mask)
        levels.append(
            {
                "label": str(label),
                "training_sample_count": len(subset),
                "characteristic_spacing": _characteristic_spacing(subset),
                "results": run_paired_benchmark(plan, schemes, criteria),
            }
        )
    return {
        "kind": "refinement_comparison",
        "reference": "independent_ground_truth",
        "physical_state": state,
        "validation_sample_count": len(validation_dataset),
        "fixed_robust_positive_validation_mask": _readonly(robust_mask, bool),
        "levels": levels,
    }


@dataclass(frozen=True)
class IllingworthRun:
    """Recorded physical history from one two-phase ternary Illingworth run.

    Composition profiles use shape ``(time, node, 2)`` and coordinates must be
    physical, finite, and strictly increasing at every accepted time.
    """

    times: np.ndarray
    interface_positions: np.ndarray
    compositions: np.ndarray | None = None
    spatial_coordinates: np.ndarray | None = None
    label: str = ""

    def __post_init__(self):
        times = np.asarray(self.times, dtype=float)
        positions = np.asarray(self.interface_positions, dtype=float)
        if (
            times.ndim != 1
            or positions.shape != times.shape
            or len(times) < 2
            or not np.all(np.isfinite(times))
            or not np.all(np.isfinite(positions))
            or not np.all(np.diff(times) > 0)
        ):
            raise ValueError("times/positions must be finite matching histories with increasing times.")

        compositions = None if self.compositions is None else np.asarray(self.compositions, dtype=float)
        coordinates = None if self.spatial_coordinates is None else np.asarray(self.spatial_coordinates, dtype=float)
        if coordinates is not None and not np.all(np.isfinite(coordinates)):
            raise ValueError("physical coordinates must be finite.")
        if compositions is not None:
            if (
                compositions.ndim != 3
                or compositions.shape[0] != len(times)
                or compositions.shape[2] != 2
                or coordinates is None
                or not np.all(np.isfinite(compositions))
            ):
                raise ValueError("compositions must be finite with shape (time, node, 2) and physical coordinates.")
            if coordinates.ndim == 1:
                coordinates = np.broadcast_to(coordinates, compositions.shape[:2])
            if coordinates.shape != compositions.shape[:2] or np.any(np.diff(coordinates, axis=1) <= 0):
                raise ValueError("physical coordinates must be strictly increasing per profile.")

        object.__setattr__(self, "times", _readonly(times))
        object.__setattr__(self, "interface_positions", _readonly(positions))
        object.__setattr__(self, "compositions", None if compositions is None else _readonly(compositions))
        object.__setattr__(self, "spatial_coordinates", None if coordinates is None else _readonly(coordinates))


def capture_illingworth_run(model, *, label=""):
    """Capture recorded physical profiles from a two-phase ternary Illingworth model."""
    history = getattr(model, "interfaceData", None)
    if history is None or not hasattr(history, "N") or getattr(model, "_z", None) is None:
        raise ValueError("expected a recorded two-phase ternary Illingworth model.")
    count = int(history.N) + 1
    return IllingworthRun(
        history._time[:count],
        history._y[:count],
        np.asarray(model.data._y[:count], dtype=float),
        np.asarray(model._z, dtype=float),
        label,
    )


def _profile_at_time(run: IllingworthRun, time: float, coordinates: np.ndarray, spatial_atol: float):
    """Interpolate a recorded physical profile without temporal or spatial extrapolation."""
    upper = int(np.searchsorted(run.times, time))
    lower = upper if upper < len(run.times) and np.isclose(run.times[upper], time, atol=1e-14, rtol=0) else upper - 1
    if lower < 0 or upper >= len(run.times):
        raise ValueError("profile comparison would extrapolate in time.")
    fraction = 0.0 if lower == upper else (time - run.times[lower]) / (run.times[upper] - run.times[lower])

    def profile(index):
        source_x = run.spatial_coordinates[index]
        if not (
            np.isclose(coordinates[0], source_x[0], atol=spatial_atol, rtol=0)
            and np.isclose(coordinates[-1], source_x[-1], atol=spatial_atol, rtol=0)
        ):
            raise ValueError("incompatible physical spatial domains; spatial extrapolation is not permitted.")
        return np.column_stack(
            [np.interp(coordinates, source_x, run.compositions[index, :, component]) for component in range(2)]
        )

    lower_profile = profile(lower)
    upper_profile = profile(upper)
    return lower_profile + fraction * (upper_profile - lower_profile)


def compare_illingworth_runs(
    reference: IllingworthRun,
    candidate: IllingworthRun,
    *,
    target_time=None,
    target_time_atol=1e-12,
    spatial_atol=1e-12,
):
    """Compare two completed two-phase histories on their common physical domain.

    No time or spatial extrapolation is performed.  Profile metrics are marked
    unavailable if their physical endpoint domains disagree.  ``completed`` is
    true only when both runs reach the requested target time.
    """
    if target_time_atol < 0 or spatial_atol < 0:
        raise ValueError("target_time_atol and spatial_atol must be nonnegative.")
    target = float(reference.times[-1] if target_time is None else target_time)
    reference_reached = reference.times[-1] >= target - target_time_atol
    candidate_reached = candidate.times[-1] >= target - target_time_atol
    common_start = max(reference.times[0], candidate.times[0])
    common_end = min(reference.times[-1], candidate.times[-1], target)
    accepted = (candidate.times >= common_start) & (candidate.times <= common_end)
    times = candidate.times[accepted]
    base = {
        "runner_succeeded": True,
        "candidate_final_time": float(candidate.times[-1]),
        "reference_final_time": float(reference.times[-1]),
        "target_time": target,
        "reference_reached_target_time": bool(reference_reached),
        "candidate_reached_target_time": bool(candidate_reached),
        "reached_target_time": bool(reference_reached and candidate_reached),
        "common_time_interval": (float(common_start), float(common_end)),
    }
    if len(times) < 2:
        return {
            **base,
            "completed": False,
            "partial": True,
            "metric_reason": "no meaningful common time interval before the requested target time",
        }

    reference_position = np.interp(times, reference.times, reference.interface_positions)
    candidate_position = candidate.interface_positions[accepted]
    position_difference = candidate_position - reference_position
    position_denominator = float(np.trapezoid(reference_position**2, times))

    composition_error = None
    metric_reason = None
    if reference.compositions is None or candidate.compositions is None:
        metric_reason = "composition history was not recorded"
    else:
        spatial_error = []
        spatial_norm = []
        try:
            for index in np.flatnonzero(accepted):
                coordinates = candidate.spatial_coordinates[index]
                reference_profile = _profile_at_time(reference, candidate.times[index], coordinates, spatial_atol)
                difference = candidate.compositions[index] - reference_profile
                spatial_error.append(float(np.trapezoid(np.sum(difference**2, axis=1), coordinates)))
                spatial_norm.append(float(np.trapezoid(np.sum(reference_profile**2, axis=1), coordinates)))
        except ValueError as error:
            metric_reason = str(error)
        else:
            final_norm = spatial_norm[-1]
            integrated_norm = float(np.trapezoid(spatial_norm, times))
            composition_error = {
                "final_absolute_l2_profile_error": float(np.sqrt(spatial_error[-1])),
                "final_normalized_l2_profile_error": None if final_norm <= 1e-300 else float(np.sqrt(spatial_error[-1] / final_norm)),
                "final_normalized_l2_profile_error_reason": "reference composition norm is effectively zero"
                if final_norm <= 1e-300
                else None,
                "space_time_normalized_l2_profile_error": None
                if integrated_norm <= 1e-300
                else float(np.sqrt(np.trapezoid(spatial_error, times) / integrated_norm)),
                "space_time_normalized_l2_profile_error_reason": "reference composition norm is effectively zero"
                if integrated_norm <= 1e-300
                else None,
                "space_time_integration": "candidate accepted times; reference interpolated in time and physical space",
            }

    maximum_error = float(np.max(np.abs(position_difference)))
    integrated_error = float(np.trapezoid(np.abs(position_difference), times))
    return {
        **base,
        "completed": bool(reference_reached and candidate_reached),
        "partial": not bool(reference_reached and candidate_reached),
        "comparison_times": times,
        "maximum_absolute_interface_position_error": maximum_error,
        "maximum_interface_position_error": maximum_error,
        "final_interface_position_error": float(abs(position_difference[-1])),
        "time_integrated_absolute_interface_position_error": integrated_error,
        "time_integrated_interface_position_error": integrated_error,
        "normalized_l2_interface_position_error": None
        if position_denominator <= 1e-300
        else float(np.sqrt(np.trapezoid(position_difference**2, times) / position_denominator)),
        "normalized_l2_interface_position_error_reason": "reference interface norm is effectively zero"
        if position_denominator <= 1e-300
        else None,
        "composition_error": composition_error,
        "metric_reason": metric_reason,
    }


def run_illingworth_benchmark(reference_runner, candidate_runners, *, target_time=None, target_time_atol=1e-12):
    """Run caller-configured two-phase cases and compare recorded histories."""
    value = reference_runner()
    reference = value if isinstance(value, IllingworthRun) else capture_illingworth_run(value, label="reference")
    results = {}
    for label, runner in candidate_runners.items():
        try:
            value = runner()
            candidate = value if isinstance(value, IllingworthRun) else capture_illingworth_run(value, label=label)
            results[label] = compare_illingworth_runs(
                reference, candidate, target_time=target_time, target_time_atol=target_time_atol
            )
        except Exception as error:
            results[label] = {
                "runner_succeeded": False,
                "completed": False,
                "reference_reached_target_time": None,
                "candidate_reached_target_time": False,
                "reached_target_time": False,
                "error": str(error),
            }
    return {
        "kind": "two_phase_ternary_illingworth_surrogate_comparison",
        "reference_label": reference.label,
        "results": results,
    }
