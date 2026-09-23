"""Cache-only utilities for the first ternary diffusivity benchmark experiment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from .MovingBoundarySurrogates import MergedPhaseDiffusivitySurrogate, _signed_cuberoot
from .SurrogateBenchmark import (
    DiffusivityDataset, SpectralCriteria, SurrogateScheme, build_neighborhoods,
    eligible_training_mask, make_independent_holdout_plan, make_loo_plan, robust_positive_mask, run_paired_benchmark,
)
from .SurrogateBenchmarkPhase2 import attach_flux_metrics, run_refinement_study


DEFAULT_GRADIENTS = np.asarray([[1., 0.], [0., 1.], [1., 1.] / np.sqrt(2.), [1., -1.] / np.sqrt(2.)])


def _composition_key(composition, atol):
    """Return a deterministic tolerance-quantized key for a composition."""
    return tuple(np.rint(np.asarray(composition, dtype=float) / atol).astype(np.int64))


def _coordinate_groups(dataset):
    """Group dataset row indices by the dataset's stable composition key."""
    atol = float(dataset.metadata.get("composition_duplicate_atol", dataset.metadata.get("duplicate_atol", 1e-14)))
    groups = defaultdict(list)
    for index, point in enumerate(dataset.compositions):
        groups[_composition_key(point, atol)].append(index)
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in groups.items()}


def _conflicting_keys(dataset):
    """Return subset-stable keys whose cached matrices disagree materially."""
    return {tuple(key) for key in dataset.metadata.get("conflicting_duplicate_composition_keys", ())}


def _source_row_key(dataset, index):
    """Order source identities naturally, with a row index fallback for synthetic data."""
    rows = dataset.metadata.get("source_rows", ())
    identity = str(rows[index]) if index < len(rows) else f"row:{index}"
    match = re.fullmatch(r"([^:]+):(\d+)", identity)
    return (match.group(1), int(match.group(2)), int(index)) if match else (identity, -1, int(index))


def _canonical_rows(dataset, keys, *, eligible_only=False, criteria=SpectralCriteria()):
    """Choose the lowest stable source identity for each requested coordinate key.

    ``eligible_only`` is used for experiment LOO so each retained canonical
    row is fit-eligible under the active spectral criteria.  Call this on the
    master dataset: :meth:`DiffusivityDataset.subset` intentionally preserves
    metadata and therefore does not reindex ``source_rows``.
    """
    groups = _coordinate_groups(dataset)
    allowed = eligible_training_mask(dataset, criteria) if eligible_only else np.ones(len(dataset), dtype=bool)
    result = []
    for key in keys:
        candidates = [int(row) for row in groups[key] if allowed[row]]
        if not candidates:
            raise ValueError("requested coordinate group has no eligible canonical row.")
        result.append(min(candidates, key=lambda row: _source_row_key(dataset, row)))
    return np.asarray(result, dtype=np.int64)


def _effective_support_rows(dataset, keys, criteria):
    """Return canonical, nonconflicting, spectrally eligible support rows."""
    groups = _coordinate_groups(dataset)
    eligible = eligible_training_mask(dataset, criteria)
    usable_keys = [tuple(key) for key in keys if tuple(key) not in _conflicting_keys(dataset)
                   and np.any(eligible[groups[tuple(key)]])]
    return _canonical_rows(dataset, usable_keys, eligible_only=True, criteria=criteria), usable_keys


def _matrix_from_repr(value):
    """Parse the NumPy ``array([[...], [...]])`` representation stored by a cache sidecar."""
    numbers = np.fromstring(re.sub(r"[^0-9eE+\-.]", " ", str(value)), sep=" ")
    if numbers.size != 4 or not np.all(np.isfinite(numbers)):
        raise ValueError("invalid sidecar diffusivity_repr; expected four finite matrix entries.")
    return numbers.reshape(2, 2)


def _duplicate_group_metadata(compositions, matrices, composition_duplicate_atol, matrix_duplicate_rtol, matrix_duplicate_atol):
    """Classify coordinate duplicates using separate composition and matrix tolerances."""
    groups = defaultdict(list)
    for index, composition in enumerate(compositions):
        groups[_composition_key(composition, composition_duplicate_atol)].append(index)
    conflicts = set()
    duplicate_groups = []
    for key, indices in groups.items():
        if len(indices) > 1:
            reference = matrices[indices[0]]
            conflict = any(not np.allclose(reference, matrices[index], rtol=matrix_duplicate_rtol,
                                            atol=matrix_duplicate_atol) for index in indices[1:])
            duplicate_groups.append({"coordinate_key": list(key), "indices": indices, "conflicting": conflict})
            if conflict:
                conflicts.add(key)
    return groups, conflicts, duplicate_groups


def load_thermocalc_surrogate_dataset(path, *, phase="BCC_B2#1", context="general", criteria=SpectralCriteria(),
                                      composition_duplicate_atol=1e-14, matrix_duplicate_rtol=1e-10,
                                      matrix_duplicate_atol=0.):
    """Load one phase/context from a cached ternary surrogate archive and its sidecar.

    Sidecar membership is retained as provenance only: matrix eligibility is
    always recomputed using ``criteria``. Coordinates are grouped with
    ``composition_duplicate_atol``; matrix consistency uses its separate
    relative/absolute tolerances. Conflicting groups remain provenance only.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        phases = [str(value) for value in archive["phases"].tolist()]
        if phase not in phases:
            raise ValueError(f"phase {phase!r} is not present in {path.name}.")
        phase_index = phases.index(phase)
        if context != "general":
            raise ValueError("this experiment loader currently supports only general-context cached matrices.")
        compositions = np.asarray(archive[f"diffusivity_compositions_general_{phase_index}"], dtype=float)
        matrices = np.asarray(archive[f"diffusivities_general_{phase_index}"], dtype=float)
        temperature = float(archive["temperature"])
        elements = [str(value) for value in archive["elements"].tolist()]
    # Retained archive positions are not the original bulk-grid indices; keep
    # their namespace distinct from sidecar ``bulk_point_index`` values.
    source_rows = [f"archive:{index}" for index in range(len(compositions))]
    provenance = ["archive"] * len(compositions)
    sidecar = path.with_name(path.stem + "_invalid_bulk_points.json")
    if sidecar.exists():
        entries = json.loads(sidecar.read_text(encoding="utf-8"))
        seen_rows = set()
        for entry in sorted(entries, key=lambda item: int(item["bulk_point_index"])):
            if str(entry.get("phase")) != phase or float(entry.get("temperature")) != temperature:
                continue
            row = int(entry["bulk_point_index"])
            if row in seen_rows:
                raise ValueError(f"duplicate source row {row} in invalid-point sidecar.")
            matrix = _matrix_from_repr(entry.get("diffusivity_repr"))
            compositions = np.vstack((compositions, np.asarray(entry["composition"], dtype=float)))
            matrices = np.concatenate((matrices, matrix[None, :, :]))
            source_rows.append(f"sidecar:{row}")
            provenance.append("invalid_sidecar")
            seen_rows.add(row)
    _, conflicts, duplicate_groups = _duplicate_group_metadata(
        compositions, matrices, composition_duplicate_atol, matrix_duplicate_rtol, matrix_duplicate_atol
    )
    dataset = DiffusivityDataset(
        compositions, matrices, phases=phase, contexts=context, temperatures=temperature,
        metadata={"source_archive": str(path), "invalid_sidecar": str(sidecar), "elements": elements,
                  "source_rows": source_rows, "provenance": provenance,
                  "conflicting_duplicate_composition_keys": [list(key) for key in sorted(conflicts)],
                  "duplicate_groups": duplicate_groups,
                  "composition_duplicate_atol": composition_duplicate_atol,
                  "matrix_duplicate_rtol": matrix_duplicate_rtol,
                  "matrix_duplicate_atol": matrix_duplicate_atol},
    )
    return dataset


def _usable_mask(dataset, criteria):
    """Select eligible unique coordinates for an interpolation fit."""
    eligible = eligible_training_mask(dataset, criteria)
    conflicts = _conflicting_keys(dataset)
    seen = set()
    keep = np.zeros(len(dataset), dtype=bool)
    atol = float(dataset.metadata.get("composition_duplicate_atol", dataset.metadata.get("duplicate_atol", 1e-14)))
    for index, point in enumerate(dataset.compositions):
        key = _composition_key(point, atol)
        if eligible[index] and key not in conflicts and key not in seen:
            keep[index] = True
            seen.add(key)
    return keep


class _MergedPredictor:
    """Benchmark predictor backed by kawin's production merged diffusivity surrogate."""
    def __init__(self, training, interpolation, criteria):
        mask = _usable_mask(training, criteria)
        if np.count_nonzero(mask) < 3:
            raise ValueError("fewer than three unique eligible points are available for interpolation.")
        self.phase = training.phases[0]
        self.temperature = float(training.temperatures[0])
        points, matrices = training.compositions[mask], training.matrices[mask]
        self.source = MergedPhaseDiffusivitySurrogate(
            elements=training.metadata.get("elements", ("W", "TI", "FE")), phase=self.phase, temperature=self.temperature,
            diffusivity_compositions={"general": points, "interface": points}, diffusivities={"general": matrices, "interface": matrices},
            diffusivity_interpolation=interpolation,
        )

    def predict(self, compositions, *, phase, context, temperature):
        return self.source.getInterdiffusivity(compositions, temperature, phase=phase, query_context=context)


class _IDWPredictor:
    """Power-two IDW in the signed cube-root matrix-entry space used by simplex-linear."""
    def __init__(self, training, criteria):
        mask = _usable_mask(training, criteria)
        self.points = training.compositions[mask]
        self.values = _signed_cuberoot(training.matrices[mask].reshape(np.count_nonzero(mask), 4))

    def predict(self, compositions, *, phase, context, temperature):
        distances = np.linalg.norm(np.asarray(compositions)[:, None, :] - self.points[None, :, :], axis=2)
        exact = distances == 0.
        transformed = np.empty((len(distances), 4), dtype=float)
        rows = np.flatnonzero(np.any(exact, axis=1))
        nonexact = np.setdiff1d(np.arange(len(distances)), rows, assume_unique=True)
        if len(nonexact):
            weights = 1. / distances[nonexact] ** 2
            transformed[nonexact] = weights @ self.values / weights.sum(axis=1, keepdims=True)
        if len(rows):
            transformed[rows] = self.values[np.argmax(exact[rows], axis=1)]
        return (transformed ** 3).reshape(-1, 2, 2)


def experiment_schemes(criteria=SpectralCriteria()):
    """Return the three paired schemes used by the W--Ti--Fe experiment."""
    return [
        SurrogateScheme("kawin_nearest", lambda training: _MergedPredictor(training, "nearest", criteria)),
        SurrogateScheme("kawin_simplex_linear", lambda training: _MergedPredictor(training, "simplex_linear", criteria)),
        SurrogateScheme("idw_signed_cuberoot_p2", lambda training: _IDWPredictor(training, criteria)),
    ]


def _fps(points, count, seed=()):
    """Deterministic farthest-point indices, preserving explicit seed indices."""
    selected = list(dict.fromkeys(int(index) for index in seed))
    available = [index for index in range(len(points)) if index not in selected]
    if not selected and available:
        selected.append(available.pop(0))
    if count > len(points):
        raise ValueError("requested more spatial locations than are available.")
    while len(selected) < count:
        distances = np.min(np.linalg.norm(points[available, None] - points[np.asarray(selected)][None], axis=2), axis=1)
        maximum = np.max(distances)
        selected.append(available[int(np.flatnonzero(np.isclose(distances, maximum))[0])])
        available.remove(selected[-1])
    return np.asarray(selected, dtype=np.int64)


def build_experiment_splits(dataset, criteria=SpectralCriteria(), *, coarse_count=64, medium_count=256, validation_count=200):
    """Create reproducible nested levels and an inside-hull coordinate-group holdout.

    All eligible-support hull vertices are seeded with hindsight to guarantee
    a shared interpolation domain. Remaining coarse/medium locations are
    selected by farthest-point sampling over raw coordinates before normal
    eligibility/conflict filtering. Duplicate rows share one location; raw
    index arrays retain every represented calculation for cost accounting.
    """
    from scipy.spatial import ConvexHull, Delaunay
    groups = _coordinate_groups(dataset)
    conflicts = _conflicting_keys(dataset)
    usable = _usable_mask(dataset, criteria)
    usable_by_key = {key: rows[np.flatnonzero(usable[rows])[0]] for key, rows in groups.items() if np.any(usable[rows])}
    usable_keys = list(usable_by_key)
    usable_points = np.asarray([dataset.compositions[usable_by_key[key]] for key in usable_keys])
    hull = ConvexHull(usable_points)
    seed = hull.vertices
    if len(seed) > coarse_count:
        raise ValueError("coarse_count is smaller than the eligible convex-hull vertex set.")
    all_keys = list(groups)
    all_points = np.asarray([dataset.compositions[groups[key][0]] for key in all_keys])
    seed_keys = [usable_keys[index] for index in seed]
    seed_locations = [all_keys.index(key) for key in seed_keys]
    coarse_locations = _fps(all_points, coarse_count, seed_locations)
    coarse_keys = [all_keys[index] for index in coarse_locations]
    coarse_support_keys = [key for key in coarse_keys if key in usable_by_key]
    triangulation = Delaunay(np.asarray([dataset.compositions[usable_by_key[key]] for key in coarse_support_keys]))
    # The validation population is allowed to include spectrally invalid raw
    # records, but its coordinates must lie inside the eligible coarse hull.
    coarse_key_set = set(coarse_keys)
    # Ambiguous duplicate-coordinate groups remain available for raw-cost
    # reporting and are excluded from fitting, but are not valid independent
    # ground truth and therefore cannot enter validation.
    all_remaining = np.asarray([index for index, key in enumerate(all_keys) if key not in coarse_key_set and key not in conflicts], dtype=np.int64)
    all_inside = all_remaining[triangulation.find_simplex(all_points[all_remaining]) >= 0]
    if len(all_inside) < validation_count:
        raise ValueError("not enough non-coarse coordinate groups lie inside the coarse eligible hull for validation.")
    validation_locations = all_inside[_fps(all_points[all_inside], validation_count)]
    validation_keys = [all_keys[index] for index in validation_locations]
    training_keys = [key for key in all_keys if key not in set(validation_keys)]
    training_points = np.asarray([dataset.compositions[groups[key][0]] for key in training_keys])
    coarse_local = [training_keys.index(key) for key in coarse_keys]
    medium_locations = _fps(training_points, medium_count, coarse_local)
    medium_keys = [training_keys[index] for index in medium_locations]

    # Each selected coordinate expands back to every original row.  This
    # preserves raw calculation counts without ever putting a validation key
    # into a training level.
    def expand(selected_keys):
        return np.asarray([row for key in selected_keys for row in groups[key]], dtype=np.int64)

    validation_raw = expand(validation_keys)
    validation_canonical = _canonical_rows(dataset, validation_keys, criteria=criteria)
    return {
        "coarse": expand(coarse_keys), "medium": expand(medium_keys), "fine": expand(training_keys),
        "validation": validation_canonical, "validation_raw": validation_raw,
        "coordinate_keys": {"coarse": [list(key) for key in coarse_keys], "medium": [list(key) for key in medium_keys],
                            "fine": [list(key) for key in training_keys], "validation": [list(key) for key in validation_keys]},
        "selection_location_counts": {"coarse": len(coarse_keys), "medium": len(medium_keys),
                                      "fine": len(training_keys), "validation": len(validation_keys)},
        "conflicting_coordinate_groups_excluded_from_validation": len(conflicts),
    }


def _spacing(dataset, criteria):
    mask = _usable_mask(dataset, criteria)
    points = dataset.compositions[mask]
    if len(points) < 2:
        return np.nan
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    distances[np.diag_indices_from(distances)] = np.inf
    return float(np.median(np.min(distances, axis=1)))


def _raw_spacing(dataset):
    """Return spacing of unique raw coordinates, avoiding duplicate zero distances."""
    groups = _coordinate_groups(dataset)
    points = np.asarray([dataset.compositions[rows[0]] for rows in groups.values()])
    if len(points) < 2:
        return np.nan
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    distances[np.diag_indices_from(distances)] = np.inf
    return float(np.median(np.min(distances, axis=1)))


def run_experiment(dataset, criteria=SpectralCriteria(), *, split_kwargs=None):
    """Run the controlled post-hoc refinement, flux, and medium LOO studies.

    ``split_kwargs`` is primarily useful for small synthetic regression cases;
    production defaults retain the approved 64/256/1327/200 design.
    """
    splits = build_experiment_splits(dataset, criteria, **(split_kwargs or {}))
    validation = dataset.subset(splits["validation"])
    training = dataset.subset(splits["fine"])
    levels = {name: np.asarray([np.where(splits["fine"] == index)[0][0] for index in indices], dtype=np.int64)
              for name in ("coarse", "medium", "fine") for indices in (splits[name],)}
    schemes = experiment_schemes(criteria)
    support_master_indices, support_keys = _effective_support_rows(dataset, splits["coordinate_keys"]["fine"], criteria)
    effective_support = dataset.subset(support_master_indices)
    # Establish headline validation robustness from precisely the canonical
    # support cloud available to interpolators, not raw provenance duplicates.
    effective_plan = make_independent_holdout_plan(effective_support, validation, criteria)
    effective_robust_mask = effective_plan.validation_robust_positive_mask
    refinement = run_refinement_study(training, validation, schemes, levels, criteria,
                                      validation_robust_positive_mask=effective_robust_mask)
    for level in refinement["levels"]:
        level["results"] = attach_flux_metrics(level["results"], DEFAULT_GRADIENTS)
        subset = training.subset(levels[level["label"]])
        level["raw_training_sample_count"] = len(subset)
        level["raw_training_coordinate_count"] = len(_coordinate_groups(subset))
        level["raw_characteristic_spacing"] = _raw_spacing(subset)
        level["fit_eligible_training_sample_count"] = int(np.count_nonzero(_usable_mask(subset, criteria)))
        level["fit_eligible_coordinate_count"] = int(np.count_nonzero(_usable_mask(subset, criteria)))
        level["fit_eligible_characteristic_spacing"] = _spacing(subset, criteria)
    medium_master_indices = splits["medium"]
    loo_master_indices, medium_usable_keys = _effective_support_rows(dataset, splits["coordinate_keys"]["medium"], criteria)
    # Experiment LOO means leave one unique usable composition coordinate out,
    # rather than leaving out one raw cache row while its duplicate remains.
    loo_dataset = dataset.subset(loo_master_indices)
    loo = attach_flux_metrics(run_paired_benchmark(make_loo_plan(loo_dataset, criteria), schemes, criteria), DEFAULT_GRADIENTS)
    robust = refinement["fixed_robust_positive_validation_mask"]
    classes = [record.classification for record in validation.spectral_records(criteria)]
    validation_counts = {kind: classes.count(kind) for kind in ("positive", "near_boundary", "invalid")}
    validation_counts["gt_positive"] = validation_counts["positive"]
    validation_counts["robust_positive"] = int(np.count_nonzero(robust))
    level_manifest = {}
    for name in ("coarse", "medium", "fine"):
        raw_indices = splits[name]
        raw_subset = dataset.subset(raw_indices)
        eligible_local = np.flatnonzero(_usable_mask(raw_subset, criteria))
        level_manifest[name] = {
            "raw_master_row_indices": raw_indices.tolist(),
            "coordinate_keys": splits["coordinate_keys"][name],
            "selection_location_count": splits["selection_location_counts"][name],
            "raw_sample_count": len(raw_subset), "raw_coordinate_count": len(_coordinate_groups(raw_subset)),
            "raw_characteristic_spacing": _raw_spacing(raw_subset),
            "fit_eligible_local_indices": eligible_local.tolist(),
            "fit_eligible_master_row_indices": raw_indices[eligible_local].tolist(),
            "fit_eligible_count": len(eligible_local), "fit_eligible_coordinate_count": len(eligible_local),
            "fit_eligible_characteristic_spacing": _spacing(raw_subset, criteria),
        }
    phase, context, temperature = dataset.phases[0], dataset.contexts[0], float(dataset.temperatures[0])
    manifest = {
        "dataset_metadata": dict(dataset.metadata), "state": {"phase": phase, "context": context, "temperature": temperature},
        "duplicate_tolerances": {
            "composition_duplicate_atol": dataset.metadata.get("composition_duplicate_atol", dataset.metadata.get("duplicate_atol", 1e-14)),
            "matrix_duplicate_rtol": dataset.metadata.get("matrix_duplicate_rtol", 1e-10),
            "matrix_duplicate_atol": dataset.metadata.get("matrix_duplicate_atol", 0.),
        },
        "spectral_criteria": asdict(criteria), "validation_master_row_indices": splits["validation"].tolist(),
        "validation_raw_master_row_indices": splits["validation_raw"].tolist(),
        "validation_coordinate_raw_master_row_indices": [
            _coordinate_groups(dataset)[tuple(key)].tolist() for key in splits["coordinate_keys"]["validation"]
        ],
        "validation_coordinate_keys": splits["coordinate_keys"]["validation"],
        "validation_selection_location_count": splits["selection_location_counts"]["validation"],
        "conflicting_coordinate_groups_excluded_from_validation": splits["conflicting_coordinate_groups_excluded_from_validation"],
        "robust_positive_validation_mask": robust.tolist(),
        "robust_positive_validation_master_row_indices": splits["validation"][robust].tolist(),
        "robust_positive_support_master_row_indices": support_master_indices.tolist(),
        "robust_positive_support_coordinate_keys": [list(key) for key in support_keys],
        "refinement_levels": level_manifest,
        "loo": {"definition": "leave one unique usable composition coordinate out",
                "canonical_master_row_indices": loo_master_indices.tolist(),
                "coordinate_keys": [list(key) for key in medium_usable_keys],
                "coordinate_raw_master_row_indices": [
                    _coordinate_groups(dataset)[key].tolist() for key in medium_usable_keys
                ],
                "raw_medium_master_row_indices": medium_master_indices.tolist()},
        "schemes": [
            {"name": "kawin_nearest", "implementation": "MergedPhaseDiffusivitySurrogate", "interpolation": "nearest"},
            {"name": "kawin_simplex_linear", "implementation": "MergedPhaseDiffusivitySurrogate", "interpolation": "simplex_linear"},
            {"name": "idw_signed_cuberoot_p2", "implementation": "benchmark_idw", "transform": "signed_cuberoot", "power": 2},
        ], "gradient_directions": DEFAULT_GRADIENTS.tolist(), "validation_class_counts": validation_counts,
        "source_identifiers": {key: dataset.metadata.get(key) for key in ("source_archive", "invalid_sidecar", "source_rows", "provenance")},
        "experimental_design": "controlled post-hoc refinement: eligible hull vertices are seeded with hindsight for a shared domain, then remaining raw-coordinate locations are sampled before eligibility filtering.",
    }
    manifest["warnings"] = [f"validation {name} stratum has fewer than 10 rows" for name, count in validation_counts.items() if count < 10]
    try:
        manifest["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        manifest["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        manifest["git_commit"] = None
        manifest["git_dirty"] = None
    return {"manifest": manifest, "refinement": refinement, "loo": loo}
