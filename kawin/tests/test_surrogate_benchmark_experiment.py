import json
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest

from kawin.diffusion import DiffusivityDataset, SpectralCriteria
from kawin.diffusion.SurrogateBenchmarkExperiment import (
    _IDWPredictor, _canonical_rows, _coordinate_groups, _duplicate_group_metadata, _usable_mask, build_experiment_splits, experiment_schemes,
    load_thermocalc_surrogate_dataset, run_experiment,
)


def _dataset():
    points = np.asarray([[i / 10, j / 10] for i in range(1, 9) for j in range(1, 9) if i + j < 10], dtype=float)
    matrices = np.asarray([[[1 + x, 0.1], [0.05, 1 + y]] for x, y in points])
    return DiffusivityDataset(points, matrices, phases="BCC_B2#1", contexts="general", temperatures=1973., metadata={"elements": ["W", "TI", "FE"]})


def test_nested_splits_seed_hull_and_keep_validation_out_of_training():
    dataset = _dataset()
    splits = build_experiment_splits(dataset, coarse_count=12, medium_count=24, validation_count=8)
    assert set(splits["coarse"]).issubset(splits["medium"])
    assert set(splits["medium"]).issubset(splits["fine"])
    assert not set(splits["validation"]) & set(splits["fine"])
    assert len(splits["coarse"]) == 12 and len(splits["medium"]) == 24


def test_idw_uses_signed_cuberoot_space_and_production_schemes_run():
    dataset = _dataset()
    predictor = _IDWPredictor(dataset, SpectralCriteria())
    result = predictor.predict(dataset.compositions[:2], phase="BCC_B2#1", context="general", temperature=1973.)
    np.testing.assert_allclose(result, dataset.matrices[:2])
    for scheme in experiment_schemes():
        prediction = scheme.factory(dataset).predict(dataset.compositions[:1], phase="BCC_B2#1", context="general", temperature=1973.)
        assert prediction.shape == (1, 2, 2)


def test_archive_loader_reclassifies_sidecar_and_rejects_duplicate_source_rows(tmp_path):
    archive = tmp_path / "cache.npz"
    np.savez(archive, phases=np.asarray(["BCC_B2#1", "LIQUID#1"]), elements=np.asarray(["W", "TI", "FE"]), temperature=np.asarray(1973.),
             diffusivity_compositions_general_0=np.asarray([[.1, .1], [.2, .1], [.1, .2]]),
             diffusivities_general_0=np.broadcast_to(np.eye(2), (3, 2, 2)),
             diffusivity_compositions_general_1=np.asarray([[.1, .1]]), diffusivities_general_1=np.asarray([np.eye(2)]))
    sidecar = archive.with_name(archive.stem + "_invalid_bulk_points.json")
    sidecar.write_text(json.dumps([{"bulk_point_index": 5, "composition": [.2, .2], "phase": "BCC_B2#1", "temperature": 1973., "diffusivity_repr": "array([[1., 0.], [0., 1.]])"}]), encoding="utf-8")
    dataset = load_thermocalc_surrogate_dataset(archive)
    assert len(dataset) == 4
    assert _usable_mask(dataset, SpectralCriteria())[-1]
    assert dataset.metadata["provenance"][-1] == "invalid_sidecar"
    duplicate = {"bulk_point_index": 1, "composition": [.2, .2], "phase": "BCC_B2#1", "temperature": 1973., "diffusivity_repr": "array([[1., 0.], [0., -1.]])"}
    sidecar.write_text(json.dumps([duplicate, duplicate]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate source row"):
        load_thermocalc_surrogate_dataset(archive)


def test_coordinate_groups_prevent_train_validation_leakage_and_preserve_raw_rows():
    base = _dataset()
    points = np.vstack((base.compositions, base.compositions[[10, 11]]))
    matrices = np.concatenate((base.matrices, base.matrices[[10, 11]]))
    dataset = DiffusivityDataset(points, matrices, phases="BCC_B2#1", contexts="general", temperatures=1973.,
                                 metadata={"elements": ["W", "TI", "FE"], "duplicate_atol": 1e-14})
    splits = build_experiment_splits(dataset, coarse_count=12, medium_count=24, validation_count=8)
    atol = dataset.metadata["duplicate_atol"]
    key = lambda row: tuple(np.rint(dataset.compositions[row] / atol).astype(np.int64))
    validation_keys = {key(row) for row in splits["validation"]}
    for level in ("coarse", "medium", "fine"):
        assert not validation_keys & {key(row) for row in splits[level]}
    assert len(splits["fine"]) > splits["selection_location_counts"]["fine"]
    repeated = build_experiment_splits(dataset, coarse_count=12, medium_count=24, validation_count=8)
    for name in ("coarse", "medium", "fine", "validation"):
        np.testing.assert_array_equal(splits[name], repeated[name])


def test_validation_uses_canonical_unique_nonconflicting_coordinate_rows():
    base = _dataset()
    points = np.vstack((base.compositions, base.compositions[[10, 10]]))
    matrices = np.concatenate((base.matrices, base.matrices[[10, 10]]))
    duplicate_key = tuple(np.rint(base.compositions[10] / 1e-14).astype(np.int64))
    matrices[-1] = np.diag([3., 4.])
    dataset = DiffusivityDataset(points, matrices, phases="BCC_B2#1", contexts="general", temperatures=1973., metadata={
        "elements": ["W", "TI", "FE"], "duplicate_atol": 1e-14,
        "source_rows": [f"archive:{index}" for index in range(len(base))] + ["sidecar:2", "sidecar:1"],
        "conflicting_duplicate_composition_keys": [list(duplicate_key)],
    })
    splits = build_experiment_splits(dataset, coarse_count=12, medium_count=24, validation_count=8)
    key = lambda row: tuple(np.rint(dataset.compositions[row] / 1e-14).astype(np.int64))
    assert duplicate_key not in {key(row) for row in splits["validation"]}
    assert len({key(row) for row in splits["validation"]}) == len(splits["validation"]) == 8
    assert len(splits["validation_raw"]) == 8
    assert splits["conflicting_coordinate_groups_excluded_from_validation"] == 1
    assert duplicate_key in {key(row) for row in splits["fine"]}


def test_canonical_row_identity_and_loo_coordinate_population_are_deterministic():
    base = _dataset()
    points = np.vstack((base.compositions, base.compositions[[10, 10]]))
    matrices = np.concatenate((base.matrices, base.matrices[[10, 10]]))
    duplicate_key = tuple(np.rint(base.compositions[10] / 1e-14).astype(np.int64))
    matrices[-1] = np.diag([3., 4.])
    dataset = DiffusivityDataset(points, matrices, phases="BCC_B2#1", contexts="general", temperatures=1973., metadata={
        "elements": ["W", "TI", "FE"], "duplicate_atol": 1e-14,
        "source_rows": [f"archive:{index}" for index in range(len(base))] + ["sidecar:9", "sidecar:1"],
        "conflicting_duplicate_composition_keys": [list(duplicate_key)],
    })
    canonical = DiffusivityDataset(
        [[.1, .1], [.1, .1], [.2, .1], [.1, .2]], [np.eye(2)] * 4,
        phases="BCC_B2#1", contexts="general", temperatures=1973.,
        metadata={"duplicate_atol": 1e-14, "source_rows": ["archive:9", "archive:2", "archive:3", "archive:4"]},
    )
    canonical_key = tuple(np.rint(np.asarray([.1, .1]) / 1e-14).astype(np.int64))
    assert _canonical_rows(canonical, [canonical_key])[0] == 1
    # The conflicting duplicate is excluded from canonical LOO and validation.
    results = run_experiment(dataset, split_kwargs={"coarse_count": 12, "medium_count": 24, "validation_count": 8})
    loo = results["manifest"]["loo"]
    support_keys = {tuple(key) for key in results["manifest"]["robust_positive_support_coordinate_keys"]}
    assert duplicate_key not in support_keys
    assert len(results["manifest"]["robust_positive_validation_mask"]) == 8
    loo_keys = [tuple(key) for key in loo["coordinate_keys"]]
    assert duplicate_key not in loo_keys
    assert len(loo_keys) == len(set(loo_keys)) == len(loo["canonical_master_row_indices"])
    assert all(index in rows for index, rows in zip(loo["canonical_master_row_indices"], loo["coordinate_raw_master_row_indices"]))
    assert len(results["loo"]["kawin_nearest"]["rows"]) == len(loo_keys)


def test_frozen_manifest_reuses_exact_split_and_robust_identities():
    dataset = _dataset()
    kwargs = {"coarse_count": 12, "medium_count": 24, "validation_count": 8}
    original = run_experiment(dataset, split_kwargs=kwargs)
    replay = run_experiment(dataset, frozen_manifest=original["manifest"])
    for level in ("coarse", "medium", "fine"):
        assert replay["manifest"]["refinement_levels"][level]["raw_master_row_indices"] == original["manifest"]["refinement_levels"][level]["raw_master_row_indices"]
    assert replay["manifest"]["validation_master_row_indices"] == original["manifest"]["validation_master_row_indices"]
    assert replay["manifest"]["robust_positive_validation_mask"] == original["manifest"]["robust_positive_validation_mask"]


def test_conflicting_duplicate_keys_remain_excluded_after_subsetting():
    dataset = DiffusivityDataset(
        [[.1, .1], [.1, .1], [.2, .1], [.1, .2]],
        [np.eye(2), np.diag([1., 2.]), np.eye(2), np.eye(2)],
        phases="BCC_B2#1", contexts="general", temperatures=1973.,
        metadata={"duplicate_atol": 1e-14, "conflicting_duplicate_composition_keys": [[10_000_000_000_000, 10_000_000_000_000]]},
    )
    subset = dataset.subset([1, 2, 3])
    assert not _usable_mask(subset, SpectralCriteria())[0]
    assert np.count_nonzero(_usable_mask(subset, SpectralCriteria())) == 2


def test_composition_and_matrix_duplicate_tolerances_are_independent():
    compositions = np.asarray([[.1, .2], [.1 + 4e-15, .2], [.3, .2], [.3 + 4e-15, .2]])
    matrices = np.asarray([np.eye(2) * 1e-30, np.eye(2) * 2e-30, np.eye(2), np.eye(2) * (1 + 1e-12)])
    _, conflicts, details = _duplicate_group_metadata(compositions, matrices, 1e-14, 1e-10, 0.)
    assert len(details) == 2
    assert details[0]["conflicting"]  # Tiny absolute values but a factor-of-two mismatch.
    assert not details[1]["conflicting"]
    assert len(conflicts) == 1


def test_nonconflicting_duplicate_validation_has_one_result_and_retains_raw_rows():
    base = _dataset()
    dataset = DiffusivityDataset(
        np.vstack((base.compositions, base.compositions)), np.concatenate((base.matrices, base.matrices)),
        phases="BCC_B2#1", contexts="general", temperatures=1973.,
        metadata={"elements": ["W", "TI", "FE"], "composition_duplicate_atol": 1e-14,
                  "source_rows": [f"archive:{index}" for index in range(len(base))] + [f"sidecar:{index}" for index in range(len(base))]},
    )
    results = run_experiment(dataset, split_kwargs={"coarse_count": 12, "medium_count": 12, "validation_count": 24})
    manifest = results["manifest"]
    assert len(manifest["validation_master_row_indices"]) == 24
    assert len(manifest["validation_raw_master_row_indices"]) == 48
    assert all(len(rows) == 2 for rows in manifest["validation_coordinate_raw_master_row_indices"])
    for report in results["refinement"]["levels"][0]["results"].values():
        assert len(report["rows"]) == 24


def test_experiment_manifest_and_loo_flux_outputs_are_reproducible():
    results = run_experiment(_dataset(), split_kwargs={"coarse_count": 12, "medium_count": 24, "validation_count": 8})
    manifest = results["manifest"]
    assert manifest["state"] == {"phase": "BCC_B2#1", "context": "general", "temperature": 1973.}
    assert len(manifest["validation_master_row_indices"]) == 8
    assert len(manifest["robust_positive_validation_mask"]) == 8
    assert {"coarse", "medium", "fine"} == set(manifest["refinement_levels"])
    assert manifest["refinement_levels"]["fine"]["raw_sample_count"] >= manifest["refinement_levels"]["fine"]["fit_eligible_count"]
    for report in results["loo"].values():
        assert "flux_summary" in report
        assert all("flux_absolute_error" in row and "flux_low_signal" in row for row in report["rows"])

    script = Path(__file__).resolve().parents[2] / "examples" / "ternaryExamples" / "benchmark_diffusivity_surrogates.py"
    spec = importlib.util.spec_from_file_location("benchmark_diffusivity_surrogates_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arrays = module._loo_detail_arrays(results)
    assert arrays["flux_absolute_error"].shape[1] == 4
    assert arrays["gradient_directions"].shape == (4, 2)
    class _NonClosingStringIO(io.StringIO):
        def close(self):
            pass

    class _MemoryPath:
        def open(self, *args, **kwargs):
            return stream

    stream = _NonClosingStringIO()
    module._write_summary(_MemoryPath(), results)
    text = stream.getvalue()
    for stratum in ("overall", "robust_positive", "gt_positive", "near_boundary", "gt_invalid"):
        assert f",{stratum}," in text


def test_cli_module_imports_and_parses_help(capsys):
    script = Path(__file__).resolve().parents[2] / "examples" / "ternaryExamples" / "benchmark_diffusivity_surrogates.py"
    spec = importlib.util.spec_from_file_location("benchmark_diffusivity_surrogates_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit, match="0"):
        module.main(["--help"])
    assert "cache-only" in capsys.readouterr().out
