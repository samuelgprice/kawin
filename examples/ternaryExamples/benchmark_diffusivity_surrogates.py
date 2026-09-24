"""Run the cache-only W--Ti--Fe ternary diffusivity surrogate experiment.

Example:
``python benchmark_diffusivity_surrogates.py --dataset <archive.npz> --output results``.
No Thermo-Calc calculations are performed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

# pycalphad's plugin discovery imports modules visible on ``sys.path``.  When
# this file is run directly, its directory exposes the unrelated
# ``pycalphad_default_phase_adapter.py`` example and causes a circular import.
_EXAMPLE_DIRECTORY = str(Path(__file__).resolve().parent)
_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != Path(_EXAMPLE_DIRECTORY)]
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

import numpy as np

from kawin.diffusion import DiffusivityDataset, SpectralCriteria
from kawin.diffusion.SurrogateBenchmarkExperiment import DEFAULT_GRADIENTS, load_thermocalc_surrogate_dataset, run_experiment


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _detail_arrays(results):
    """Convert paired rows to portable fixed-shape arrays for later inspection."""
    rows = []
    for level in results["refinement"]["levels"]:
        for scheme, report in level["results"].items():
            rows.extend((level["label"], scheme, row) for row in report["rows"])
    return {
        "level": np.asarray([item[0] for item in rows]), "scheme": np.asarray([item[1] for item in rows]),
        "composition": np.asarray([item[2]["composition"] for item in rows]),
        "truth_matrix": np.asarray([item[2]["truth_matrix"] for item in rows]),
        "predicted_matrix": np.asarray([item[2]["predicted_matrix"] for item in rows]),
        "truth_classification": np.asarray([item[2]["truth_classification"] for item in rows]),
        "prediction_failure": np.asarray([str(item[2]["prediction_failure"]) for item in rows]),
        "robust_positive": np.asarray([item[2]["robust_positive"] for item in rows]),
        "frobenius_relative_error": np.asarray([np.nan if item[2]["frobenius_relative_error"] is None else item[2]["frobenius_relative_error"] for item in rows]),
        "spectral_relative_error": np.asarray([np.nan if item[2]["spectral_relative_error"] is None else item[2]["spectral_relative_error"] for item in rows]),
        "operator_relative_error": np.asarray([np.nan if item[2]["operator_relative_error"] is None else item[2]["operator_relative_error"] for item in rows]),
        "minimum_eigenvalue_absolute_error": np.asarray([np.nan if item[2]["minimum_eigenvalue_absolute_error"] is None else item[2]["minimum_eigenvalue_absolute_error"] for item in rows]),
        "minimum_eigenvalue_signed_error": np.asarray([np.nan if item[2]["minimum_eigenvalue_signed_error"] is None else item[2]["minimum_eigenvalue_signed_error"] for item in rows]),
        "flux_absolute_error": np.asarray([item[2]["flux_absolute_error"] for item in rows]),
        "flux_relative_error": np.asarray([item[2]["flux_relative_error"] for item in rows]),
        "flux_low_signal": np.asarray([[False if value is None else bool(value) for value in item[2]["flux_low_signal"]] for item in rows]),
        "flux_low_signal_available": np.asarray([[value is not None for value in item[2]["flux_low_signal"]] for item in rows]),
    }


def _loo_detail_arrays(results):
    """Serialize LOO pointwise rows using the same portable matrix layout."""
    rows = [(scheme, row) for scheme, report in results["loo"].items() for row in report["rows"]]
    return {
        "scheme": np.asarray([item[0] for item in rows]),
        "composition": np.asarray([item[1]["composition"] for item in rows]),
        "truth_matrix": np.asarray([item[1]["truth_matrix"] for item in rows]),
        "predicted_matrix": np.asarray([item[1]["predicted_matrix"] for item in rows]),
        "prediction_failure": np.asarray([str(item[1]["prediction_failure"]) for item in rows]),
        "flux_absolute_error": np.asarray([item[1]["flux_absolute_error"] for item in rows]),
        "flux_relative_error": np.asarray([item[1]["flux_relative_error"] for item in rows]),
        "flux_low_signal": np.asarray([[False if value is None else bool(value) for value in item[1]["flux_low_signal"]] for item in rows]),
        "flux_low_signal_available": np.asarray([[value is not None for value in item[1]["flux_low_signal"]] for item in rows]),
        "gradient_directions": np.asarray(DEFAULT_GRADIENTS),
    }


def _write_summary(path, results):
    """Write compact long-form aggregate metrics for each scheme and level."""
    columns = ["level", "scheme", "stratum", "metric", "statistic", "value"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for level in results["refinement"]["levels"]:
            for scheme, report in level["results"].items():
                summary = report["summary"]
                for key in ("raw_training_sample_count", "raw_training_coordinate_count", "fit_eligible_training_sample_count", "fit_eligible_coordinate_count", "characteristic_spacing", "raw_characteristic_spacing", "fit_eligible_characteristic_spacing"):
                    writer.writerow(dict(level=level["label"], scheme=scheme, stratum="training", metric=key, statistic="value", value=level.get(key)))
                for key in ("gt_sample_count", "eligible_training_pool_count", "robust_positive_spectral_invalid_rate", "robust_positive_unusable_rate", "negative_prediction_count", "complex_prediction_count", "nonfinite_prediction_count"):
                    writer.writerow(dict(level=level["label"], scheme=scheme, stratum="overall", metric=key, statistic="value", value=summary.get(key)))
                strata = {"overall": {key: value for key, value in summary.items() if isinstance(value, dict) and key != "strata"}, **summary["strata"]}
                for stratum, metrics in strata.items():
                    for metric, statistics in metrics.items():
                        for statistic, value in statistics.items():
                            writer.writerow(dict(level=level["label"], scheme=scheme, stratum=stratum, metric=metric, statistic=statistic, value=value))
                flux_strata = {"overall": report["flux_summary"]["overall"], **report["flux_summary"]["strata"]}
                for stratum, metrics in flux_strata.items():
                    for metric, statistics in metrics.items():
                        if isinstance(statistics, dict):
                            for statistic, value in statistics.items():
                                writer.writerow(dict(level=level["label"], scheme=scheme, stratum=f"flux_{stratum}", metric=metric, statistic=statistic, value=value))
                        else:
                            writer.writerow(dict(level=level["label"], scheme=scheme, stratum=f"flux_{stratum}", metric=metric, statistic="value", value=statistics))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", default="BCC_B2#1")
    parser.add_argument("--context", default="general")
    parser.add_argument("--frozen-manifest", help="Existing manifest whose exact dataset splits and robust mask must be reused")
    args = parser.parse_args(argv)
    source = Path(args.dataset)
    dataset = DiffusivityDataset.load(source) if source.name.endswith(".dataset.npz") else load_thermocalc_surrogate_dataset(source, phase=args.phase, context=args.context)
    results = run_experiment(dataset, SpectralCriteria(), frozen_manifest=args.frozen_manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(results["manifest"], indent=2, default=_json_default), encoding="utf-8")
    _write_summary(output / "summary.csv", results)
    np.savez_compressed(output / "pointwise_results.npz", **_detail_arrays(results))
    loo_summary = {"gradient_directions": DEFAULT_GRADIENTS.tolist(), "schemes": {
        key: {"summary": value["summary"], "flux_summary": value["flux_summary"]}
        for key, value in results["loo"].items()
    }}
    (output / "loo_summary.json").write_text(json.dumps(loo_summary, indent=2, default=_json_default), encoding="utf-8")
    np.savez_compressed(output / "loo_pointwise_results.npz", **_loo_detail_arrays(results))
    print(json.dumps(results["manifest"], indent=2, default=_json_default))


if __name__ == "__main__":
    main()
