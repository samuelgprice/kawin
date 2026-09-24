"""Visualize completed cache-only ternary diffusivity benchmark artifacts.

This script is post-processing only: it reads saved predictions and the source
cache recorded in ``manifest.json``; it neither reruns Thermo-Calc nor refits a
surrogate. Example: ``python analyze_diffusivity_benchmark.py --benchmark ...
--output ...``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

_EXAMPLE_DIRECTORY = str(Path(__file__).resolve().parent)
_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
sys.path = [item for item in sys.path if Path(item or ".").resolve() != Path(_EXAMPLE_DIRECTORY)]
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

import matplotlib.pyplot as plt
import numpy as np

from kawin.diffusion.SurrogateBenchmarkAnalysis import (
    LEVELS, SCHEMES, fine_simplex_failures, load_benchmark_analysis,
    matched_failure_success_summary, negative_prediction_breakdown, pointwise_level_rows,
)
from kawin.diffusion.SurrogateBenchmark import robust_positive_mask


def _available_schemes(pointwise):
    """Return recorded schemes in the documented display order."""
    available = set(pointwise["scheme"])
    return tuple(scheme for scheme in SCHEMES if scheme in available)


def _write_csv(path, rows):
    """Write records with nested diagnostics JSON-encoded for portable tables."""
    rows = list(rows)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, default=lambda item: item.tolist() if isinstance(item, np.ndarray) else item.item()) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def _summary_value(rows, level, scheme, stratum, metric, statistic):
    return next((float(row["value"]) for row in rows if row["level"] == level and row["scheme"] == scheme and row["stratum"] == stratum and row["metric"] == metric and row["statistic"] == statistic), np.nan)


def _refinement_table(data):
    """Flatten experiment-specific support counts and key aggregate metrics."""
    output = []
    schemes = _available_schemes(data["pointwise"])
    for level in LEVELS:
        metadata = data["manifest"]["refinement_levels"][level]
        for scheme in schemes:
            row = {"level": level, "scheme": scheme, "selection_location_count": metadata["selection_location_count"],
                   "raw_sample_count": metadata["raw_sample_count"], "fit_eligible_count": metadata["fit_eligible_count"],
                   "fit_eligible_location_count": metadata["fit_eligible_coordinate_count"],
                   "raw_characteristic_spacing": metadata["raw_characteristic_spacing"],
                   "fit_eligible_characteristic_spacing": metadata["fit_eligible_characteristic_spacing"]}
            for metric in ("frobenius_relative_error", "spectral_relative_error", "operator_relative_error", "rms_log_eigenvalue_error"):
                for statistic in ("median", "p95"):
                    row[f"{metric}_{statistic}"] = _summary_value(data["summary"], level, scheme, "robust_positive", metric, statistic)
            for metric in ("absolute_flux_error", "relative_flux_error"):
                for statistic in ("median", "p95"):
                    row[f"{metric}_{statistic}"] = _summary_value(data["summary"], level, scheme, "flux_overall", metric, statistic)
            for metric in ("robust_positive_spectral_invalid_rate", "robust_positive_unusable_rate"):
                row[metric] = _summary_value(data["summary"], level, scheme, "overall", metric, "value")
            output.append(row)
    return output


def _save_refinement_figure(table, figures):
    """Plot metric convergence against raw cost and actual eligible support."""
    metrics = ("frobenius_relative_error", "spectral_relative_error", "operator_relative_error", "relative_flux_error",
               "robust_positive_spectral_invalid_rate", "robust_positive_unusable_rate")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    schemes = tuple(dict.fromkeys(row["scheme"] for row in table))
    for axis, metric in zip(axes.flat, metrics):
        for scheme in schemes:
            rows = [row for row in table if row["scheme"] == scheme]
            values = [row.get(f"{metric}_median", row.get(metric)) for row in rows]
            axis.plot([row["raw_sample_count"] for row in rows], values, "o-", label=scheme)
        axis.set_xscale("log")
        if metric not in ("robust_positive_spectral_invalid_rate", "robust_positive_unusable_rate"):
            axis.set_yscale("log")
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("raw GT samples")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(figures / "refinement_vs_raw_cost.png", dpi=180)
    for axis, metric in zip(axes.flat, metrics):
        axis.clear()
        for scheme in schemes:
            rows = [row for row in table if row["scheme"] == scheme]
            axis.plot([row["fit_eligible_location_count"] for row in rows], [row.get(f"{metric}_median", row.get(metric)) for row in rows], "o-", label=scheme)
        axis.set_xscale("log")
        if metric not in ("robust_positive_spectral_invalid_rate", "robust_positive_unusable_rate"):
            axis.set_yscale("log")
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("fit-eligible locations")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(figures / "refinement_vs_effective_support.png", dpi=180)
    plt.close(fig)


def _save_maps(data, figures):
    """Save GT classification/margin and fine prediction/error composition maps."""
    dataset, criteria, pointwise = data["dataset"], data["criteria"], data["pointwise"]
    records = dataset.spectral_records(criteria)
    classes = np.asarray([record.classification for record in records])
    robust = robust_positive_mask(dataset, criteria)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for label, color, membership in (("robust-positive", "tab:green", robust), ("other positive", "tab:blue", (classes == "positive") & ~robust), ("near-boundary", "tab:orange", classes == "near_boundary"), ("invalid", "tab:red", classes == "invalid")):
        axes[0].scatter(dataset.compositions[membership, 0], dataset.compositions[membership, 1], s=8, label=label, color=color)
    axes[0].legend(fontsize=8); axes[0].set_title("GT spectral classification")
    margins = np.asarray([record.normalized_minimum_eigenvalue for record in records])
    positive = classes == "positive"
    image = axes[1].scatter(dataset.compositions[positive, 0], dataset.compositions[positive, 1], c=np.log10(np.maximum(margins[positive], 1e-300)), s=8, cmap="viridis")
    axes[1].scatter(dataset.compositions[~positive, 0], dataset.compositions[~positive, 1], s=5, color="lightgray", label="nonpositive/invalid")
    fig.colorbar(image, ax=axes[1], label="log10 normalized minimum eigenvalue")
    axes[1].set_title("GT spectral margin")
    fig.savefig(figures / "gt_classification_and_margin_maps.png", dpi=180); plt.close(fig)
    schemes = _available_schemes(pointwise)
    fig, axes = plt.subplots(len(schemes), 3, figsize=(13, 3.6 * len(schemes)), constrained_layout=True)
    for row, scheme in enumerate(schemes):
        mask = (pointwise["level"] == "fine") & (pointwise["scheme"] == scheme)
        points = pointwise["composition"][mask]
        failure = pointwise["prediction_failure"][mask] == "negative"
        robust_failure = failure & pointwise["robust_positive"][mask]
        axes[row, 0].scatter(points[:, 0], points[:, 1], c=np.where(failure, 1, 0), cmap="coolwarm", s=14); axes[row, 0].scatter(points[robust_failure, 0], points[robust_failure, 1], facecolors="none", edgecolors="k", s=38)
        frobenius = axes[row, 1].scatter(points[:, 0], points[:, 1], c=pointwise["frobenius_relative_error"][mask], norm="log", cmap="magma", s=14)
        flux = axes[row, 2].scatter(points[:, 0], points[:, 1], c=np.nanmedian(pointwise["flux_relative_error"][mask], axis=1), norm="log", cmap="magma", s=14)
        fig.colorbar(frobenius, ax=axes[row, 1], label="normalized Frobenius error")
        fig.colorbar(flux, ax=axes[row, 2], label="median relative flux error")
        axes[row, 0].set_ylabel(scheme)
    for axis, title in zip(axes[0], ("negative (outline: robust)", "Frobenius error", "median relative flux error")):
        axis.set_title(title); axis.set_xlabel("composition coordinate 1")
    for axis in axes[:, 0]: axis.set_xlabel("composition coordinate 1")
    fig.savefig(figures / "fine_failure_and_error_maps.png", dpi=180); plt.close(fig)


def _save_simplex_diagnostics(data, figures):
    """Save fine simplex spectral sensitivity plots with robust failures highlighted."""
    pointwise, criteria = data["pointwise"], data["criteria"]
    mask = (pointwise["level"] == "fine") & (pointwise["scheme"] == "kawin_simplex_linear")
    truth, prediction = pointwise["truth_matrix"][mask], pointwise["predicted_matrix"][mask]
    gt = np.asarray([np.sort(np.linalg.eigvals(value).real)[0] for value in truth])
    pred = np.asarray([np.sort(np.linalg.eigvals(value).real)[0] for value in prediction])
    margins = np.asarray([np.linalg.eigvals(value).real.min() / np.linalg.norm(value, ord=np.inf) for value in truth])
    condition = np.asarray([np.linalg.cond(value) for value in truth])
    failed = (pointwise["prediction_failure"][mask] == "negative") & pointwise["robust_positive"][mask]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    pairs = ((gt, pred, "predicted vs GT minimum eigenvalue"), (gt, pred - gt, "signed minimum-eigenvalue error"),
             (margins, pointwise["frobenius_relative_error"][mask], "Frobenius vs GT margin"), (margins, np.nanmedian(pointwise["flux_relative_error"][mask], axis=1), "flux vs GT margin"),
             (condition, pointwise["operator_relative_error"][mask], "operator vs condition"), (condition, pointwise["frobenius_relative_error"][mask], "matrix error vs condition"))
    for axis, (x, y, title) in zip(axes.flat, pairs):
        axis.scatter(x, y, s=13, alpha=.7); axis.scatter(x[failed], y[failed], s=34, facecolors="none", edgecolors="red")
        axis.set_xscale("symlog", linthresh=1e-18); axis.set_yscale("symlog", linthresh=1e-18); axis.set_title(title)
    fig.savefig(figures / "fine_simplex_spectral_diagnostics.png", dpi=180); plt.close(fig)


def _save_paired_and_loo(data, figures):
    """Save paired fine error comparisons and LOO/holdout distribution CDFs."""
    pointwise = data["pointwise"]
    schemes = _available_schemes(pointwise)
    comparisons = tuple(scheme for scheme in ("kawin_nearest", "idw_signed_cuberoot_p2", "kawin_simplex_positive_2x2") if scheme in schemes)
    fig, axes = plt.subplots(1, len(comparisons), figsize=(5 * len(comparisons), 4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    fine = {scheme: {name: pointwise[name][(pointwise["level"] == "fine") & (pointwise["scheme"] == scheme)]
                     for name in pointwise.files} for scheme in schemes}
    for axis, other in zip(axes, comparisons):
        ratio = fine["kawin_simplex_linear"]["frobenius_relative_error"] / fine[other]["frobenius_relative_error"]
        image = axis.scatter(fine[other]["composition"][:, 0], fine[other]["composition"][:, 1], c=np.log10(ratio), cmap="coolwarm", s=16)
        fig.colorbar(image, ax=axis, label="log10 error ratio")
        axis.set_title(f"log10 simplex / {other} Frob. ratio")
    fig.savefig(figures / "fine_paired_scheme_comparisons.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    loo = data["loo"]
    for axis, metric in zip(axes, ("frobenius", "minimum_eigenvalue", "relative_flux")):
        for scheme in schemes:
            medium = pointwise_level_rows(pointwise, "medium", scheme)
            independent = medium["frobenius_relative_error"] if metric == "frobenius" else (medium["minimum_eigenvalue_absolute_error"] if metric == "minimum_eigenvalue" else np.nanmedian(medium["flux_relative_error"], axis=1))
            lm = loo["scheme"] == scheme
            truth, pred = loo["truth_matrix"][lm], loo["predicted_matrix"][lm]
            values = np.linalg.norm(pred - truth, axis=(1, 2)) / np.linalg.norm(truth, axis=(1, 2)) if metric == "frobenius" else (np.abs(_minimum(pred) - _minimum(truth)) if metric == "minimum_eigenvalue" else np.nanmedian(loo["flux_relative_error"][lm], axis=1))
            for value, style in ((independent, "-"), (values, "--")):
                value = np.sort(value[np.isfinite(value)]); axis.plot(value, np.arange(1, len(value) + 1) / len(value), style, label=f"{scheme} {'medium independent' if style == '-' else 'medium LOO'}")
        axis.set_xscale("log"); axis.set_title(f"medium LOO vs medium independent: {metric}"); axis.legend(fontsize=6)
    fig.savefig(figures / "medium_loo_vs_independent_cdfs.png", dpi=180); plt.close(fig)


def _minimum(matrices):
    """Return real minimum eigenvalues for a matrix batch."""
    return np.sort(np.linalg.eigvals(matrices).real, axis=1)[:, 0]


def run_analysis(benchmark_directory, output_directory, dataset_path=None):
    """Generate report, figures, and traceable tables for a completed benchmark."""
    data = load_benchmark_analysis(benchmark_directory, dataset_path)
    output, figures, tables = Path(output_directory), Path(output_directory) / "figures", Path(output_directory) / "tables"
    figures.mkdir(parents=True, exist_ok=True); tables.mkdir(parents=True, exist_ok=True)
    refinement = _refinement_table(data); negative = negative_prediction_breakdown(data["pointwise"]); failures, vertices = fine_simplex_failures(data); matched = matched_failure_success_summary(data, failures)
    if "kawin_simplex_positive_2x2" in set(data["pointwise"]["scheme"]):
        positive_mask = ((data["pointwise"]["level"] == "fine") &
                         (data["pointwise"]["scheme"] == "kawin_simplex_positive_2x2"))
        positive_rows = {tuple(point): index for index, point in enumerate(data["pointwise"]["composition"][positive_mask])}
        positive_failure = data["pointwise"]["prediction_failure"][positive_mask]
        positive_frobenius = data["pointwise"]["frobenius_relative_error"][positive_mask]
        for row in failures:
            index = positive_rows[(row["composition_x"], row["composition_y"])]
            row["simplex_positive_2x2_prediction_failure"] = str(positive_failure[index])
            row["simplex_positive_2x2_frobenius_relative_error"] = float(positive_frobenius[index])
    _write_csv(tables / "refinement_metric_summary.csv", refinement); _write_csv(tables / "simplex_negative_predictions_by_stratum.csv", negative)
    _write_csv(tables / "simplex_fine_robust_positive_failures.csv", failures); _write_csv(tables / "simplex_failure_simplex_vertices.csv", vertices)
    _write_csv(tables / "simplex_failure_vs_success_summary.csv", [{"group": key, **value} for key, value in matched.items() if key != "matches" and isinstance(value, dict)])
    _write_csv(tables / "simplex_failure_matched_successes.csv", matched["matches"])
    _save_refinement_figure(refinement, figures); _save_maps(data, figures); _save_simplex_diagnostics(data, figures); _save_paired_and_loo(data, figures)
    lines = ["# Diffusivity benchmark analysis", "", "This report is cache-only post-processing of saved benchmark predictions; no Thermo-Calc calls or surrogate refits were made.", "",
             "## Results summary", "", f"{len(failures)} fine robust-positive simplex failures were reconstructed from the exact recorded fine effective support. Each signed-cube-root reconstruction matches the saved prediction. Total negative predictions also include GT-invalid validation rows; the stratum table below keeps those populations separate.", "",
             "## Negative predictions by stratum", "", "|Level|Stratum|Negative|Population|Fraction|", "|---|---|---:|---:|---:|"]
    lines += [f"|{row['level']}|{row['stratum']}|{row['negative_prediction_count']}|{row['stratum_count']}|{row['negative_fraction']:.3g}|" for row in negative]
    determinant_crossings = sum(row["predicted_negative_determinant"] and row["active_vertices_positive_determinant"] and row["trace_remains_positive"] for row in failures)
    near_invalid = sum(row["nearest_invalid_distance"] <= .02 + 1e-12 for row in failures)
    lines += ["", "## Determinant and locality evidence", "", f"{determinant_crossings}/{len(failures)} robust-positive failures have positive-determinant active vertices, a negative predicted determinant, and a positive predicted trace. {near_invalid}/{len(failures)} lie within 0.02 composition units of a GT-invalid cache point. This is evidence of a spectral-boundary crossing, not by itself proof of a GT discontinuity.", "",
              "## Matched success comparison", "", f"Matching rule: {matched['matching_rule']}. Success reuse count: {matched['success_reuse_count']}.", "", "```json", json.dumps({key: value for key, value in matched.items() if key != 'matches'}, indent=2), "```", "",
              "## Figures", "", "- `refinement_vs_raw_cost.png` and `refinement_vs_effective_support.png` separate cached sampling cost from usable interpolation support.",
              "- `gt_classification_and_margin_maps.png` and `fine_failure_and_error_maps.png` locate GT spectral classes, failures, and errors in composition space.",
              "- `fine_simplex_spectral_diagnostics.png` highlights the small-error/negative-minimum-eigenvalue regime; red outlines are robust-positive failures.",
              "- `medium_loo_vs_independent_cdfs.png` compares different validation populations and is not interpreted as an automatic optimistic/pessimistic ordering.",
              "- `fine_paired_scheme_comparisons.png` is pointwise paired, so colour reflects the same validation compositions across schemes."]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"output": output, "failure_count": len(failures), "negative": negative, "matched": matched}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="Completed benchmark output directory")
    parser.add_argument("--output", required=True, help="Analysis output directory")
    parser.add_argument("--dataset", help="Optional cache archive override")
    args = parser.parse_args(argv)
    result = run_analysis(args.benchmark, args.output, args.dataset)
    print(json.dumps({"output": str(result["output"]), "fine_robust_positive_failure_count": result["failure_count"]}, indent=2))


if __name__ == "__main__":
    main()
