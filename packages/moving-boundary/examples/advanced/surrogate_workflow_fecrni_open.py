"""Advanced checks for ``surrogate_workflow_fecrni_open``; see that example for assumptions and references.

The Case supplies explicit physical inputs and callbacks. No main script
is imported, and no solve or file operation runs during import.
"""

from dataclasses import dataclass
from contextlib import ExitStack
from pathlib import Path
import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
import numpy as np
from moving_boundary import surrogate_diagnostics as diagnostics
from moving_boundary.surrogate_artifacts import SurrogateArtifactBundle, SurrogateArtifactCompatibilityError, canonicalize_build_spec
from moving_boundary.surrogate_comparison import evaluate_saved_surrogate_comparison


@dataclass(frozen=True)
class Case:
    """Inputs and callbacks supplied by the main example at call time."""

    DATABASE_ENV: object
    DEFAULT_TC_CACHE: object
    ELEMENTS: object
    ExampleResult: object
    PHASE_MAPPING: object
    TC_CACHE_ENV: object
    TC_CACHE_MAX_CHARS: object
    TC_PHASES: object
    TRAINING_DISTANCE: object
    ThermoCalcUnavailableError: object
    description: str
    _backend_steps: object
    _split: object
    _stats: object
    _timed: object
    build_source: object
    check_options: object
    compare_bundles: object
    database_digest: object
    fresh_query_diagnostics: object
    load_saved_thermocalc_bundle: object
    loo_indices: object
    open_thermocalc: object
    prepare_pycalphad_bundle: object
    prepare_thermocalc_bundle: object
    require_thermocalc: object
    resolve_database: object
    resolve_tc_cache: object
    saved_bundle_diagnostics: object
    shared_spec: object
    simulate: object


def require_thermocalc(*, case):
    """Stop before any work if TC-Python cannot be imported; never imports it here."""
    try:
        available = importlib.util.find_spec("tc_python") is not None
    except (ImportError, ValueError):
        available = False
    if not available:
        raise case.ThermoCalcUnavailableError(
            "Live Thermo-Calc work (--thermocalc build or --tc-fresh-queries) needs TC-Python (tc_python) and a "
            "Thermo-Calc licence. Without them, use --thermocalc saved with an existing bundle, or --thermocalc off.")

def resolve_tc_cache(path, output_dir, *, case):
    """Short Thermo-Calc cache folder, separate from the bundles; never changes machine settings."""
    cache = Path(path or os.environ.get(case.TC_CACHE_ENV) or case.DEFAULT_TC_CACHE).resolve()
    if cache.is_relative_to(Path(output_dir).resolve()):
        raise ValueError(f"Keep the Thermo-Calc cache ({cache}) outside the bundle folder {output_dir}.")
    if len(str(cache)) > case.TC_CACHE_MAX_CHARS:
        raise ValueError(f"Thermo-Calc cache path is {len(str(cache))} characters ({cache}); Thermo-Calc truncates "
                         f"long workspace paths, so use --tc-cache-dir or {case.TC_CACHE_ENV} with at most "
                         f"{case.TC_CACHE_MAX_CHARS} characters.")
    return cache

def open_thermocalc(tdb, cache_dir, *, case):
    """Start a Thermo-Calc session for the open database (needs TC-Python and a licence).

    Uses the suffixed composition-set labels, which the adapter matches exactly. For a user
    database that declares VA without entering it, the adapter selects VA as a system element
    itself (physical components stay FE, CR, NI).
    """
    from moving_boundary.providers.thermocalc import TCPythonThermodynamics, ThermoCalcConfig

    config = ThermoCalcConfig(user_database_path=Path(tdb), elements=tuple(case.ELEMENTS), phases=tuple(case.TC_PHASES),
                              reference_element="FE", use_default_phases=False, cache_dir=Path(cache_dir))
    return TCPythonThermodynamics(config)

def load_saved_thermocalc_bundle(path, config, tdb, *, case):
    """Load a saved Thermo-Calc bundle without Thermo-Calc and check its backend-free settings.

    The bundle is verified against its own manifest; its database digest, system, tie-line and
    bulk settings must equal this configuration's, and its source engine must be thermocalc.
    """
    path = Path(path)
    hint = ("Build one with --thermocalc build on a machine with TC-Python and a licence, "
            "or point --thermocalc-bundle at an existing compatible bundle.")
    if not (path / "manifest.json").is_file():
        raise FileNotFoundError(f"No saved Thermo-Calc bundle at {path}. {hint}")
    bundle = SurrogateArtifactBundle.load_published(path)
    recorded = bundle.manifest["build_spec"]
    expected = canonicalize_build_spec(case.shared_spec(config, case.resolve_database(tdb), case.TC_PHASES))
    differences = [key for key, value in expected.items() if recorded.get(key) != value]
    if recorded.get("source", {}).get("engine") != "thermocalc":
        differences.append("source.engine")
    if differences:
        raise SurrogateArtifactCompatibilityError(
            f"Saved Thermo-Calc bundle {path} does not match these settings ({', '.join(differences)}). {hint}")
    return bundle

def _stats(values, *, case):
    """Summarize finite query discrepancies; empty sets have no mean or maximum."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {"count": int(finite.size), "max": float(finite.max()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None}

def _split(values, training, *, case):
    """Summaries of ``values`` at training and at held-out queries, reported separately."""
    values, training = np.asarray(values, dtype=np.float64), np.asarray(training, dtype=bool)
    return {"training": case._stats(values[training]), "held_out": case._stats(values[~training])}

def loo_indices(surrogate, config, *, case):
    """At most ``loo_max_per_phase`` evenly spread rows of each phase's fitted bulk samples."""
    counts = {phase: len(surrogate.get_diffusivity_samples(phase)[0]) for phase in surrogate.tieline_phases}
    return {phase: np.unique(np.linspace(0, n - 1, min(config.loo_max_per_phase, n)).round().astype(np.int64))
            for phase, n in counts.items()}

def saved_bundle_diagnostics(surrogate, config, *, case):
    """Diagnostics from the saved bundle alone: no backend is queried.

    Returns ``(summary, reports)``: construction provenance counts, leave-one-out (internal
    cross-validation on at most ``loo_max_per_phase`` samples per phase) and the heuristic
    interface/bulk consistency flags.
    """
    construction = diagnostics.evaluate_surrogate_construction_diagnostics(surrogate)
    loo = diagnostics.evaluate_diffusivity_leave_one_out(surrogate, sample_indices=case.loo_indices(surrogate, config))
    consistency = diagnostics.evaluate_interface_bulk_diffusivity_consistency(surrogate)
    summary = {
        "tie_lines": int(len(surrogate.eta_samples)),
        "fit_points": {phase: int(len(surrogate.get_diffusivity_samples(phase)[0])) for phase in surrogate.tieline_phases},
        "construction_records": len(construction["records"]),
        "leave_one_out": {phase: dict(report["summary"]) for phase, report in loo["phase_reports"].items()},
        "consistency": dict(consistency["summary"]),
    }
    return summary, {"construction": construction, "leave_one_out": loo, "consistency": consistency}

def fresh_query_diagnostics(surrogate, source, config, *, case):
    """Fresh queries of the backend that built ``surrogate``: approximation discrepancy.

    Tie-line endpoint errors, interface matrix errors and bulk matrix errors are each split
    into queries that coincide with training samples (training eta, or a bulk composition
    within ``TRAINING_DISTANCE`` of a stored sample) and held-out ones. Backend failures are
    recorded, not raised.
    """
    tielines = diagnostics.evaluate_tieline_diagnostics(
        surrogate, thermodynamics=source, eta_count=config.fresh_tieline_eta_count, on_truth_error="record")
    matrices = diagnostics.evaluate_diffusivity_diagnostics(
        surrogate, thermodynamics=source, interface_eta_count=config.fresh_interface_eta_count,
        bulk_axes=config.fresh_bulk_axes, on_truth_error="record")
    training_eta = np.asarray(tielines["training_mask"], dtype=bool)
    summary = {"tie_lines": {"queries": int(training_eta.size), "training": int(training_eta.sum()),
                             "failures": len(tielines["failures"]),
                             "endpoint_error": {phase: case._split(values, training_eta) for phase, values in
                                                tielines["truth"]["endpoint_error_norm"].items()}},
               "interface_matrices": {}, "bulk_matrices": {}, "failures": len(matrices["failures"])}
    for phase, report in matrices["interface"]["phases"].items():
        summary["interface_matrices"][phase] = case._split(report["truth"]["matrix_relative_error"], report["training_mask"])
    for phase, report in matrices["bulk"]["phases"].items():
        training = np.asarray(report["nearest_training_distance"]) <= case.TRAINING_DISTANCE
        summary["bulk_matrices"][phase] = {**case._split(report["truth"]["matrix_relative_error"], training),
                                           "invalid_predictions": int(np.count_nonzero(~report["valid"]))}
    return summary, {"tielines": tielines, "diffusivity": matrices}

def compare_bundles(reference, candidate, config, *, case):
    """Compare the pycalphad (reference) and Thermo-Calc (candidate) saved bundles.

    Uses ``PHASE_MAPPING`` because the backends label the same phases differently. A summary
    of own-path geometry, matrices on own paths and at identical compositions, and bulk
    coverage is returned with the full report. Identical-composition interface maxima are NaN
    when ``interface_cross_shared_fraction`` is 0 (no cross query had support).
    """
    report = evaluate_saved_surrogate_comparison(
        reference, candidate, interface_eta_count=config.comparison_interface_eta_count,
        bulk_grid_counts=config.comparison_bulk_grid_counts, phase_mapping=case.PHASE_MAPPING)
    error = "symmetric_relative_frobenius_error"
    summary = {
        "phase_labels": report["phase_labels"]["mapping"],
        "max_endpoint_displacement": report["tieline"]["summary"]["maximum_endpoint_displacement"],
        "interface_own_paths_max": {p: r["own_paths"]["summary"][error]["maximum"] for p, r in report["interface"]["phases"].items()},
        "interface_cross_shared_fraction": report["interface"]["summary"]["cross_shared_fraction"],
        "interface_at_reference_compositions_max": {
            p: r["at_reference_compositions"]["summary"][error]["maximum"] for p, r in report["interface"]["phases"].items()},
        "bulk_shared_max": {p: r["comparison"]["summary"][error]["maximum"] for p, r in report["bulk"]["phases"].items()},
        "bulk_coverage": dict(report["bulk"]["summary"]["coverage_counts"]),
        "build_spec_differences": [item["path"] for item in report["build_spec_differences"]],
    }
    return summary, report

def check_options(thermocalc, *, tc_fresh_queries, case):
    """Refuse inconsistent or unavailable live Thermo-Calc requests before any work starts."""
    if thermocalc not in {"off", "saved", "build"}:
        raise ValueError("thermocalc must be 'off', 'saved' or 'build'.")
    if tc_fresh_queries and thermocalc != "build":
        raise case.ThermoCalcUnavailableError(
            "--tc-fresh-queries queries Thermo-Calc live, so it needs --thermocalc build with TC-Python and a "
            "licence; saved bundles are checked offline with the saved-bundle diagnostics and the comparison.")
    if thermocalc == "build":
        case.require_thermocalc()

def _timed(timings, key, function, *, case):
    """Record elapsed seconds for one workflow step and return its result."""
    start = time.perf_counter()
    value = function()
    timings[key] = round(time.perf_counter() - start, 3)
    return value

def _backend_steps(label, bundle, action, config, fresh_source, metrics, reports, models, timings, *, case):
    """Saved-bundle diagnostics, optional fresh queries and the simulation for one bundle."""
    surrogate = bundle.surrogate
    record_path = (bundle.member_path("kinetics_diagnostics")
                   if "kinetics_diagnostics" in bundle.manifest["members"] else None)
    entry = metrics[label] = {"bundle": str(bundle.path), "action": action,
                              "build_fingerprint": bundle.manifest["build_fingerprint"]}
    entry["saved_bundle"], reports[f"{label}_saved"] = case._timed(
        timings, f"{label}_saved_diagnostics", lambda: case.saved_bundle_diagnostics(surrogate, config))
    reports[f"{label}_saved"]["sampled_diffusivities"] = {
        "surrogate": surrogate, "kinetics_diagnostics": record_path}
    if fresh_source is not None:
        entry["fresh_queries"], reports[f"{label}_fresh"] = case._timed(
            timings, f"{label}_fresh_queries", lambda: case.fresh_query_diagnostics(surrogate, fresh_source, config))
    models[label], entry["simulation"] = case._timed(timings, f"{label}_simulation", lambda: case.simulate(config, surrogate))

def run(config, *, tdb, output_dir, rebuild, thermocalc, thermocalc_bundle, tc_cache_dir, pycalphad_fresh_queries, tc_fresh_queries, case):
    """Run the whole workflow; returns an :class:`ExampleResult`.

    Options are checked first (``check_options``), so an unavailable live Thermo-Calc request
    stops before any bundle folder is created. In ``build`` mode the Thermo-Calc session is
    started before anything else for the same reason. ``saved`` never imports ``tc_python``.
    """
    case.check_options(thermocalc, tc_fresh_queries=tc_fresh_queries)
    tdb = case.resolve_database(tdb)
    metrics, reports, models, timings = {"database_sha256": case.database_digest(tdb)}, {}, {}, {}
    metrics["timings_s"] = timings
    with ExitStack() as stack:
        tc_bundle = None
        if thermocalc == "build":
            cache = case.resolve_tc_cache(tc_cache_dir, output_dir)
            therm = case._timed(timings, "thermocalc_session_start", lambda: stack.enter_context(case.open_thermocalc(tdb, cache)))
            tc_bundle, tc_action = case._timed(timings, "thermocalc_bundle", lambda: case.prepare_thermocalc_bundle(
                config, tdb, therm, output_dir=output_dir, rebuild=rebuild))
            case._backend_steps("thermocalc", tc_bundle, tc_action, config, therm if tc_fresh_queries else None,
                           metrics, reports, models, timings)
            metrics["thermocalc"]["tc_cache_dir"] = str(cache)
        elif thermocalc == "saved":
            path = Path(thermocalc_bundle) if thermocalc_bundle else Path(output_dir) / "thermocalc"
            tc_bundle = case._timed(timings, "thermocalc_bundle", lambda: case.load_saved_thermocalc_bundle(path, config, tdb))
            case._backend_steps("thermocalc", tc_bundle, "loaded saved", config, None, metrics, reports, models, timings)
    pyc_bundle, pyc_action = case._timed(timings, "pycalphad_bundle", lambda: case.prepare_pycalphad_bundle(
        config, tdb, output_dir=output_dir, rebuild=rebuild))
    fresh = case._timed(timings, "pycalphad_fresh_source", lambda: case.build_source(tdb)) if pycalphad_fresh_queries else None
    case._backend_steps("pycalphad", pyc_bundle, pyc_action, config, fresh, metrics, reports, models, timings)
    if tc_bundle is not None:
        metrics["comparison"], reports["comparison"] = case._timed(
            timings, "comparison", lambda: case.compare_bundles(pyc_bundle, tc_bundle, config))
    timings["total"] = round(sum(timings.values()), 3)
    return case.ExampleResult(config, models["pycalphad"], None, metrics, reports, models)

def report(result, *, case):
    """Print the summaries, saying what each kind of result can show."""
    print("Saved-bundle diagnostics (no backend): provenance, leave-one-out cross-validation, consistency heuristic.")
    print("Fresh queries (same backend): approximation discrepancy; training and held-out queries are separate.")
    print("Comparison: differences between the two surrogates; neither backend is ground truth.")
    print("Simulation: inventory drift over the history, final-state coverage and interface position; "
          "not an accuracy check.")
    print(json.dumps(result.metrics, indent=2, default=float))

def figures(result, *, renderer, case):
    """Plotly figures for sampled values, leave-one-out, fresh queries and comparison.

    Needs the optional ``moving-boundary[diagnostics]`` extra; without it the diagnostics
    plotting functions raise an ImportError naming it. Returns ``{name: figure}``.
    """
    from moving_boundary.surrogate_comparison import plot_saved_surrogate_comparison

    output = {}
    for label in ("pycalphad", "thermocalc"):
        if f"{label}_saved" not in result.reports:
            continue
        sampled = result.reports[f"{label}_saved"]["sampled_diffusivities"]
        output[f"{label} sampled diffusivities"] = diagnostics.plot_selected_diffusivity_calculations(
            sampled["surrogate"], sampled["kinetics_diagnostics"], renderer=renderer)
        loo = result.reports[f"{label}_saved"]["leave_one_out"]
        for phase in loo["phases"]:
            output[f"{label} leave-one-out {phase}"] = diagnostics.plot_diffusivity_leave_one_out(loo, phase, renderer=renderer)
        if f"{label}_fresh" in result.reports:
            output[f"{label} tie lines"] = diagnostics.plot_tieline_diagnostics(
                result.reports[f"{label}_fresh"]["tielines"], renderer=renderer)
    if "comparison" in result.reports:
        output["comparison"] = plot_saved_surrogate_comparison(result.reports["comparison"], renderer=renderer)
    return output

def settings_from_command_line(settings, *, case):
    """Apply command-line options from a terminal; inside a notebook kernel, change nothing.

    ``settings`` is a dict with the Settings-cell names in lower case; a modified copy is returned.
    """
    if "ipykernel" in sys.modules:
        return dict(settings)
    parser = argparse.ArgumentParser(description=case.description.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--tdb", default=settings["tdb"], help=f"database (default: bundled; or set {case.DATABASE_ENV})")
    parser.add_argument("--output-dir", type=Path, default=settings["output_dir"], help="where bundles are kept")
    parser.add_argument("--rebuild", action="store_true", help="rebuild and replace existing bundles")
    parser.add_argument("--thermocalc", choices=("off", "saved", "build"), default=settings["thermocalc"])
    parser.add_argument("--thermocalc-bundle", type=Path, default=settings["thermocalc_bundle"],
                        help="saved Thermo-Calc bundle (default: <output-dir>/thermocalc)")
    parser.add_argument("--tc-cache-dir", type=Path, default=settings["tc_cache_dir"],
                        help=f"short Thermo-Calc cache folder (or set {case.TC_CACHE_ENV})")
    parser.add_argument("--no-pycalphad-fresh-queries", action="store_true", help="skip fresh pycalphad queries")
    parser.add_argument("--tc-fresh-queries", action="store_true", help="fresh Thermo-Calc queries (live, build mode)")
    parser.add_argument("--plot", action="store_true", help="show Plotly figures (needs the diagnostics extra)")
    args = parser.parse_args()
    try:
        case.resolve_database(args.tdb)
        case.check_options(args.thermocalc, tc_fresh_queries=args.tc_fresh_queries)
    except (FileNotFoundError, case.ThermoCalcUnavailableError) as exc:
        parser.error(str(exc))
    return {**settings, "tdb": args.tdb, "output_dir": args.output_dir, "rebuild": args.rebuild,
            "thermocalc": args.thermocalc, "thermocalc_bundle": args.thermocalc_bundle,
            "tc_cache_dir": args.tc_cache_dir, "pycalphad_fresh_queries": not args.no_pycalphad_fresh_queries,
            "tc_fresh_queries": args.tc_fresh_queries, "plot": args.plot}
