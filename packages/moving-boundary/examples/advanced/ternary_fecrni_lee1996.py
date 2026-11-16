"""Advanced checks for ``ternary_fecrni_lee1996``; see that example for assumptions and references.

The Case supplies explicit physical inputs and callbacks. No main script
is imported, and no solve or file operation runs during import.
"""

from dataclasses import dataclass
from moving_boundary.plotting import plot_interface_history
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from scipy.spatial import Delaunay
from moving_boundary.analytic import FixedMatrixTernaryDiffusivity, compare_model_to_similarity, recorded_history, semi_infinite_validity_time


# Numerical accuracy limits for FROZEN_ANALYTIC only: 50/200 nodes, semi_log_dt=0.04, t to
# 100 h, window t >= 1e4 s, frozen matrices on a semi-infinite domain. k_rel: |k-k_exact| /
# |k_exact|. interface_rel: max|s-s_exact| / max|s_exact-s0|. eta_abs: max|eta-eta_exact|.
# profile_linf: max|X-X_exact| at five recorded times. inventory_rel: |final-initial| / initial.
FROZEN_ANALYTIC_LIMITS = {"k_rel": 5.0e-3, "interface_rel": 1.0e-2, "eta_abs": 2.0e-3,
                          "profile_linf": 2.0e-2, "inventory_rel": 1.0e-8}

# Workflow properties for LAGGED_FAST and LAGGED_DEMO only: thresholds, not accuracy claims.
# Spread is (max-min)/max|entry| of a phase's matrix entries over accepted states, largest
# entry. Interpolation is the median relative difference from the nearest training matrix at
# centroids of the simplices the accepted path crosses. Lagged-vs-frozen is max|s_lagged -
# s_frozen| / max|s_frozen - s0| on identical geometry, closure, grids and time schedule.
# Outside-hull counts accepted or attempted bulk compositions outside the retained samples'
# hull; retained fraction is retained samples / requested grid points (it can exceed 1, since
# tie-line endpoints are added and unusable points dropped); centroid range is the
# fraction of crossed-simplex centroids whose matrix lies within its vertices' range.
LAGGED_THRESHOLDS = {"inventory_rel_max": 1.0e-8, "matrix_spread_min": 0.2, "simplices_crossed_min": 3,
                     "interpolation_vs_nearest_min": 1.0e-3, "lagged_vs_frozen_min": 5.0e-3,
                     "outside_hull_max": 0, "retained_fraction_min": 0.9,
                     "centroids_within_vertex_range_min": 0.9, "training_point_relative_error_max": 1.0e-8}


@dataclass(frozen=True)
class Case:
    """Inputs and callbacks supplied by the main example at call time."""

    CONFIGS: object
    DATABASE_ENV: object
    ExampleResult: object
    FROZEN_ANALYTIC: object
    FROZEN_ANALYTIC_LIMITS: object
    LAGGED_THRESHOLDS: object
    PHASES: object
    RecordingDiffusivity: object
    TEMPERATURE: object
    THRESHOLD_CONFIGS: object
    description: str
    _accepted_states: object
    build_model: object
    case_geometry: object
    check_limits: object
    exact_solution: object
    freeze_matrices: object
    interface_matrices: object
    inventory_relative_drift: object
    lagged_vs_frozen: object
    make_model: object
    prepare_surrogate: object
    resolve_database: object
    retained_general_samples: object
    simplex_bulk_points: object
    workflow_report: object


def _accepted_states(model, max_times, *, case):
    """Compositions at nodes and midpoints of accepted states, for each phase."""
    times, _ = recorded_history(model.interfaceData)
    states = {phase: [] for phase in case.PHASES}
    for time in times[np.unique(np.linspace(0, len(times) - 1, min(len(times), max_times)).astype(int))]:
        for phase, state in zip(case.PHASES, model.getTransformedState(float(time))):
            states[phase].append(np.vstack((state, 0.5 * (state[1:] + state[:-1]))))
    return {phase: np.vstack(values) for phase, values in states.items()}

def workflow_report(model, surrogate, recorder, *, case):
    """Check that the lagged run stayed inside its data and really used it, per phase.

    Builds its own triangulation from the retained samples. Interpolation evidence
    comes from centroids of the simplices crossed by accepted states: the returned
    matrix should differ from the nearest training matrix there, and lie within its
    simplex's vertex range, while a training point returns its own matrix.
    """
    report = {"interpolation": surrogate.diffusivityInterpolation}
    accepted = case._accepted_states(model)
    for phase in case.PHASES:
        points, matrices = case.retained_general_samples(surrogate, phase)
        triangulation = Delaunay(points)
        path = accepted[phase]
        inside = triangulation.find_simplex(path) >= 0
        values = np.asarray(surrogate.getInterdiffusivity(path[inside], case.TEMPERATURE, phase=phase, query_context="general"), dtype=np.float64)
        entries = values.reshape(-1, 4)
        spread = (entries.max(axis=0) - entries.min(axis=0)) / np.maximum(np.abs(entries).max(axis=0), 1.0e-300)
        simplices = np.unique(triangulation.find_simplex(path[inside]))
        vertices = triangulation.simplices[simplices]
        centroids = points[vertices].mean(axis=1)
        returned = np.asarray(surrogate.getInterdiffusivity(centroids, case.TEMPERATURE, phase=phase, query_context="general"), dtype=np.float64)
        nearest = matrices[np.argmin(np.linalg.norm(points[None, :, :] - centroids[:, None, :], axis=2), axis=1)]
        versus_nearest = np.max(np.abs(returned - nearest), axis=(1, 2)) / np.max(np.abs(nearest), axis=(1, 2))
        low, high = matrices[vertices].min(axis=1), matrices[vertices].max(axis=1)
        slack = 1.0e-9 * np.maximum(np.abs(low), np.abs(high))
        node = vertices[0]
        recovered = np.asarray(surrogate.getInterdiffusivity(points[node], case.TEMPERATURE, phase=phase, query_context="general"), dtype=np.float64)
        report[phase] = {
            "retained_samples": len(points),
            "accepted_outside_hull": int(np.count_nonzero(~inside)),
            "matrix_spread": float(np.max(spread)),
            "simplices_crossed": len(simplices),
            "interpolation_vs_nearest_median": float(np.median(versus_nearest)),
            "centroids_within_vertex_range": float(np.mean(np.all((returned >= low - slack) & (returned <= high + slack), axis=(1, 2)))),
            "training_point_relative_error": float(np.max(np.abs(recovered - matrices[node]) / np.abs(matrices[node]))),
        }
        if recorder is not None:
            attempted = np.vstack(recorder.attempted[phase]) if recorder.attempted[phase] else np.empty((0, 2))
            report[phase]["attempted_queries"] = len(attempted)
            report[phase]["attempted_outside_hull"] = int(np.count_nonzero(triangulation.find_simplex(attempted) < 0)) if len(attempted) else 0
    return report

def lagged_vs_frozen(model, surrogate, config, *, case):
    """Compare with frozen matrices, using identical geometry, closure, grids and schedule.

    Returns ``max|s_lagged - s_frozen| / max|s_frozen - s0|``.
    """
    s0, length = case.case_geometry()
    frozen = case.make_model(config, FixedMatrixTernaryDiffusivity(case.interface_matrices(surrogate, model.initialEta), case.PHASES),
                        surrogate, s0, length, "phase_uniform")
    frozen.solve(config.duration, minDtFrac=config.min_dt_frac)
    times, lagged_s = recorded_history(model.interfaceData)
    frozen_times, frozen_s = recorded_history(frozen.interfaceData)
    if not np.array_equal(times, frozen_times):
        raise ValueError("Lagged and frozen runs must use the same time schedule.")
    return float(np.max(np.abs(lagged_s - frozen_s)) / np.max(np.abs(frozen_s - frozen_s[0])))

def run(config, *, tdb, output_dir, rebuild, surrogate, case):
    """Prepare the surrogate (unless given), solve, and report what this configuration can show.

    Lagged runs add ``lagged_vs_frozen`` to the metrics only when ``config.compare_frozen``.
    """
    metrics = {}
    if surrogate is None:
        bundle, action = case.prepare_surrogate(config, tdb, output_dir=output_dir, rebuild=rebuild)
        surrogate = bundle.surrogate
        metrics.update(surrogate_action=action, bundle=str(bundle.path), build_fingerprint=bundle.manifest["build_fingerprint"])
    if config.frozen:
        matrices, s0, length = case.freeze_matrices(config, surrogate)
        model = case.make_model(config, FixedMatrixTernaryDiffusivity(matrices, case.PHASES), surrogate, s0, length, "phase_uniform")
        model.solve(config.duration, minDtFrac=config.min_dt_frac)
        metrics["inventory_rel"] = case.inventory_relative_drift(model)
        metrics["validity_time_left"] = semi_infinite_validity_time(matrices[case.PHASES[0]], s0)
        metrics["validity_time_right"] = semi_infinite_validity_time(matrices[case.PHASES[1]], length - s0)
        exact = None
        if config.duration > config.compare_from:
            exact = case.exact_solution(model, surrogate, matrices)
            comparison = compare_model_to_similarity(model, exact, t_min=config.compare_from)
            metrics.update(k_rel=comparison["k_rel_error"], interface_rel=comparison["interface_rel_error"],
                           eta_abs=comparison["eta_max_abs_error"], profile_linf=float(np.max(comparison["profile_linf"])))
        return case.ExampleResult(config, model, exact, metrics)
    recorder = case.RecordingDiffusivity(surrogate)
    model = case.build_model(config, surrogate, thermodynamics=recorder)
    model.solve(config.duration, minDtFrac=config.min_dt_frac)
    metrics["inventory_rel_max"] = case.inventory_relative_drift(model)
    metrics["workflow"] = case.workflow_report(model, surrogate, recorder)
    if config.compare_frozen:
        metrics["lagged_vs_frozen"] = case.lagged_vs_frozen(model, surrogate, config)
    return case.ExampleResult(config, model, None, metrics)

def check_limits(result, *, case):
    """Compare metrics with the limits that apply; returns ``(name, value, rule, passed)`` rows.

    Limits apply only to the configurations they were set for. ``FROZEN_ANALYTIC`` uses
    ``FROZEN_ANALYTIC_LIMITS`` (each metric at most its limit). ``LAGGED_FAST`` and
    ``LAGGED_DEMO`` (``THRESHOLD_CONFIGS``) use ``LAGGED_THRESHOLDS``: the interpolation mode,
    conservation, and per-phase coverage and interpolation evidence. Rows for
    ``lagged_vs_frozen`` and attempted queries appear only when those were measured. Any
    other configuration, including ``PRODUCTION_B`` and modified copies, returns no rows:
    no validated limits apply to it. The tests assert that every row passes.
    """
    metrics, rows = result.metrics, []
    compare = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
               ">=": lambda a, b: a >= b, "==": lambda a, b: a == b}

    def add(name, value, symbol, limit):
        rows.append((name, value, f"{symbol} {limit:g}" if isinstance(limit, (int, float)) else f"{symbol} {limit}",
                     bool(compare[symbol](value, limit))))

    if result.config == case.FROZEN_ANALYTIC:
        for name, limit in case.FROZEN_ANALYTIC_LIMITS.items():
            add(name, metrics[name], "<=", limit)
        return rows
    if result.config not in case.THRESHOLD_CONFIGS:
        return rows
    limits, workflow = case.LAGGED_THRESHOLDS, metrics["workflow"]
    add("interpolation", workflow["interpolation"], "==", result.config.interpolation)
    add("inventory_rel_max", metrics["inventory_rel_max"], "<", limits["inventory_rel_max"])
    requested = len(case.simplex_bulk_points(result.config.bulk_spacing))
    for phase in case.PHASES:
        phase_report = workflow[phase]
        add(f"{phase} retained_fraction", phase_report["retained_samples"] / requested, ">", limits["retained_fraction_min"])
        add(f"{phase} accepted_outside_hull", phase_report["accepted_outside_hull"], "<=", limits["outside_hull_max"])
        if "attempted_queries" in phase_report:
            add(f"{phase} attempted_queries", phase_report["attempted_queries"], ">", 0)
            add(f"{phase} attempted_outside_hull", phase_report["attempted_outside_hull"], "<=", limits["outside_hull_max"])
        add(f"{phase} matrix_spread", phase_report["matrix_spread"], ">", limits["matrix_spread_min"])
        add(f"{phase} simplices_crossed", phase_report["simplices_crossed"], ">=", limits["simplices_crossed_min"])
        add(f"{phase} interpolation_vs_nearest_median", phase_report["interpolation_vs_nearest_median"], ">",
            limits["interpolation_vs_nearest_min"])
        add(f"{phase} centroids_within_vertex_range", phase_report["centroids_within_vertex_range"], ">",
            limits["centroids_within_vertex_range_min"])
        add(f"{phase} training_point_relative_error", phase_report["training_point_relative_error"], "<",
            limits["training_point_relative_error_max"])
    if "lagged_vs_frozen" in metrics:
        add("lagged_vs_frozen", metrics["lagged_vs_frozen"], ">", limits["lagged_vs_frozen_min"])
    return rows

def report(result, *, case):
    """Print the final state, the metrics, and each applicable limit with whether it was met.

    Says so explicitly when no validated limits apply to the configuration.
    """
    print(f"final interface {result.model.getInterfacePosition():.6e} m, eta {result.model.getInterfaceEta():.6f}")
    print(json.dumps(result.metrics, indent=2, default=float))
    rows = case.check_limits(result)
    if not rows:
        print(f"No validated limits apply to {result.config.name} with these settings; the metrics have no pass/fail.")
        return
    print("accuracy limits (numerics only):" if result.config.frozen else "workflow thresholds (not accuracy):")
    for name, value, rule, passed in rows:
        shown = f"{value:.3g}" if isinstance(value, (int, float, np.floating, np.integer)) else str(value)
        print(f"  {name}: {shown} {rule}: {'within limit' if passed else 'OUTSIDE LIMIT'}")

def plot(result, *, case):
    """Plot the interface history and the final phase profiles; returns the figure.

    For ``FROZEN_ANALYTIC`` the exact interface is shown too.
    """
    import matplotlib.pyplot as plt

    model = result.model
    times, interface = recorded_history(model.interfaceData)
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    hours = times / 3600.0
    plot_interface_history(model, ax=left, time_scale=1.0 / 3600.0, position_scale=1.0e6, label="model")
    if result.exact is not None:
        left.plot(hours, result.exact.interface_position(times) * 1.0e6, "--", label="exact (frozen matrices)")
    left.set(xlabel="time (h)", ylabel="interface position (um)", title=f"{result.config.name}: interface")
    left.legend()
    for x, c in model.getPhysicalPhaseProfiles():
        right.plot(x * 1.0e6, c[:, 1], color="C0")
        right.plot(x * 1.0e6, c[:, 2], color="C1")
    right.axvline(float(model.getInterfacePosition()) * 1.0e6, color="0.6", lw=0.8)
    right.set(xlabel="position (um)", ylabel="mole fraction", title="Final profiles (BCC left, FCC right)")
    right.legend(["X_CR", "X_NI"])
    figure.tight_layout()
    return figure

def settings_from_command_line(tdb, config, output_dir, rebuild, plot_figures, *, case):
    """Apply command-line options from a terminal; inside a notebook kernel, change nothing.

    A Jupyter kernel's ``sys.argv`` holds the kernel's own arguments, so they are not parsed.
    """
    if "ipykernel" in sys.modules:
        return tdb, config, output_dir, rebuild, plot_figures
    parser = argparse.ArgumentParser(description=case.description.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--tdb", default=tdb, help=f"thermodynamic database (or set {case.DATABASE_ENV})")
    parser.add_argument("--config", choices=sorted(case.CONFIGS), default=config.name)
    parser.add_argument("--output-dir", type=Path, default=output_dir, help="where bundles are kept")
    parser.add_argument("--rebuild", action="store_true", help="rebuild and replace an existing bundle")
    parser.add_argument("--plot", action="store_true", help="show plots")
    args = parser.parse_args()
    try:
        case.resolve_database(args.tdb)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    return args.tdb, case.CONFIGS[args.config], args.output_dir, args.rebuild, args.plot


class RecordingDiffusivity:
    """Passes queries through and records attempted bulk compositions per phase.

    Attempted queries include retried and rejected steps, so ``workflow_report`` uses
    them only for a coverage check; claims about accepted steps use accepted states.
    """

    def __init__(self, inner, phases):
        self.inner = inner
        self.attempted = {phase: [] for phase in phases}

    def __getattr__(self, name):
        """Delegate provider metadata and methods without changing queries."""
        return getattr(self.inner, name)

    def getInterdiffusivity(self, x, T=None, phase=None, query_context=None, **kwargs):
        """Record attempted bulk compositions, including rejected/retried steps."""
        if query_context != "interface":
            self.attempted[phase].append(np.atleast_2d(np.asarray(x, dtype=np.float64)).copy())
        return self.inner.getInterdiffusivity(x, T, phase=phase, query_context=query_context, **kwargs)
