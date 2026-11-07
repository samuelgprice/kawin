"""How does a Fe-Cr-Ni couple evolve with composition-dependent diffusivity?

A Lee & Oh (1996) case B couple at 1373 K: a thin BCC layer against FCC. Thermodynamics
and mobilities come from a CALPHAD database through pycalphad, sampled once into a
reusable surrogate. This is the full workflow:

    select source -> sample surrogate -> save -> reload -> build model -> solve -> inspect

**You supply the database** with ``--tdb PATH`` (or ``MB_EXAMPLE_TDB``): a TDB with
FE, CR and NI, BCC_A2 and FCC_A1, and mobilities. This example was developed with a
Fe-Cr-Ni database assembled from Lee (1993) with Lee & Oh (1996) mobilities; see
``examples/README.md`` for why it isn't included yet.

**Configurations, and what each one can tell you:**

- ``LAGGED_DEMO`` (default) and ``LAGGED_FAST``: the real workflow, with bulk matrices
  from ``composition_dependent_lagged`` and ``simplex_positive_2x2`` interpolation.
  There is no exact solution, so these make **no accuracy claim**. They show that the
  workflow runs, conserves solute, stays inside the sampled data, and really uses
  composition-dependent matrices.
- ``FROZEN_ANALYTIC``: frozen matrices sampled from the same surrogate, on an
  effectively semi-infinite domain, compared with the exact similarity solution. This
  checks the numerics with realistic data and closure. It does **not** validate
  composition-dependent Fe-Cr-Ni physics.
- ``FROZEN_FAST``: a very short frozen-matrix run with a minimal surrogate.

**Ordering, basis and units.** Elements are ``[FE, CR, NI]`` with FE dependent;
compositions are ``[X_CR, X_NI]`` mole fractions. BCC_A2 lies left of FCC_A1. Lengths
are in m, times in s, temperature in K, diffusivities in m^2/s. The domain is half of
the symmetric experimental couple, so the left wall is a symmetry plane.

**Files.** The bundle goes in ``--output-dir`` (default
``./moving_boundary_example_outputs/fecrni_lee1996/<config>/``). A compatible bundle
is reloaded without sampling; an incompatible or damaged one stops with an error;
``--rebuild`` replaces it. Nothing is ever rebuilt silently.

**Settings you would change:** ``bulk_spacing`` and ``samples_per_side`` (sampling
cost against accuracy), node counts, ``semi_log_dt``, ``duration``, the far-field
compositions and ``terminal_thin_phase_policy`` (``"raise"`` stops at a thin-phase
limit, ``"continue"`` proceeds past it; neither changes the phase topology).

**Limitations.** These settings are deliberately coarse and not converged. A production
case B run uses about 101 tie lines, a 0.01 bulk grid (5151 points), 100/1000 nodes and
``semiLog_dt = 0.25 / 160`` to 1000 h. Assumes equal molar volumes, no Kirkendall flow
and a fixed temperature. With ``validity_policy="raise"`` a bulk query lacking fit
support raises an error; with ``"legacy"`` it silently falls back to the nearest
sample, so coverage is worth checking (see ``workflow_report``). Sampling also prints
progress from the surrogate builder.

Run ``python ternary_fecrni_lee1996.py --tdb PATH`` (about 40 s, measured on one
machine), or add ``--config LAGGED_FAST`` (about 15 s).
"""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import tempfile

import numpy as np
from scipy.spatial import Delaunay

from moving_boundary.analytic import (
    FixedMatrixTernaryDiffusivity,
    compare_model_to_similarity,
    recorded_history,
    semi_infinite_validity_time,
    solve_similarity_roots,
)
from moving_boundary.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from moving_boundary.mesh.TransformedGrids import two_phase_landau_grids
from moving_boundary.providers.pycalphad import create_pycalphad_thermodynamics_source
from moving_boundary.solvers import MovingBoundaryIllingworthTernaryFD1DModel
from moving_boundary.surrogate_artifacts import (
    SurrogateArtifactBundle,
    SurrogateArtifactCompatibilityError,
    prepare_surrogate_artifact,
)
from moving_boundary.surrogates import TernaryMovingBoundaryThermodynamicsSurrogate

ELEMENTS = ["FE", "CR", "NI"]
INDEPENDENT = ["CR", "NI"]
PHASES = ["BCC_A2", "FCC_A1"]  # left, right
TEMPERATURE = 1373.0
PROBE_POINT = np.array([0.45, 0.21])  # a composition inside the two-phase region
A_BULK = np.array([0.39696, 0.0001])  # far-field [X_CR, X_NI] in BCC_A2 (case B)
B_BULK = np.array([0.139297, 0.142387])  # far-field [X_CR, X_NI] in FCC_A1 (case B)
HALF_LENGTH = 2000.0e-6 + 124.1e-6 / 2.0
B_PHASE_LENGTH = 2000.0e-6
SIZING_TIME = 100.0 * 3600.0  # semi-infinite domains are sized for this time
DEFAULT_OUTPUT = Path("moving_boundary_example_outputs") / "fecrni_lee1996"
DATABASE_ENV = "MB_EXAMPLE_TDB"


@dataclass(frozen=True)
class Config:
    """One workflow configuration; see the module docstring for what each can show.

    ``bulk_spacing`` is the composition grid for bulk diffusivities; ``None`` samples
    only the two far fields, which can't support simplex interpolation.
    """

    name: str
    frozen: bool = False
    bulk_spacing: float | None = 0.05
    samples_per_side: int = 8
    interpolation: str = "simplex_positive_2x2"
    validity_policy: str = "raise"
    left_nodes: int = 21
    right_nodes: int = 41
    mesh_nodes: int = 61
    semi_log_dt: float = 0.05
    fixed_dt: float | None = None
    duration: float = 10.0 * 3600.0
    terminal_thin_phase_width: float = 1.0e-7
    semi_infinite: bool = False
    resize_after_probe: bool = False
    compare_from: float = 1.0e4


@dataclass(frozen=True)
class ExampleResult:
    """A solved example: its settings, model, exact solution (if any) and named metrics."""

    config: Config
    model: object
    exact: object
    metrics: dict


LAGGED_FAST = Config("LAGGED_FAST")
LAGGED_DEMO = Config("LAGGED_DEMO", bulk_spacing=0.02, samples_per_side=16, left_nodes=41, right_nodes=161,
                     semi_log_dt=0.025, duration=100.0 * 3600.0)
FROZEN_ANALYTIC = Config("FROZEN_ANALYTIC", frozen=True, left_nodes=50, right_nodes=200, semi_log_dt=0.04,
                         duration=SIZING_TIME, terminal_thin_phase_width=1.0e-9, semi_infinite=True)
FROZEN_FAST = Config("FROZEN_FAST", frozen=True, bulk_spacing=None, samples_per_side=2, interpolation="nearest",
                     validity_policy="legacy", mesh_nodes=401, fixed_dt=600.0, duration=3600.0,
                     terminal_thin_phase_width=1.0e-9, semi_infinite=True, resize_after_probe=True)
CONFIGS = {config.name: config for config in (LAGGED_DEMO, LAGGED_FAST, FROZEN_ANALYTIC, FROZEN_FAST)}

# Numerical accuracy limits for FROZEN_ANALYTIC only: 50/200 nodes, semi_log_dt=0.04, t to
# 100 h, window t >= 1e4 s, frozen matrices on a semi-infinite domain. k_rel: |k-k_exact| /
# |k_exact|. interface_rel: max|s-s_exact| / max|s_exact-s0|. eta_abs: max|eta-eta_exact|.
# profile_linf: max|X-X_exact| at five recorded times. inventory_rel: |final-initial| / initial.
FROZEN_ANALYTIC_LIMITS = {"k_rel": 5.0e-3, "interface_rel": 1.0e-2, "eta_abs": 2.0e-3,
                          "profile_linf": 2.0e-2, "inventory_rel": 1.0e-8}

# Workflow properties for LAGGED_*: thresholds, not accuracy claims. Spread is
# (max-min)/max|entry| of a phase's matrix entries over accepted states, largest entry.
# Interpolation is the median relative difference from the nearest training matrix at
# centroids of the simplices the accepted path crosses. Lagged-vs-frozen is max|s_lagged -
# s_frozen| / max|s_frozen - s0| on identical geometry, closure, grids and time schedule.
LAGGED_THRESHOLDS = {"inventory_rel_max": 1.0e-8, "matrix_spread_min": 0.2, "simplices_crossed_min": 3,
                     "interpolation_vs_nearest_min": 1.0e-3, "lagged_vs_frozen_min": 5.0e-3}


# ---------------------------------------------------------------------------
# Thermodynamic source and surrogate
# ---------------------------------------------------------------------------

def resolve_database(tdb=None):
    """Return the database path from the argument or ``MB_EXAMPLE_TDB``, or explain what's needed."""
    tdb = tdb or os.environ.get(DATABASE_ENV)
    if not tdb:
        raise FileNotFoundError(f"This example needs a thermodynamic database: pass --tdb PATH or set {DATABASE_ENV}.")
    path = Path(tdb)
    if not path.is_file():
        raise FileNotFoundError(f"Thermodynamic database not found: {path}")
    return path


def database_digest(path):
    """SHA-256 with line endings normalized, so one database hashes alike on every platform."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build_source(tdb):
    """Select the thermodynamic source. Thermo-Calc users would create their provider here."""
    return create_pycalphad_thermodynamics_source(tdb, ELEMENTS, PHASES, use_default_phases=False, g_offset=0.0)


def simplex_bulk_points(spacing, floor=1.0e-5):
    """Regular grid of ``[X_CR, X_NI]`` over the ternary simplex, kept ``floor`` from its edges."""
    count = int(round(1.0 / spacing))
    points = np.array([[i * spacing, j * spacing] for i in range(count + 1) for j in range(count + 1 - i)])
    points = np.maximum(points, floor)
    over = points.sum(axis=1) > 1.0 - floor
    points[over] *= (1.0 - floor) / points[over].sum(axis=1, keepdims=True)
    return np.unique(points, axis=0)


def _bulk_points(config):
    """The requested bulk points; the surrogate adds tie-line endpoints and drops unusable ones."""
    return np.stack((A_BULK, B_BULK)) if config.bulk_spacing is None else simplex_bulk_points(config.bulk_spacing)


def sample_surrogate(config, tdb):
    """Sample tie lines from the probe point and bulk diffusivities. This is the slow step."""
    return TernaryMovingBoundaryThermodynamicsSurrogate.from_database(
        thermodynamics=build_source(tdb),
        elements=ELEMENTS,
        phases=PHASES,
        tieline_phases=PHASES,
        temperature=TEMPERATURE,
        probe_point=PROBE_POINT,
        probe_samples_per_side=config.samples_per_side,
        precipitate_phase=PHASES[1],
        diffusivity_bulk_points=_bulk_points(config),
        diffusivity_interpolation=config.interpolation,
        skip_failed_bulk_calculations=config.bulk_spacing is not None,
        drop_invalid_bulk_matrices=config.bulk_spacing is not None,
        validity_policy=config.validity_policy,
        min_composition=1.0e-10,
    )


def build_spec(config, tdb):
    """Everything that defines the sampled data; a reload requires an exact match."""
    return {
        "system": {"elements": ELEMENTS, "phases": PHASES, "temperature": TEMPERATURE},
        "source": {"engine": "pycalphad", "version": importlib.metadata.version("pycalphad"), "database_sha256": database_digest(tdb)},
        "tie_lines": {"probe_point": PROBE_POINT, "samples_per_side": config.samples_per_side},
        "bulk": {"points": _bulk_points(config), "interpolation": config.interpolation, "validity_policy": config.validity_policy},
    }


def prepare_surrogate(config, tdb, *, output_dir=DEFAULT_OUTPUT, rebuild=False):
    """Build or strictly reload the verified bundle; returns ``(bundle, action)``.

    Dropped bulk points are stored as reports inside the bundle when they can occur.
    """
    tdb = resolve_database(tdb)
    bundle_path = Path(output_dir) / config.name
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    spec = build_spec(config, tdb)
    roles = () if config.bulk_spacing is None else ("failed_bulk_points", "invalid_bulk_points")
    if not rebuild and (bundle_path / "manifest.json").exists():
        try:
            return prepare_surrogate_artifact(bundle_path, spec, _no_builder, reload=True, required_members=roles), "reloaded"
        except SurrogateArtifactCompatibilityError as exc:
            raise SurrogateArtifactCompatibilityError(f"{exc}\nUse --rebuild to replace {bundle_path}.") from exc
    with tempfile.TemporaryDirectory() as directory:
        def builder():
            surrogate = sample_surrogate(config, tdb)
            members = {}
            for role in roles:
                report = Path(directory) / f"{role}.json"
                report.write_text(json.dumps(surrogate.metadata[role], indent=2) + "\n", encoding="utf-8")
                members[role] = {"path": report, "schema_version": 1}
            return surrogate, members
        return prepare_surrogate_artifact(bundle_path, spec, builder, required_members=roles), "built"


def _no_builder():
    """A strict reload never builds."""
    raise AssertionError("A strict reload must not call its builder.")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@contextmanager
def _legacy_validity(surrogate):
    """Temporarily allow interface queries at any eta, for probing outside normal solve use."""
    previous = surrogate.validity_policy
    surrogate.validity_policy = "legacy"
    try:
        yield surrogate
    finally:
        surrogate.validity_policy = previous


def interface_matrices(surrogate, eta):
    """The surrogate's interface matrices at the tie-line ends for ``eta``, keyed by phase."""
    left, right = surrogate.interface_compositions(float(eta))
    with _legacy_validity(surrogate):
        return {
            PHASES[0]: np.asarray(surrogate.getInterdiffusivity(left, TEMPERATURE, phase=PHASES[0], query_context="interface"), dtype=np.float64).reshape(2, 2),
            PHASES[1]: np.asarray(surrogate.getInterdiffusivity(right, TEMPERATURE, phase=PHASES[1], query_context="interface"), dtype=np.float64).reshape(2, 2),
        }


def semi_infinite_geometry(matrices):
    """``(s0, domain_length)`` with each phase ``8 sqrt(lambda_max t)`` wide for ``SIZING_TIME``."""
    widths = [8.0 * np.sqrt(float(np.max(np.real(np.linalg.eigvals(matrices[phase])))) * SIZING_TIME) for phase in PHASES]
    return widths[0] + 1.0e-12, widths[0] + widths[1]  # nudged off the physical mesh nodes


def case_geometry():
    """``(s0, domain_length)`` of the real half-couple, with the interface off the mesh nodes."""
    return HALF_LENGTH - B_PHASE_LENGTH + 1.0e-12, HALF_LENGTH


def _time_options(config):
    """Fixed steps for a very short run, otherwise a log-spaced schedule."""
    if config.fixed_dt is not None:
        return {"time_step": config.fixed_dt, "dt_mode": "fixed"}
    return {"time_step": 1.0, "dt_mode": "semi_log", "semiLog_dt": config.semi_log_dt, "semiLogT0": 1.0e-6}


def make_model(config, thermodynamics, closure, s0, length, mode, *, time_options=None):
    """The two-phase model; the closure is always the surrogate's tie-line family."""
    mesh = CartesianFD1D(INDEPENDENT, [0.0, length], config.mesh_nodes)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(s0, A_BULK, B_BULK), INDEPENDENT)]))
    u_grid, v_grid = two_phase_landau_grids(config.left_nodes, config.right_nodes, method="linear", spacing_ratio=0.1)
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=ELEMENTS,
        phases=PHASES,
        thermodynamics=thermodynamics,
        temperature=TEMPERATURE,
        interfacePosition=s0,
        interface_equilibrium=closure,
        initial_eta_method="instantaneous_balance",
        initial_eta_bracket=(1.0e-3, 1.0 - 1.0e-3),
        initial_eta_guess=0.5,
        bulk_diffusivity_mode=mode,
        tolerance=1.0e-12,
        max_iterations=25,
        max_step_retries=8,
        terminal_thin_phase_width=config.terminal_thin_phase_width,
        terminal_thin_phase_policy="raise",
        record=True,
        record_pq_data=True,
        transformed_u_grid=u_grid,
        transformed_v_grid=v_grid,
        **(time_options or _time_options(config)),
    )


def freeze_matrices(config, surrogate):
    """Freeze matrices at the solver's own initial tie line; returns ``(matrices, s0, length)``.

    Sizes the domain with mid-tie-line matrices, finds the initial tie line with a probe
    model, then freezes those matrices. ``FROZEN_FAST`` resizes the domain afterwards;
    the analytic run keeps the size so the initial tie line stays consistent with it.
    """
    s0, length = semi_infinite_geometry(interface_matrices(surrogate, 0.5))
    probe_options = {"time_step": 1.0, "dt_mode": "semi_log", "semiLog_dt": config.semi_log_dt, "semiLogT0": 1.0e-6}
    with _legacy_validity(surrogate):
        probe = make_model(config, surrogate, surrogate, s0, length, "phase_uniform", time_options=probe_options)
        probe.setup()
        matrices = interface_matrices(surrogate, probe.getInterfaceEta())
    if config.resize_after_probe:
        s0, length = semi_infinite_geometry(matrices)
    return matrices, s0, length


def build_model(config=LAGGED_DEMO, surrogate=None, *, thermodynamics=None, tdb=None, output_dir=DEFAULT_OUTPUT):
    """Build the model for one configuration; nothing is solved yet.

    ``surrogate`` is prepared from the database if omitted. ``thermodynamics`` optionally
    replaces the bulk provider of a lagged model, such as with a recording wrapper.
    """
    if surrogate is None:
        surrogate = prepare_surrogate(config, tdb, output_dir=output_dir)[0].surrogate
    if config.frozen:
        matrices, s0, length = freeze_matrices(config, surrogate)
        return make_model(config, FixedMatrixTernaryDiffusivity(matrices, PHASES), surrogate, s0, length, "phase_uniform")
    s0, length = case_geometry()
    return make_model(config, surrogate if thermodynamics is None else thermodynamics, surrogate, s0, length,
                      "composition_dependent_lagged")


# ---------------------------------------------------------------------------
# Inspecting a lagged run
# ---------------------------------------------------------------------------

class RecordingDiffusivity:
    """Passes queries through and records attempted bulk compositions per phase.

    Attempted queries include retried and rejected steps, so ``workflow_report`` uses
    them only for a coverage check; claims about accepted steps use accepted states.
    """

    def __init__(self, inner):
        self.inner = inner
        self.attempted = {phase: [] for phase in PHASES}

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def getInterdiffusivity(self, x, T=None, phase=None, query_context=None, **kwargs):
        if query_context != "interface":
            self.attempted[phase].append(np.atleast_2d(np.asarray(x, dtype=np.float64)).copy())
        return self.inner.getInterdiffusivity(x, T, phase=phase, query_context=query_context, **kwargs)


def retained_general_samples(surrogate, phase):
    """The samples actually fitted: requested points plus tie-line ends, minus unusable ones.

    Reads a private helper because no public accessor exposes the fitted set.
    """
    return surrogate._fit_diffusivity_samples("general", phase)


def _accepted_states(model, max_times=60):
    """Compositions at nodes and midpoints of accepted states, for each phase."""
    times, _ = recorded_history(model.interfaceData)
    states = {phase: [] for phase in PHASES}
    for time in times[np.unique(np.linspace(0, len(times) - 1, min(len(times), max_times)).astype(int))]:
        for phase, state in zip(PHASES, model.getTransformedState(float(time))):
            states[phase].append(np.vstack((state, 0.5 * (state[1:] + state[:-1]))))
    return {phase: np.vstack(values) for phase, values in states.items()}


def workflow_report(model, surrogate, recorder=None):
    """Check that the lagged run stayed inside its data and really used it, per phase.

    Builds its own triangulation from the retained samples. Interpolation evidence
    comes from centroids of the simplices crossed by accepted states: the returned
    matrix should differ from the nearest training matrix there, and lie within its
    simplex's vertex range, while a training point returns its own matrix.
    """
    report = {"interpolation": surrogate.diffusivityInterpolation}
    accepted = _accepted_states(model)
    for phase in PHASES:
        points, matrices = retained_general_samples(surrogate, phase)
        triangulation = Delaunay(points)
        path = accepted[phase]
        inside = triangulation.find_simplex(path) >= 0
        values = np.asarray(surrogate.getInterdiffusivity(path[inside], TEMPERATURE, phase=phase, query_context="general"), dtype=np.float64)
        entries = values.reshape(-1, 4)
        spread = (entries.max(axis=0) - entries.min(axis=0)) / np.maximum(np.abs(entries).max(axis=0), 1.0e-300)
        simplices = np.unique(triangulation.find_simplex(path[inside]))
        vertices = triangulation.simplices[simplices]
        centroids = points[vertices].mean(axis=1)
        returned = np.asarray(surrogate.getInterdiffusivity(centroids, TEMPERATURE, phase=phase, query_context="general"), dtype=np.float64)
        nearest = matrices[np.argmin(np.linalg.norm(points[None, :, :] - centroids[:, None, :], axis=2), axis=1)]
        versus_nearest = np.max(np.abs(returned - nearest), axis=(1, 2)) / np.max(np.abs(nearest), axis=(1, 2))
        low, high = matrices[vertices].min(axis=1), matrices[vertices].max(axis=1)
        slack = 1.0e-9 * np.maximum(np.abs(low), np.abs(high))
        node = vertices[0]
        recovered = np.asarray(surrogate.getInterdiffusivity(points[node], TEMPERATURE, phase=phase, query_context="general"), dtype=np.float64)
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


def lagged_vs_frozen(model, surrogate, config):
    """Compare with frozen matrices, using identical geometry, closure, grids and schedule.

    Returns ``max|s_lagged - s_frozen| / max|s_frozen - s0|``.
    """
    s0, length = case_geometry()
    frozen = make_model(config, FixedMatrixTernaryDiffusivity(interface_matrices(surrogate, model.initialEta), PHASES),
                        surrogate, s0, length, "phase_uniform")
    frozen.solve(config.duration, minDtFrac=1e-14)
    times, lagged_s = recorded_history(model.interfaceData)
    frozen_times, frozen_s = recorded_history(frozen.interfaceData)
    if not np.array_equal(times, frozen_times):
        raise ValueError("Lagged and frozen runs must use the same time schedule.")
    return float(np.max(np.abs(lagged_s - frozen_s)) / np.max(np.abs(frozen_s - frozen_s[0])))


def inventory_relative_drift(model):
    """Largest component ``|final - initial| / initial`` of the solute inventory."""
    _, inventory = recorded_history(model.inventoryData)
    return float(np.max(np.abs(inventory[-1] - inventory[0]) / np.abs(inventory[0])))


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def exact_solution(model, surrogate, matrices=None):
    """The exact solution for frozen matrices, using the root closest to the initial tie line."""
    matrices = interface_matrices(surrogate, model.initialEta) if matrices is None else matrices
    roots = solve_similarity_roots(matrices[PHASES[0]], matrices[PHASES[1]], A_BULK, B_BULK, surrogate, s0=model.initialInterfacePosition)
    return min(roots, key=lambda root: abs(root.eta - float(model.initialEta)))


def run(config=LAGGED_DEMO, *, tdb=None, output_dir=DEFAULT_OUTPUT, rebuild=False, surrogate=None):
    """Prepare the surrogate (unless given), solve, and report what this configuration can show."""
    metrics = {}
    if surrogate is None:
        bundle, action = prepare_surrogate(config, tdb, output_dir=output_dir, rebuild=rebuild)
        surrogate = bundle.surrogate
        metrics.update(surrogate_action=action, bundle=str(bundle.path), build_fingerprint=bundle.manifest["build_fingerprint"])
    if config.frozen:
        matrices, s0, length = freeze_matrices(config, surrogate)
        model = make_model(config, FixedMatrixTernaryDiffusivity(matrices, PHASES), surrogate, s0, length, "phase_uniform")
        model.solve(config.duration, minDtFrac=1e-14)
        metrics["inventory_rel"] = inventory_relative_drift(model)
        metrics["validity_time_left"] = semi_infinite_validity_time(matrices[PHASES[0]], s0)
        metrics["validity_time_right"] = semi_infinite_validity_time(matrices[PHASES[1]], length - s0)
        exact = None
        if config.duration > config.compare_from:
            exact = exact_solution(model, surrogate, matrices)
            comparison = compare_model_to_similarity(model, exact, t_min=config.compare_from)
            metrics.update(k_rel=comparison["k_rel_error"], interface_rel=comparison["interface_rel_error"],
                           eta_abs=comparison["eta_max_abs_error"], profile_linf=float(np.max(comparison["profile_linf"])))
        return ExampleResult(config, model, exact, metrics)
    recorder = RecordingDiffusivity(surrogate)
    model = build_model(config, surrogate, thermodynamics=recorder)
    model.solve(config.duration, minDtFrac=1e-14)
    metrics["inventory_rel_max"] = inventory_relative_drift(model)
    metrics["workflow"] = workflow_report(model, surrogate, recorder)
    metrics["lagged_vs_frozen"] = lagged_vs_frozen(model, surrogate, config)
    return ExampleResult(config, model, None, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tdb", help=f"thermodynamic database (or set {DATABASE_ENV})")
    parser.add_argument("--config", choices=sorted(CONFIGS), default=LAGGED_DEMO.name)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="where bundles are kept")
    parser.add_argument("--rebuild", action="store_true", help="rebuild and replace an existing bundle")
    args = parser.parse_args()
    try:
        result = run(CONFIGS[args.config], tdb=args.tdb, output_dir=args.output_dir, rebuild=args.rebuild)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    print(f"final interface {result.model.getInterfacePosition():.6e} m, eta {result.model.getInterfaceEta():.6f}")
    print(json.dumps(result.metrics, indent=2, default=float))
