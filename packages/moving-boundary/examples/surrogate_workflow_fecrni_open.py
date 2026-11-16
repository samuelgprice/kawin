# %% [markdown]
# # How do I check, compare and use Fe-Cr-Ni surrogates built from an open database?
#
# One open Fe-Cr-Ni database, two thermodynamic backends: pycalphad (always) and, optionally,
# Thermo-Calc through TC-Python. For each backend this example
#
#     builds or reloads a surrogate bundle -> runs diagnostics -> runs a short simulation
#
# and then compares the two saved bundles. Physical case: 1373 K, the far-field compositions of
# couple B of Kajihara, Lim and Kikuchi [1] as used by Lee and Oh [2], a BCC_A2 layer against
# FCC_A1, on the real half-couple geometry.
#
# **Database.** `FeCrNi_MatCalc_open.tdb`, next to this file, is an existing adapted excerpt of
# the MatCalc open steel databases [3, 4], copied unchanged (ODbL 1.0 / DbCL 1.0; attribution
# and the excerpt's history are in `examples/README.md` and `NOTICE`). Another database can be
# passed with `--tdb` or `MB_OPEN_FECRNI_TDB`; it needs FE, CR, NI, BCC_A2, FCC_A1 and
# mobilities, and the compositions below may then need changing.
#
# **What each step can and cannot tell you.** Neither backend is ground truth.
#
# - *From a saved bundle alone* (no backend needed, so this also works for a saved Thermo-Calc
#   bundle without a licence): construction diagnostics (provenance of the sampling),
#   leave-one-out (internal cross-validation: each stored bulk matrix predicted from a refit
#   without it; not error against new data) and the interface/bulk consistency check (a
#   heuristic signal).
# - *Fresh queries of the same backend* (pycalphad by default; Thermo-Calc only with
#   `--tc-fresh-queries`): the surrogate's approximation discrepancy at the queried points. A
#   query that coincides with a training sample only shows that the stored value is reproduced
#   (exactly in the process that built the bundle; in a new process, to the equilibrium
#   solver's tolerance, about 1e-9 relative in development runs), so training and held-out
#   queries are reported separately.
# - *Saved-bundle comparison*: differences between the two surrogates. Tie lines are paired by
#   normalized eta, not by matched physical conditions; matrices are also compared at identical
#   compositions. The two backends label phases differently (Thermo-Calc composition sets are
#   `BCC_A2#1` and `FCC_A1#1`), so the comparison uses an explicit `phase_mapping`. With
#   `validity_policy="raise"` a surrogate answers interface queries only on its own tie-line
#   path (within its interface query tolerance, 1e-8), so when the two paths differ by more than
#   that, the identical-composition interface view is empty (`interface_cross_shared_fraction`
#   0) and identical-composition matrices are compared in the bulk only.
# - *Simulation*: a short lagged ternary run from each bundle, with the largest absolute and
#   relative solute-inventory drift over the recorded history, whether the final state lies
#   inside the sampled data, and the interface position. Similar interface motion or good
#   conservation does not establish surrogate accuracy or profile accuracy; there is no exact
#   solution here.
#
# **Thermo-Calc modes** (`--thermocalc`): `off` (default) uses pycalphad only. `saved` reuses an
# existing Thermo-Calc bundle (`<output>/thermocalc/` or `--thermocalc-bundle PATH`); it never
# imports `tc_python`. `build` builds, or reuses after a live check, a Thermo-Calc bundle and is
# the only mode that needs TC-Python and a licence. Thermo-Calc truncates long workspace paths,
# so its cache goes to a short folder of its own (`--tc-cache-dir`, `MB_TC_CACHE_DIR`, default
# `<system temp>/mb_fecrni_open_tc`), separate from the bundles. Thermo-Calc-derived bundles are
# local results; don't commit or redistribute them.
#
# **Ordering, basis and units.** Elements `[FE, CR, NI]` with FE dependent; compositions are
# `[X_CR, X_NI]` mole fractions. BCC_A2 lies left of FCC_A1. SI units. Diffusivity matrices are
# in the frame and basis each backend returns. During development, rebuilding Thermo-Calc's
# matrices with kawin's volume-fixed formula supported the same frame, FE reference and
# `[CR, NI]` basis at six tested points, within a residual of about 1e-6 to 6e-6 of the largest
# entry; that is not a general proof.
#
# **Files.** Bundles go to `~/moving_boundary_example_outputs/surrogate_workflow_fecrni_open/
# <backend>/` (`--output-dir` to change). A compatible bundle is reloaded; an incompatible or
# damaged one stops with an error, and `--rebuild` replaces it.
#
# **Limitations.** The settings (17 tie lines, a 0.05 bulk grid, 1 h simulated) are sized to run
# in about half a minute, not for converged results. Leave-one-out errors at these settings are
# of order 5% on average and up to about 30% for single samples (development run), so treat
# matrices between samples accordingly. Equal molar volumes, no Kirkendall flow, fixed
# temperature.
#
# **Running.** From a terminal: `python surrogate_workflow_fecrni_open.py [--thermocalc
# off|saved|build] [--plot]`, or run the cells in order in a Jupyter or VS Code Interactive
# window, editing the Settings cell first.
"""How do I check, compare and use Fe-Cr-Ni surrogates built from an open database? See the notes above."""

# %%

from pathlib import Path
import sys

# Run notebook cells with the extraction repository root as the working directory.
sys.path.append(str(Path("packages/moving-boundary/examples").resolve()))

from advanced import surrogate_workflow_fecrni_open as _advanced
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import tempfile
import time

import numpy as np
from scipy.spatial import Delaunay

from moving_boundary import surrogate_diagnostics as diagnostics
from moving_boundary.analytic import recorded_history
from moving_boundary.run_diagnostics import inventory_drift as measure_inventory_drift, evaluate_bulk_coverage
from moving_boundary.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from moving_boundary.mesh.TransformedGrids import two_phase_landau_grids
from moving_boundary.providers.pycalphad import create_pycalphad_thermodynamics_source
from moving_boundary.solvers import MovingBoundaryIllingworthTernaryFD1DModel
from moving_boundary.surrogate_artifacts import (
    SurrogateArtifactBundle,
    SurrogateArtifactCompatibilityError,
    SurrogateArtifactIntegrityError,
    canonicalize_build_spec,
    prepare_surrogate_artifact,
    load_or_build_surrogate_artifact,
    write_surrogate_diagnostic_members,
    load_or_build_diagnostic_surrogate,
)
from moving_boundary.surrogate_comparison import evaluate_saved_surrogate_comparison
from moving_boundary.surrogates import TernaryMovingBoundaryThermodynamicsSurrogate

ELEMENTS = ["FE", "CR", "NI"]
INDEPENDENT = ["CR", "NI"]
PHASES = ["BCC_A2", "FCC_A1"]  # left, right (pycalphad labels)
TC_PHASES = ["BCC_A2#1", "FCC_A1#1"]  # the same phases as Thermo-Calc composition sets
PHASE_MAPPING = dict(zip(PHASES, TC_PHASES))  # comparison: pycalphad (reference) -> Thermo-Calc
TEMPERATURE = 1373.0  # [1, 2]
PROBE_POINT = np.array([0.45, 0.21])  # inside the BCC_A2 + FCC_A1 field of this database at 1373 K
# Couple B [2, Table 3], from [1], as in ternary_fecrni_lee1996.py. Both are single-phase in this
# database at 1373 K (checked with pycalphad during development).
A_BULK = np.array([0.39696, 0.0001])  # far-field [X_CR, X_NI] in BCC_A2
B_BULK = np.array([0.139297, 0.142387])  # far-field [X_CR, X_NI] in FCC_A1
HALF_LENGTH = 2000.0e-6 + 124.1e-6 / 2.0  # half of the symmetric couple [2, Table 3]
B_PHASE_LENGTH = 2000.0e-6
DATABASE = Path(_advanced.__file__).resolve().parent.parent / "FeCrNi_MatCalc_open.tdb"
DATABASE_SHA256 = "f713f965a83292eaa91399a28720d13efb0298d6ef107dd5e6062bbe88d99061"  # bundled bytes
DATABASE_ENV = "MB_OPEN_FECRNI_TDB"
TC_CACHE_ENV = "MB_TC_CACHE_DIR"
TC_CACHE_MAX_CHARS = 150  # Thermo-Calc adds ~52 characters below this and truncates paths at 256
DEFAULT_OUTPUT = Path.home() / "moving_boundary_example_outputs" / "surrogate_workflow_fecrni_open"
DEFAULT_TC_CACHE = Path(tempfile.gettempdir()) / "mb_fecrni_open_tc"
TRAINING_DISTANCE = 1.0e-12  # a bulk query this close to a stored sample counts as training


@dataclass(frozen=True)
class Config:
    """Sampling, diagnostic and simulation settings.

    ``samples_per_side`` gives ``2 * samples_per_side + 1`` tie lines; ``bulk_spacing`` is the
    simplex grid of bulk diffusivity samples. Fresh queries use ``fresh_tieline_eta_count`` and
    ``fresh_interface_eta_count`` evenly spaced etas (their ends coincide with training tie
    lines, interior ones need not) and bulk compositions on ``fresh_bulk_axes``, chosen between
    the bulk grid nodes. ``loo_max_per_phase`` bounds the leave-one-out sample.
    """

    name: str = "WORKFLOW"
    samples_per_side: int = 8
    bulk_spacing: float = 0.05
    interpolation: str = "simplex_positive_2x2"
    validity_policy: str = "raise"
    loo_max_per_phase: int = 20
    fresh_tieline_eta_count: int = 8
    fresh_interface_eta_count: int = 6
    fresh_bulk_axes: tuple = ((0.175, 0.325, 0.475), (0.025, 0.125, 0.275))
    comparison_interface_eta_count: int = 21
    comparison_bulk_grid_counts: tuple = (11, 11)
    left_nodes: int = 21
    right_nodes: int = 41
    mesh_nodes: int = 61
    semi_log_dt: float = 0.05
    duration: float = 3600.0
    terminal_thin_phase_width: float = 1.0e-7
    terminal_thin_phase_policy: str = "raise"
    min_dt_frac: float = 1.0e-14


@dataclass
class ExampleResult:
    """A completed workflow: settings, the pycalphad model, no exact solution, and metrics.

    ``metrics`` holds JSON-friendly summaries per backend and for the comparison, plus
    ``timings_s``; ``reports`` holds the full diagnostic and comparison reports; ``models``
    holds the solved model of each backend.
    """

    config: Config
    model: object
    exact: object
    metrics: dict
    reports: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)


WORKFLOW = Config()

# %% [markdown]
# ## Settings
#
# `TDB = None` uses the bundled open database. `THERMOCALC` is `"off"`, `"saved"` or `"build"`.

# %%
TDB = None
CONFIG = WORKFLOW
OUTPUT_DIR = DEFAULT_OUTPUT
REBUILD = False
THERMOCALC = "off"
THERMOCALC_BUNDLE = None  # saved mode: defaults to OUTPUT_DIR / "thermocalc"
TC_CACHE_DIR = None  # build mode: defaults to MB_TC_CACHE_DIR or DEFAULT_TC_CACHE
PYCALPHAD_FRESH_QUERIES = True
TC_FRESH_QUERIES = False
PLOT = False


# %% [markdown]
# ## Database, sources and bundles

# %%

class ThermoCalcUnavailableError(RuntimeError):
    """Live Thermo-Calc work was requested where TC-Python or a licence is unavailable."""


def resolve_database(tdb=None):
    """Return the database: the argument, ``MB_OPEN_FECRNI_TDB`` or the bundled file."""
    tdb = tdb or os.environ.get(DATABASE_ENV) or DATABASE
    path = Path(tdb)
    if not path.is_file():
        raise FileNotFoundError(
            f"Thermodynamic database not found: {path}. Pass --tdb PATH or set {DATABASE_ENV} to a TDB with "
            "FE, CR, NI, BCC_A2, FCC_A1 and mobilities; without either, the bundled FeCrNi_MatCalc_open.tdb is used.")
    return path


def database_digest(path):
    """SHA-256 with line endings normalized, so one database hashes alike on every platform."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build_source(tdb):
    """The pycalphad source: restricted to BCC_A2 and FCC_A1, ``g_offset = 0``."""
    return create_pycalphad_thermodynamics_source(tdb, ELEMENTS, PHASES, use_default_phases=False, g_offset=0.0)


from moving_boundary.sampling import ternary_simplex_points


def simplex_bulk_points(spacing, floor=1.0e-5):
    """Return the package's ordered, buffered ternary composition grid."""
    return ternary_simplex_points(spacing, floor)


def sample_surrogate(source, phases, config, *, kinetics_records=None):
    """Sample tie lines from the probe point and forced-phase bulk diffusivities (the slow step).

    ``phases`` are the backend's own labels for BCC_A2 and FCC_A1, in that order.
    New builds may capture genuine selected-calculation records into ``kinetics_records``.
    Capture uses the same calculations; it never evaluates the fitted surrogate.
    """
    capture = nullcontext() if kinetics_records is None else source.captureKineticsDiagnostics(kinetics_records.append)
    with capture:
        return TernaryMovingBoundaryThermodynamicsSurrogate.from_database(
            thermodynamics=source, elements=ELEMENTS, phases=list(phases), tieline_phases=list(phases),
            temperature=TEMPERATURE, probe_point=PROBE_POINT, probe_samples_per_side=config.samples_per_side,
            precipitate_phase=phases[1], diffusivity_bulk_points=simplex_bulk_points(config.bulk_spacing),
            diffusivity_interpolation=config.interpolation, skip_failed_bulk_calculations=True,
            drop_invalid_bulk_matrices=True, validity_policy=config.validity_policy, min_composition=1.0e-10)


def shared_spec(config, tdb, phases):
    """Spec fields that need no backend: system, database, tie-line and bulk sampling settings.

    A saved Thermo-Calc bundle is accepted offline only if these match exactly.
    """
    return {
        "system": {"elements": ELEMENTS, "phases": list(phases), "temperature": TEMPERATURE},
        "database_sha256": database_digest(tdb),
        "tie_lines": {"probe_point": PROBE_POINT, "samples_per_side": config.samples_per_side},
        "bulk": {"points": simplex_bulk_points(config.bulk_spacing), "interpolation": config.interpolation,
                 "validity_policy": config.validity_policy},
    }


def pycalphad_spec(config, tdb):
    """Everything that defines the pycalphad bundle; a reload requires an exact match."""
    return {**shared_spec(config, tdb, PHASES),
            "source": {"engine": "pycalphad", "version": importlib.metadata.version("pycalphad")},
            "package": {"moving_boundary": importlib.metadata.version("moving-boundary")}}


def thermocalc_spec(config, tdb, therm):
    """Everything that defines the Thermo-Calc bundle, including the live system selection."""
    return {**shared_spec(config, tdb, TC_PHASES),
            "source": {"engine": "thermocalc", "tc_python": therm.getRuntimeVersion(),
                       "system_selection": therm.getSystemSelection()},
            "package": {"moving_boundary": importlib.metadata.version("moving-boundary")}}


ROLES = ("failed_bulk_points", "invalid_bulk_points")


def _publish_or_reload(bundle_path, spec, build, rebuild):
    """Strictly reuse or publish samples and genuine captured calculation records."""
    return load_or_build_diagnostic_surrogate(
        bundle_path, spec, build, rebuild=rebuild, report_roles=ROLES,
        capture_kinetics=True, backend=spec["source"]["engine"], elements=ELEMENTS)


def _no_builder():
    """A strict reload never builds."""
    raise AssertionError("A strict reload must not call its builder.")


def prepare_pycalphad_bundle(config, tdb, *, output_dir=DEFAULT_OUTPUT, rebuild=False):
    """Build or strictly reload the pycalphad bundle in ``output_dir/pycalphad``; ``(bundle, action)``."""
    tdb = resolve_database(tdb)
    return _publish_or_reload(Path(output_dir) / "pycalphad", pycalphad_spec(config, tdb),
                              lambda records: sample_surrogate(build_source(tdb), PHASES, config, kinetics_records=records), rebuild)


def prepare_thermocalc_bundle(config, tdb, therm, *, output_dir=DEFAULT_OUTPUT, rebuild=False):
    """Build or strictly reload (after a live spec check) the bundle in ``output_dir/thermocalc``."""
    tdb = resolve_database(tdb)
    return _publish_or_reload(Path(output_dir) / "thermocalc", thermocalc_spec(config, tdb, therm),
                              lambda records: sample_surrogate(therm, TC_PHASES, config, kinetics_records=records), rebuild)


# %% [markdown]
# ## Diagnostics and comparison

# %%


# %% [markdown]
# ## Simulation

# %%

def build_model(config=WORKFLOW, surrogate=None, *, tdb=None, output_dir=DEFAULT_OUTPUT):
    """The lagged two-phase model on the half-couple; nothing is solved yet.

    Uses the surrogate's own phase labels, so a Thermo-Calc bundle (``BCC_A2#1``, ``FCC_A1#1``)
    works unchanged. ``surrogate`` defaults to the pycalphad bundle.
    """
    if surrogate is None:
        surrogate = prepare_pycalphad_bundle(config, tdb, output_dir=output_dir)[0].surrogate
    phases = list(surrogate.tieline_phases)
    s0 = HALF_LENGTH - B_PHASE_LENGTH + 1.0e-12  # interface nudged off the mesh nodes
    mesh = CartesianFD1D(INDEPENDENT, [0.0, HALF_LENGTH], config.mesh_nodes)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(s0, A_BULK, B_BULK), INDEPENDENT)]))
    u_grid, v_grid = two_phase_landau_grids(config.left_nodes, config.right_nodes, method="linear", spacing_ratio=0.1)
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh, elements=ELEMENTS, phases=phases, thermodynamics=surrogate, temperature=TEMPERATURE,
        interfacePosition=s0, interface_equilibrium=surrogate, initial_eta_method="instantaneous_balance",
        initial_eta_bracket=(1.0e-3, 1.0 - 1.0e-3), initial_eta_guess=0.5,
        bulk_diffusivity_mode="composition_dependent_lagged", tolerance=1.0e-12, max_iterations=25,
        max_step_retries=8, terminal_thin_phase_width=config.terminal_thin_phase_width,
        terminal_thin_phase_policy=config.terminal_thin_phase_policy, record=True, record_pq_data=True,
        transformed_u_grid=u_grid, transformed_v_grid=v_grid, time_step=1.0, dt_mode="semi_log",
        semiLog_dt=config.semi_log_dt, semiLogT0=1.0e-6)


def run_core(config=WORKFLOW, *, tdb=None, output_dir=DEFAULT_OUTPUT, rebuild=False):
    """Prepare one pycalphad bundle, solve once and inspect conservation/coverage.

    No fresh-query, cross-validation or Thermo-Calc workflow runs here.
    """
    bundle, action = prepare_pycalphad_bundle(config, tdb, output_dir=output_dir, rebuild=rebuild)
    model, metrics = simulate(config, bundle.surrogate)
    metrics["action"] = action
    return ExampleResult(config, model, None, metrics)


def inventory_drift(inventory):
    """Largest absolute and component-relative drift over the whole history."""
    return measure_inventory_drift(inventory, reduction="max")


def simulate(config, surrogate):
    """Solve the lagged model; returns ``(model, summary)``.

    The summary holds the interface position and eta, the largest absolute and relative
    solute-inventory drift over the whole recorded history (``inventory_drift``), and per phase
    the number of compositions of the final state only that lie outside the hull of the fitted
    bulk samples (earlier states are not checked). These are workflow checks, not accuracy.
    """
    model = build_model(config, surrogate)
    model.solve(config.duration, minDtFrac=config.min_dt_frac)
    times, positions = recorded_history(model.interfaceData)
    _, inventory = recorded_history(model.inventoryData)
    absolute_drift, relative_drift = inventory_drift(inventory)
    coverage = evaluate_bulk_coverage(surrogate, dict(zip(
        surrogate.tieline_phases, model.getTransformedState(float(times[-1])))))
    outside = {phase: report["outside"] for phase, report in coverage.items()}
    positions = np.ravel(positions)
    return model, {"final_time": float(times[-1]), "recorded_steps": int(len(times)),
                   "initial_interface_m": float(positions[0]), "final_interface_m": float(positions[-1]),
                   "final_eta": float(model.getInterfaceEta()),
                   "inventory_abs_max": absolute_drift, "inventory_rel_max": relative_drift,
                   "final_state_outside_hull": outside}


# %% [markdown]
# ## Advanced checks and compatibility entry points
#
# The introductory path above needs only `run_core`. These wrappers retain
# the original detailed workflow and command-line behavior.

# %%
def require_thermocalc():
    """Stop before any work if TC-Python cannot be imported; never imports it here. See the advanced companion."""
    return _advanced.require_thermocalc(case=_advanced_case())


def resolve_tc_cache(path=None, output_dir=DEFAULT_OUTPUT):
    """Short Thermo-Calc cache folder, separate from the bundles; never changes machine settings. See the advanced companion."""
    return _advanced.resolve_tc_cache(path, output_dir, case=_advanced_case())


def open_thermocalc(tdb, cache_dir):
    """Start a Thermo-Calc session for the open database (needs TC-Python and a licence). See the advanced companion."""
    return _advanced.open_thermocalc(tdb, cache_dir, case=_advanced_case())


def load_saved_thermocalc_bundle(path, config, tdb):
    """Load a saved Thermo-Calc bundle without Thermo-Calc and check its backend-free settings. See the advanced companion."""
    return _advanced.load_saved_thermocalc_bundle(path, config, tdb, case=_advanced_case())


def _stats(values):
    """Run this step of the detailed workflow. See the advanced companion."""
    return _advanced._stats(values, case=_advanced_case())


def _split(values, training):
    """Summaries of ``values`` at training and at held-out queries, reported separately. See the advanced companion."""
    return _advanced._split(values, training, case=_advanced_case())


def loo_indices(surrogate, config):
    """At most ``loo_max_per_phase`` evenly spread rows of each phase's fitted bulk samples. See the advanced companion."""
    return _advanced.loo_indices(surrogate, config, case=_advanced_case())


def saved_bundle_diagnostics(surrogate, config):
    """Diagnostics from the saved bundle alone: no backend is queried. See the advanced companion."""
    return _advanced.saved_bundle_diagnostics(surrogate, config, case=_advanced_case())


def fresh_query_diagnostics(surrogate, source, config):
    """Fresh queries of the backend that built ``surrogate``: approximation discrepancy. See the advanced companion."""
    return _advanced.fresh_query_diagnostics(surrogate, source, config, case=_advanced_case())


def compare_bundles(reference, candidate, config):
    """Compare the pycalphad (reference) and Thermo-Calc (candidate) saved bundles. See the advanced companion."""
    return _advanced.compare_bundles(reference, candidate, config, case=_advanced_case())


def _advanced_case():
    """Supply current case inputs to the optional detailed workflow. See the advanced companion."""
    return _advanced.Case(
        DATABASE_ENV=DATABASE_ENV,
        DEFAULT_TC_CACHE=DEFAULT_TC_CACHE,
        ELEMENTS=ELEMENTS,
        ExampleResult=ExampleResult,
        PHASE_MAPPING=PHASE_MAPPING,
        TC_CACHE_ENV=TC_CACHE_ENV,
        TC_CACHE_MAX_CHARS=TC_CACHE_MAX_CHARS,
        TC_PHASES=TC_PHASES,
        TRAINING_DISTANCE=TRAINING_DISTANCE,
        ThermoCalcUnavailableError=ThermoCalcUnavailableError,
        description=__doc__,
        _backend_steps=_backend_steps,
        _split=_split,
        _stats=_stats,
        _timed=_timed,
        build_source=build_source,
        check_options=check_options,
        compare_bundles=compare_bundles,
        database_digest=database_digest,
        fresh_query_diagnostics=fresh_query_diagnostics,
        load_saved_thermocalc_bundle=load_saved_thermocalc_bundle,
        loo_indices=loo_indices,
        open_thermocalc=open_thermocalc,
        prepare_pycalphad_bundle=prepare_pycalphad_bundle,
        prepare_thermocalc_bundle=prepare_thermocalc_bundle,
        require_thermocalc=require_thermocalc,
        resolve_database=resolve_database,
        resolve_tc_cache=resolve_tc_cache,
        saved_bundle_diagnostics=saved_bundle_diagnostics,
        shared_spec=shared_spec,
        simulate=simulate,
    )


def check_options(thermocalc="off", *, tc_fresh_queries=False):
    """Refuse inconsistent or unavailable live Thermo-Calc requests before any work starts. See the advanced companion."""
    return _advanced.check_options(thermocalc, tc_fresh_queries=tc_fresh_queries, case=_advanced_case())


def _timed(timings, key, function):
    """Run this step of the detailed workflow. See the advanced companion."""
    return _advanced._timed(timings, key, function, case=_advanced_case())


def _backend_steps(label, bundle, action, config, fresh_source, metrics, reports, models, timings):
    """Saved-bundle diagnostics, optional fresh queries and the simulation for one bundle. See the advanced companion."""
    return _advanced._backend_steps(label, bundle, action, config, fresh_source, metrics, reports, models, timings, case=_advanced_case())


def run(config=WORKFLOW, *, tdb=None, output_dir=DEFAULT_OUTPUT, rebuild=False, thermocalc="off",
        thermocalc_bundle=None, tc_cache_dir=None, pycalphad_fresh_queries=True, tc_fresh_queries=False):
    """Run the whole workflow; returns an :class:`ExampleResult`. See the advanced companion."""
    return _advanced.run(config, tdb=tdb, output_dir=output_dir, rebuild=rebuild, thermocalc=thermocalc, thermocalc_bundle=thermocalc_bundle, tc_cache_dir=tc_cache_dir, pycalphad_fresh_queries=pycalphad_fresh_queries, tc_fresh_queries=tc_fresh_queries, case=_advanced_case())


def report(result):
    """Print the summaries, saying what each kind of result can show. See the advanced companion."""
    return _advanced.report(result, case=_advanced_case())


def figures(result, *, renderer=None):
    """Plotly figures for sampled values, leave-one-out, fresh queries and comparison. See the advanced companion."""
    return _advanced.figures(result, renderer=renderer, case=_advanced_case())


def settings_from_command_line(settings):
    """Apply command-line options from a terminal; inside a notebook kernel, change nothing. See the advanced companion."""
    return _advanced.settings_from_command_line(settings, case=_advanced_case())


# %% [markdown]
# ## Running

# %%


# %% [markdown]
# ## Run
#
# For the introductory path use `result = run_core(CONFIG)` after the Settings
# and model cells. The compatibility entry point below also runs the detailed
# checks in `advanced/surrogate_workflow_fecrni_open.py`.
#
# Builds or reloads each bundle, runs its diagnostics and simulation, compares the bundles when
# there are two, and prints the summaries with step timings.

# %%
if __name__ == "__main__":
    SETTINGS = settings_from_command_line(dict(
        tdb=TDB, output_dir=OUTPUT_DIR, rebuild=REBUILD, thermocalc=THERMOCALC, thermocalc_bundle=THERMOCALC_BUNDLE,
        tc_cache_dir=TC_CACHE_DIR, pycalphad_fresh_queries=PYCALPHAD_FRESH_QUERIES, tc_fresh_queries=TC_FRESH_QUERIES,
        plot=PLOT))
    result = run(CONFIG, **{key: value for key, value in SETTINGS.items() if key != "plot"})
    report(result)

# %% [markdown]
# Plot (optional; needs `moving-boundary[diagnostics]`).

# %%
if __name__ == "__main__" and SETTINGS["plot"]:
    for name, figure in figures(result, renderer="browser").items():
        figure.show()

# %% [markdown]
# ## References
#
# 1. M. Kajihara, C.-B. Lim, M. Kikuchi, "Experimental study on dissolution of alpha phase in
#    gamma/alpha/gamma diffusion couples of the Fe-Cr-Ni system," ISIJ International 33 (1993)
#    498-507. Source of the diffusion couples and their 1373 K anneals.
# 2. B.-J. Lee, K. H. Oh, "Numerical treatment of the moving interface in diffusional
#    reactions," Z. Metallkd. 87 (1996) 195-204. Table 3: couple B compositions and thicknesses,
#    from [1].
# 3. E. Povoden-Karadeniz, MatCalc thermodynamic steel database mc_fe, release 2.060, TU Wien,
#    MatCalc open databases, https://www.matcalc.at/index.php/databases/open-databases (ODbL 1.0 /
#    DbCL 1.0). Thermodynamic parameters of the bundled database.
# 4. E. Povoden-Karadeniz, MatCalc diffusion mobility database for steel (mc_fe), TU Wien, same
#    source and licence as [3]. Mobility parameters of the bundled database; they match releases
#    2.012, 2.015 and 2.016 alike, so no single release is identified.
