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

from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import tempfile
import time

import numpy as np
from scipy.spatial import Delaunay

from moving_boundary import surrogate_diagnostics as diagnostics
from moving_boundary.analytic import recorded_history
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
DATABASE = Path(__file__).resolve().parent / "FeCrNi_MatCalc_open.tdb"
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


def _project_to_simplex(z):
    """Row-wise Euclidean projection onto ``{z >= 0, sum(z) = 1}`` (sort-based)."""
    ordered = -np.sort(-z, axis=1)
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    active = np.count_nonzero(ordered - cumulative / np.arange(1, z.shape[1] + 1) > 0, axis=1)
    shift = cumulative[np.arange(len(z)), active - 1] / active
    return np.maximum(z - shift[:, None], 0.0)


def simplex_bulk_points(spacing, floor=1.0e-5):
    """Regular ``[X_CR, X_NI]`` grid over the ternary simplex, every component >= ``floor``.

    The same grid as ``ternary_fecrni_lee1996.simplex_bulk_points``: grid points are projected
    onto the buffered simplex so all three components are treated alike, kept in grid order
    and rounded to 10 decimals.
    """
    count = int(round(1.0 / spacing))
    grid = np.array([[i, j, count - i - j] for i in range(count + 1) for j in range(count + 1 - i)], dtype=np.float64) / count
    scale = 1.0 - 3.0 * floor
    z = np.maximum((grid - floor) / scale, 0.0)
    z = np.where(z.sum(axis=1, keepdims=True) > 1.0, _project_to_simplex(z), z)
    return np.round(floor + scale * z, 10)[:, :2]


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
    """Reload strictly, or build/publish samples plus hash-checked calculation records.

    Existing bundles without a diagnostics sidecar remain usable; only new builds capture
    records. ``build(records)`` appends records from the actual source calculations.
    """
    bundle_path = Path(bundle_path)
    if not rebuild:
        hint = f"Use --rebuild (REBUILD = True) to replace {bundle_path}."
        if (bundle_path / "manifest.json").exists():
            try:
                return prepare_surrogate_artifact(bundle_path, spec, _no_builder, reload=True, required_members=ROLES), "reloaded"
            except (SurrogateArtifactCompatibilityError, SurrogateArtifactIntegrityError) as exc:
                raise type(exc)(f"{exc}\n{hint}") from exc
        if bundle_path.is_dir() and any(bundle_path.iterdir()):
            raise SurrogateArtifactIntegrityError(f"{bundle_path} has files but no manifest.json.\n{hint}")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        def builder():
            records = []
            surrogate = build(records)
            members = {}
            for role in ROLES:
                report = Path(directory) / f"{role}.json"
                report.write_text(json.dumps(surrogate.metadata[role], indent=2) + "\n", encoding="utf-8")
                members[role] = {"path": report, "schema_version": 1}
            record_path = Path(directory) / "kinetics_diagnostics.jsonl"
            with record_path.open("w", encoding="utf-8") as stream:
                stream.write(json.dumps({"record_type": "metadata", "schema_version": 1,
                                         "backend": spec["source"]["engine"], "elements": ELEMENTS}) + "\n")
                for record in records:
                    stream.write(json.dumps({**record, "backend": spec["source"]["engine"]}) + "\n")
            members["kinetics_diagnostics"] = {"path": record_path, "schema_version": 1}
            return surrogate, members
        return prepare_surrogate_artifact(bundle_path, spec, builder, required_members=ROLES), "built"


def _no_builder():
    """A strict reload never builds."""
    raise AssertionError("A strict reload must not call its builder.")


def prepare_pycalphad_bundle(config, tdb, *, output_dir=DEFAULT_OUTPUT, rebuild=False):
    """Build or strictly reload the pycalphad bundle in ``output_dir/pycalphad``; ``(bundle, action)``."""
    tdb = resolve_database(tdb)
    return _publish_or_reload(Path(output_dir) / "pycalphad", pycalphad_spec(config, tdb),
                              lambda records: sample_surrogate(build_source(tdb), PHASES, config, kinetics_records=records), rebuild)


def require_thermocalc():
    """Stop before any work if TC-Python cannot be imported; never imports it here."""
    try:
        available = importlib.util.find_spec("tc_python") is not None
    except (ImportError, ValueError):
        available = False
    if not available:
        raise ThermoCalcUnavailableError(
            "Live Thermo-Calc work (--thermocalc build or --tc-fresh-queries) needs TC-Python (tc_python) and a "
            "Thermo-Calc licence. Without them, use --thermocalc saved with an existing bundle, or --thermocalc off.")


def resolve_tc_cache(path=None, output_dir=DEFAULT_OUTPUT):
    """Short Thermo-Calc cache folder, separate from the bundles; never changes machine settings."""
    cache = Path(path or os.environ.get(TC_CACHE_ENV) or DEFAULT_TC_CACHE).resolve()
    if cache.is_relative_to(Path(output_dir).resolve()):
        raise ValueError(f"Keep the Thermo-Calc cache ({cache}) outside the bundle folder {output_dir}.")
    if len(str(cache)) > TC_CACHE_MAX_CHARS:
        raise ValueError(f"Thermo-Calc cache path is {len(str(cache))} characters ({cache}); Thermo-Calc truncates "
                         f"long workspace paths, so use --tc-cache-dir or {TC_CACHE_ENV} with at most "
                         f"{TC_CACHE_MAX_CHARS} characters.")
    return cache


def open_thermocalc(tdb, cache_dir):
    """Start a Thermo-Calc session for the open database (needs TC-Python and a licence).

    Uses the suffixed composition-set labels, which the adapter matches exactly. For a user
    database that declares VA without entering it, the adapter selects VA as a system element
    itself (physical components stay FE, CR, NI).
    """
    from moving_boundary.providers.thermocalc import TCPythonThermodynamics, ThermoCalcConfig

    config = ThermoCalcConfig(user_database_path=Path(tdb), elements=tuple(ELEMENTS), phases=tuple(TC_PHASES),
                              reference_element="FE", use_default_phases=False, cache_dir=Path(cache_dir))
    return TCPythonThermodynamics(config)


def prepare_thermocalc_bundle(config, tdb, therm, *, output_dir=DEFAULT_OUTPUT, rebuild=False):
    """Build or strictly reload (after a live spec check) the bundle in ``output_dir/thermocalc``."""
    tdb = resolve_database(tdb)
    return _publish_or_reload(Path(output_dir) / "thermocalc", thermocalc_spec(config, tdb, therm),
                              lambda records: sample_surrogate(therm, TC_PHASES, config, kinetics_records=records), rebuild)


def load_saved_thermocalc_bundle(path, config, tdb):
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
    expected = canonicalize_build_spec(shared_spec(config, resolve_database(tdb), TC_PHASES))
    differences = [key for key, value in expected.items() if recorded.get(key) != value]
    if recorded.get("source", {}).get("engine") != "thermocalc":
        differences.append("source.engine")
    if differences:
        raise SurrogateArtifactCompatibilityError(
            f"Saved Thermo-Calc bundle {path} does not match these settings ({', '.join(differences)}). {hint}")
    return bundle


# %% [markdown]
# ## Diagnostics and comparison

# %%

def _stats(values):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {"count": int(finite.size), "max": float(finite.max()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None}


def _split(values, training):
    """Summaries of ``values`` at training and at held-out queries, reported separately."""
    values, training = np.asarray(values, dtype=np.float64), np.asarray(training, dtype=bool)
    return {"training": _stats(values[training]), "held_out": _stats(values[~training])}


def loo_indices(surrogate, config):
    """At most ``loo_max_per_phase`` evenly spread rows of each phase's fitted bulk samples."""
    counts = {phase: len(surrogate._fitSupport[phase]["points"]) for phase in surrogate.tieline_phases}
    return {phase: np.unique(np.linspace(0, n - 1, min(config.loo_max_per_phase, n)).round().astype(np.int64))
            for phase, n in counts.items()}


def saved_bundle_diagnostics(surrogate, config):
    """Diagnostics from the saved bundle alone: no backend is queried.

    Returns ``(summary, reports)``: construction provenance counts, leave-one-out (internal
    cross-validation on at most ``loo_max_per_phase`` samples per phase) and the heuristic
    interface/bulk consistency flags.
    """
    construction = diagnostics.evaluate_surrogate_construction_diagnostics(surrogate)
    loo = diagnostics.evaluate_diffusivity_leave_one_out(surrogate, sample_indices=loo_indices(surrogate, config))
    consistency = diagnostics.evaluate_interface_bulk_diffusivity_consistency(surrogate)
    summary = {
        "tie_lines": int(len(surrogate.eta_samples)),
        "fit_points": {phase: int(len(surrogate._fitSupport[phase]["points"])) for phase in surrogate.tieline_phases},
        "construction_records": len(construction["records"]),
        "leave_one_out": {phase: dict(report["summary"]) for phase, report in loo["phase_reports"].items()},
        "consistency": dict(consistency["summary"]),
    }
    return summary, {"construction": construction, "leave_one_out": loo, "consistency": consistency}


def fresh_query_diagnostics(surrogate, source, config):
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
                             "endpoint_error": {phase: _split(values, training_eta) for phase, values in
                                                tielines["truth"]["endpoint_error_norm"].items()}},
               "interface_matrices": {}, "bulk_matrices": {}, "failures": len(matrices["failures"])}
    for phase, report in matrices["interface"]["phases"].items():
        summary["interface_matrices"][phase] = _split(report["truth"]["matrix_relative_error"], report["training_mask"])
    for phase, report in matrices["bulk"]["phases"].items():
        training = np.asarray(report["nearest_training_distance"]) <= TRAINING_DISTANCE
        summary["bulk_matrices"][phase] = {**_split(report["truth"]["matrix_relative_error"], training),
                                           "invalid_predictions": int(np.count_nonzero(~report["valid"]))}
    return summary, {"tielines": tielines, "diffusivity": matrices}


def compare_bundles(reference, candidate, config):
    """Compare the pycalphad (reference) and Thermo-Calc (candidate) saved bundles.

    Uses ``PHASE_MAPPING`` because the backends label the same phases differently. A summary
    of own-path geometry, matrices on own paths and at identical compositions, and bulk
    coverage is returned with the full report. Identical-composition interface maxima are NaN
    when ``interface_cross_shared_fraction`` is 0 (no cross query had support).
    """
    report = evaluate_saved_surrogate_comparison(
        reference, candidate, interface_eta_count=config.comparison_interface_eta_count,
        bulk_grid_counts=config.comparison_bulk_grid_counts, phase_mapping=PHASE_MAPPING)
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


def inventory_drift(inventory):
    """Largest solute-inventory drift over every recorded time and component.

    ``inventory`` has one row per recorded time and one column per independent component;
    row 0 is the initial inventory. Returns ``(absolute, relative)``: the maximum of
    ``|I(t, k) - I(0, k)|`` (inventory units, here mole fraction times m) and the maximum of
    that difference divided by ``|I(0, k)|``. An excursion that later returns to the initial
    value is still counted.
    """
    inventory = np.atleast_2d(np.asarray(inventory, dtype=np.float64))
    change = np.abs(inventory - inventory[0])
    return float(np.max(change)), float(np.max(change / np.abs(inventory[0])))


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
    outside = {}
    for phase, state in zip(surrogate.tieline_phases, model.getTransformedState(float(times[-1]))):
        points = np.asarray(surrogate._fitSupport[phase]["points"], dtype=np.float64)
        outside[phase] = int(np.count_nonzero(Delaunay(points).find_simplex(np.asarray(state)) < 0))
    positions = np.ravel(positions)
    return model, {"final_time": float(times[-1]), "recorded_steps": int(len(times)),
                   "initial_interface_m": float(positions[0]), "final_interface_m": float(positions[-1]),
                   "final_eta": float(model.getInterfaceEta()),
                   "inventory_abs_max": absolute_drift, "inventory_rel_max": relative_drift,
                   "final_state_outside_hull": outside}


# %% [markdown]
# ## Running

# %%

def check_options(thermocalc="off", *, tc_fresh_queries=False):
    """Refuse inconsistent or unavailable live Thermo-Calc requests before any work starts."""
    if thermocalc not in {"off", "saved", "build"}:
        raise ValueError("thermocalc must be 'off', 'saved' or 'build'.")
    if tc_fresh_queries and thermocalc != "build":
        raise ThermoCalcUnavailableError(
            "--tc-fresh-queries queries Thermo-Calc live, so it needs --thermocalc build with TC-Python and a "
            "licence; saved bundles are checked offline with the saved-bundle diagnostics and the comparison.")
    if thermocalc == "build":
        require_thermocalc()


def _timed(timings, key, function):
    start = time.perf_counter()
    value = function()
    timings[key] = round(time.perf_counter() - start, 3)
    return value


def _backend_steps(label, bundle, action, config, fresh_source, metrics, reports, models, timings):
    """Saved-bundle diagnostics, optional fresh queries and the simulation for one bundle."""
    surrogate = bundle.surrogate
    record_path = (bundle.member_path("kinetics_diagnostics")
                   if "kinetics_diagnostics" in bundle.manifest["members"] else None)
    entry = metrics[label] = {"bundle": str(bundle.path), "action": action,
                              "build_fingerprint": bundle.manifest["build_fingerprint"]}
    entry["saved_bundle"], reports[f"{label}_saved"] = _timed(
        timings, f"{label}_saved_diagnostics", lambda: saved_bundle_diagnostics(surrogate, config))
    reports[f"{label}_saved"]["sampled_diffusivities"] = {
        "surrogate": surrogate, "kinetics_diagnostics": record_path}
    if fresh_source is not None:
        entry["fresh_queries"], reports[f"{label}_fresh"] = _timed(
            timings, f"{label}_fresh_queries", lambda: fresh_query_diagnostics(surrogate, fresh_source, config))
    models[label], entry["simulation"] = _timed(timings, f"{label}_simulation", lambda: simulate(config, surrogate))


def run(config=WORKFLOW, *, tdb=None, output_dir=DEFAULT_OUTPUT, rebuild=False, thermocalc="off",
        thermocalc_bundle=None, tc_cache_dir=None, pycalphad_fresh_queries=True, tc_fresh_queries=False):
    """Run the whole workflow; returns an :class:`ExampleResult`.

    Options are checked first (``check_options``), so an unavailable live Thermo-Calc request
    stops before any bundle folder is created. In ``build`` mode the Thermo-Calc session is
    started before anything else for the same reason. ``saved`` never imports ``tc_python``.
    """
    check_options(thermocalc, tc_fresh_queries=tc_fresh_queries)
    tdb = resolve_database(tdb)
    metrics, reports, models, timings = {"database_sha256": database_digest(tdb)}, {}, {}, {}
    metrics["timings_s"] = timings
    with ExitStack() as stack:
        tc_bundle = None
        if thermocalc == "build":
            cache = resolve_tc_cache(tc_cache_dir, output_dir)
            therm = _timed(timings, "thermocalc_session_start", lambda: stack.enter_context(open_thermocalc(tdb, cache)))
            tc_bundle, tc_action = _timed(timings, "thermocalc_bundle", lambda: prepare_thermocalc_bundle(
                config, tdb, therm, output_dir=output_dir, rebuild=rebuild))
            _backend_steps("thermocalc", tc_bundle, tc_action, config, therm if tc_fresh_queries else None,
                           metrics, reports, models, timings)
            metrics["thermocalc"]["tc_cache_dir"] = str(cache)
        elif thermocalc == "saved":
            path = Path(thermocalc_bundle) if thermocalc_bundle else Path(output_dir) / "thermocalc"
            tc_bundle = _timed(timings, "thermocalc_bundle", lambda: load_saved_thermocalc_bundle(path, config, tdb))
            _backend_steps("thermocalc", tc_bundle, "loaded saved", config, None, metrics, reports, models, timings)
    pyc_bundle, pyc_action = _timed(timings, "pycalphad_bundle", lambda: prepare_pycalphad_bundle(
        config, tdb, output_dir=output_dir, rebuild=rebuild))
    fresh = _timed(timings, "pycalphad_fresh_source", lambda: build_source(tdb)) if pycalphad_fresh_queries else None
    _backend_steps("pycalphad", pyc_bundle, pyc_action, config, fresh, metrics, reports, models, timings)
    if tc_bundle is not None:
        metrics["comparison"], reports["comparison"] = _timed(
            timings, "comparison", lambda: compare_bundles(pyc_bundle, tc_bundle, config))
    timings["total"] = round(sum(timings.values()), 3)
    return ExampleResult(config, models["pycalphad"], None, metrics, reports, models)


def report(result):
    """Print the summaries, saying what each kind of result can show."""
    print("Saved-bundle diagnostics (no backend): provenance, leave-one-out cross-validation, consistency heuristic.")
    print("Fresh queries (same backend): approximation discrepancy; training and held-out queries are separate.")
    print("Comparison: differences between the two surrogates; neither backend is ground truth.")
    print("Simulation: inventory drift over the history, final-state coverage and interface position; "
          "not an accuracy check.")
    print(json.dumps(result.metrics, indent=2, default=float))


def figures(result, *, renderer=None):
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


def settings_from_command_line(settings):
    """Apply command-line options from a terminal; inside a notebook kernel, change nothing.

    ``settings`` is a dict with the Settings-cell names in lower case; a modified copy is returned.
    """
    if "ipykernel" in sys.modules:
        return dict(settings)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--tdb", default=settings["tdb"], help=f"database (default: bundled; or set {DATABASE_ENV})")
    parser.add_argument("--output-dir", type=Path, default=settings["output_dir"], help="where bundles are kept")
    parser.add_argument("--rebuild", action="store_true", help="rebuild and replace existing bundles")
    parser.add_argument("--thermocalc", choices=("off", "saved", "build"), default=settings["thermocalc"])
    parser.add_argument("--thermocalc-bundle", type=Path, default=settings["thermocalc_bundle"],
                        help="saved Thermo-Calc bundle (default: <output-dir>/thermocalc)")
    parser.add_argument("--tc-cache-dir", type=Path, default=settings["tc_cache_dir"],
                        help=f"short Thermo-Calc cache folder (or set {TC_CACHE_ENV})")
    parser.add_argument("--no-pycalphad-fresh-queries", action="store_true", help="skip fresh pycalphad queries")
    parser.add_argument("--tc-fresh-queries", action="store_true", help="fresh Thermo-Calc queries (live, build mode)")
    parser.add_argument("--plot", action="store_true", help="show Plotly figures (needs the diagnostics extra)")
    args = parser.parse_args()
    try:
        resolve_database(args.tdb)
        check_options(args.thermocalc, tc_fresh_queries=args.tc_fresh_queries)
    except (FileNotFoundError, ThermoCalcUnavailableError) as exc:
        parser.error(str(exc))
    return {**settings, "tdb": args.tdb, "output_dir": args.output_dir, "rebuild": args.rebuild,
            "thermocalc": args.thermocalc, "thermocalc_bundle": args.thermocalc_bundle,
            "tc_cache_dir": args.tc_cache_dir, "pycalphad_fresh_queries": not args.no_pycalphad_fresh_queries,
            "tc_fresh_queries": args.tc_fresh_queries, "plot": args.plot}


# %% [markdown]
# ## Run
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
