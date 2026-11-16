# %% [markdown]
# # How does a Fe-Cr-Ni couple evolve with composition-dependent diffusivity?
#
# Couple B of the gamma/alpha/gamma Fe-Cr-Ni diffusion couples of Kajihara, Lim and Kikuchi
# [1] at 1373 K, as simulated by Lee and Oh [2] (their Table 3): a thin BCC layer against FCC.
# Thermodynamics and mobilities come from a CALPHAD database through pycalphad, sampled once
# into a reusable surrogate. This is the full workflow:
#
#     select source -> sample surrogate -> save -> reload -> build model -> solve -> inspect
#
# **You supply the database**: set `TDB` in the Settings cell (or `MB_EXAMPLE_TDB`, or `--tdb`
# from a terminal). It needs FE, CR and NI, BCC_A2 and FCC_A1, and mobilities. This example was
# developed with a Fe-Cr-Ni database assembled from Lee's thermodynamic assessment [3] with the
# mobilities listed by Lee and Oh [2] (Table 1), which they took from Hoglund et al. [4], based
# on data of Fridberg et al. [5]; see `examples/README.md` for why it isn't included yet.
#
# **Configurations, and what each one can tell you:**
#
# - `LAGGED_DEMO` (default) and `LAGGED_FAST`: the real workflow, with bulk matrices from
#   `composition_dependent_lagged` and `simplex_positive_2x2` interpolation. There is no exact
#   solution, so these make **no accuracy claim**. They show that the workflow runs, conserves
#   solute, stays inside the sampled data, and really uses composition-dependent matrices.
# - `FROZEN_ANALYTIC`: frozen matrices sampled from the same surrogate, on an effectively
#   semi-infinite domain, compared with the exact similarity solution [6], the kind of analytical
#   solution Kajihara and Kikuchi [7] applied to these couples. This checks the numerics
#   with realistic data and closure. It does **not** validate composition-dependent Fe-Cr-Ni
#   physics.
# - `FROZEN_FAST`: a very short frozen-matrix run with a minimal surrogate.
# - `PRODUCTION_B`: the resolution, duration and thin-phase policy of the research case B run
#   (see Limitations). It is **untested here** and, like the lagged demos, makes no accuracy
#   claim. It skips the frozen twin. Extrapolated from short runs on one machine, not run end to
#   end: about 5 min to sample and 4 min to solve.
#
# **Ordering, basis and units.** Elements are `[FE, CR, NI]` with FE dependent; compositions are
# `[X_CR, X_NI]` mole fractions. BCC_A2 lies left of FCC_A1. Lengths are in m, times in s,
# temperature in K, diffusivities in m^2/s. The domain is half of the symmetric experimental
# couple, so the left wall is a symmetry plane.
#
# **Files.** Bundles go in `OUTPUT_DIR`, by default
# `~/moving_boundary_example_outputs/fecrni_lee1996/<config>/`, wherever you run from. A
# compatible bundle is reloaded without sampling; an incompatible or damaged one stops with an
# error, as does a bundle folder with files but no `manifest.json` (usually an interrupted
# build); `REBUILD = True` (or `--rebuild`) replaces it. Nothing is ever rebuilt silently.
#
# A bundle also records the installed `moving-boundary` version, so upgrading the package stops
# reloads of bundles built by an earlier release. An editable install keeps one version while
# its code changes, so there, use `REBUILD` after changing surrogate code.
#
# **Settings you would change.** Numerical settings are fields of `Config`: `bulk_spacing` and
# `samples_per_side` (sampling cost against accuracy), node counts, `semi_log_dt`, `duration`,
# `terminal_thin_phase_policy` (`"raise"` stops at a thin-phase limit, `"continue"` proceeds
# past it; neither changes the phase topology), and `compare_frozen` (lagged runs only: also
# solve a frozen-matrix twin, roughly doubling the solve time). Make a variant with
# `dataclasses.replace(LAGGED_DEMO, duration=...)`. The physical case is set by module
# constants below the imports: `TEMPERATURE`, the far-field compositions `A_BULK` and
# `B_BULK`, `PROBE_POINT` and the geometry. Pass/fail limits apply only to the named
# configurations they were set for; any other settings report their metrics without them.
#
# **Limitations.** The demo settings are deliberately coarse and not converged. A production
# case B run, `PRODUCTION_B`, uses about 101 tie lines, a 0.01 bulk grid (5151 points),
# 100/1000 nodes and `semiLog_dt = 0.25 / 160` to 1000 h. Assumes equal molar volumes, no Kirkendall flow and a
# fixed temperature. With `validity_policy="raise"` a bulk query lacking fit support raises an
# error; with `"legacy"` it silently falls back to the nearest sample, so coverage is worth
# checking (see `workflow_report`). Sampling also prints progress from the surrogate builder.
#
# **Running.** Run the cells in order in a Jupyter or VS Code Interactive window, editing the
# Settings cell first. From a terminal:
# `python ternary_fecrni_lee1996.py --tdb PATH [--config LAGGED_FAST] [--plot]`. `LAGGED_DEMO`
# takes about 40 s and `LAGGED_FAST` about 15 s, measured on one machine.
"""How does a Fe-Cr-Ni couple evolve with composition-dependent diffusivity? See the notes above."""

# %%

from pathlib import Path
import sys

# Run notebook cells with the extraction repository root as the working directory.
sys.path.append(str(Path("packages/moving-boundary/examples").resolve()))

from advanced import ternary_fecrni_lee1996 as _advanced
from contextlib import contextmanager
from dataclasses import dataclass
import argparse
import hashlib
import importlib.metadata
import json
import os
import tempfile

import numpy as np
from scipy.spatial import Delaunay
from moving_boundary.run_diagnostics import inventory_drift as measure_inventory_drift

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
    SurrogateArtifactIntegrityError,
    prepare_surrogate_artifact,
    load_or_build_surrogate_artifact,
    write_surrogate_diagnostic_members,
    load_or_build_diagnostic_surrogate,
)
from moving_boundary.surrogates import TernaryMovingBoundaryThermodynamicsSurrogate

ELEMENTS = ["FE", "CR", "NI"]
INDEPENDENT = ["CR", "NI"]
PHASES = ["BCC_A2", "FCC_A1"]  # left, right
TEMPERATURE = 1373.0  # [1, 2]
PROBE_POINT = np.array([0.45, 0.21])  # a composition inside the two-phase region
# Couple B [2, Table 3], from [1]: alpha 38 wt.% Cr; gamma 13 wt.% Cr, 15 wt.% Ni; converted to
# mole fractions with the database's atomic masses. Ni in alpha is 1e-4 rather than the nominal 0.
A_BULK = np.array([0.39696, 0.0001])  # far-field [X_CR, X_NI] in BCC_A2 (case B)
B_BULK = np.array([0.139297, 0.142387])  # far-field [X_CR, X_NI] in FCC_A1 (case B)
# Half of the symmetric couple [2, Table 3]: half of the 124.1 um alpha sheet and the 2000 um gamma plate.
HALF_LENGTH = 2000.0e-6 + 124.1e-6 / 2.0
B_PHASE_LENGTH = 2000.0e-6
SIZING_TIME = 100.0 * 3600.0  # semi-infinite domains are sized for this time
DEFAULT_OUTPUT = Path.home() / "moving_boundary_example_outputs" / "fecrni_lee1996"
DATABASE_ENV = "MB_EXAMPLE_TDB"


@dataclass(frozen=True)
class Config:
    """One workflow configuration; see the module docstring for what each can show.

    ``bulk_spacing`` is the composition grid for bulk diffusivities; ``None`` samples
    only the two far fields, which can't support simplex interpolation.
    ``terminal_thin_phase_policy`` is passed to the solver unchanged. ``compare_frozen``
    applies to lagged runs only and adds a frozen-matrix twin solve. ``min_dt_frac`` is
    the solver's ``minDtFrac`` for every solve this configuration runs.
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
    terminal_thin_phase_policy: str = "raise"
    compare_frozen: bool = True
    min_dt_frac: float = 1.0e-14
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
# The research case B settings. Untested here; 50 per side gives about 101 tie lines.
PRODUCTION_B = Config("PRODUCTION_B", bulk_spacing=0.01, samples_per_side=50, left_nodes=100, right_nodes=1000,
                      semi_log_dt=0.25 / 160.0, duration=1000.0 * 3600.0, terminal_thin_phase_policy="continue",
                      compare_frozen=False, min_dt_frac=1.0e-16)
CONFIGS = {config.name: config for config in (LAGGED_DEMO, LAGGED_FAST, FROZEN_ANALYTIC, FROZEN_FAST, PRODUCTION_B)}


# Configuration-specific checks are documented in the advanced companion.
FROZEN_ANALYTIC_LIMITS = _advanced.FROZEN_ANALYTIC_LIMITS


# Configuration-specific checks are documented in the advanced companion.
LAGGED_THRESHOLDS = _advanced.LAGGED_THRESHOLDS
THRESHOLD_CONFIGS = (LAGGED_FAST, LAGGED_DEMO)

# %% [markdown]
# ## Settings
#
# Set `TDB` to your database path (or leave it `None` and set `MB_EXAMPLE_TDB`), and pick a
# configuration: `LAGGED_DEMO`, `LAGGED_FAST`, `FROZEN_ANALYTIC`, `FROZEN_FAST` or `PRODUCTION_B`.

# %%
TDB = None
CONFIG = LAGGED_DEMO
OUTPUT_DIR = DEFAULT_OUTPUT
REBUILD = False
PLOT = True


# %% [markdown]
# ## Thermodynamic source and surrogate

# %%

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


from moving_boundary.sampling import ternary_simplex_points


def simplex_bulk_points(spacing, floor=1.0e-5):
    """Return the package's ordered, buffered ternary composition grid."""
    return ternary_simplex_points(spacing, floor)


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
    """Everything that defines the sampled data; a reload requires an exact match.

    Records the installed ``moving-boundary`` version but not its source, so a bundle
    survives code edits within one version (such as in an editable install).
    """
    return {
        "system": {"elements": ELEMENTS, "phases": PHASES, "temperature": TEMPERATURE},
        "source": {"engine": "pycalphad", "version": importlib.metadata.version("pycalphad"), "database_sha256": database_digest(tdb)},
        "package": {"moving_boundary": importlib.metadata.version("moving-boundary")},
        "tie_lines": {"probe_point": PROBE_POINT, "samples_per_side": config.samples_per_side},
        "bulk": {"points": _bulk_points(config), "interpolation": config.interpolation, "validity_policy": config.validity_policy},
    }


def prepare_surrogate(config, tdb, *, output_dir=DEFAULT_OUTPUT, rebuild=False):
    """Build or strictly reuse the surrogate and required reports; return ``(bundle, action)``."""
    tdb = resolve_database(tdb)
    roles = () if config.bulk_spacing is None else ("failed_bulk_points", "invalid_bulk_points")
    return load_or_build_diagnostic_surrogate(
        Path(output_dir) / config.name, build_spec(config, tdb),
        lambda records: sample_surrogate(config, tdb), rebuild=rebuild, report_roles=roles)


def _no_builder():
    """A strict reload never builds."""
    raise AssertionError("A strict reload must not call its builder.")


# %% [markdown]
# ## Models

# %%

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
        terminal_thin_phase_policy=config.terminal_thin_phase_policy,
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


# %% [markdown]
# ## Short workflow
#
# Solve once with basic checks. Use the original `run` below for full validation.

# %%
def run_core(config=LAGGED_DEMO, *, tdb=None, output_dir=DEFAULT_OUTPUT, rebuild=False, surrogate=None):
    """Prepare a surrogate and solve once; report final inventory drift only.

    The selected configuration still controls frozen/lagged model construction.
    Exact checks, query recording and a frozen twin belong to ``run``.
    """
    if surrogate is None:
        surrogate = prepare_surrogate(config, tdb, output_dir=output_dir, rebuild=rebuild)[0].surrogate
    model = build_model(config, surrogate)
    model.solve(config.duration, minDtFrac=config.min_dt_frac)
    return ExampleResult(config, model, None, {"inventory_rel": inventory_relative_drift(model)})


class RecordingDiffusivity(_advanced.RecordingDiffusivity):
    """Record attempted bulk queries for this case; accepted-state checks remain separate."""

    def __init__(self, inner):
        super().__init__(inner, PHASES)


def retained_general_samples(surrogate, phase):
    """The samples actually fitted: requested points plus tie-line ends, minus unusable ones.

    Uses the public accessor, which returns copies of the retained usable rows.
    """
    return surrogate.get_diffusivity_samples(phase)


def inventory_relative_drift(model):
    """Largest component-relative final-minus-initial inventory drift."""
    _, inventory = recorded_history(model.inventoryData)
    return measure_inventory_drift(inventory, reduction="final")[1]


# %% [markdown]
# ## Advanced checks and compatibility entry points
#
# The introductory path above needs only `run_core`. These wrappers retain
# the original detailed workflow and command-line behavior.

# %%
def _advanced_case():
    """Supply current case inputs to the optional detailed workflow. See the advanced companion."""
    return _advanced.Case(
        CONFIGS=CONFIGS,
        DATABASE_ENV=DATABASE_ENV,
        ExampleResult=ExampleResult,
        FROZEN_ANALYTIC=FROZEN_ANALYTIC,
        FROZEN_ANALYTIC_LIMITS=FROZEN_ANALYTIC_LIMITS,
        LAGGED_THRESHOLDS=LAGGED_THRESHOLDS,
        PHASES=PHASES,
        RecordingDiffusivity=RecordingDiffusivity,
        TEMPERATURE=TEMPERATURE,
        THRESHOLD_CONFIGS=THRESHOLD_CONFIGS,
        description=__doc__,
        _accepted_states=_accepted_states,
        build_model=build_model,
        case_geometry=case_geometry,
        check_limits=check_limits,
        exact_solution=exact_solution,
        freeze_matrices=freeze_matrices,
        interface_matrices=interface_matrices,
        inventory_relative_drift=inventory_relative_drift,
        lagged_vs_frozen=lagged_vs_frozen,
        make_model=make_model,
        prepare_surrogate=prepare_surrogate,
        resolve_database=resolve_database,
        retained_general_samples=retained_general_samples,
        simplex_bulk_points=simplex_bulk_points,
        workflow_report=workflow_report,
    )


def _accepted_states(model, max_times=60):
    """Compositions at nodes and midpoints of accepted states, for each phase. See the advanced companion."""
    return _advanced._accepted_states(model, max_times, case=_advanced_case())


def workflow_report(model, surrogate, recorder=None):
    """Check that the lagged run stayed inside its data and really used it, per phase. See the advanced companion."""
    return _advanced.workflow_report(model, surrogate, recorder, case=_advanced_case())


def lagged_vs_frozen(model, surrogate, config):
    """Compare with frozen matrices, using identical geometry, closure, grids and schedule. See the advanced companion."""
    return _advanced.lagged_vs_frozen(model, surrogate, config, case=_advanced_case())


def run(config=LAGGED_DEMO, *, tdb=None, output_dir=DEFAULT_OUTPUT, rebuild=False, surrogate=None):
    """Prepare the surrogate (unless given), solve, and report what this configuration can show. See the advanced companion."""
    return _advanced.run(config, tdb=tdb, output_dir=output_dir, rebuild=rebuild, surrogate=surrogate, case=_advanced_case())


def check_limits(result):
    """Compare metrics with the limits that apply; returns ``(name, value, rule, passed)`` rows. See the advanced companion."""
    return _advanced.check_limits(result, case=_advanced_case())


def report(result):
    """Print the final state, the metrics, and each applicable limit with whether it was met. See the advanced companion."""
    return _advanced.report(result, case=_advanced_case())


def plot(result):
    """Plot the interface history and the final phase profiles; returns the figure. See the advanced companion."""
    return _advanced.plot(result, case=_advanced_case())


def settings_from_command_line(tdb, config, output_dir, rebuild, plot_figures):
    """Apply command-line options from a terminal; inside a notebook kernel, change nothing. See the advanced companion."""
    return _advanced.settings_from_command_line(tdb, config, output_dir, rebuild, plot_figures, case=_advanced_case())


# %% [markdown]
# ## Running

# %%

def exact_solution(model, surrogate, matrices=None):
    """The exact solution for frozen matrices, using the root closest to the initial tie line
    (more than one root can exist [8])."""
    matrices = interface_matrices(surrogate, model.initialEta) if matrices is None else matrices
    roots = solve_similarity_roots(matrices[PHASES[0]], matrices[PHASES[1]], A_BULK, B_BULK, surrogate, s0=model.initialInterfacePosition)
    return min(roots, key=lambda root: abs(root.eta - float(model.initialEta)))


# %% [markdown]
# ## Run, one step per cell
#
# 1. Prepare the surrogate: sample and publish it, or reload a compatible bundle.

# %%
if __name__ == "__main__":
    TDB, CONFIG, OUTPUT_DIR, REBUILD, PLOT = settings_from_command_line(TDB, CONFIG, OUTPUT_DIR, REBUILD, PLOT)
    bundle, action = prepare_surrogate(CONFIG, TDB, output_dir=OUTPUT_DIR, rebuild=REBUILD)
    surrogate = bundle.surrogate
    print(f"{action} {bundle.path}")
    print(f"tie lines: {len(surrogate.eta_samples)}, interpolation: {surrogate.diffusivityInterpolation}, "
          f"validity policy: {surrogate.validity_policy}")

# %% [markdown]
# 2. Build the model, solve, and check what this configuration can show. Lagged runs also
#    solve a frozen-matrix twin for comparison, unless `compare_frozen` is off.

# %%
if __name__ == "__main__":
    result = run(CONFIG, surrogate=surrogate)
    report(result)

# %% [markdown]
# 3. Plot.

# %%
if __name__ == "__main__" and PLOT:
    import matplotlib.pyplot as plt

    plot(result)
    plt.show()

# %% [markdown]
# ## References
#
# 1. M. Kajihara, C.-B. Lim, M. Kikuchi, "Experimental study on dissolution of alpha phase in
#    gamma/alpha/gamma diffusion couples of the Fe-Cr-Ni system," ISIJ International 33 (1993)
#    498-507. Source of the diffusion couples and their 1373 K anneals.
# 2. B.-J. Lee, K. H. Oh, "Numerical treatment of the moving interface in diffusional
#    reactions," Z. Metallkd. 87 (1996) 195-204. Table 3 (couple compositions and thicknesses,
#    from [1]) and Table 1 (mobilities, from [4]).
# 3. B.-J. Lee, J. Korean Inst. Met. Mater. 31 (1993) 107 (Fe-Cr-Ni thermodynamic assessment,
#    as cited in [2], note it is difficult to find a copy of thisn online).
# 4. L. Hoglund, B. Jonsson, J. Agren, TRITA-MAC-0444, Royal Institute of Technology,
#    Stockholm (1990), as cited in [2].
# 5. J. Fridberg, L.-E. Torndahl, M. Hillert, Jernkont. Ann. 153 (1969) 263, as cited in [2].
# 6. J. S. Kirkaldy, "Diffusion in multicomponent metallic systems: III. The motion of planar
#    phase interfaces," Can. J. Phys. 36 (1958) 917-925, doi:10.1139/p58-098. Similarity
#    solutions for planar interfaces with constant diffusivities (FROZEN_ANALYTIC).
# 7. M. Kajihara, M. Kikuchi, "Analysis of dissolution of alpha phase in gamma/alpha/gamma
#    diffusion couples of the Fe-Cr-Ni system using analytical solutions for semi-infinite
#    diffusion couples," Acta Metall. Mater. 43 (1995) 807-820. The same couples analysed with
#    constant-diffusivity semi-infinite solutions.
# 8. P. Maugis, W. D. Hopfe, J. E. Morral, J. S. Kirkaldy, "Multiple interface velocity
#    solutions for ternary biphase infinite diffusion couples," Acta Mater. 45 (1997)
#    1941-1954. More than one interface velocity can satisfy the same couple.
