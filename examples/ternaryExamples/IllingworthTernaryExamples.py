# %%
"""
Interactive Fe-Cr-Ni examples for the ternary Illingworth moving-boundary solver.

This file is intentionally written as a Jupyter/VS Code cell script. Run cells
top-to-bottom, then edit the configuration cell to try different tie-line
sampling, fixed diffusivity matrices, mesh sizes, and solve times.
"""

# If you run this file in IPython/Jupyter, these are often useful:
# %matplotlib inline
# %config InlineBackend.figure_format = "svg"

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from kawin.diffusion import (
    MovingBoundaryIllingworthTernaryFD1DModel,
    TernaryMovingBoundaryThermodynamicsSurrogate,
)
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.solver import explicitEulerIterator
from kawin.thermo import MulticomponentThermodynamics


def _find_repo_root(start: Path) -> Path:
    """Finds the repository root when the file is run from a notebook kernel."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "kawin").exists():
            return candidate
    return Path.cwd()


THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = _find_repo_root(THIS_DIR)
EXAMPLES_DIR = REPO_ROOT / "examples"


# %%
# Editable case configuration

ELEMENTS = ["FE", "CR", "NI"]
INDEPENDENT_ELEMENTS = ["CR", "NI"]
PHASES = ["BCC_A2", "FCC_A1"]
TIELINE_PHASES = ("BCC_A2", "FCC_A1")
TEMPERATURE = 1373.0
TDB_PATH = EXAMPLES_DIR / "FeCrNi_Lee1993_L_style_ternary_checked_withMobility.tdb"

# Probe compositions are independent components [X(CR), X(NI)].
# These values are adapted from the Lee/Oh ternary example but kept here as
# plain editable knobs.
PROBE_START = np.array([0.1233, 0.0001], dtype=np.float64)
PROBE_END = np.array([0.4993, 0.2257], dtype=np.float64)
ETA_SAMPLES = np.linspace(0.0, 1.0, 21)
INITIAL_ETA_METHOD = "instantaneous_balance"
INITIAL_ETA_BRACKET = (1e-3, 1-1e-3)
INITIAL_ETA_GUESS = None
INITIAL_VELOCITY_GUESS = None
PLOT_LEE_OH_FIG9_DATA = True
LEE_OH_FIG9_LOWER_CR_PATH = EXAMPLES_DIR / "leeAndOh1996_data" / "fig9_lowerCurve_Cr.csv"
LEE_OH_FIG9_UPPER_NI_PATH = EXAMPLES_DIR / "leeAndOh1996_data" / "fig9_upperCurve_Ni.csv"
LEE_OH_FIG9_TIME_UNIT_SECONDS = 3600.0

LENGTH = 30.0e-6
NODES = 61
INTERFACE_POSITION = 12.0e-6 + 1.0e-12
LEFT_BULK = np.array([0.38, 0.001], dtype=np.float64) # np.array([0.31391446, 0.06419736], dtype=np.float64) 
RIGHT_BULK = np.array([0.13, 0.15], dtype=np.float64) # np.array([0.22596712, 0.11443033], dtype=np.float64)

idealized_comp = LEFT_BULK * (INTERFACE_POSITION/LENGTH) + RIGHT_BULK * (1 - INTERFACE_POSITION/LENGTH)

# Top-level solver diffusivity source:
#   "fixed"    -> preserve the original example behavior: build the interface
#                 surrogate, then solve with one frozen 2x2 matrix per phase.
#   "variable" -> use the surrogate itself as the bulk diffusivity provider.
DIFFUSIVITY_SOURCE = ["fixed", "variable"][1]

# Fixed-source matrix selection:
#   "explicit"       -> set FIXED_DIFFUSIVITY_MATRICES below.
#   "sample_tieline" -> sample one tie-line from the real thermodynamics object
#                       at DIFFUSIVITY_SAMPLE_ETA and freeze one matrix per phase.
DIFFUSIVITY_MODE = "sample_tieline"
DIFFUSIVITY_SAMPLE_ETA = 0.5
FIXED_DIFFUSIVITY_MATRICES = None
# Example explicit form:
# FIXED_DIFFUSIVITY_MATRICES = {
#     "BCC_A2": np.array([[1.0e-15, 0.0], [0.0, 1.0e-15]], dtype=np.float64),
#     "FCC_A1": np.array([[1.0e-16, 0.0], [0.0, 1.0e-16]], dtype=np.float64),
# }

# Variable-source surrogate options:
#   VARIABLE_DIFFUSIVITY_INTERPOLATION = "continuous_grid" -> smooth interface
#       and regular-grid bulk interpolation. This is the recommended variable
#       path for composition-dependent lagged/Picard solves.
#   VARIABLE_DIFFUSIVITY_INTERPOLATION = "nearest" -> legacy nearest-neighbor
#       surrogate diffusivity using the same grid samples as extra bulk points.
VARIABLE_DIFFUSIVITY_INTERPOLATION = "continuous_grid"
VARIABLE_DIFFUSIVITY_BULK_MODE = ["lagged", "picard"][1]
# Alternative:
# VARIABLE_DIFFUSIVITY_BULK_MODE = "lagged"
VARIABLE_DIFFUSIVITY_BULK_CR_AXIS = np.linspace(0.10, 0.55, 19)
VARIABLE_DIFFUSIVITY_BULK_NI_AXIS = np.linspace(0.0001, 0.25, 19)
# Set this to an explicit ``(cr_axis, ni_axis)`` tuple to override the two axes
# above without editing both variables.
VARIABLE_DIFFUSIVITY_BULK_GRIDS = None
VARIABLE_DIFFUSIVITY_BULK_PICARD_RTOL = None
VARIABLE_DIFFUSIVITY_BULK_PICARD_ATOL = 1.0e-12
VARIABLE_DIFFUSIVITY_BULK_PICARD_MAX_ITERATIONS = 25
VARIABLE_DIFFUSIVITY_BULK_PICARD_RELAXATION = 1.0

# Time stepping:
#   DT_MODE = "fixed"    -> advance by FIXED_TIME_STEP.
#   DT_MODE = "semi_log" -> advance to semi-log-spaced target times starting
#                           at SEMI_LOG_T0 with natural-log spacing SEMI_LOG_DT.
#                           SEMI_LOG_BASE_TIME_STEP remains a positive fallback
#                           scale used internally by the model.
DT_MODE = "semi_log"
FIXED_TIME_STEP = 1.0
SEMI_LOG_BASE_TIME_STEP = 1.0
SEMI_LOG_DT = 0.25 / 5
SEMI_LOG_T0 = 1.0e-6
SOLVE_TIME = [3600*1e0, 3600*1e2, 3600*1e3][2]
PHASE_A_NODES = None
PHASE_B_NODES = None
TOLERANCE = 3.0e-13
MAX_ITERATIONS = 25
VERBOSE = True
VERBOSE_INTERVAL = 10
MIN_DT_FRAC = 1.0e-16

# Leave this False when you only want to build the surrogate and model without
# starting a solve after running all cells.
RUN_SOLVE = True


# %%
# Helper classes and functions


_OVERRIDE_KEY_ALIASES = {
    "diffusivity_mode": "DIFFUSIVITY_MODE",
    "diffusivity_sample_eta": "DIFFUSIVITY_SAMPLE_ETA",
    "diffusivity_source": "DIFFUSIVITY_SOURCE",
    "dt_mode": "DT_MODE",
    "fixed_time_step": "FIXED_TIME_STEP",
    "fixed_diffusivity_matrices": "FIXED_DIFFUSIVITY_MATRICES",
    "left_bulk": "LEFT_BULK",
    "length": "LENGTH",
    "max_iterations": "MAX_ITERATIONS",
    "min_dt_frac": "MIN_DT_FRAC",
    "nodes": "NODES",
    "phase_a_nodes": "PHASE_A_NODES",
    "phase_b_nodes": "PHASE_B_NODES",
    "plot_lee_oh_fig9_data": "PLOT_LEE_OH_FIG9_DATA",
    "right_bulk": "RIGHT_BULK",
    "run_solve": "RUN_SOLVE",
    "semi_log_base_time_step": "SEMI_LOG_BASE_TIME_STEP",
    "semi_log_dt": "SEMI_LOG_DT",
    "semi_log_t0": "SEMI_LOG_T0",
    "solve_time": "SOLVE_TIME",
    "tolerance": "TOLERANCE",
    "verbose": "VERBOSE",
    "verbose_interval": "VERBOSE_INTERVAL",
    "variable_diffusivity_bulk_cr_axis": "VARIABLE_DIFFUSIVITY_BULK_CR_AXIS",
    "variable_diffusivity_bulk_grids": "VARIABLE_DIFFUSIVITY_BULK_GRIDS",
    "variable_diffusivity_bulk_mode": "VARIABLE_DIFFUSIVITY_BULK_MODE",
    "variable_diffusivity_bulk_ni_axis": "VARIABLE_DIFFUSIVITY_BULK_NI_AXIS",
    "variable_diffusivity_bulk_picard_atol": "VARIABLE_DIFFUSIVITY_BULK_PICARD_ATOL",
    "variable_diffusivity_bulk_picard_max_iterations": "VARIABLE_DIFFUSIVITY_BULK_PICARD_MAX_ITERATIONS",
    "variable_diffusivity_bulk_picard_relaxation": "VARIABLE_DIFFUSIVITY_BULK_PICARD_RELAXATION",
    "variable_diffusivity_bulk_picard_rtol": "VARIABLE_DIFFUSIVITY_BULK_PICARD_RTOL",
    "variable_diffusivity_interpolation": "VARIABLE_DIFFUSIVITY_INTERPOLATION",
}


def _refresh_derived_config():
    """Updates derived example globals after temporary configuration changes."""
    global idealized_comp
    idealized_comp = LEFT_BULK * (INTERFACE_POSITION / LENGTH) + RIGHT_BULK * (1 - INTERFACE_POSITION / LENGTH)


def _normalize_override_key(key):
    key = str(key)
    if key in globals():
        return key
    upper_key = key.upper()
    if upper_key in globals():
        return upper_key
    if key in _OVERRIDE_KEY_ALIASES:
        return _OVERRIDE_KEY_ALIASES[key]
    raise KeyError(f"Unknown Illingworth ternary example override '{key}'.")


@contextmanager
def _temporary_config(overrides=None):
    """Temporarily applies module-level example configuration overrides."""
    if not overrides:
        yield
        return
    normalized = {_normalize_override_key(key): value for key, value in dict(overrides).items()}
    old_values = {key: globals()[key] for key in normalized}
    try:
        globals().update(normalized)
        _refresh_derived_config()
        yield
    finally:
        globals().update(old_values)
        _refresh_derived_config()


class FixedMatrixTernaryDiffusivity:
    """
    Thermodynamics-like object that returns fixed 2x2 matrices per phase.

    The ternary Illingworth example uses this object as the model
    ``thermodynamics`` while a separate tie-line surrogate supplies interface
    compositions through ``interface_equilibrium``.
    """

    def __init__(self, diffusivity_matrices, phases, temperature=None):
        self.phases = list(phases)
        self.temperature = None if temperature is None else float(temperature)
        self.diffusivity_matrices = {}
        for phase in self.phases:
            if phase not in diffusivity_matrices:
                raise ValueError(f"Missing fixed diffusivity matrix for phase '{phase}'.")
            matrix = np.asarray(diffusivity_matrices[phase], dtype=np.float64)
            if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"Fixed diffusivity matrix for phase '{phase}' must be finite with shape (2, 2).")
            self.diffusivity_matrices[phase] = matrix.copy()

    def clearCache(self):
        """Compatibility no-op for diffusion model setup."""
        return

    def getInterdiffusivity(self, x, T=None, phase=None, query_context=None, **kwargs):
        """Returns the fixed matrix for ``phase`` and tiles for batched queries."""
        if phase is None:
            phase = self.phases[0]
        if phase not in self.diffusivity_matrices:
            raise ValueError(f"Unknown phase '{phase}'. Expected one of {self.phases}.")
        if self.temperature is not None and T is not None:
            T_values = np.asarray(T, dtype=np.float64)
            if not np.allclose(T_values, self.temperature, rtol=0.0, atol=1.0e-8):
                raise ValueError(f"Fixed diffusivity object is isothermal at {self.temperature}; received T={T}.")

        matrix = self.diffusivity_matrices[phase]
        values = np.asarray(x, dtype=np.float64)
        if values.ndim <= 1:
            return matrix.copy()
        return np.tile(matrix, (values.shape[0], 1, 1))


def build_source_thermodynamics():
    """Builds the real Fe-Cr-Ni thermodynamics object used only for sampling."""
    if not TDB_PATH.exists():
        raise FileNotFoundError(f"Could not find Fe-Cr-Ni TDB at {TDB_PATH}.")
    return MulticomponentThermodynamics(str(TDB_PATH), ELEMENTS, PHASES)


def _coerce_diffusivity_source():
    source = str(DIFFUSIVITY_SOURCE)
    if source not in {"fixed", "variable"}:
        raise ValueError("DIFFUSIVITY_SOURCE must be either 'fixed' or 'variable'.")
    return source


def _normalize_variable_diffusivity_bulk_mode(mode):
    """Normalizes user-facing variable diffusivity solve mode aliases."""
    mode = str(mode)
    mode_aliases = {
        "lagged": "composition_dependent_lagged",
        "picard": "composition_dependent_implicit",
        "implicit": "composition_dependent_implicit",
    }
    mode = mode_aliases.get(mode, mode)
    if mode not in {"composition_dependent_lagged", "composition_dependent_implicit"}:
        raise ValueError(
            "VARIABLE_DIFFUSIVITY_BULK_MODE must be either "
            "'lagged'/'composition_dependent_lagged' or 'picard'/'composition_dependent_implicit'."
        )
    return mode


def _coerce_variable_diffusivity_bulk_mode():
    return _normalize_variable_diffusivity_bulk_mode(VARIABLE_DIFFUSIVITY_BULK_MODE)


def get_variable_diffusivity_solver_options():
    """Returns Illingworth bulk-diffusivity kwargs for variable diffusivity solves."""
    if _coerce_diffusivity_source() == "fixed":
        return {"bulk_diffusivity_mode": "phase_uniform"}
    return {
        "bulk_diffusivity_mode": _coerce_variable_diffusivity_bulk_mode(),
        "bulk_picard_rtol": VARIABLE_DIFFUSIVITY_BULK_PICARD_RTOL,
        "bulk_picard_atol": float(VARIABLE_DIFFUSIVITY_BULK_PICARD_ATOL),
        "bulk_picard_max_iterations": int(VARIABLE_DIFFUSIVITY_BULK_PICARD_MAX_ITERATIONS),
        "bulk_picard_relaxation": float(VARIABLE_DIFFUSIVITY_BULK_PICARD_RELAXATION),
    }


def get_variable_diffusivity_bulk_grids():
    """Returns the regular [CR, NI] grid axes used to train variable bulk diffusivity."""
    grids = VARIABLE_DIFFUSIVITY_BULK_GRIDS
    if grids is None:
        grids = (VARIABLE_DIFFUSIVITY_BULK_CR_AXIS, VARIABLE_DIFFUSIVITY_BULK_NI_AXIS)
    axes = tuple(np.asarray(axis, dtype=np.float64).reshape(-1) for axis in grids)
    if len(axes) != 2:
        raise ValueError("VARIABLE_DIFFUSIVITY_BULK_GRIDS must contain exactly two axes.")
    for label, axis in zip(INDEPENDENT_ELEMENTS, axes):
        if axis.size < 2:
            raise ValueError(f"Variable diffusivity {label} grid axis must contain at least two values.")
        if not np.all(np.isfinite(axis)) or not np.all(np.diff(axis) > 0.0):
            raise ValueError(f"Variable diffusivity {label} grid axis must be finite and strictly increasing.")
    if axes[0][-1] + axes[1][-1] >= 1.0:
        raise ValueError("Variable diffusivity grid rectangle must lie inside the ternary composition simplex.")
    return tuple(axis.copy() for axis in axes)


def get_surrogate_diffusivity_sampling_options():
    """
    Returns diffusivity sampling kwargs for ``TernaryMovingBoundaryThermodynamicsSurrogate``.

    Fixed-source solves keep the light tie-line-only surrogate used by the
    original example. Variable-source solves additionally sample a regular bulk
    diffusivity grid so the surrogate can supply composition-dependent bulk
    matrices to the Illingworth solver.
    """
    if _coerce_diffusivity_source() == "fixed":
        return {}

    interpolation = str(VARIABLE_DIFFUSIVITY_INTERPOLATION)
    if interpolation not in {"nearest", "continuous_grid"}:
        raise ValueError("VARIABLE_DIFFUSIVITY_INTERPOLATION must be either 'nearest' or 'continuous_grid'.")
    grids = get_variable_diffusivity_bulk_grids()
    return {
        "diffusivity_interpolation": interpolation,
        "diffusivity_bulk_grids": grids,
    }


def build_tieline_surrogate(source_thermodynamics):
    """Samples the Fe-Cr-Ni tie-line family and optional variable diffusivity grid."""
    return TernaryMovingBoundaryThermodynamicsSurrogate.from_database(
        thermodynamics=source_thermodynamics,
        elements=ELEMENTS,
        phases=PHASES,
        tieline_phases=TIELINE_PHASES,
        temperature=TEMPERATURE,
        probe_start=PROBE_START,
        probe_end=PROBE_END,
        eta_samples=ETA_SAMPLES,
        precipitate_phase=TIELINE_PHASES[1],
        validation_database=TDB_PATH,
        **get_surrogate_diffusivity_sampling_options(),
    )


def _print_diffusivity_matrices(matrices):
    print("Fixed diffusivity matrices used by the Illingworth solve:")
    for phase in TIELINE_PHASES:
        print(f"{phase}:\n{np.asarray(matrices[phase], dtype=np.float64)}")


def select_fixed_diffusivity_matrices(source_thermodynamics, tieline_surrogate, *, print_matrices=True):
    """
    Selects one fixed 2x2 diffusivity matrix per phase.

    ``DIFFUSIVITY_MODE='explicit'`` requires the user to set
    ``FIXED_DIFFUSIVITY_MATRICES``. ``'sample_tieline'`` samples the real
    thermodynamics object once at ``DIFFUSIVITY_SAMPLE_ETA`` and freezes the
    resulting matrices.
    """
    if DIFFUSIVITY_MODE == "explicit":
        if FIXED_DIFFUSIVITY_MATRICES is None:
            raise ValueError(
                "DIFFUSIVITY_MODE is 'explicit', so FIXED_DIFFUSIVITY_MATRICES must be a dict "
                "with one 2x2 matrix for each phase in TIELINE_PHASES."
            )
        matrices = {phase: np.asarray(FIXED_DIFFUSIVITY_MATRICES[phase], dtype=np.float64) for phase in TIELINE_PHASES}
    elif DIFFUSIVITY_MODE == "sample_tieline":
        if DIFFUSIVITY_SAMPLE_ETA is None:
            raise ValueError("DIFFUSIVITY_SAMPLE_ETA must be set when DIFFUSIVITY_MODE is 'sample_tieline'.")
        interface_compositions = tieline_surrogate.interface_compositions(float(DIFFUSIVITY_SAMPLE_ETA))
        matrices = {
            phase: np.asarray(
                source_thermodynamics.getInterdiffusivity(composition, TEMPERATURE, phase=phase),
                dtype=np.float64,
            ).reshape(2, 2)
            for phase, composition in zip(TIELINE_PHASES, interface_compositions)
        }
    else:
        raise ValueError("DIFFUSIVITY_MODE must be either 'explicit' or 'sample_tieline'.")

    if print_matrices:
        _print_diffusivity_matrices(matrices)
    return matrices


def build_case_context(overrides=None, *, print_matrices=True):
    """
    Builds reusable thermodynamics and diffusivity objects for this case.

    The expensive Fe-Cr-Ni database and surrogate setup is independent of mesh
    density and timestep controls, so convergence sweeps can build this context
    once and pass it to ``run_case`` for each numerical variant.
    """
    with _temporary_config(overrides):
        diffusivity_source = _coerce_diffusivity_source()
        source_thermodynamics = build_source_thermodynamics()
        tieline_surrogate = build_tieline_surrogate(source_thermodynamics)
        fixed_diffusivity_matrices = None
        fixed_diffusivity = None
        solver_options = get_variable_diffusivity_solver_options()
        if diffusivity_source == "fixed":
            fixed_diffusivity_matrices = select_fixed_diffusivity_matrices(
                source_thermodynamics,
                tieline_surrogate,
                print_matrices=print_matrices,
            )
            fixed_diffusivity = FixedMatrixTernaryDiffusivity(
                fixed_diffusivity_matrices,
                phases=TIELINE_PHASES,
                temperature=TEMPERATURE,
            )
            solver_thermodynamics = fixed_diffusivity
        else:
            solver_thermodynamics = tieline_surrogate
            if print_matrices:
                print("Using variable surrogate diffusivity for the Illingworth solve.")
                print(f"Surrogate diffusivity interpolation = {VARIABLE_DIFFUSIVITY_INTERPOLATION}")
                print(f"Solver bulk diffusivity mode = {solver_options['bulk_diffusivity_mode']}")
        return {
            "bulk_diffusivity_mode": solver_options["bulk_diffusivity_mode"],
            "diffusivity_source": diffusivity_source,
            "source_thermodynamics": source_thermodynamics,
            "tieline_surrogate": tieline_surrogate,
            "fixed_diffusivity_matrices": fixed_diffusivity_matrices,
            "fixed_diffusivity": fixed_diffusivity,
            "solver_thermodynamics": solver_thermodynamics,
            "solver_options": solver_options,
        }


def make_mesh():
    """Builds the ternary step-profile mesh for [X(CR), X(NI)]."""
    mesh = CartesianFD1D(INDEPENDENT_ELEMENTS, [0.0, LENGTH], NODES)
    profile = ProfileBuilder(
        [
            (
                StepProfile1D(INTERFACE_POSITION, LEFT_BULK, RIGHT_BULK),
                INDEPENDENT_ELEMENTS,
            )
        ]
    )
    mesh.setResponseProfile(profile)
    return mesh


def get_time_step_options():
    """Returns constructor kwargs for the selected Illingworth timestep mode."""
    if DT_MODE == "fixed":
        if not np.isfinite(FIXED_TIME_STEP) or FIXED_TIME_STEP <= 0.0:
            raise ValueError("FIXED_TIME_STEP must be positive and finite when DT_MODE is 'fixed'.")
        return {
            "time_step": float(FIXED_TIME_STEP),
            "dt_mode": "fixed",
            "semiLog_dt": None,
            "semiLogT0": None,
        }
    if DT_MODE == "semi_log":
        if not np.isfinite(SEMI_LOG_BASE_TIME_STEP) or SEMI_LOG_BASE_TIME_STEP <= 0.0:
            raise ValueError("SEMI_LOG_BASE_TIME_STEP must be positive and finite when DT_MODE is 'semi_log'.")
        if not np.isfinite(SEMI_LOG_DT) or SEMI_LOG_DT <= 0.0:
            raise ValueError("SEMI_LOG_DT must be positive and finite when DT_MODE is 'semi_log'.")
        if not np.isfinite(SEMI_LOG_T0) or SEMI_LOG_T0 <= 0.0:
            raise ValueError("SEMI_LOG_T0 must be positive and finite when DT_MODE is 'semi_log'.")
        return {
            "time_step": float(SEMI_LOG_BASE_TIME_STEP),
            "dt_mode": "semi_log",
            "semiLog_dt": float(SEMI_LOG_DT),
            "semiLogT0": float(SEMI_LOG_T0),
        }
    raise ValueError("DT_MODE must be either 'fixed' or 'semi_log'.")


def build_model(tieline_surrogate, solver_thermodynamics, *, solver_options=None, bulk_diffusivity_mode=None):
    """Constructs the ternary Illingworth model without starting the solve."""
    time_step_options = get_time_step_options()
    solver_options = {} if solver_options is None else dict(solver_options)
    if bulk_diffusivity_mode is not None:
        solver_options["bulk_diffusivity_mode"] = bulk_diffusivity_mode
    solver_options.setdefault("bulk_diffusivity_mode", "phase_uniform")
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=make_mesh(),
        elements=ELEMENTS,
        phases=list(TIELINE_PHASES),
        thermodynamics=solver_thermodynamics,
        temperature=TEMPERATURE,
        interfacePosition=INTERFACE_POSITION,
        interface_equilibrium=tieline_surrogate,
        initial_eta_method=INITIAL_ETA_METHOD,
        initial_eta_bracket=INITIAL_ETA_BRACKET,
        initial_eta_guess=INITIAL_ETA_GUESS,
        initial_velocity_guess=INITIAL_VELOCITY_GUESS,
        time_step=time_step_options["time_step"],
        dt_mode=time_step_options["dt_mode"],
        semiLog_dt=time_step_options["semiLog_dt"],
        semiLogT0=time_step_options["semiLogT0"],
        phase_a_nodes=PHASE_A_NODES,
        phase_b_nodes=PHASE_B_NODES,
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        **solver_options,
        record=True,
    )


def run_case(overrides=None, context=None, make_plots=False):
    """
    Builds and optionally solves one configured ternary Illingworth example.

    Parameters in ``overrides`` temporarily replace module-level configuration
    values such as ``NODES``, ``DT_MODE``, ``SEMI_LOG_DT``, ``FIXED_TIME_STEP``,
    and ``SOLVE_TIME``. Passing a context from ``build_case_context`` reuses the
    database-derived surrogate and diffusivity object.
    """
    figures = {}
    with _temporary_config(overrides):
        if context is None:
            context = build_case_context(print_matrices=VERBOSE)
        model = build_model(
            context["tieline_surrogate"],
            context["solver_thermodynamics"],
            solver_options=context["solver_options"],
        )

        if DT_MODE == "fixed":
            n_steps = int(np.ceil(SOLVE_TIME / FIXED_TIME_STEP))
        elif DT_MODE == "semi_log":
            if SEMI_LOG_DT is None or SEMI_LOG_T0 is None:
                raise ValueError("semiLog_dt and semiLogT0 must be set when dt_mode is 'semi_log'.")
            if SEMI_LOG_DT <= 0 or SEMI_LOG_T0 <= 0:
                raise ValueError("semiLog_dt and semiLogT0 must be positive when dt_mode is 'semi_log'.")
            if SOLVE_TIME <= SEMI_LOG_T0:
                n_steps = 1
            else:
                n_steps = int(np.ceil((np.log(SOLVE_TIME) - np.log(SEMI_LOG_T0)) / SEMI_LOG_DT)) + 1
        print(f"Estimated number of time-steps: {n_steps}")

        if RUN_SOLVE:
            model.solve(
                SOLVE_TIME,
                iterator=explicitEulerIterator,
                verbose=VERBOSE,
                vIt=VERBOSE_INTERVAL,
                minDtFrac=MIN_DT_FRAC,
            )
        if make_plots and model.currentTime > 0:
            figures["interface_position"] = plot_interface_position(
                model,
                plot_lee_oh_fig9_data=PLOT_LEE_OH_FIG9_DATA,
                xlims=(1e-1, 1e7),
            )
            figures["integrated_inventory"] = plot_integrated_inventory(model)
            figures["interface_compositions"] = plot_interface_compositions(model, context["tieline_surrogate"])
        return {
            "model": model,
            "context": context,
            "figures": figures,
            "overrides": {} if overrides is None else dict(overrides),
        }


def print_initial_eta_estimate(model):
    """Prints the model-selected initial eta and interface compositions."""
    estimate = model.initialEtaEstimate
    if estimate is None:
        print("Initial eta estimate is not available until model.setup() or model.solve() has run.")
        return
    print(f"Initial eta method = {estimate.method}")
    print(f"Estimated initial eta = {estimate.eta}")
    print(f"Initial Stefan residual norm = {estimate.residual_norm}")
    print(f"Initial fitted interface velocity = {estimate.velocity}")
    if estimate.branch is not None:
        print(f"Initial swept-inventory branch = {estimate.branch}")
    print(f"Initial residual vector [CR, NI] = {estimate.residual}")
    print(f"Initial flux imbalance [CR, NI] = {estimate.flux_delta}")
    print(f"Initial eta bracket = {estimate.bracket}")
    print(f"Initial eta solver iterations/function calls = {estimate.iterations}/{estimate.function_calls}")
    print(f"{TIELINE_PHASES[0]} interface composition [CR, NI] = {estimate.left_interface_composition}")
    print(f"{TIELINE_PHASES[1]} interface composition [CR, NI] = {estimate.right_interface_composition}")


def _show_if_interactive():
    if plt.get_backend().lower() != "agg":
        plt.show()


def _load_lee_oh_fig9_curve(path):
    """Loads a two-column Lee and Oh Fig. 9 digitized interface-position curve."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find Lee and Oh Fig. 9 data at {path}.")
    values = np.loadtxt(path, delimiter=",", dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape((-1, 2))
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Lee and Oh Fig. 9 data at {path} must contain finite two-column data.")
    return values[np.argsort(values[:, 0])]


def _plot_lee_oh_fig9_interface_data(ax, *, scale_time=1.0):
    """Overlays digitized Lee and Oh Fig. 9 normalized interface-position data."""
    curves = [
        (LEE_OH_FIG9_LOWER_CR_PATH, "Lee & Oh Fig. 9 lower curve (Cr)", "dimgray"),
        (LEE_OH_FIG9_UPPER_NI_PATH, "Lee & Oh Fig. 9 upper curve (Ni)", "k"),
    ]
    for path, label, color in curves:
        data = _load_lee_oh_fig9_curve(path)
        times = data[:, 0] * float(LEE_OH_FIG9_TIME_UNIT_SECONDS) / float(scale_time)
        ax.plot(times, data[:, 1], linestyle="-", linewidth=1.5, color=color, label=label)


def plot_interface_position(model, *, scale_time=1.0, normalize_to=None, plot_lee_oh_fig9_data=False, xlims=None):
    """Plots normalized interface position with optional Lee and Oh Fig. 9 data."""
    times = np.asarray(model.interfaceData._time[: model.interfaceData.N + 1], dtype=np.float64) / scale_time
    positions = np.asarray(model.interfaceData._y[: model.interfaceData.N + 1], dtype=np.float64)
    reference_position = float(positions[0] if normalize_to is None else normalize_to)
    if not np.isfinite(reference_position) or abs(reference_position) <= 1.0e-300:
        raise ValueError("Cannot normalize interface position by a zero or non-finite reference position.")
    normalized_positions = positions / reference_position
    positive_time = times > 0.0

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times[positive_time], normalized_positions[positive_time], marker="o", linewidth=1.5, label="Illingworth ternary")
    if plot_lee_oh_fig9_data:
        _plot_lee_oh_fig9_interface_data(ax, scale_time=scale_time)
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(f"time / {scale_time:g}")
    ax.set_ylabel("normalized interface position")
    ax.set_title("Ternary Illingworth normalized interface position")
    if xlims is not None:
        ax.set_xlim(*xlims)
    ax.legend()
    fig.tight_layout()
    _show_if_interactive()
    return fig, ax


def plot_diffusivity_comparison_interface_positions(
    results,
    *,
    scale_time=1.0,
    normalize_to=None,
    plot_lee_oh_fig9_data=False,
    xlims=None,
):
    """Plots normalized interface position histories for diffusivity comparison runs."""
    runs = list(results["runs"] if isinstance(results, dict) else results)
    if not runs:
        raise ValueError("At least one diffusivity comparison run is required.")

    fig, ax = plt.subplots(figsize=(14, 8))
    for run in runs:
        times = np.asarray(run["times"], dtype=np.float64) / scale_time
        positions = np.asarray(run["interface_position"], dtype=np.float64)
        reference_position = float(positions[0] if normalize_to is None else normalize_to)
        if not np.isfinite(reference_position) or abs(reference_position) <= 1.0e-300:
            raise ValueError("Cannot normalize interface position by a zero or non-finite reference position.")
        positive_time = times > 0.0
        ax.plot(
            times[positive_time],
            (positions / reference_position)[positive_time],
            linestyle="-",
            linewidth=1.0,
            label=run["label"],
        )

    if plot_lee_oh_fig9_data:
        _plot_lee_oh_fig9_interface_data(ax, scale_time=scale_time)
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(f"time / {scale_time:g}")
    ax.set_ylabel("normalized interface position")
    ax.set_title("Ternary Illingworth diffusivity comparison")
    if xlims is not None:
        ax.set_xlim(*xlims)
    ax.legend()
    fig.tight_layout()
    _show_if_interactive()
    return fig, ax


def plot_integrated_inventory(model, *, scale_time=1.0):
    """Plots domain-average CR and NI composition from inventory divided by R."""
    times = np.asarray(model.inventoryData._time[: model.inventoryData.N + 1], dtype=np.float64) / scale_time
    inventory = np.asarray(model.inventoryData._y[: model.inventoryData.N + 1], dtype=np.float64)
    average_composition = inventory / float(model._R)
    positive_time = times > 0.0

    fig, ax_left = plt.subplots(figsize=(7, 4))
    ax_right = ax_left.twinx()
    axes = [ax_left, ax_right]
    lines = []
    for i, (element, ax) in enumerate(zip(INDEPENDENT_ELEMENTS, axes)):
        color = f"C{i}"
        (line,) = ax.plot(times[positive_time], average_composition[positive_time, i], marker="o", linewidth=1.5, color=color, label=element)
        ax.axhline(average_composition[0, i], color=color, linestyle="--", linewidth=1, alpha=0.7)
        ax.set_ylabel(f"{element} average composition", color=color)
        ax.tick_params(axis="y", labelcolor=color)
        lines.append(line)
        compDiff_arr = average_composition[:, i]-average_composition[0, i]
        compDiffPercent_arr = (compDiff_arr/average_composition[0, i])*100
        print(f"{element} min and max comp diff from initial:         {compDiff_arr.min():.4}, {compDiff_arr.max():.4}")
        print(f"{element} min and max percent comp diff from initial: {compDiffPercent_arr.min():.4}%, {compDiffPercent_arr.max():.4}%")
    for i, (element, ax) in enumerate(zip(INDEPENDENT_ELEMENTS, axes)):
        print(f"{element} absolute and percent comp diff from idealized: {average_composition[0, i]-idealized_comp[i]:.4}, {((average_composition[0, i]-idealized_comp[i])/idealized_comp[i])*100:.4}%")

    ax_left.set_xscale("log")
    ax_left.set_xlabel(f"time / {scale_time:g}")
    ax_left.set_title("Domain-average independent-component composition")
    ax_left.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    _show_if_interactive()
    return fig, (ax_left, ax_right)


def plot_interface_compositions(model, tieline_surrogate, *, scale_time=1.0):
    """Plots phase-side interface compositions evaluated from recorded eta."""
    times = np.asarray(model.etaData._time[: model.etaData.N + 1], dtype=np.float64) / scale_time
    etas = np.asarray(model.etaData._y[: model.etaData.N + 1], dtype=np.float64)
    positive_time = times > 0.0
    interface_compositions = np.asarray(
        [tieline_surrogate.interface_compositions(float(eta)) for eta in etas],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for component_index, (element, ax) in enumerate(zip(INDEPENDENT_ELEMENTS, axes)):
        for phase_index, phase in enumerate(TIELINE_PHASES):
            ax.plot(
                times[positive_time],
                interface_compositions[positive_time, phase_index, component_index],
                marker="o",
                linewidth=1.5,
                label=phase,
            )
        ax.set_xscale("log")
        ax.set_xlabel(f"time / {scale_time:g}")
        ax.set_ylabel(f"{element} interface composition")
        ax.set_title(element)
        ax.legend()

    fig.suptitle("Ternary Illingworth interface compositions")
    fig.tight_layout()
    _show_if_interactive()
    return fig, axes


def print_final_equilibrium_estimates(model, tieline_surrogate):
    """Prints idealized, initial-inventory, and calculated final interface positions."""
    left_idealized, right_idealized, meta_idealized = tieline_surrogate.getTielineOfGlobalComposition(
        idealized_comp,
        T=TEMPERATURE,
        returnMeta=True,
    )
    initial_comp = model.inventoryData._y[0] / model._R
    left_initial, right_initial, meta_initial = tieline_surrogate.getTielineOfGlobalComposition(
        initial_comp,
        T=TEMPERATURE,
        returnMeta=True,
    )
    final_idealized_fraction = (
        meta_idealized["phase_fraction"]
        if meta_idealized["phase_fraction_phase"] == "BCC_A2"
        else 1 - meta_idealized["phase_fraction"]
    )
    final_initial_fraction = (
        meta_initial["phase_fraction"]
        if meta_initial["phase_fraction_phase"] == "BCC_A2"
        else 1 - meta_initial["phase_fraction"]
    )
    print(f"idealized final normalized interface position:  {(final_idealized_fraction * model._R) / model.interfaceData._y[0]}")
    print(f"initial final normalized interface position:    {(final_initial_fraction * model._R) / model.interfaceData._y[0]}")
    print(f"calculated final normalized interface position: {model.interfaceData._y[-1] / model.interfaceData._y[0]}")
    return {
        "idealized": (left_idealized, right_idealized, meta_idealized),
        "initial_inventory": (left_initial, right_initial, meta_initial),
    }


def default_diffusivity_comparison_config():
    """
    Returns a mutable default config for fixed/variable diffusivity comparisons.

    The default sweep first runs a fine variable-diffusivity Picard reference,
    then runs the same fine surrogate with lagged bulk diffusivity, fixed
    diffusivity cases sampled at several tie-line eta values, and coarser
    variable diffusivity cases for both lagged and Picard bulk solves. Three
    surrogate parameter sets vary the regular bulk grid density and tie-line
    eta sampling.
    """
    return {
        "base_overrides": {
            "VERBOSE": False,
            "PLOT_LEE_OH_FIG9_DATA": False,
        },
        "fixed_overrides": {
            "DIFFUSIVITY_SOURCE": "fixed",
            "DIFFUSIVITY_MODE": DIFFUSIVITY_MODE,
        },
        "fixed_sample_etas": [0.4, 0.5, 0.6],
        "reference_parameter_set": {
            "label": "reference_grid51_eta101",
            "VARIABLE_DIFFUSIVITY_INTERPOLATION": "continuous_grid",
            "VARIABLE_DIFFUSIVITY_BULK_CR_AXIS": np.linspace(0.10, 0.55, 51),
            "VARIABLE_DIFFUSIVITY_BULK_NI_AXIS": np.linspace(0.0001, 0.25, 51),
            "ETA_SAMPLES": np.linspace(0.0, 1.0, 101),
            "VARIABLE_DIFFUSIVITY_BULK_MODE": "picard",
        },
        "reference_comparison_modes": ["lagged"],
        "variable_modes": ["lagged", "picard"],
        "surrogate_parameter_sets": [
            {
                "label": "grid11_eta21",
                "VARIABLE_DIFFUSIVITY_INTERPOLATION": "continuous_grid",
                "VARIABLE_DIFFUSIVITY_BULK_CR_AXIS": np.linspace(0.10, 0.55, 11),
                "VARIABLE_DIFFUSIVITY_BULK_NI_AXIS": np.linspace(0.0001, 0.25, 11),
                "ETA_SAMPLES": np.linspace(0.0, 1.0, 21),
            },
            {
                "label": "grid101_eta201",
                "VARIABLE_DIFFUSIVITY_INTERPOLATION": "continuous_grid",
                "VARIABLE_DIFFUSIVITY_BULK_CR_AXIS": np.linspace(0.10, 0.55, 101),
                "VARIABLE_DIFFUSIVITY_BULK_NI_AXIS": np.linspace(0.0001, 0.25, 101),
                "ETA_SAMPLES": np.linspace(0.0, 1.0, 201),
            },
            # {
            #     "label": "grid19_eta21",
            #     "VARIABLE_DIFFUSIVITY_INTERPOLATION": "continuous_grid",
            #     "VARIABLE_DIFFUSIVITY_BULK_CR_AXIS": np.linspace(0.10, 0.55, 19),
            #     "VARIABLE_DIFFUSIVITY_BULK_NI_AXIS": np.linspace(0.0001, 0.25, 19),
            #     "ETA_SAMPLES": np.linspace(0.0, 1.0, 21),
            # },
            # {
            #     "label": "grid19_eta31",
            #     "VARIABLE_DIFFUSIVITY_INTERPOLATION": "continuous_grid",
            #     "VARIABLE_DIFFUSIVITY_BULK_CR_AXIS": np.linspace(0.10, 0.55, 19),
            #     "VARIABLE_DIFFUSIVITY_BULK_NI_AXIS": np.linspace(0.0001, 0.25, 19),
            #     "ETA_SAMPLES": np.linspace(0.0, 1.0, 31),
            # },
        ],
        "analysis_time_count": 64,
        "keep_models": True,
        "print_matrices": False,
        "progress": True,
    }


def run_diffusivity_comparison(config=None):
    """
    Compares fixed and coarser variable diffusivity solves to a fine reference.

    The returned dictionary contains the normalized config, the fine Picard
    reference run as the baseline, all fine-reference run records, all fixed
    run records, all run records, common analysis times, and a quantitative
    summary. Variable surrogates are built once per surrogate parameter set and
    reused for each requested lagged/Picard solve mode.
    """
    config = _normalize_diffusivity_comparison_config(config)
    progress = bool(config["progress"])
    runs = []
    reference_runs = []
    fixed_runs = []

    reference_set = config["reference_parameter_set"]
    reference_label = str(reference_set["label"])
    reference_mode = _normalize_variable_diffusivity_bulk_mode(reference_set["VARIABLE_DIFFUSIVITY_BULK_MODE"])
    reference_overrides = _comparison_overrides(
        config["base_overrides"],
        {key: value for key, value in reference_set.items() if key != "label"},
        {"DIFFUSIVITY_SOURCE": "variable", "VARIABLE_DIFFUSIVITY_BULK_MODE": reference_mode},
    )
    if progress:
        print(f"[reference:{reference_label}] building fine surrogate context and running {_short_bulk_mode_label(reference_mode)}")
    reference_context = build_case_context(
        overrides=reference_overrides,
        print_matrices=bool(config["print_matrices"]),
    )
    reference_run = _run_diffusivity_comparison_case(
        label=_label_with_bulk_mode(reference_label, reference_mode),
        overrides=reference_overrides,
        context=reference_context,
        keep_model=bool(config["keep_models"]),
        surrogate_label=reference_label,
    )
    reference_runs.append(reference_run)
    runs.append(reference_run)

    for mode in config["reference_comparison_modes"]:
        normalized_mode = _normalize_variable_diffusivity_bulk_mode(mode)
        if normalized_mode == reference_mode:
            continue
        run_context = _context_with_bulk_mode(reference_context, normalized_mode)
        run_overrides = _comparison_overrides(
            reference_overrides,
            {"VARIABLE_DIFFUSIVITY_BULK_MODE": mode},
        )
        if progress:
            print(f"[reference:{reference_label}] running {_short_bulk_mode_label(normalized_mode)}")
        comparison_run = _run_diffusivity_comparison_case(
            label=_label_with_bulk_mode(reference_label, normalized_mode),
            overrides=run_overrides,
            context=run_context,
            keep_model=bool(config["keep_models"]),
            surrogate_label=reference_label,
        )
        reference_runs.append(comparison_run)
        runs.append(comparison_run)

    for fixed_eta in config["fixed_sample_etas"]:
        fixed_overrides = _comparison_overrides(
            config["base_overrides"],
            config["fixed_overrides"],
            {"DIFFUSIVITY_SAMPLE_ETA": fixed_eta},
        )
        label = f"fixed_eta{float(fixed_eta):g}"
        if progress:
            print(f"[fixed:{float(fixed_eta):g}] building context and running")
        fixed_context = build_case_context(
            overrides=fixed_overrides,
            print_matrices=bool(config["print_matrices"]),
        )
        fixed_run = _run_diffusivity_comparison_case(
            label=label,
            overrides=fixed_overrides,
            context=fixed_context,
            keep_model=bool(config["keep_models"]),
        )
        fixed_runs.append(fixed_run)
        runs.append(fixed_run)
    baseline = reference_run

    for parameter_set in config["surrogate_parameter_sets"]:
        parameter_label = str(parameter_set["label"])
        build_overrides = _comparison_overrides(
            config["base_overrides"],
            {key: value for key, value in parameter_set.items() if key != "label"},
            {"DIFFUSIVITY_SOURCE": "variable", "VARIABLE_DIFFUSIVITY_BULK_MODE": config["variable_modes"][0]},
        )
        if progress:
            print(f"[variable:{parameter_label}] building surrogate context")
        variable_context = build_case_context(
            overrides=build_overrides,
            print_matrices=bool(config["print_matrices"]),
        )

        for mode in config["variable_modes"]:
            normalized_mode = _normalize_variable_diffusivity_bulk_mode(mode)
            run_context = _context_with_bulk_mode(variable_context, normalized_mode)
            run_overrides = _comparison_overrides(
                build_overrides,
                {"VARIABLE_DIFFUSIVITY_BULK_MODE": mode},
            )
            label = f"{parameter_label}_{_short_bulk_mode_label(normalized_mode)}"
            if progress:
                print(f"[variable:{parameter_label}] running {mode}")
            runs.append(
                _run_diffusivity_comparison_case(
                    label=label,
                    overrides=run_overrides,
                    context=run_context,
                    keep_model=bool(config["keep_models"]),
                    surrogate_label=parameter_label,
                )
            )

    analysis_times = _common_comparison_times(runs, count=config["analysis_time_count"], include_zero=True)
    summary = summarize_diffusivity_comparison(runs, baseline, analysis_times)
    return {
        "config": config,
        "baseline": baseline,
        "reference_run": reference_run,
        "reference_runs": reference_runs,
        "fixed_runs": fixed_runs,
        "runs": runs,
        "analysis_times": analysis_times,
        "summary": summary,
    }


def summarize_diffusivity_comparison(runs, baseline=None, analysis_times=None):
    """
    Returns a quantitative table comparing runs to the selected baseline.

    A ``pandas.DataFrame`` is returned when pandas is available; otherwise the
    fallback is a list of dictionaries with the same columns.
    """
    runs = list(runs)
    if not runs:
        raise ValueError("At least one comparison run is required.")
    baseline = runs[0] if baseline is None else baseline
    if analysis_times is None:
        analysis_times = _common_comparison_times(runs, count=64, include_zero=True)
    rows = [_diffusivity_comparison_summary_row(run, baseline, analysis_times) for run in runs]
    try:
        import pandas as pd

        return pd.DataFrame(rows)
    except ImportError:
        return rows


def _normalize_diffusivity_comparison_config(config):
    defaults = default_diffusivity_comparison_config()
    if config is not None:
        merged = dict(defaults)
        for key, value in dict(config).items():
            if key in {"base_overrides", "fixed_overrides"}:
                merged[key] = {**dict(defaults.get(key, {})), **dict(value)}
            elif key == "reference_parameter_set":
                merged[key] = {**dict(defaults.get(key, {})), **dict(value)}
            elif key == "surrogate_parameter_sets":
                merged[key] = [dict(item) for item in value]
            else:
                merged[key] = value
        defaults = merged
    defaults["base_overrides"] = dict(defaults.get("base_overrides", {}))
    defaults["fixed_overrides"] = dict(defaults.get("fixed_overrides", {}))
    defaults["fixed_sample_etas"] = [float(eta) for eta in defaults.get("fixed_sample_etas", [])]
    defaults["reference_parameter_set"] = dict(defaults.get("reference_parameter_set", {}))
    defaults["reference_comparison_modes"] = list(defaults.get("reference_comparison_modes", []))
    defaults["variable_modes"] = list(defaults.get("variable_modes", ["lagged", "picard"]))
    defaults["surrogate_parameter_sets"] = [dict(item) for item in defaults.get("surrogate_parameter_sets", [])]
    defaults["analysis_time_count"] = int(defaults.get("analysis_time_count", 64))
    defaults["keep_models"] = bool(defaults.get("keep_models", True))
    defaults["print_matrices"] = bool(defaults.get("print_matrices", False))
    defaults["progress"] = bool(defaults.get("progress", True))
    defaults["reference_parameter_set"].setdefault("label", "reference")
    defaults["reference_parameter_set"].setdefault("VARIABLE_DIFFUSIVITY_BULK_MODE", "picard")
    if not defaults["fixed_sample_etas"]:
        raise ValueError("At least one fixed diffusivity sample eta is required.")
    if not defaults["variable_modes"]:
        raise ValueError("At least one variable diffusivity mode is required.")
    if not defaults["surrogate_parameter_sets"]:
        raise ValueError("At least one surrogate parameter set is required.")
    for i, parameter_set in enumerate(defaults["surrogate_parameter_sets"]):
        parameter_set.setdefault("label", f"surrogate_{i}")
    return defaults


def _comparison_overrides(*overrides):
    merged = {}
    for override in overrides:
        if override:
            merged.update(dict(override))
    return merged


def _context_with_bulk_mode(context, bulk_mode):
    run_context = dict(context)
    solver_options = dict(run_context["solver_options"])
    solver_options["bulk_diffusivity_mode"] = bulk_mode
    run_context["bulk_diffusivity_mode"] = bulk_mode
    run_context["solver_options"] = solver_options
    return run_context


def _run_diffusivity_comparison_case(label, overrides, context, *, keep_model=True, surrogate_label=None):
    start = time.perf_counter()
    run = run_case(overrides=overrides, context=context, make_plots=False)
    runtime_s = time.perf_counter() - start
    record = _extract_diffusivity_comparison_record(
        label,
        run["model"],
        run["context"],
        overrides,
        runtime_s=runtime_s,
        surrogate_label=surrogate_label,
    )
    if keep_model:
        record["model"] = run["model"]
    return record


def _extract_diffusivity_comparison_record(label, model, context, overrides, *, runtime_s=0.0, surrogate_label=None):
    n_interface = model.interfaceData.N + 1
    n_eta = model.etaData.N + 1
    n_inventory = model.inventoryData.N + 1
    p_final = np.asarray(model.pData._y[model.pData.N], dtype=np.float64).copy() if model.pData is not None else None
    q_final = np.asarray(model.qData._y[model.qData.N], dtype=np.float64).copy() if model.qData is not None else None
    grids = context["tieline_surrogate"].diffusivityBulkGridAxes
    grid_counts = None if grids is None else tuple(int(axis.size) for axis in grids)
    return {
        "label": str(label),
        "surrogate_label": surrogate_label,
        "diffusivity_source": context["diffusivity_source"],
        "bulk_diffusivity_mode": context["bulk_diffusivity_mode"],
        "surrogate_interpolation": getattr(context["tieline_surrogate"], "diffusivityInterpolation", None),
        "surrogate_grid_counts": grid_counts,
        "overrides": dict(overrides),
        "runtime_s": float(runtime_s),
        "step_count": int(max(n_interface, 1) - 1),
        "final_time": float(model.currentTime),
        "times": np.asarray(model.interfaceData._time[:n_interface], dtype=np.float64),
        "interface_position": np.asarray(model.interfaceData._y[:n_interface], dtype=np.float64),
        "eta_times": np.asarray(model.etaData._time[:n_eta], dtype=np.float64),
        "eta": np.asarray(model.etaData._y[:n_eta], dtype=np.float64),
        "inventory_times": np.asarray(model.inventoryData._time[:n_inventory], dtype=np.float64),
        "inventory": np.asarray(model.inventoryData._y[:n_inventory], dtype=np.float64),
        "u_grid": np.asarray(model._u_grid, dtype=np.float64).copy(),
        "v_grid": np.asarray(model._v_grid, dtype=np.float64).copy(),
        "p_final": p_final,
        "q_final": q_final,
        "bulk_provider_calls": int(getattr(model, "_lastBulkDiffusivityProviderCalls", 0)),
        "bulk_face_matrices_evaluated": int(getattr(model, "_lastBulkFaceMatricesEvaluated", 0)),
        "bulk_left_picard_iterations": int(getattr(model, "_lastBulkLeftPicardIterations", 0)),
        "bulk_right_picard_iterations": int(getattr(model, "_lastBulkRightPicardIterations", 0)),
        "bulk_converged": getattr(model, "_lastBulkConverged", None),
        "bulk_failure_reason": getattr(model, "_lastBulkFailureReason", None),
    }


def _diffusivity_comparison_summary_row(run, baseline, analysis_times):
    run_s = _interp_comparison_scalar(run["times"], run["interface_position"], analysis_times)
    base_s = _interp_comparison_scalar(baseline["times"], baseline["interface_position"], analysis_times)
    run_eta = _interp_comparison_scalar(run["eta_times"], run["eta"], analysis_times)
    base_eta = _interp_comparison_scalar(baseline["eta_times"], baseline["eta"], analysis_times)
    initial_s = float(baseline["interface_position"][0])
    profile_errors = _comparison_profile_errors(run, baseline)
    final_inventory = np.asarray(run["inventory"][-1], dtype=np.float64)
    baseline_inventory = np.asarray(baseline["inventory"][-1], dtype=np.float64)
    grid_counts = run["surrogate_grid_counts"]
    return {
        "label": run["label"],
        "surrogate_label": run["surrogate_label"],
        "diffusivity_source": run["diffusivity_source"],
        "bulk_diffusivity_mode": run["bulk_diffusivity_mode"],
        "surrogate_interpolation": run["surrogate_interpolation"],
        "diffusivity_sample_eta": run["overrides"].get("DIFFUSIVITY_SAMPLE_ETA", np.nan),
        "bulk_grid_cr_count": np.nan if grid_counts is None else grid_counts[0],
        "bulk_grid_ni_count": np.nan if grid_counts is None else grid_counts[1],
        "runtime_s": run["runtime_s"],
        "step_count": run["step_count"],
        "final_time": run["final_time"],
        "final_interface_position": float(run["interface_position"][-1]),
        "final_normalized_interface_position": float(run["interface_position"][-1] / run["interface_position"][0]),
        "final_eta": float(run["eta"][-1]),
        "final_inventory_cr": float(final_inventory[0]),
        "final_inventory_ni": float(final_inventory[1]),
        "delta_final_interface_position": float(run["interface_position"][-1] - baseline["interface_position"][-1]),
        "delta_final_normalized_interface_position": float(
            run["interface_position"][-1] / run["interface_position"][0]
            - baseline["interface_position"][-1] / baseline["interface_position"][0]
        ),
        "delta_final_eta": float(run["eta"][-1] - baseline["eta"][-1]),
        "delta_final_inventory_cr": float(final_inventory[0] - baseline_inventory[0]),
        "delta_final_inventory_ni": float(final_inventory[1] - baseline_inventory[1]),
        "max_normalized_interface_delta": float(np.max(np.abs((run_s - base_s) / initial_s))),
        "max_eta_delta": float(np.max(np.abs(run_eta - base_eta))),
        "final_profile_linf_delta": profile_errors["final_profile_linf_delta"],
        "final_profile_rms_delta": profile_errors["final_profile_rms_delta"],
        "bulk_provider_calls": run["bulk_provider_calls"],
        "bulk_face_matrices_evaluated": run["bulk_face_matrices_evaluated"],
        "bulk_left_picard_iterations": run["bulk_left_picard_iterations"],
        "bulk_right_picard_iterations": run["bulk_right_picard_iterations"],
        "bulk_converged": run["bulk_converged"],
        "bulk_failure_reason": run["bulk_failure_reason"],
    }


def _common_comparison_times(runs, count=64, include_zero=True):
    if not runs:
        return np.asarray([], dtype=np.float64)
    count = max(2, int(count))
    final_time = min(float(np.max(run["times"])) for run in runs)
    positive_starts = []
    for run in runs:
        times = np.asarray(run["times"], dtype=np.float64)
        positives = times[times > 0.0]
        if positives.size:
            positive_starts.append(float(positives[0]))
    if not positive_starts or final_time <= 0.0:
        return np.asarray([0.0], dtype=np.float64) if include_zero else np.asarray([], dtype=np.float64)
    start = max(positive_starts)
    if start >= final_time:
        values = np.asarray([final_time], dtype=np.float64)
    else:
        values = np.geomspace(start, final_time, count)
    if include_zero:
        values = np.concatenate(([0.0], values))
    return np.unique(values)


def _interp_comparison_scalar(times, values, query_times):
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    query_times = np.asarray(query_times, dtype=np.float64)
    if times.size == 0:
        return np.full_like(query_times, np.nan, dtype=np.float64)
    return np.interp(query_times, times, values)


def _comparison_profile_errors(run, baseline):
    if run.get("p_final") is None or baseline.get("p_final") is None:
        return {
            "final_profile_linf_delta": np.nan,
            "final_profile_rms_delta": np.nan,
        }
    p_interp = _interp_comparison_profile(run["u_grid"], run["p_final"], baseline["u_grid"])
    q_interp = _interp_comparison_profile(run["v_grid"], run["q_final"], baseline["v_grid"])
    p_delta = p_interp - np.asarray(baseline["p_final"], dtype=np.float64)
    q_delta = q_interp - np.asarray(baseline["q_final"], dtype=np.float64)
    combined = np.concatenate((p_delta, q_delta), axis=0)
    return {
        "final_profile_linf_delta": float(np.max(np.abs(combined))),
        "final_profile_rms_delta": float(np.sqrt(np.mean(combined * combined))),
    }


def _interp_comparison_profile(source_grid, source_profile, target_grid):
    source_grid = np.asarray(source_grid, dtype=np.float64)
    source_profile = np.asarray(source_profile, dtype=np.float64)
    target_grid = np.asarray(target_grid, dtype=np.float64)
    values = np.empty((target_grid.size, source_profile.shape[1]), dtype=np.float64)
    for component in range(source_profile.shape[1]):
        values[:, component] = np.interp(target_grid, source_grid, source_profile[:, component])
    return values


def _short_bulk_mode_label(mode):
    if mode == "composition_dependent_lagged":
        return "lagged"
    if mode == "composition_dependent_implicit":
        return "picard"
    return str(mode)


def _label_with_bulk_mode(label, mode):
    short_mode = _short_bulk_mode_label(mode)
    label = str(label)
    if label.endswith(f"_{short_mode}"):
        return label
    return f"{label}_{short_mode}"


def run_interactive_example(overrides=None, timeProfiling=False):
    """
    Runs the original notebook-style Fe-Cr-Ni example flow.

    This preserves the old script behavior for direct execution while keeping
    module imports free of thermodynamics setup, solving, and plotting side
    effects.
    """
    with _temporary_config(overrides):
        context = build_case_context(print_matrices=True)
        tieline_surrogate = context["tieline_surrogate"]
        print(f"Built tie-line surrogate with eta bounds {tieline_surrogate.eta_bounds}.")
        initial_eta_bracket_to_print = tieline_surrogate.eta_bounds if INITIAL_ETA_BRACKET is None else INITIAL_ETA_BRACKET
        print(f"Initial eta will be estimated with method '{INITIAL_ETA_METHOD}' over {initial_eta_bracket_to_print}.")

        run = run_case(context=context, make_plots=False)
        model = run["model"]
        print("Built MovingBoundaryIllingworthTernaryFD1DModel.")
        print(f"Initial interface position = {INTERFACE_POSITION}")
        print("Initial eta and inventory estimates will be available after setup/solve.")

        if timeProfiling:
            return run
        
        if model.currentTime > 0:
            print_initial_eta_estimate(model)
            print(f"Finished solve at t = {model.currentTime}.")
            print(f"Final interface position = {model.getInterfacePosition()}")
            print(f"Final integrated inventory [CR, NI] = {model.getTotalInventory()}")
            run["figures"]["interface_position"] = plot_interface_position(
                model,
                plot_lee_oh_fig9_data=PLOT_LEE_OH_FIG9_DATA,
                xlims=(1e-1, 1e7),
            )
            run["figures"]["integrated_inventory"] = plot_integrated_inventory(model)
            run["figures"]["interface_compositions"] = plot_interface_compositions(model, tieline_surrogate)
            run["final_equilibrium_estimates"] = print_final_equilibrium_estimates(model, tieline_surrogate)
        else:
            print("RUN_SOLVE is False. Set RUN_SOLVE = True in the configuration cell to run the solve.")
            print("No solve has been run yet, so there are no histories to plot.")
        return run


def run_convergence_demo():
    """Runs a small two-node/timestep convergence sweep for interactive use."""
    from examples.ternaryExamples import IllingworthTernaryConvergence as conv

    cfg = conv.default_convergence_config()
    cfg["nodes"] = [31, 61, 121]
    cfg["semi_log_dt"] = [0.1, 0.05, 0.025, 0.01]
    results = conv.run_convergence_sweep(cfg)
    summary = conv.summarize_convergence(results)
    return results, summary


def run_diffusivity_comparison_demo(
    *,
    fixed_sample_etas=None,
    reference_parameter_set=None,
    reference_comparison_modes=None,
    plot_normalized_interface_positions=False,
    interface_plot_scale_time=1.0,
    interface_plot_normalize_to=None,
    interface_plot_lee_oh_fig9_data=False,
    interface_plot_xlims=None,
):
    """
    Runs the default fixed-vs-variable diffusivity comparison sweep.

    ``fixed_sample_etas`` overrides the default fixed-diffusivity tie-line eta
    sweep, which is ``[0.4, 0.5, 0.6]``.
    ``reference_parameter_set`` can override the fine Picard reference settings
    used as the quantitative baseline.
    ``reference_comparison_modes`` controls which additional fine-reference
    solve modes are included alongside the baseline; the default is ``["lagged"]``.
    When ``plot_normalized_interface_positions`` is true, a shared line plot of
    normalized interface position histories is stored in
    ``results["figures"]["normalized_interface_positions"]``.
    """
    cfg = default_diffusivity_comparison_config()
    if fixed_sample_etas is not None:
        cfg["fixed_sample_etas"] = list(fixed_sample_etas)
    if reference_parameter_set is not None:
        cfg["reference_parameter_set"] = {**cfg["reference_parameter_set"], **dict(reference_parameter_set)}
    if reference_comparison_modes is not None:
        cfg["reference_comparison_modes"] = list(reference_comparison_modes)
    results = run_diffusivity_comparison(cfg)
    if plot_normalized_interface_positions:
        results.setdefault("figures", {})["normalized_interface_positions"] = plot_diffusivity_comparison_interface_positions(
            results,
            scale_time=interface_plot_scale_time,
            normalize_to=interface_plot_normalize_to,
            plot_lee_oh_fig9_data=interface_plot_lee_oh_fig9_data,
            xlims=interface_plot_xlims,
        )
    return results, results["summary"]


if __name__ == "__main__":
#     def debugInPlace():
#         try:
#             import debugpy
#             # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#             debugpy.listen(5678)
#             print("Waiting for debugger attach")
#             debugpy.wait_for_client()
#             debugpy.breakpoint()
#             print('break on this line')
#         except:
#             pass
#     debugInPlace()
    run_results = run_interactive_example(timeProfiling=False)

# %%
# if __name__ == "__main__":
#     results, summary = run_convergence_demo()
# %%
if __name__ == "__main__":
    results, summary = run_diffusivity_comparison_demo(
        plot_normalized_interface_positions=True,
        interface_plot_xlims=(1e-1, 1e7),
    )
# %%

# %%
def run_surrogate_validation_demo(make_plot=True):
    """
    Builds a variable-diffusivity surrogate and runs dense validation reports.

    The surrogate stores ``TDB_PATH`` as validation metadata during construction,
    so ``compare_diffusivity_to_ground_truth`` can rebuild the thermodynamics
    source without passing a database or thermodynamics object here.
    """
    validation_context = build_case_context(
        overrides={
            "DIFFUSIVITY_SOURCE": "variable",
            "VARIABLE_DIFFUSIVITY_INTERPOLATION": "continuous_grid",
            "VARIABLE_DIFFUSIVITY_BULK_CR_AXIS": np.linspace(0.10, 0.55, 19),
            "VARIABLE_DIFFUSIVITY_BULK_NI_AXIS": np.linspace(0.0001, 0.25, 19),
            "ETA_SAMPLES": np.linspace(0.0, 1.0, 21),
            "RUN_SOLVE": False,
            "VERBOSE": False,
        },
        print_matrices=False,
    )
    validation_surrogate = validation_context["tieline_surrogate"]
    print(f"Stored validation database: {validation_surrogate.metadata.get('validation_database')}")

    matrix_report = validation_surrogate.validate_diffusivity_matrices(
        matrix_interface_eta_count=401,
        matrix_bulk_grid_counts=(151, 151),
    )
    print(matrix_report["summary"])
    print(matrix_report["interface"]["summary"])
    print(matrix_report["bulk"]["summary"])

    truth_report = validation_surrogate.compare_diffusivity_to_ground_truth(
        error_interface_eta_count=201,
        error_bulk_grid_counts=(61, 61),
    )
    print(truth_report["summary"])
    print(truth_report["interface"]["summary"])
    print(truth_report["bulk"]["summary"])

    figures = {}
    if make_plot:
        bulkOrInterface = ["bulk", "interface"][1]
        alpha_BOI = truth_report[bulkOrInterface]["phases"][TIELINE_PHASES[0]]
        alpha_BOI_distance = alpha_BOI["nearest_training_distance"]
        alpha_BOI_max_rel_error = np.max(alpha_BOI["relative_error"], axis=(1, 2))
        alpha_BOI_max_abs_error = np.max(alpha_BOI["absolute_error"], axis=(1, 2))

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(alpha_BOI_distance, alpha_BOI_max_rel_error, s=12)
        ax.set_xlabel(f"distance to nearest {bulkOrInterface} training point")
        ax.set_ylabel("max elementwise relative diffusivity error")
        ax.set_yscale("log")
        fig.tight_layout()
        _show_if_interactive()
        figures[f"alpha_{bulkOrInterface}_relError_distance"] = (fig, ax)

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.scatter(alpha_BOI_distance, alpha_BOI_max_abs_error, s=12)
        ax2.set_xlabel(f"distance to nearest {bulkOrInterface} training point")
        ax2.set_ylabel("max elementwise absolute diffusivity error")
        ax2.set_yscale("log")
        fig2.tight_layout()
        _show_if_interactive()
        figures[f"alpha_{bulkOrInterface}_absError_distance"] = (fig2, ax2)

        from typing import Any


        from numpy.typing import ArrayLike, NDArray


        def normalized_matrix_errors(
            true_matrices: ArrayLike,
            predicted_matrices: ArrayLike,
            *,
            denominator_floor: float = 0.0,
        ) -> dict[str, NDArray[np.float64]]:
            """
            Calculate absolute and normalized Frobenius and spectral matrix errors.

            Parameters
            ----------
            true_matrices
                Ground-truth matrices with shape (..., m, n).
            predicted_matrices
                Surrogate matrices with the same shape.
            denominator_floor
                Minimum denominator used for normalized errors. This has the same
                units as the matrix entries.

            Returns
            -------
            dict
                Arrays with shape true_matrices.shape[:-2].
            """
            true = np.asarray(true_matrices)
            predicted = np.asarray(predicted_matrices)

            if true.shape != predicted.shape:
                raise ValueError(
                    "True and predicted matrices must have the same shape; "
                    f"received {true.shape} and {predicted.shape}."
                )

            if true.ndim < 2:
                raise ValueError("Inputs must contain at least one matrix.")

            if denominator_floor < 0:
                raise ValueError("denominator_floor must be nonnegative.")

            if not np.all(np.isfinite(true)):
                raise ValueError("Ground-truth matrices contain nonfinite values.")

            if not np.all(np.isfinite(predicted)):
                raise ValueError("Predicted matrices contain nonfinite values.")

            difference = predicted - true
            matrix_axes = (-2, -1)

            frobenius_absolute = np.linalg.norm(
                difference,
                ord="fro",
                axis=matrix_axes,
            )
            spectral_absolute = np.linalg.norm(
                difference,
                ord=2,
                axis=matrix_axes,
            )

            frobenius_reference = np.linalg.norm(
                true,
                ord="fro",
                axis=matrix_axes,
            )
            spectral_reference = np.linalg.norm(
                true,
                ord=2,
                axis=matrix_axes,
            )

            # np.finfo(...).tiny prevents division by zero even when the requested
            # physical floor is zero.
            numerical_floor = np.finfo(np.result_type(true, predicted, float)).tiny
            floor = max(float(denominator_floor), numerical_floor)
            
            return {
                "frobenius_absolute": frobenius_absolute,
                "frobenius_normalized": (
                    frobenius_absolute
                    / np.maximum(frobenius_reference, floor)
                ),
                "spectral_absolute": spectral_absolute,
                "spectral_normalized": (
                    spectral_absolute
                    / np.maximum(spectral_reference, floor)
                ),
                "frobenius_reference": frobenius_reference,
                "spectral_reference": spectral_reference,
            }

        errors_alpha_BOI = normalized_matrix_errors(
            true_matrices=truth_report[bulkOrInterface]['phases'][TIELINE_PHASES[0]]['truth_matrices'].copy(),
            predicted_matrices=truth_report[bulkOrInterface]['phases'][TIELINE_PHASES[0]]['surrogate_matrices'].copy(),
            denominator_floor=1e-30,
        )

        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.scatter(alpha_BOI_distance, errors_alpha_BOI["spectral_normalized"], s=12, alpha=0.5)
        ax3.set_xlabel(f"distance to nearest {bulkOrInterface} training point")
        ax3.set_ylabel("spectral_normalized error")
        ax3.set_yscale("log")
        fig3.tight_layout()
        _show_if_interactive()
        figures[f"alpha_{bulkOrInterface}_spectralNorm_distance"] = (fig3, ax3)

        fig4, ax4 = plt.subplots(figsize=(5, 4))
        ax4.scatter(alpha_BOI_distance, errors_alpha_BOI["frobenius_normalized"], s=12, alpha=0.5)
        ax4.set_xlabel(f"distance to nearest {bulkOrInterface} training point")
        ax4.set_ylabel("frobenius_normalized error")
        ax4.set_yscale("log")
        fig4.tight_layout()
        _show_if_interactive()
        figures[f"alpha_{bulkOrInterface}_frobeniusNorm_distance"] = (fig4, ax4)

        fig5, ax5 = plt.subplots(figsize=(5, 4))
        ax5.scatter(alpha_BOI_max_abs_error, errors_alpha_BOI["frobenius_normalized"], s=12, alpha=0.5)
        ax5.set_xlabel(f"alpha_{bulkOrInterface}_max_abs_error")
        ax5.set_ylabel("frobenius_normalized error")
        ax5.set_xscale("log")
        ax5.set_yscale("log")
        fig5.tight_layout()
        _show_if_interactive()
        figures[f"alpha_{bulkOrInterface}_max_abs_error vs frobeniusNorm"] = (fig5, ax5)

        fig5, ax5 = plt.subplots(figsize=(5, 4))
        ax5.scatter(alpha_BOI_max_rel_error, errors_alpha_BOI["frobenius_normalized"], s=12, alpha=0.5)
        ax5.set_xlabel(f"alpha_{bulkOrInterface}_max_rel_error")
        ax5.set_ylabel("frobenius_normalized error")
        ax5.set_xscale("log")
        ax5.set_yscale("log")
        fig5.tight_layout()
        _show_if_interactive()
        figures[f"alpha_{bulkOrInterface}_max_rel_error vs frobeniusNorm"] = (fig5, ax5)

        fig5, ax5 = plt.subplots(figsize=(5, 4))
        ax5.scatter(errors_alpha_BOI["spectral_normalized"], errors_alpha_BOI["frobenius_normalized"], s=12, alpha=0.5)
        ax5.set_xlabel("spectral_normalized error")
        ax5.set_ylabel("frobenius_normalized error")
        ax5.set_xscale("log")
        ax5.set_yscale("log")
        fig5.tight_layout()
        _show_if_interactive()
        figures["spectral_Norm vs frobeniusNorm"] = (fig5, ax5)


    return {
        "context": validation_context,
        "surrogate": validation_surrogate,
        "matrix_report": matrix_report,
        "truth_report": truth_report,
        "figures": figures,
    }


if __name__ == "__main__":
    validation_results = run_surrogate_validation_demo()
# %%
