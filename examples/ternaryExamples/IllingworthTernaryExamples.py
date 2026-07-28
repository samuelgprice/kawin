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

from pathlib import Path

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
INITIAL_ETA_METHOD = ["stefan_cross_brentq", "instantaneous_balance"][1]
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
LEFT_BULK = np.array([0.38, 0.001], dtype=np.float64)
RIGHT_BULK = np.array([0.13, 0.15], dtype=np.float64)

idealized_comp = LEFT_BULK * (INTERFACE_POSITION/LENGTH) + RIGHT_BULK * (1 - INTERFACE_POSITION/LENGTH)

# Two diffusivity modes are supported:
#   "explicit"       -> set FIXED_DIFFUSIVITY_MATRICES below.
#   "sample_tieline" -> sample one tie-line from the real thermodynamics object
#                       and freeze one matrix per phase for the solver.
DIFFUSIVITY_MODE = "sample_tieline"
DIFFUSIVITY_SAMPLE_ETA = 0.5
FIXED_DIFFUSIVITY_MATRICES = None
# Example explicit form:
# FIXED_DIFFUSIVITY_MATRICES = {
#     "BCC_A2": np.array([[1.0e-15, 0.0], [0.0, 1.0e-15]], dtype=np.float64),
#     "FCC_A1": np.array([[1.0e-16, 0.0], [0.0, 1.0e-16]], dtype=np.float64),
# }

# Time stepping:
#   DT_MODE = "fixed"    -> advance by FIXED_TIME_STEP.
#   DT_MODE = "semi_log" -> advance to semi-log-spaced target times starting
#                           at SEMI_LOG_T0 with natural-log spacing SEMI_LOG_DT.
#                           SEMI_LOG_BASE_TIME_STEP remains a positive fallback
#                           scale used internally by the model.
DT_MODE = "semi_log"
if DT_MODE=="fixed":
    FIXED_TIME_STEP = 1.0
else:
    SEMI_LOG_BASE_TIME_STEP = 1.0
    SEMI_LOG_DT = 0.25 / 10
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


def build_tieline_surrogate(source_thermodynamics):
    """Samples the Fe-Cr-Ni tie-line family used by the interface solver."""
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
    )


def _print_diffusivity_matrices(matrices):
    print("Fixed diffusivity matrices used by the Illingworth solve:")
    for phase in TIELINE_PHASES:
        print(f"{phase}:\n{np.asarray(matrices[phase], dtype=np.float64)}")


def select_fixed_diffusivity_matrices(source_thermodynamics, tieline_surrogate):
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

    _print_diffusivity_matrices(matrices)
    return matrices


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


def build_model(tieline_surrogate, fixed_diffusivity):
    """Constructs the ternary Illingworth model without starting the solve."""
    time_step_options = get_time_step_options()
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=make_mesh(),
        elements=ELEMENTS,
        phases=list(TIELINE_PHASES),
        thermodynamics=fixed_diffusivity,
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
        record=True,
    )


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


# %%
# Build Fe-Cr-Ni thermodynamics and the tie-line surrogate

source_thermodynamics = build_source_thermodynamics()
tieline_surrogate = build_tieline_surrogate(source_thermodynamics)

print(f"Built tie-line surrogate with eta bounds {tieline_surrogate.eta_bounds}.")
initial_eta_bracket_to_print = tieline_surrogate.eta_bounds if INITIAL_ETA_BRACKET is None else INITIAL_ETA_BRACKET
print(f"Initial eta will be estimated with method '{INITIAL_ETA_METHOD}' over {initial_eta_bracket_to_print}.")


# %%
# Choose fixed diffusivity matrices

fixed_diffusivity_matrices = select_fixed_diffusivity_matrices(source_thermodynamics, tieline_surrogate)
fixed_diffusivity = FixedMatrixTernaryDiffusivity(
    fixed_diffusivity_matrices,
    phases=TIELINE_PHASES,
    temperature=TEMPERATURE,
)


# %%
# Build the mesh and model

model = build_model(tieline_surrogate, fixed_diffusivity)
print("Built MovingBoundaryIllingworthTernaryFD1DModel.")
print(f"Initial interface position = {INTERFACE_POSITION}")
print("Initial eta and inventory estimates will be available after setup/solve.")


# %%
# Solve

if RUN_SOLVE:
    model.solve(
        SOLVE_TIME,
        iterator=explicitEulerIterator,
        verbose=VERBOSE,
        vIt=VERBOSE_INTERVAL,
        minDtFrac=MIN_DT_FRAC,
    )
    print_initial_eta_estimate(model)
    print(f"Finished solve at t = {model.currentTime}.")
    print(f"Final interface position = {model.getInterfacePosition()}")
    print(f"Final integrated inventory [CR, NI] = {model.getTotalInventory()}")
else:
    print("RUN_SOLVE is False. Set RUN_SOLVE = True in the configuration cell to run the solve.")


# %%
# Plot interface position over time

if model.currentTime > 0:
    fig, ax = plot_interface_position(model, plot_lee_oh_fig9_data=PLOT_LEE_OH_FIG9_DATA, xlims=(1e-1, 1e7))
else:
    print("No solve has been run yet, so there is no interface-position history to plot.")


# %%
# Plot integrated mass/inventory of CR and NI over time

if model.currentTime > 0:
    plot_integrated_inventory(model)
else:
    print("No solve has been run yet, so there is no inventory history to plot.")


# %%
# Plot interface compositions over time

if model.currentTime > 0:
    plot_interface_compositions(model, tieline_surrogate)
else:
    print("No solve has been run yet, so there is no interface-composition history to plot.")

# %%
left, right, meta = tieline_surrogate.getTielineOfGlobalComposition(
    idealized_comp,
    T=TEMPERATURE,
    returnMeta=True,
)
left, right, meta
finalIdealizedFraction = meta['phase_fraction'] if meta['phase_fraction_phase']=="BCC_A2" else 1-meta['phase_fraction']
print(f"idealized final normalized interface position: {(finalIdealizedFraction * model._R) / model.interfaceData._y[0]}")
print(f"calculated final normalized interface position: {model.interfaceData._y[-1] / model.interfaceData._y[0]}")

#%%
