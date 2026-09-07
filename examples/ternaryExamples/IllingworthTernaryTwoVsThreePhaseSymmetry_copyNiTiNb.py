# %%

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "kawin").exists():
            return candidate
    return Path.cwd()


THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = _find_repo_root(THIS_DIR)
EXAMPLES_DIR = REPO_ROOT / "examples"

# Running this file directly places examples/ternaryExamples first on
# sys.path.  That lets pycalphad plugin discovery import the neighboring
# pycalphad_default_phase_adapter.py as a top-level module, before
# kawin.thermo has finished importing, which creates a circular import.
_this_dir_resolved = THIS_DIR.resolve()


def _is_this_dir_sys_path_entry(entry):
    path = Path.cwd() if entry == "" else Path(entry)
    return path.resolve() == _this_dir_resolved


sys.path[:] = [
    entry
    for entry in sys.path
    if not _is_this_dir_sys_path_entry(entry)
]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kawin.diffusion import (
    MovingBoundaryIllingworthTernaryFD1DModel,
    TernaryMovingBoundaryThermodynamicsSurrogate,
)
from kawin.diffusion.MovingBoundaryIllingworthTernaryThreePhaseFDM import (
    MovingBoundaryIllingworthTernaryThreePhaseFD1DModel,
)
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.solver import explicitEulerIterator
from kawin.thermo import MulticomponentThermodynamics
from examples.ThermoCalc.tc_python_adapter import TCPythonThermodynamics, ThermoCalcConfig
from examples.ternaryExamples.pycalphad_default_phase_adapter import create_pycalphad_thermodynamics_source

OUTPUTS = REPO_ROOT / "examples" / "ThermoCalc" / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)


# %%
# Fe-Cr-Ni case configuration adapted from IllingworthTernaryExamples.py
THERM_ENGINE = ["PYCALPAHD", "TC"][-1]

TC_USE_DEFAULT_PHASES = True
GLOBAL_MINIMIZATION_MAX_GRID_POINTS = 2000
PYCALPHAD_USE_DEFAULT_PHASES = True
PYCALPHAD_EQUILIBRIUM_PHASES = None
G_OFFSET = 0.0
TDB_PATH = ""

ELEMENTS = ["NB", "NI", "TI"]
INDEPENDENT_ELEMENTS = ["NI", "TI"]
REFERENCE_ELEMENT = "NB"

PHASE_A = "BCC_B2#2"
PHASE_B = "BCC_B2#1"
TIELINE_PHASES = (PHASE_A, PHASE_B)

TEMPERATURE = 1398.0

TIELINE_SURROGATE_BUILD_MODE = "seed_point"
# Probe compositions are independent components [X(CR), X(NI)].
PROBE_START = np.array([-1, -1], dtype=np.float64)
PROBE_END = np.array([-1, -1], dtype=np.float64)
ETA_SAMPLES = np.linspace(0.0, 1.0, 21)

PROBE_POINT = np.array([0.26, 0.20], dtype=np.float64)
PROBE_SAMPLES_PER_SIDE = 50
PROBE_BOUNDARY_MARGIN = 1.0e-3
PROBE_BOUNDARY_SEARCH_STEP = 1.0e-2
PROBE_BOUNDARY_XTOL = 1.0e-6
PROBE_MAX_SEARCH_STEPS = 200

INITIAL_ETA_BRACKET = (1.0e-3, 1.0 - 1.0e-3)
INITIAL_ETA_GUESS = None
INITIAL_VELOCITY_GUESS_2PHASE = None
INITIAL_VELOCITY_GUESS_3PHASE = None

# Two-phase geometry: [0, HALF_LENGTH].
HALF_LENGTH = 200.0e-6
TWO_PHASE_NODES = 201
INTERFACE_POSITION = 100.0e-6 + 1.0e-12

# The symmetric three-phase geometry is [0, 2*HALF_LENGTH].
THREE_PHASE_LENGTH = 2.0 * HALF_LENGTH
THREE_PHASE_NODES = 2 * TWO_PHASE_NODES - 1
INTERFACE_POSITIONS_3PHASE = np.array(
    [INTERFACE_POSITION, 2.0 * HALF_LENGTH - INTERFACE_POSITION],
    dtype=np.float64,
)

# Initial bulk values: BCC on the outer regions, FCC in the middle.
A_BULK = np.array([0.4995, 0.4995], dtype=np.float64)
B_BULK = np.array([0.001, 0.001], dtype=np.float64) # np.array([0.001, 0.625], dtype=np.float64) #np.array([0.001, 0.6], dtype=np.float64)

# Freeze one phase matrix sampled at this tie line.  Using identical matrices
# in both solvers is intentional: this benchmark is primarily a solver
# reduction/symmetry check rather than a variable-D comparison.
DIFFUSIVITY_SAMPLE_ETA = 0.5

# Time stepping.  semi_log mirrors the style of IllingworthTernaryExamples.py
# while keeping both solvers on the same requested target-time schedule.
DT_MODE = "semi_log"
FIXED_TIME_STEP = 1.0
SEMI_LOG_BASE_TIME_STEP = 1.0
SEMI_LOG_DT = 0.25 / 10.0 # 0.25 / 10.0
SEMI_LOG_T0 = 1.0e-6
# SOLVE_TIME = 3600.0*1e3
SOLVE_TIME = 3600*24 #6.92

TOLERANCE = 1.0e-12
RESIDUAL_TOLERANCE = None
MAX_ITERATIONS = 25
VERBOSE = True
VERBOSE_INTERVAL = 10
MIN_DT_FRAC = 1.0e-16
TERMINAL_THIN_PHASE_POLICY='continue'

# Plot/output controls
TIME_SCALE = 1
TIME_LABEL = "s"
SAVE_FIGURES = False
FIGURE_DIR = THIS_DIR / "illingworth_two_vs_three_phase_figures"

# For an exact reduction test, explicitly match the transformed-grid physical
# resolution.  The three-phase FCC interval has twice the physical width of
# the two-phase FCC interval, so it gets twice as many transformed intervals.
#
# The values below reproduce the node counts that the 31-node, 30 um
# two-phase physical mesh would naturally give near a 12 um interface.
PHASE_A_NODES_2 = 100
PHASE_B_NODES_2 = 100
PHASE_A_NODES_3 = PHASE_A_NODES_2
PHASE_B_NODES_3 = 2 * (PHASE_B_NODES_2 - 1) + 1
PHASE_C_NODES_3 = PHASE_A_NODES_2

U_A_2 = np.linspace(0.0, 1.0, PHASE_A_NODES_2)
V_B_2 = np.linspace(0.0, 1.0, PHASE_B_NODES_2)

U_A_3 = U_A_2.copy()
U_B_3 = np.linspace(0.0, 1.0, PHASE_B_NODES_3)
U_C_3 = U_A_2.copy()


# %%
# Small helper objects

class FixedMatrixTernaryDiffusivity:
    """Thermodynamics-like object returning one fixed 2x2 matrix per phase."""

    def __init__(self, diffusivity_matrices, phases, temperature=None):
        self.phases = list(phases)
        self.temperature = None if temperature is None else float(temperature)
        self.diffusivity_matrices = {}

        for phase in set(self.phases):
            if phase not in diffusivity_matrices:
                raise ValueError(f"Missing fixed diffusivity matrix for phase '{phase}'.")
            matrix = np.asarray(diffusivity_matrices[phase], dtype=np.float64)
            if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
                raise ValueError(
                    f"Fixed diffusivity matrix for phase '{phase}' must be finite "
                    "with shape (2, 2)."
                )
            self.diffusivity_matrices[phase] = matrix.copy()

    def clearCache(self):
        return

    def getInterdiffusivity(self, x, T=None, phase=None, query_context=None, **kwargs):
        if phase is None:
            phase = self.phases[0]
        if phase not in self.diffusivity_matrices:
            raise ValueError(
                f"Unknown phase '{phase}'. Expected one of "
                f"{sorted(self.diffusivity_matrices)}."
            )

        if self.temperature is not None and T is not None:
            T_values = np.asarray(T, dtype=np.float64)
            if not np.allclose(
                T_values, self.temperature, rtol=0.0, atol=1.0e-8
            ):
                raise ValueError(
                    f"Fixed diffusivity object is isothermal at "
                    f"{self.temperature}; received T={T}."
                )

        matrix = self.diffusivity_matrices[phase]
        values = np.asarray(x, dtype=np.float64)
        if values.ndim <= 1:
            return matrix.copy()
        return np.tile(matrix, (values.shape[0], 1, 1))


class ReversedInterfaceEquilibrium:
    """
    Reverses an A|B equilibrium closure to B|A while preserving eta.

    If the base closure returns
        (c_A(eta), c_B(eta)),
    this wrapper returns
        (c_B(eta), c_A(eta)).

    This is what makes the right FCC|BCC interface use exactly the same
    physical tie line as the left BCC|FCC interface.
    """

    def __init__(self, base):
        self.base = base

    @property
    def eta_bounds(self):
        return self.base.eta_bounds

    def interface_compositions(self, eta):
        c_a, c_b = self.base.interface_compositions(float(eta))
        return (
            np.asarray(c_b, dtype=np.float64).copy(),
            np.asarray(c_a, dtype=np.float64).copy(),
        )


class SymmetricABAProfile:
    """Piecewise-constant A|B|A initial profile on a one-dimensional mesh."""

    def __init__(self, s_left, s_right, a_value, b_value):
        self.s_left = float(s_left)
        self.s_right = float(s_right)
        self.a_value = np.asarray(a_value, dtype=np.float64).reshape(2)
        self.b_value = np.asarray(b_value, dtype=np.float64).reshape(2)

    def __call__(self, z):
        x = np.atleast_2d(z)[:, 0]
        y = np.tile(self.a_value, (len(x), 1))
        in_b = (x > self.s_left) & (x < self.s_right)
        y[in_b] = self.b_value
        return y




# %%
# Thermodynamics / surrogate

def _make_tc_config(phases):
    """Returns a TC-Python config for one ordered two-phase interface."""
    return ThermoCalcConfig(
        thermodynamic_database="TCHEA5",
        kinetic_database="MOBHEA4",
        elements=ELEMENTS,
        phases=tuple(phases),
        reference_element=REFERENCE_ELEMENT,
        use_default_phases=TC_USE_DEFAULT_PHASES,
        global_minimization_max_grid_points=GLOBAL_MINIMIZATION_MAX_GRID_POINTS,
        cache_dir=OUTPUTS / "tc_cache",
    )

def build_source_thermodynamics():
    if THERM_ENGINE=="PYCALPHAD":
        if not TDB_PATH.exists():
            raise FileNotFoundError(f"Could not find Fe-Cr-Ni TDB at {TDB_PATH}.")
        therm_ab = create_pycalphad_thermodynamics_source(
                str(TDB_PATH),
                list(ELEMENTS),
                list(TIELINE_PHASES),
                use_default_phases=PYCALPHAD_USE_DEFAULT_PHASES,
                equilibrium_phases=PYCALPHAD_EQUILIBRIUM_PHASES,
                g_offset=G_OFFSET,
            ),
            
    elif THERM_ENGINE=="TC":
        therm_ab = TCPythonThermodynamics(_make_tc_config(TIELINE_PHASES))
    else:
        raise ValueError(f"THERM_ENGINE should be one of above options. Instead got: {THERM_ENGINE}")
    return therm_ab

def _tieline_surrogate_probe_kwargs(probe_start, probe_end, probe_point):
    """Returns from_database probe arguments for line or seed-point surrogate builds."""
    mode = str(TIELINE_SURROGATE_BUILD_MODE).lower()
    if mode == "line":
        return {
            "probe_start": np.asarray(probe_start, dtype=np.float64),
            "probe_end": np.asarray(probe_end, dtype=np.float64),
            "eta_samples": np.asarray(ETA_SAMPLES, dtype=np.float64),
        }
    if mode == "seed_point":
        return {
            "probe_point": np.asarray(probe_point, dtype=np.float64),
            "probe_samples_per_side": PROBE_SAMPLES_PER_SIDE,
            "probe_boundary_search_step": PROBE_BOUNDARY_SEARCH_STEP,
            "probe_boundary_xtol": PROBE_BOUNDARY_XTOL,
            "probe_max_search_steps": PROBE_MAX_SEARCH_STEPS,
        }
    raise ValueError("TIELINE_SURROGATE_BUILD_MODE must be 'line' or 'seed_point'.")

def build_tieline_surrogate(source_thermodynamics):
    pair_ab = TIELINE_PHASES
    with source_thermodynamics:
            kwargs_ab = {}
            # if CASE_NAME in [CASE_FE_CR_NI_PYCALPHAD, CASE_FE_CR_NI_TC]:
            #     kwargs_ab["validation_database"] = TDB_PATH
            surrogate_ab = TernaryMovingBoundaryThermodynamicsSurrogate.from_database(
                thermodynamics=source_thermodynamics,
                elements=ELEMENTS,
                phases=pair_ab,
                tieline_phases=pair_ab,
                temperature=TEMPERATURE,
                **_tieline_surrogate_probe_kwargs(PROBE_START, PROBE_END, PROBE_POINT),
                precipitate_phase=pair_ab[1],
                # **diffusivity_sampling,
                **kwargs_ab,
            )
    return surrogate_ab
    # return TernaryMovingBoundaryThermodynamicsSurrogate.from_database(
    #     thermodynamics=source_thermodynamics,
    #     elements=ELEMENTS,
    #     phases=list(TIELINE_PHASES),
    #     tieline_phases=TIELINE_PHASES,
    #     temperature=TEMPERATURE,
    #     # probe_start=PROBE_START,
    #     # probe_end=PROBE_END,
    #     # eta_samples=ETA_SAMPLES,
    #     probe_point=PROBE_POINT,
    #     probe_samples_per_side=PROBE_SAMPLES_PER_SIDE,
    #     probe_boundary_search_step=PROBE_BOUNDARY_SEARCH_STEP,
    #     probe_boundary_xtol=PROBE_BOUNDARY_XTOL,
    #     probe_max_search_steps=PROBE_MAX_SEARCH_STEPS,
    #     precipitate_phase=TIELINE_PHASES[1],
    # )


def select_fixed_diffusivity_matrices(source_thermodynamics, tieline_surrogate):
    c_a, c_b = tieline_surrogate.interface_compositions(
        float(DIFFUSIVITY_SAMPLE_ETA)
    )
    matrices = {
        PHASE_A: np.asarray(
            source_thermodynamics.getInterdiffusivity(
                c_a, TEMPERATURE, phase=PHASE_A
            ),
            dtype=np.float64,
        ).reshape(2, 2),
        PHASE_B: np.asarray(
            source_thermodynamics.getInterdiffusivity(
                c_b, TEMPERATURE, phase=PHASE_B
            ),
            dtype=np.float64,
        ).reshape(2, 2),
    }

    print("Fixed diffusivity matrices used by BOTH comparison models:")
    for phase, matrix in matrices.items():
        print(f"{phase}:\n{matrix}")

    return matrices


# %%
# Mesh/model construction

def make_two_phase_mesh():
    mesh = CartesianFD1D(
        INDEPENDENT_ELEMENTS,
        [0.0, HALF_LENGTH],
        TWO_PHASE_NODES,
    )
    profile = ProfileBuilder(
        [
            (
                StepProfile1D(
                    INTERFACE_POSITION,
                    A_BULK,
                    B_BULK,
                ),
                INDEPENDENT_ELEMENTS,
            )
        ]
    )
    mesh.setResponseProfile(profile)
    return mesh


def make_three_phase_mesh():
    mesh = CartesianFD1D(
        INDEPENDENT_ELEMENTS,
        [0.0, THREE_PHASE_LENGTH],
        THREE_PHASE_NODES,
    )
    profile = ProfileBuilder(
        [
            (
                SymmetricABAProfile(
                    INTERFACE_POSITIONS_3PHASE[0],
                    INTERFACE_POSITIONS_3PHASE[1],
                    A_BULK,
                    B_BULK,
                ),
                INDEPENDENT_ELEMENTS,
            )
        ]
    )
    mesh.setResponseProfile(profile)
    return mesh


def get_time_step_options():
    if DT_MODE == "fixed":
        return {
            "time_step": float(FIXED_TIME_STEP),
            "dt_mode": "fixed",
            "semiLog_dt": None,
            "semiLogT0": None,
        }

    if DT_MODE == "semi_log":
        return {
            "time_step": float(SEMI_LOG_BASE_TIME_STEP),
            "dt_mode": "semi_log",
            "semiLog_dt": float(SEMI_LOG_DT),
            "semiLogT0": float(SEMI_LOG_T0),
        }

    raise ValueError("DT_MODE must be either 'fixed' or 'semi_log'.")


def build_two_phase_model(tieline_surrogate, fixed_diffusivity):
    time_options = get_time_step_options()
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=make_two_phase_mesh(),
        elements=ELEMENTS,
        phases=[PHASE_A, PHASE_B],
        thermodynamics=fixed_diffusivity,
        temperature=TEMPERATURE,
        interfacePosition=INTERFACE_POSITION,
        interface_equilibrium=tieline_surrogate,
        initial_eta_method="instantaneous_balance",
        initial_eta_bracket=INITIAL_ETA_BRACKET,
        initial_eta_guess=INITIAL_ETA_GUESS,
        initial_velocity_guess=INITIAL_VELOCITY_GUESS_2PHASE,
        time_step=time_options["time_step"],
        dt_mode=time_options["dt_mode"],
        semiLog_dt=time_options["semiLog_dt"],
        semiLogT0=time_options["semiLogT0"],
        tolerance=TOLERANCE,
        residual_tolerance=RESIDUAL_TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        terminal_thin_phase_policy=TERMINAL_THIN_PHASE_POLICY,
        record=True,
        record_pq_data=True,
        transformed_u_grid=U_A_2,
        transformed_v_grid=V_B_2,
    )


def build_three_phase_model(tieline_surrogate, fixed_diffusivity):
    time_options = get_time_step_options()

    reverse_equilibrium = ReversedInterfaceEquilibrium(tieline_surrogate)

    return MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=make_three_phase_mesh(),
        elements=ELEMENTS,
        phases=[PHASE_A, PHASE_B, PHASE_A],
        thermodynamics=fixed_diffusivity,
        temperature=TEMPERATURE,
        interfacePositions=INTERFACE_POSITIONS_3PHASE,
        interface_equilibria=(tieline_surrogate, reverse_equilibrium),
        initial_eta_method="instantaneous_balance",
        initial_eta_brackets=(
            INITIAL_ETA_BRACKET,
            INITIAL_ETA_BRACKET,
        ),
        initial_eta_guess=(
            None
            if INITIAL_ETA_GUESS is None
            else np.array([INITIAL_ETA_GUESS, INITIAL_ETA_GUESS])
        ),
        initial_velocity_guess=INITIAL_VELOCITY_GUESS_3PHASE,
        time_step=time_options["time_step"],
        dt_mode=time_options["dt_mode"],
        semiLog_dt=time_options["semiLog_dt"],
        semiLogT0=time_options["semiLogT0"],
        tolerance=TOLERANCE,
        residual_tolerance=RESIDUAL_TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        terminal_thin_phase_policy=TERMINAL_THIN_PHASE_POLICY,
        record=True,
        record_pq_data=True,
        transformed_grids=(U_A_3, U_B_3, U_C_3),
    )


# %%
# History / comparison helpers

def _history(history):
    n = history.N + 1
    t = np.asarray(history._time[:n], dtype=np.float64)
    y = np.asarray(history._y[:n], dtype=np.float64)
    return t, y


def _interp_columns(target_t, source_t, source_y):
    source_y = np.asarray(source_y, dtype=np.float64)
    if source_y.ndim == 1:
        return np.interp(target_t, source_t, source_y)
    return np.column_stack(
        [
            np.interp(target_t, source_t, source_y[:, j])
            for j in range(source_y.shape[1])
        ]
    )


def _relative_l2(error, reference):
    error = np.asarray(error, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    denom = float(np.linalg.norm(reference.ravel()))
    return float(np.linalg.norm(error.ravel()) / max(denom, 1.0e-300))


def _max_relative(error, reference):
    error = np.asarray(error, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    scale = max(float(np.max(np.abs(reference))), 1.0e-300)
    return float(np.max(np.abs(error)) / scale)


def _final_independent_profile(model):
    return np.asarray(model.data.y(model.currentTime), dtype=np.float64)


def _physical_coordinates(model):
    return np.asarray(model.mesh.z, dtype=np.float64).reshape(-1)


def compare_models(two_phase, three_phase):
    t2, s2 = _history(two_phase.interfaceData)
    t3, s3 = _history(three_phase.interfaceData)

    _, eta2 = _history(two_phase.etaData)
    t3_eta, eta3 = _history(three_phase.etaData)

    _, inv2 = _history(two_phase.inventoryData)
    _, inv3 = _history(three_phase.inventoryData)

    # Restrict to times represented by both histories, then interpolate the
    # three-phase solution to two-phase record times.
    common_mask = (t2 >= t3[0]) & (t2 <= t3[-1])
    tc = t2[common_mask]
    s2c = s2[common_mask]
    eta2c = eta2[common_mask]

    s3c = _interp_columns(tc, t3, s3)
    eta3c = _interp_columns(tc, t3_eta, eta3)

    s3_left = s3c[:, 0]
    s3_right_equivalent = 2.0 * HALF_LENGTH - s3c[:, 1]

    b_half_2 = HALF_LENGTH - s2c
    b_half_3 = 0.5 * (s3c[:, 1] - s3c[:, 0])

    interface_left_error = s3_left - s2c
    interface_right_error = s3_right_equivalent - s2c
    interface_symmetry_error = s3c[:, 0] + s3c[:, 1] - THREE_PHASE_LENGTH
    b_width_error = b_half_3 - b_half_2

    eta_left_error = eta3c[:, 0] - eta2c
    eta_right_error = eta3c[:, 1] - eta2c
    eta_symmetry_error = eta3c[:, 0] - eta3c[:, 1]

    # Final physical profiles.  The physical meshes were constructed with the
    # same dx, but interpolate anyway so this remains robust to future edits.
    x2 = _physical_coordinates(two_phase)
    c2 = _final_independent_profile(two_phase)

    x3 = _physical_coordinates(three_phase)
    c3 = _final_independent_profile(three_phase)

    c3_on_x2 = np.column_stack(
        [np.interp(x2, x3, c3[:, j]) for j in range(c3.shape[1])]
    )
    profile_reduction_error = c3_on_x2 - c2

    # Mirror the full three-phase profile around x=HALF_LENGTH.
    x3_mirror = THREE_PHASE_LENGTH - x3
    c3_mirror = np.column_stack(
        [
            np.interp(x3, x3_mirror[::-1], c3[::-1, j])
            for j in range(c3.shape[1])
        ]
    )
    profile_symmetry_error = c3 - c3_mirror

    inv2_drift = inv2 - inv2[0]
    inv3_drift = inv3 - inv3[0]

    metrics = {
        "max_abs_left_interface_error_m": float(
            np.max(np.abs(interface_left_error))
        ),
        "max_abs_right_mirrored_interface_error_m": float(
            np.max(np.abs(interface_right_error))
        ),
        "max_abs_three_phase_interface_symmetry_error_m": float(
            np.max(np.abs(interface_symmetry_error))
        ),
        "max_abs_half_B_width_error_m": float(
            np.max(np.abs(b_width_error))
        ),
        "max_abs_left_eta_error": float(np.max(np.abs(eta_left_error))),
        "max_abs_right_eta_error": float(np.max(np.abs(eta_right_error))),
        "max_abs_three_phase_eta_symmetry_error": float(
            np.max(np.abs(eta_symmetry_error))
        ),
        "final_profile_relative_L2_error": _relative_l2(
            profile_reduction_error, c2
        ),
        "final_profile_max_relative_error": _max_relative(
            profile_reduction_error, c2
        ),
        "final_three_phase_profile_symmetry_relative_L2_error": _relative_l2(
            profile_symmetry_error, c3
        ),
        "max_abs_two_phase_inventory_drift": float(
            np.max(np.abs(inv2_drift))
        ),
        "max_abs_three_phase_inventory_drift": float(
            np.max(np.abs(inv3_drift))
        ),
    }

    arrays = {
        "time": tc,
        "s2": s2c,
        "s3": s3c,
        "s3_right_equivalent": s3_right_equivalent,
        "interface_left_error": interface_left_error,
        "interface_right_error": interface_right_error,
        "interface_symmetry_error": interface_symmetry_error,
        "b_half_2": b_half_2,
        "b_half_3": b_half_3,
        "b_width_error": b_width_error,
        "eta2": eta2c,
        "eta3": eta3c,
        "eta_left_error": eta_left_error,
        "eta_right_error": eta_right_error,
        "eta_symmetry_error": eta_symmetry_error,
        "x2": x2,
        "c2": c2,
        "x3": x3,
        "c3": c3,
        "c3_on_x2": c3_on_x2,
        "profile_reduction_error": profile_reduction_error,
        "profile_symmetry_error": profile_symmetry_error,
        "t_inv2": _history(two_phase.inventoryData)[0],
        "inv2_drift": inv2_drift,
        "t_inv3": _history(three_phase.inventoryData)[0],
        "inv3_drift": inv3_drift,
    }

    return metrics, arrays


def print_metrics(metrics):
    print("\n" + "=" * 78)
    print("TWO-PHASE / THREE-PHASE SYMMETRY COMPARISON")
    print("=" * 78)

    interface_metrics = [
        "max_abs_left_interface_error_m",
        "max_abs_right_mirrored_interface_error_m",
        "max_abs_three_phase_interface_symmetry_error_m",
        "max_abs_half_B_width_error_m",
    ]
    eta_metrics = [
        "max_abs_left_eta_error",
        "max_abs_right_eta_error",
        "max_abs_three_phase_eta_symmetry_error",
    ]
    profile_metrics = [
        "final_profile_relative_L2_error",
        "final_profile_max_relative_error",
        "final_three_phase_profile_symmetry_relative_L2_error",
    ]
    conservation_metrics = [
        "max_abs_two_phase_inventory_drift",
        "max_abs_three_phase_inventory_drift",
    ]

    for title, names in (
        ("Interface/width metrics", interface_metrics),
        ("Eta metrics", eta_metrics),
        ("Final profile metrics", profile_metrics),
        ("Inventory metrics", conservation_metrics),
    ):
        print(f"\n{title}")
        for name in names:
            print(f"  {name:52s} = {metrics[name]:.6e}")

    print(
        "\nInterpretation:\n"
        "  - Left-interface, mirrored-right-interface, B-half-width, and eta\n"
        "    errors test reduction of A|B|A to the two-phase A|B problem.\n"
        "  - Three-phase symmetry errors test mirror symmetry independently\n"
        "    of the two-phase implementation.\n"
        "  - Profile errors compare the left half of the final A|B|A profile\n"
        "    with the final A|B profile.\n"
        "  - Inventory drift tests conservation, but is not itself a\n"
        "    two-vs-three reduction metric.\n"
    )


# %%
# Plotting

def _positive_time_mask(t):
    return np.asarray(t) > 0.0


def _save_or_show(fig, filename):
    fig.tight_layout()
    if SAVE_FIGURES:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURE_DIR / filename
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved {path}")
    return fig


def plot_interface_comparison(arrays):
    t = arrays["time"] / TIME_SCALE
    mask = _positive_time_mask(t)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(t[mask], 1.0e6 * arrays["s2"][mask], label="2-phase A|B")
    ax.plot(
        t[mask],
        1.0e6 * arrays["s3"][mask, 0],
        "--",
        label="3-phase left A|B",
    )
    ax.plot(
        t[mask],
        1.0e6 * arrays["s3_right_equivalent"][mask],
        ":",
        label="3-phase right B|A, mirrored",
    )
    ax.set_xscale("log")
    ax.set_xlabel(f"time / {TIME_LABEL}")
    ax.set_ylabel("left-equivalent interface position / µm")
    ax.set_title("Interface-position reduction check")
    ax.legend()
    return _save_or_show(fig, "interface_comparison.png")


def plot_middle_width_comparison(arrays):
    t = arrays["time"] / TIME_SCALE
    mask = _positive_time_mask(t)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        t[mask],
        1.0e6 * arrays["b_half_2"][mask],
        label="2-phase FCC width",
    )
    ax.plot(
        t[mask],
        1.0e6 * arrays["b_half_3"][mask],
        "--",
        label="half of 3-phase FCC width",
    )
    ax.set_xscale("log")
    ax.set_xlabel(f"time / {TIME_LABEL}")
    ax.set_ylabel("FCC half-width / µm")
    ax.set_title("Middle-phase width reduction check")
    ax.legend()
    return _save_or_show(fig, "middle_width_comparison.png")


def plot_eta_comparison(arrays):
    t = arrays["time"] / TIME_SCALE
    mask = _positive_time_mask(t)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(t[mask], arrays["eta2"][mask], label="2-phase eta")
    ax.plot(
        t[mask],
        arrays["eta3"][mask, 0],
        "--",
        label="3-phase left eta",
    )
    ax.plot(
        t[mask],
        arrays["eta3"][mask, 1],
        ":",
        label="3-phase right eta",
    )
    ax.set_xscale("log")
    ax.set_xlabel(f"time / {TIME_LABEL}")
    ax.set_ylabel("tie-line parameter eta")
    ax.set_title("Interface tie-line reduction check")
    ax.legend()
    return _save_or_show(fig, "eta_comparison.png")


def plot_final_profiles(arrays):
    x2_um = 1.0e6 * arrays["x2"]
    x3_um = 1.0e6 * arrays["x3"]
    c2 = arrays["c2"]
    c3 = arrays["c3"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.25), sharex=False)

    for j, (element, ax) in enumerate(zip(INDEPENDENT_ELEMENTS, axes)):
        ax.plot(
            x3_um,
            c3[:, j],
            linewidth=1.2,
            alpha=0.55,
            label="3-phase full A|B|A",
        )
        ax.plot(
            x2_um,
            c2[:, j],
            linewidth=2.0,
            label="2-phase A|B",
        )
        ax.plot(
            x2_um,
            arrays["c3_on_x2"][:, j],
            "--",
            linewidth=1.5,
            label="3-phase left half",
        )
        ax.axvline(1.0e6 * HALF_LENGTH, linestyle=":", linewidth=1.0)
        ax.set_xlabel("position / µm")
        ax.set_ylabel(f"X({element})")
        ax.set_title(element)
        ax.legend()

    fig.suptitle(f"Final composition profiles at t = {SOLVE_TIME / 3600.0:g} h")
    return _save_or_show(fig, "final_profile_comparison.png")


def plot_discrepancies(arrays):
    t = arrays["time"] / TIME_SCALE
    mask = _positive_time_mask(t)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.25))

    ax = axes[0]
    ax.plot(
        t[mask],
        1.0e6 * np.abs(arrays["interface_left_error"][mask]),
        label="left vs 2-phase",
    )
    ax.plot(
        t[mask],
        1.0e6 * np.abs(arrays["interface_right_error"][mask]),
        label="mirrored right vs 2-phase",
    )
    ax.plot(
        t[mask],
        1.0e6 * np.abs(arrays["interface_symmetry_error"][mask]),
        label="3-phase mirror symmetry",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"time / {TIME_LABEL}")
    ax.set_ylabel("absolute interface discrepancy / µm")
    ax.set_title("Interface discrepancies")
    ax.legend()

    ax = axes[1]
    ax.plot(
        t[mask],
        np.abs(arrays["eta_left_error"][mask]),
        label="left eta vs 2-phase",
    )
    ax.plot(
        t[mask],
        np.abs(arrays["eta_right_error"][mask]),
        label="right eta vs 2-phase",
    )
    ax.plot(
        t[mask],
        np.abs(arrays["eta_symmetry_error"][mask]),
        label="3-phase eta symmetry",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"time / {TIME_LABEL}")
    ax.set_ylabel("absolute eta discrepancy")
    ax.set_title("Eta discrepancies")
    ax.legend()

    return _save_or_show(fig, "discrepancies.png")


def plot_inventory_drift(arrays):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.25))

    for j, (element, ax) in enumerate(zip(INDEPENDENT_ELEMENTS, axes)):
        t2 = arrays["t_inv2"] / TIME_SCALE
        t3 = arrays["t_inv3"] / TIME_SCALE
        m2 = _positive_time_mask(t2)
        m3 = _positive_time_mask(t3)

        ax.plot(
            t2[m2],
            np.abs(arrays["inv2_drift"][m2, j]),
            label="2-phase",
        )
        ax.plot(
            t3[m3],
            np.abs(arrays["inv3_drift"][m3, j]),
            "--",
            label="3-phase",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"time / {TIME_LABEL}")
        ax.set_ylabel(f"|inventory drift in {element}|")
        ax.set_title(element)
        ax.legend()

    fig.suptitle("Conservation diagnostics")
    return _save_or_show(fig, "inventory_drift.png")


# %%
# Build the shared Fe-Cr-Ni thermodynamic data

source_thermodynamics = build_source_thermodynamics()
tieline_surrogate = build_tieline_surrogate(source_thermodynamics)

print(f"Built Fe-Cr-Ni tie-line surrogate with eta bounds {tieline_surrogate.eta_bounds}.")

fixed_diffusivity_matrices = select_fixed_diffusivity_matrices(
    source_thermodynamics,
    tieline_surrogate,
)

fixed_diffusivity = FixedMatrixTernaryDiffusivity(
    fixed_diffusivity_matrices,
    phases=[PHASE_A, PHASE_B],
    temperature=TEMPERATURE,
)


# %%
# Build matched models

two_phase_model = build_two_phase_model(
    tieline_surrogate,
    fixed_diffusivity,
)
three_phase_model = build_three_phase_model(
    tieline_surrogate,
    fixed_diffusivity,
)

print("\nTwo-phase geometry:")
print(f"  domain = [0, {HALF_LENGTH:.8e}] m")
print(f"  initial interface = {INTERFACE_POSITION:.8e} m")
print(
    f"  initial FCC width = "
    f"{HALF_LENGTH - INTERFACE_POSITION:.8e} m"
)

print("\nThree-phase geometry:")
print(f"  domain = [0, {THREE_PHASE_LENGTH:.8e}] m")
print(f"  initial interfaces = {INTERFACE_POSITIONS_3PHASE} m")
print(
    f"  initial FCC width = "
    f"{INTERFACE_POSITIONS_3PHASE[1] - INTERFACE_POSITIONS_3PHASE[0]:.8e} m"
)
print(
    "  FCC width ratio (3-phase / 2-phase) = "
    f"{(INTERFACE_POSITIONS_3PHASE[1] - INTERFACE_POSITIONS_3PHASE[0]) / (HALF_LENGTH - INTERFACE_POSITION):.12g}"
)

print("\nTransformed-grid counts:")
print(
    f"  2-phase: A={len(U_A_2)}, B={len(V_B_2)}; "
    f"3-phase: A={len(U_A_3)}, B={len(U_B_3)}, C={len(U_C_3)}"
)


# %%
# Solve both cases
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

print("\nSolving two-phase A|B case...")
two_phase_model.solve(
    SOLVE_TIME,
    iterator=explicitEulerIterator,
    verbose=VERBOSE,
    vIt=VERBOSE_INTERVAL,
    minDtFrac=MIN_DT_FRAC,
)

print("\nSolving symmetric three-phase A|B|A case...")
three_phase_model.solve(
    SOLVE_TIME,
    iterator=explicitEulerIterator,
    verbose=VERBOSE,
    vIt=VERBOSE_INTERVAL,
    minDtFrac=MIN_DT_FRAC,
)

print("\nSolve complete.")
print(
    f"  two-phase final time   = {two_phase_model.currentTime:.12g} s"
)
print(
    f"  three-phase final time = {three_phase_model.currentTime:.12g} s"
)


# %%
# Quantitative comparison

metrics, comparison = compare_models(two_phase_model, three_phase_model)
print_metrics(metrics)


# %%
# Plots

plot_interface_comparison(comparison)
plot_middle_width_comparison(comparison)
plot_eta_comparison(comparison)
plot_final_profiles(comparison)
plot_discrepancies(comparison)
plot_inventory_drift(comparison)

plt.show()

# %%
