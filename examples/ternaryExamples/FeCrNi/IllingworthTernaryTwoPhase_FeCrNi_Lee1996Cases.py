# %%

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

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
    callable_source_sha256,
    evaluate_interface_bulk_diffusivity_consistency,
    file_sha256,
    plot_selected_diffusivity_calculations,
    plot_surrogate_diagnostics,
    prepare_surrogate_artifact,
)
import kawin.diffusion.MovingBoundarySurrogates as moving_boundary_surrogates_module
import kawin.diffusion.SurrogateArtifacts as surrogate_artifacts_module
import kawin.diffusion.SurrogateDiagnostics as surrogate_diagnostics_module
import kawin.thermo.FreeEnergyHessian as free_energy_hessian_module
import kawin.thermo.Mobility as mobility_module
import kawin.thermo.Thermodynamics as pycalphad_thermodynamics_module
from kawin.diffusion.MovingBoundaryIllingworthTernaryThreePhaseFDM import (
    MovingBoundaryIllingworthTernaryThreePhaseFD1DModel,
)
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.diffusion.mesh.TransformedGrids import two_phase_landau_grids
from kawin.solver import explicitEulerIterator
from kawin.thermo import MulticomponentThermodynamics
from examples.ThermoCalc.tc_python_adapter import TCPythonThermodynamics, ThermoCalcConfig
import examples.ThermoCalc.tc_python_adapter as tc_python_adapter_module
from examples.ternaryExamples.pycalphad_default_phase_adapter import create_pycalphad_thermodynamics_source
import examples.ternaryExamples.pycalphad_default_phase_adapter as pycalphad_adapter_module
from examples.debugInPlace import debugInPlace

OUTPUTS = REPO_ROOT / "examples" / "ThermoCalc" / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
# Rebuilding is intentionally the default. Set this to True only to require an
# existing bundle that exactly matches the current settings and implementation.
RELOAD_TIELINE_SURROGATE = True
# %%
# Repository paths


# %%
# Fe-Cr-Ni case configuration adapted from IllingworthTernaryExamples.py
THERM_ENGINE = ["PYCALPHAD", "TC"][0]

TC_USE_DEFAULT_PHASES = False #True
DEFAULT_REMOVE_CACHE=False
GLOBAL_MINIMIZATION_MAX_GRID_POINTS = 2000
TC_EQUILIBRIUM_QTHISS_RETRY_GRID_POINTS = () #(20_000, 200_000)  # Applies to all global-minimization calculations; set to () to disable.
TC_KINETICS_DISABLE_GLOBAL_MINIMIZATION = True
TC_KINETICS_DISABLE_POSITIVE_DEFINITE_HESSIAN = False
TC_KINETICS_CONSTRAIN_SINGLE_COMPOSITION_SET = True  # Requires local kinetics minimization to prevent set splitting.
TC_KINETICS_MULTISTART_MODE = "global_scout"
TC_KINETICS_SESSION_RESTART_INTERVAL = 100
TC_CAPTURE_DIFFUSIVITY_DIAGNOSTICS = True
TC_DROP_FAILED_BULK_CALCULATIONS = True
TC_DROP_INVALID_BULK_MATRICES = True
TC_CHECK_INTERFACE_BULK_DIFFUSIVITY_CONSISTENCY = True
TC_INTERFACE_BULK_MATRIX_RELATIVE_THRESHOLD = 0.75
TC_INTERFACE_BULK_LOCAL_OUTLIER_FACTOR = 5.0
TC_INTERFACE_BULK_MAX_DISTANCE_RATIO = 1.5
TC_INTERFACE_BULK_SITE_FRACTION_ABSOLUTE_THRESHOLD = 0.1
PYCALPHAD_USE_DEFAULT_PHASES = False
PYCALPHAD_EQUILIBRIUM_PHASES = None
G_OFFSET = 0.0
TDB_PATH = EXAMPLES_DIR / "FeCrNi_Lee1993_L_style_ternary_checked_withMobility.tdb"

systemStr = "FeCrNi"

if systemStr =="FeCrNi":
    ELEMENTS = ["FE", "CR", "NI"]
    INDEPENDENT_ELEMENTS = ["CR", "NI"]
    REFERENCE_ELEMENT = "FE"


if systemStr =="FeCrNi":
    PHASE_A = "BCC_A2"
    PHASE_B = "FCC_A1"


TIELINE_PHASES = (PHASE_A, PHASE_B)
TC_KINETICS_MULTISTART_PHASES = (PHASE_A,)
if systemStr =="FeCrNi":
    TEMPERATURE = 1373.0


TIELINE_SURROGATE_BUILD_MODE = "seed_point"
# Probe compositions are independent components [X(CR), X(NI)].
PROBE_START = np.array([-1, -1], dtype=np.float64)
PROBE_END = np.array([-1, -1], dtype=np.float64)
ETA_SAMPLES = np.linspace(0.0, 1.0, 21)

if systemStr =="FeCrNi":
    PROBE_POINT = np.array([0.45, 0.21], dtype=np.float64)

PROBE_SAMPLES_PER_SIDE = 50
PROBE_BOUNDARY_MARGIN = 1.0e-3
PROBE_BOUNDARY_SEARCH_STEP = 1.0e-2
PROBE_BOUNDARY_XTOL = 1.0e-6
PROBE_MAX_SEARCH_STEPS = 200

INITIAL_ETA_BRACKET = (1.0e-3, 1.0 - 1.0e-3)
INITIAL_ETA_GUESS = None
if systemStr =="FeCrNi":
    INITIAL_ETA_GUESS = 0.5
INITIAL_VELOCITY_GUESS_2PHASE = None
INITIAL_VELOCITY_GUESS_3PHASE = None

# Two-phase geometry: [0, HALF_LENGTH].
multiplier=8
stepDT_multiplier = multiplier#/4
HALF_LENGTH = np.round(30e-6, 16) #200.0e-6 * multiplier
TWO_PHASE_NODES = 61
B_PHASE_LENGTH = 18e-6 #100e-6 * multiplier
if systemStr =="FeCrNi":
    INTERFACE_POSITION = (HALF_LENGTH-(B_PHASE_LENGTH) + 1.0e-12) 


# Optional constant molar volumes [A, B, A] in m^3/mol. For example,
# ``(7.1e-6, 7.8e-6, 7.1e-6)`` enables physical molar diagnostics and
# stress-free motion of the three-phase right material boundary. The two
# outer A volumes must match to preserve the intended A|B|A symmetry. ``None``
# retains the legacy normalized equal-volume comparison.
if systemStr =="FeCrNi":
    PHASE_MOLAR_VOLUMES = None

# Initial bulk values: A on the outer regions, B in the middle.
if systemStr =="FeCrNi":
    A_BULK = np.array([0.38, 0.001], dtype=np.float64)
    B_BULK = np.array([0.13, 0.15], dtype=np.float64)

# ``phase_uniform`` freezes one matrix per phase at the sampled tie line.
# ``composition_dependent_lagged`` samples a phase-specific diffusivity field
# and evaluates its face matrices from the accepted old-time profile.  The
# latter matches the nonconstant-diffusivity option in
# IllingworthTernaryThreePhaseNiTiNb_TC.py.
BULK_DIFFUSIVITY_MODE = ["phase_uniform", "composition_dependent_lagged", "composition_dependent_implicit"][1]
DIFFUSIVITY_INTERPOLATION =  "simplex_positive_2x2" #"nearest" "simplex_linear"

if systemStr =="FeCrNi":
    FECRNI_BCC_DIFFUSIVITY_MATRIX = None

# The default coarse simplex-valid ternary compositions avoid sampling invalid
# Fe-Cr-Ni points while keeping the initial TC-Python build practical. Refine
# flagged or trajectory-adjacent regions separately when higher resolution is
# needed.
# Set this to an explicit ``(n, 2)`` array to use custom samples.
BULK_DIFFUSIVITY_POINTS = None
BULK_DIFFUSIVITY_GRIDS = None
BULK_DIFFUSIVITY_GRID_SPACING = 0.02

# Freeze one phase matrix sampled at this tie line when using ``phase_uniform``.
# Keeping those matrices identical in both solvers supports the original
# reduction/symmetry comparison.
DIFFUSIVITY_SAMPLE_ETA = 0.5

# Time stepping.  semi_log mirrors the style of IllingworthTernaryExamples.py
# while keeping both solvers on the same requested target-time schedule.
DT_MODE = "semi_log"
FIXED_TIME_STEP = 1.0
SEMI_LOG_BASE_TIME_STEP = 1.0

SEMI_LOG_T0 = 1.0e-6
if systemStr =="FeCrNi":
    SEMI_LOG_DT = 0.25 / (20.0 * 2) # 0.25 / 10.0
    SOLVE_TIME = 1e3*3600

TOLERANCE = 1.0e-12 #1.0e-12
RESIDUAL_TOLERANCE = None
MAX_ITERATIONS = 25
if systemStr =="FeCrNi":
    MAX_STEP_RETRIES = 8
VERBOSE = True
VERBOSE_INTERVAL = 10
MIN_DT_FRAC = 1.0e-16
TERMINAL_THIN_PHASE_WIDTH=1e-7
TERMINAL_THIN_PHASE_POLICY='continue'

# Plot/output controls
TIME_SCALE = 1
TIME_LABEL = "s"
SAVE_FIGURES = False
FIGURE_DIR = THIS_DIR / "illingworth_two_vs_three_phase_figures"
PLOT_LEE_OH_FIG9_COMPARISON = True
LEE_OH_FIG9_LOWER_CR_PATH = EXAMPLES_DIR / "leeAndOh1996_data" / "fig9_lowerCurve_Cr.csv"
LEE_OH_FIG9_UPPER_NI_PATH = EXAMPLES_DIR / "leeAndOh1996_data" / "fig9_upperCurve_Ni.csv"
LEE_OH_FIG9_TIME_UNIT_SECONDS = 3600.0
LEE_OH_FIG9_XLIMS_HOURS = (1.0e-3, 1.0e3)
LEE_OH_FIG9_EQUILIBRIUM_NORMALIZED_ALPHA = 0.5

# For an exact reduction test, explicitly match the transformed-grid physical
# resolution.  The three-phase B interval has twice the physical width of
# the two-phase B interval, so it gets twice as many transformed intervals.
#
# The values below reproduce the node counts that the 31-node, 30 um
# two-phase physical mesh would naturally give near a 12 um interface.
PHASE_A_NODES_2 = 200
PHASE_B_NODES_2 = 200


U_A_2, V_B_2 = two_phase_landau_grids(PHASE_A_NODES_2, PHASE_B_NODES_2, method="uniform")


# %%
# Small helper objects
def _validate_2x2_matrix(matrix, label):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite 2x2 matrix.")
    return matrix


def _validate_positive_2x2_matrix(matrix, label):
    """
    Validates a finite 2x2 matrix with positive real eigenvalues.

    The ternary Illingworth bulk solver requires interdiffusivity matrices whose
    normalized eigenvalues stay positive. Continuous surrogate splines are
    checked at build time so runtime evaluation can remain a cheap array call.
    """
    matrix = _validate_2x2_matrix(matrix, label)
    scale = float(np.linalg.norm(matrix, ord=np.inf))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{label} must have nonzero norm.")
    eigenvalues = np.linalg.eigvals(matrix / scale)
    if np.any(np.abs(np.imag(eigenvalues)) > 1e-12) or np.any(np.real(eigenvalues) <= 1e-14):
        debugInPlace()
        raise ValueError(f"{label} must have positive real eigenvalues.")
    return matrix

class FixedPhaseDiffusivityThermodynamics:
    """
    Thermodynamics wrapper that supplies one fixed phase diffusivity matrix.

    This is used for the Fe-Cr-Ni LIQUID phase because the checked-in Lee TDB
    has BCC/FCC mobility terms but no LIQUID mobility terms. Equilibrium and
    all non-fixed diffusivity queries are delegated to the wrapped kawin
    thermodynamics object.
    """

    def __init__(self, thermodynamics, fixed_phase, diffusivity_matrix):
        self.thermodynamics = thermodynamics
        self.fixed_phase = str(fixed_phase)
        self.diffusivity_matrix = _validate_positive_2x2_matrix(diffusivity_matrix, "diffusivity_matrix")
        self.elements = list(getattr(thermodynamics, "elements", ELEMENTS))
        self.phases = list(getattr(thermodynamics, "phases", ()))
        self._kinetics_diagnostic_callback = None
        self._kinetics_diagnostic_index = 0

    def __enter__(self):
        """Enters the wrapped context manager when it has one, otherwise no-ops."""
        enter = getattr(self.thermodynamics, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        """Exits the wrapped context manager when it has one."""
        exit_ = getattr(self.thermodynamics, "__exit__", None)
        if exit_ is not None:
            return exit_(exc_type, exc, traceback)
        return False

    def clearCache(self):
        """Clears the wrapped thermodynamics cache when available."""
        clear_cache = getattr(self.thermodynamics, "clearCache", None)
        if clear_cache is not None:
            clear_cache()

    @contextmanager
    def captureKineticsDiagnostics(self, callback):
        """Capture delegated and fixed-matrix diffusivity queries in one order."""
        if not callable(callback):
            raise TypeError("callback must be callable.")
        previous_callback = self._kinetics_diagnostic_callback
        previous_index = self._kinetics_diagnostic_index
        self._kinetics_diagnostic_callback = callback
        self._kinetics_diagnostic_index = 0

        def forward(record):
            forwarded = dict(record)
            forwarded["query_index"] = self._kinetics_diagnostic_index
            callback(forwarded)
            self._kinetics_diagnostic_index += 1

        capture = getattr(self.thermodynamics, "captureKineticsDiagnostics", None)
        scope = capture(forward) if capture is not None else nullcontext()
        try:
            with scope:
                yield self
        finally:
            self._kinetics_diagnostic_callback = previous_callback
            self._kinetics_diagnostic_index = previous_index

    def getInterfacialComposition(self, *args, **kwargs):
        """Delegates tie-line equilibrium queries to the wrapped thermodynamics object."""
        return self.thermodynamics.getInterfacialComposition(*args, **kwargs)

    def getEquilibriumData(self, *args, **kwargs):
        """Delegates stable-phase equilibrium queries when the wrapped object supports them."""
        equilibrium_data = getattr(self.thermodynamics, "getEquilibriumData", None)
        if equilibrium_data is None:
            raise AttributeError("Wrapped thermodynamics object does not provide getEquilibriumData.")
        return equilibrium_data(*args, **kwargs)

    def getInterdiffusivity(self, x, T=None, phase=None, query_context=None, **kwargs):
        """Returns the fixed matrix for ``fixed_phase`` and delegates all other phases."""
        if str(phase) == self.fixed_phase:
            values = np.asarray(x, dtype=np.float64)
            rows = np.atleast_2d(values)
            if self._kinetics_diagnostic_callback is not None:
                for row in rows:
                    full_composition = [1.0 - float(np.sum(row)), *row.tolist()]
                    self._kinetics_diagnostic_callback({
                        "record_type": "kinetics",
                        "query_index": self._kinetics_diagnostic_index,
                        "cache_hit": False,
                        "requested_phase": self.fixed_phase,
                        "temperature": None if T is None else float(T),
                        "input_composition": row.tolist(),
                        "interdiffusivity": self.diffusivity_matrix.tolist(),
                        "tracer_diffusivities": None,
                        "thermodynamic_factors": None,
                        "mobilities": None,
                        "stable_composition_sets": [self.fixed_phase],
                        "phase_composition": full_composition,
                        "site_fractions": None,
                        "kinetics_strategy": "fixed_matrix",
                        "selected_seed_index": None,
                        "selected_source_composition_set": self.fixed_phase,
                        "selected_gibbs_energy": None,
                        "diagnostic_errors": {
                            "phase_state": "Fixed matrix does not require a kinetics equilibrium."
                        },
                    })
                    self._kinetics_diagnostic_index += 1
            if values.ndim == 2:
                return np.broadcast_to(self.diffusivity_matrix, (values.shape[0], 2, 2)).copy()
            return self.diffusivity_matrix.copy()
        try:
            return self.thermodynamics.getInterdiffusivity(x, T, phase=phase, query_context=query_context, **kwargs)
        except TypeError:
            return self.thermodynamics.getInterdiffusivity(x, T, phase=phase, **kwargs)


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

    This is what makes the right A|B interface use exactly the same
    physical tie line as the left B|A interface.
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
        equilibrium_qthiss_retry_grid_points=TC_EQUILIBRIUM_QTHISS_RETRY_GRID_POINTS,
        kinetics_disable_global_minimization=TC_KINETICS_DISABLE_GLOBAL_MINIMIZATION,
        kinetics_disable_positive_definite_hessian=TC_KINETICS_DISABLE_POSITIVE_DEFINITE_HESSIAN,
        kinetics_constrain_single_composition_set=TC_KINETICS_CONSTRAIN_SINGLE_COMPOSITION_SET,
        kinetics_multistart_mode=TC_KINETICS_MULTISTART_MODE,
        kinetics_multistart_phases=TC_KINETICS_MULTISTART_PHASES,
        kinetics_session_restart_interval=TC_KINETICS_SESSION_RESTART_INTERVAL,
        cache_dir=OUTPUTS / "tc_cache",
    )

def build_source_thermodynamics():
    if THERM_ENGINE=="PYCALPHAD":
        if not TDB_PATH.exists():
            raise FileNotFoundError(f"Could not find TDB at {TDB_PATH}.")
        therm_ab = create_pycalphad_thermodynamics_source(
                str(TDB_PATH),
                list(ELEMENTS),
                list(TIELINE_PHASES),
                use_default_phases=PYCALPHAD_USE_DEFAULT_PHASES,
                equilibrium_phases=PYCALPHAD_EQUILIBRIUM_PHASES,
                g_offset=G_OFFSET,
            )

    elif THERM_ENGINE=="TC":
        if FECRNI_BCC_DIFFUSIVITY_MATRIX is not None:
            therm_ab = FixedPhaseDiffusivityThermodynamics(
                        TCPythonThermodynamics(_make_tc_config(TIELINE_PHASES), default_remove_cache=DEFAULT_REMOVE_CACHE),
                        "BCC_A2",
                        FECRNI_BCC_DIFFUSIVITY_MATRIX,
                )
        else:
            therm_ab = TCPythonThermodynamics(_make_tc_config(TIELINE_PHASES), default_remove_cache=DEFAULT_REMOVE_CACHE)

    else:
        raise ValueError(f"THERM_ENGINE should be one of above options. Instead got: {THERM_ENGINE}")
    return therm_ab


def _thermodynamics_backend(source_thermodynamics):
    """Return the supported backend name for a possibly wrapped source."""
    if isinstance(source_thermodynamics, TCPythonThermodynamics):
        return "tc_python"
    wrapped = getattr(source_thermodynamics, "thermodynamics", None)
    if isinstance(wrapped, TCPythonThermodynamics):
        return "tc_python"
    if isinstance(source_thermodynamics, MulticomponentThermodynamics):
        return "pycalphad"
    if isinstance(wrapped, MulticomponentThermodynamics):
        return "pycalphad"
    raise TypeError(
        "Fe-Cr-Ni surrogate artifacts require a TC-Python or PyCalphad thermodynamics source."
    )


def _calculation_capture_source(source_thermodynamics):
    """Return the object that emits exact interdiffusivity calculation records."""
    if hasattr(source_thermodynamics, "captureKineticsDiagnostics"):
        return source_thermodynamics
    wrapped = getattr(source_thermodynamics, "thermodynamics", None)
    if wrapped is not None and hasattr(wrapped, "captureKineticsDiagnostics"):
        return wrapped
    raise TypeError("The selected thermodynamics source does not support kinetics diagnostics capture.")


def _thermodynamics_context(source_thermodynamics):
    """Use a source context manager when available and otherwise no-op."""
    if hasattr(source_thermodynamics, "__enter__") and hasattr(source_thermodynamics, "__exit__"):
        return source_thermodynamics
    return nullcontext(source_thermodynamics)

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


def _surrogate_diffusivity_sampling_kwargs():
    """Returns bulk-diffusivity sampling kwargs for the selected interpolation."""
    interpolation = str(DIFFUSIVITY_INTERPOLATION)
    bulk_points = BULK_DIFFUSIVITY_POINTS
    if bulk_points is None and interpolation in {"nearest", "simplex_linear", "simplex_positive_2x2"}:
        import pickle

        with open(
            EXAMPLES_DIR
            / "ternaryExamples"
            / f"allValid_3Element_compositions_{BULK_DIFFUSIVITY_GRID_SPACING:g}inc_projTo1eminus4.pkl",
            "rb",
        ) as bulk_diffusivity_file:
            bulk_points = pickle.load(bulk_diffusivity_file)[:, :-1].copy()
        if systemStr =="FeCrNi":
            points_toSkip = np.array([
                                    [2, 2],
                                    ])

        def in2d(arr1_input, arr2_input):
            arr1 = arr1_input.copy()
            arr2 = arr2_input.copy()
            nrows_arr1, ncols_arr1 = arr1.shape
            dtype_arr1={'names':['f{}'.format(i) for i in range(ncols_arr1)], 'formats':ncols_arr1 * [arr1.dtype]}

            arr2 = np.array(arr2.tolist()).copy()
            nrows_arr2, ncols_arr2= arr2.shape
            dtype_arr2={'names':['f{}'.format(i) for i in range(ncols_arr2)], 'formats':ncols_arr2 * [arr2.dtype]}
            return np.isin(arr1.view(dtype_arr1), arr2.view(dtype_arr2))
        bulk_points = bulk_points[np.invert(in2d(bulk_points, points_toSkip).ravel())].copy()

    if interpolation == "nearest":
        return {
            "diffusivity_interpolation": interpolation,
            "diffusivity_bulk_points": np.asarray(bulk_points, dtype=np.float64),
        }
    if interpolation in {"simplex_linear", "simplex_positive_2x2", "continuous_grid"}:
        grids = (
            None
            if BULK_DIFFUSIVITY_GRIDS is None
            else tuple(np.asarray(axis, dtype=np.float64) for axis in BULK_DIFFUSIVITY_GRIDS)
        )
        kwargs = {
            "diffusivity_interpolation": interpolation,
            "diffusivity_bulk_grids": grids,
        }
        if interpolation in ["simplex_linear", "simplex_positive_2x2"]:
            kwargs["diffusivity_bulk_points"] = np.asarray(
                bulk_points,
                dtype=np.float64,
            )
        return kwargs
    raise ValueError(
        "DIFFUSIVITY_INTERPOLATION must be 'nearest', 'continuous_grid', 'simplex_positive_2x2', or "
        "'simplex_linear'."
    )


def _diffusivity_artifact_stem():
    """Name surrogate artifacts by backend, temperature, strategy, and grid."""
    backend = str(THERM_ENGINE).strip().lower().replace("-", "_")
    if backend not in {"tc", "pycalphad"}:
        raise ValueError(f"Unsupported THERM_ENGINE for surrogate artifacts: {THERM_ENGINE!r}.")
    stem = f"{systemStr}_{backend}_{TEMPERATURE:g}K"
    if backend == "tc":
        global_enabled = int(not (
            TC_KINETICS_DISABLE_GLOBAL_MINIMIZATION or TC_KINETICS_CONSTRAIN_SINGLE_COMPOSITION_SET
        ))
        hessian_forced = int(not TC_KINETICS_DISABLE_POSITIVE_DEFINITE_HESSIAN)
        stem += f"_global_{global_enabled}_hessian_{hessian_forced}"
        if TC_KINETICS_CONSTRAIN_SINGLE_COMPOSITION_SET:
            stem += "_single_set_1"
        if TC_KINETICS_MULTISTART_MODE != "off":
            stem += f"_multistart_{TC_KINETICS_MULTISTART_MODE}"
            phase_tag = "-".join(
                str(phase).replace("#", "set").lower()
                for phase in TC_KINETICS_MULTISTART_PHASES
            )
            stem += f"_phases_{phase_tag}"
    else:
        stem += f"_default_phases_{int(PYCALPHAD_USE_DEFAULT_PHASES)}"
        stem += f"_g_offset_{str(G_OFFSET).replace('.', 'p')}"
    stem += f"_bulk_{str(BULK_DIFFUSIVITY_GRID_SPACING).replace('.', 'p')}"
    return stem


def _record_interface_bulk_consistency(surrogate, *, kinetics_path=None, report_path=None):
    """Persist a compact warning report for discontinuous matrices or site states.

    This is a heuristic comparison against independently calculated nearby
    bulk samples. Selected-result kinetics diagnostics supply the labeled
    sublattice site fractions. The report records enough state data to
    investigate every flag without storing the full NumPy arrays in metadata.
    """
    kinetics_path = (
        OUTPUTS / f"{_diffusivity_artifact_stem()}_kinetics.jsonl"
        if kinetics_path is None else Path(kinetics_path)
    )
    report = evaluate_interface_bulk_diffusivity_consistency(
        surrogate,
        matrix_relative_threshold=TC_INTERFACE_BULK_MATRIX_RELATIVE_THRESHOLD,
        local_outlier_factor=TC_INTERFACE_BULK_LOCAL_OUTLIER_FACTOR,
        max_distance_ratio=TC_INTERFACE_BULK_MAX_DISTANCE_RATIO,
        kinetics_diagnostics=kinetics_path,
        site_fraction_absolute_threshold=TC_INTERFACE_BULK_SITE_FRACTION_ABSOLUTE_THRESHOLD,
    )
    flags = [
        {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in item.items()
        }
        for item in report["flags"]
    ]
    site_fraction_flags = [
        {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in item.items()
        }
        for item in report["site_fraction_flags"]
    ]
    payload = {
        "schema_version": 1,
        "kind": report["kind"],
        "temperature": report["temperature"],
        "settings": report["settings"],
        "summary": report["summary"],
        "phase_summaries": {
            phase: phase_report["summary"] | {
                "excluded_general_interface_prefix_count": phase_report[
                    "excluded_general_interface_prefix_count"
                ],
            }
            for phase, phase_report in report["phase_reports"].items()
        },
        "flags": flags,
        "site_fraction_settings": report["site_fraction_settings"],
        "site_fraction_phase_summaries": {
            phase: phase_report["summary"] | {
                "excluded_general_interface_prefix_count": phase_report[
                    "excluded_general_interface_prefix_count"
                ],
            }
            for phase, phase_report in report["site_fraction_phase_reports"].items()
        },
        "site_fraction_flags": site_fraction_flags,
        "site_fraction_matching_failures": report["site_fraction_matching_failures"],
    }
    surrogate.metadata["interface_bulk_diffusivity_consistency"] = payload
    report_path = (
        OUTPUTS / f"{_diffusivity_artifact_stem()}_interface_bulk_consistency.json"
        if report_path is None else Path(report_path)
    )
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if flags or site_fraction_flags or report["site_fraction_matching_failures"]:
        details = []
        if flags:
            worst = max(flags, key=lambda item: item["matrix_relative_difference"])
            details.append(
                f"{len(flags)} matrix sample(s); worst phase={worst['phase']}, "
                f"interface_index={worst['interface_index']}, "
                f"relative_difference={worst['matrix_relative_difference']:.3g}"
            )
        if site_fraction_flags:
            worst_site = max(
                site_fraction_flags,
                key=lambda item: item["maximum_site_fraction_difference"],
            )
            details.append(
                f"{len(site_fraction_flags)} site-fraction sample(s); worst phase={worst_site['phase']}, "
                f"interface_index={worst_site['interface_index']}, "
                f"maximum_difference={worst_site['maximum_site_fraction_difference']:.3g}"
            )
        if report["site_fraction_matching_failures"]:
            details.append(
                f"{len(report['site_fraction_matching_failures'])} site-fraction record match failure(s)"
            )
        print(
            "\n"
            + "WARNING: interface/bulk diffusivity and site-fraction consistency check flagged "
            + "; ".join(details) + ". "
            f"Details: {report_path}"
        )
    else:
        print(
            "Interface/bulk diffusivity and site-fraction consistency check passed; "
            f"details: {report_path}"
        )
    return report


@contextmanager
def _capture_surrogate_kinetics(source_thermodynamics, path=None):
    """Stream exact construction-time source diffusivity queries to JSON Lines."""
    if systemStr != "FeCrNi":
        raise ValueError("This diagnostics writer is configured for Fe-Cr-Ni.")
    path = (
        OUTPUTS / f"{_diffusivity_artifact_stem()}_kinetics.jsonl"
        if path is None else Path(path)
    )
    backend = _thermodynamics_backend(source_thermodynamics)
    capture_source = _calculation_capture_source(source_thermodynamics)
    if backend == "tc_python":
        tc_source = (
            source_thermodynamics
            if isinstance(source_thermodynamics, TCPythonThermodynamics)
            else source_thermodynamics.thermodynamics
        )
        backend_version = tc_source.getRuntimeVersion()
        config = tc_source.config.to_metadata()
        mobility_definition = "mobility_of_component_in_phase"
    else:
        pycalphad_source = (
            source_thermodynamics
            if isinstance(source_thermodynamics, MulticomponentThermodynamics)
            else source_thermodynamics.thermodynamics
        )
        backend_version = importlib.metadata.version("pycalphad")
        config = {
            "source_type": type(source_thermodynamics).__name__,
            "phases": list(getattr(source_thermodynamics, "phases", pycalphad_source.phases)),
            "g_offset": float(pycalphad_source.gOffset),
            "equilibrium_sampling_density": int(pycalphad_source.pDens),
            "local_sampling_density": int(pycalphad_source.local_pDens),
        }
        mobility_definition = (
            "mobility_from_composition_set when mobility parameters are available; "
            "otherwise tracer_diffusivity/(R*T)"
        )
    with path.open("w", encoding="utf-8", newline="\n") as output:
        header = {
            "record_type": "metadata",
            "schema_version": 2,
            "backend": backend,
            "backend_version": backend_version,
            "tc_python_version": backend_version if backend == "tc_python" else None,
            "config": config,
            "temperature_unit": "K",
            "composition_unit": "mole_fraction",
            "diffusivity_unit": "m^2/s",
            "mobility_unit": "m^2/(J*s)",
            "mobility_definition": mobility_definition,
            "factor_unit": "J/mol",
            "factor_definition": "d(mu_i - mu_reference)/d(x_j)",
            "matrix_elements": list(INDEPENDENT_ELEMENTS),
            "tracer_and_phase_composition_elements": list(ELEMENTS),
            "mobility_elements": list(ELEMENTS),
            "stable_phase_scope": (
                "forced_kinetics_equilibrium"
                if backend == "tc_python" else "pycalphad_local_single_phase_equilibrium"
            ),
        }
        output.write(json.dumps(header, allow_nan=False) + "\n")
        output.flush()

        def write_record(record):
            output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()

        with capture_source.captureKineticsDiagnostics(write_record):
            yield path


def plot_selected_diffusivity_calculations_for_run(surrogate=None, *, artifact=None, renderer="browser"):
    """Plot selected source records used by all four diffusivity datasets.

    Tabs cross both phases with the interface and general training contexts.
    Hover includes the stored matrix and selected sublattice state; surrogate
    predictions, failed attempts, and rejected candidates are excluded.
    ``surrogate`` defaults to ``result['surrogate_ab']`` after the example run.
    """
    if surrogate is None:
        surrogate = result["surrogate_ab"]
    if artifact is None:
        artifact = globals().get("tieline_surrogate_artifact")
    if artifact is not None:
        path = artifact.member_path("kinetics_diagnostics")
    else:
        sidecar = surrogate.metadata.get("kinetics_diagnostics_sidecar")
        if not sidecar:
            raise ValueError("Surrogate metadata does not identify a kinetics diagnostics sidecar.")
        path = OUTPUTS / sidecar
    return plot_selected_diffusivity_calculations(surrogate, path, renderer=renderer)


def _finite_numeric_mapping(values, label):
    """Return a stable string-to-float mapping for strict artifact identity."""
    output = {}
    for key, value in dict(values).items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cannot safely fingerprint {label} value {key!s}={value!r}.") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"Cannot safely fingerprint non-finite {label} value {key!s}.")
        output[str(key)] = numeric
    return output


def _surrogate_build_spec(source_thermodynamics, diffusivity_sampling):
    """Describe every numerical input used to construct the Fe-Cr-Ni surrogate.

    Large sampling arrays are left as arrays for the artifact canonicalizer,
    which records their dtype, shape, and digest without expanding them into
    the manifest. Purely locational cache and output paths are excluded.
    """
    backend = _thermodynamics_backend(source_thermodynamics)
    configured_engine = str(THERM_ENGINE).strip().upper()
    configured_backends = {"PYCALPHAD": "pycalphad", "TC": "tc_python"}
    if configured_engine not in configured_backends:
        raise ValueError(f"Unsupported THERM_ENGINE for surrogate artifacts: {THERM_ENGINE!r}.")
    configured_backend = configured_backends[configured_engine]
    if backend != configured_backend:
        raise ValueError(
            f"THERM_ENGINE selects {configured_backend!r}, but the source uses {backend!r}."
        )
    implementation_sources = {
        "moving_boundary_surrogates": file_sha256(moving_boundary_surrogates_module.__file__),
        "surrogate_artifacts": file_sha256(surrogate_artifacts_module.__file__),
        "surrogate_diagnostics": file_sha256(surrogate_diagnostics_module.__file__),
    }
    if backend == "tc_python":
        tc_source = (
            source_thermodynamics
            if isinstance(source_thermodynamics, TCPythonThermodynamics)
            else source_thermodynamics.thermodynamics
        )
        runtime_version = tc_source.getRuntimeVersion()
        if not runtime_version:
            raise ValueError("Cannot safely identify the installed TC-Python runtime for surrogate reuse.")
        config = tc_source.config.to_metadata()
        config.pop("cache_dir", None)
        config.pop("timeout_seconds", None)
        user_database_path = config.pop("user_database_path", None)
        database_identity = {
            "thermodynamic_database": config.get("thermodynamic_database"),
            "kinetic_database": config.get("kinetic_database"),
            "user_database": None,
        }
        if user_database_path is not None:
            database_path = Path(user_database_path)
            database_identity["user_database"] = {
                "name": database_path.name,
                "sha256": file_sha256(database_path),
            }
        runtime_identity = {
            "backend": backend,
            "tc_python_version": runtime_version,
            "databases": database_identity,
            "thermocalc_config": config,
            "default_remove_cache": tc_source.default_remove_cache,
        }
        implementation_sources["tc_python_adapter"] = file_sha256(
            tc_python_adapter_module.__file__
        )
    else:
        if not TDB_PATH.is_file():
            raise ValueError(f"Cannot fingerprint missing PyCalphad database: {TDB_PATH}")
        pycalphad_source = (
            source_thermodynamics
            if isinstance(source_thermodynamics, MulticomponentThermodynamics)
            else source_thermodynamics.thermodynamics
        )
        try:
            runtime_version = importlib.metadata.version("pycalphad")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError("Cannot safely identify the installed PyCalphad runtime.") from exc
        runtime_identity = {
            "backend": backend,
            "pycalphad_version": runtime_version,
            "database": {"name": TDB_PATH.name, "sha256": file_sha256(TDB_PATH)},
            "source_type": (
                f"{type(source_thermodynamics).__module__}."
                f"{type(source_thermodynamics).__qualname__}"
            ),
            "use_default_phases": bool(PYCALPHAD_USE_DEFAULT_PHASES),
            "equilibrium_phases": list(
                getattr(source_thermodynamics, "equilibrium_phases", pycalphad_source.phases)
            ),
            "phase_amount_tolerance": getattr(
                source_thermodynamics, "phase_amount_tolerance", None
            ),
            "g_offset": float(pycalphad_source.gOffset),
            "equilibrium_sampling_density": int(pycalphad_source.pDens),
            "local_sampling_density": int(pycalphad_source.local_pDens),
            "driving_force_sampling_density": int(pycalphad_source.sampling_pDens),
            "mobility_correction": _finite_numeric_mapping(
                pycalphad_source.mobility_correction, "mobility correction"
            ),
            "parameters": _finite_numeric_mapping(
                pycalphad_source._parameters, "symbolic parameter"
            ),
        }
        implementation_sources.update({
            "pycalphad_adapter": file_sha256(pycalphad_adapter_module.__file__),
            "pycalphad_thermodynamics": file_sha256(pycalphad_thermodynamics_module.__file__),
            "mobility": file_sha256(mobility_module.__file__),
            "free_energy_hessian": file_sha256(free_energy_hessian_module.__file__),
        })
    fixed_matrix = getattr(source_thermodynamics, "diffusivity_matrix", FECRNI_BCC_DIFFUSIVITY_MATRIX)
    probe_parameters = _tieline_surrogate_probe_kwargs(PROBE_START, PROBE_END, PROBE_POINT)
    recipe_functions = (
        _make_tc_config,
        build_source_thermodynamics,
        FixedPhaseDiffusivityThermodynamics.getInterdiffusivity,
        FixedPhaseDiffusivityThermodynamics.captureKineticsDiagnostics,
        _tieline_surrogate_probe_kwargs,
        _surrogate_diffusivity_sampling_kwargs,
        _thermodynamics_backend,
        _calculation_capture_source,
        _thermodynamics_context,
        _record_interface_bulk_consistency,
        _capture_surrogate_kinetics,
        _finite_numeric_mapping,
        _surrogate_build_spec,
        build_tieline_surrogate,
        prepare_tieline_surrogate,
    )
    return {
        "system": {
            "name": systemStr,
            "thermodynamics_engine": THERM_ENGINE,
            "elements": ELEMENTS,
            "independent_elements": INDEPENDENT_ELEMENTS,
            "reference_element": REFERENCE_ELEMENT,
            "phases": TIELINE_PHASES,
            "temperature": TEMPERATURE,
        },
        "runtime": {
            **runtime_identity,
        },
        "tieline_sampling": {
            "mode": TIELINE_SURROGATE_BUILD_MODE,
            "parameters": probe_parameters,
        },
        "diffusivity_sampling": {
            "mode": BULK_DIFFUSIVITY_MODE,
            "interpolation": DIFFUSIVITY_INTERPOLATION,
            "parameters": diffusivity_sampling,
            "fixed_phase": getattr(source_thermodynamics, "fixed_phase", None),
            "fixed_matrix": fixed_matrix,
            "fixed_sample_eta": DIFFUSIVITY_SAMPLE_ETA,
            "skip_failed_bulk_calculations": TC_DROP_FAILED_BULK_CALCULATIONS,
            "drop_invalid_bulk_matrices": TC_DROP_INVALID_BULK_MATRICES,
            "validity_policy": "raise",
            "min_composition": 1.0e-10,
        },
        "diagnostics": {
            "capture_kinetics": TC_CAPTURE_DIFFUSIVITY_DIAGNOSTICS,
            "check_interface_bulk_consistency": TC_CHECK_INTERFACE_BULK_DIFFUSIVITY_CONSISTENCY,
            "matrix_relative_threshold": TC_INTERFACE_BULK_MATRIX_RELATIVE_THRESHOLD,
            "local_outlier_factor": TC_INTERFACE_BULK_LOCAL_OUTLIER_FACTOR,
            "max_distance_ratio": TC_INTERFACE_BULK_MAX_DISTANCE_RATIO,
            "site_fraction_absolute_threshold": TC_INTERFACE_BULK_SITE_FRACTION_ABSOLUTE_THRESHOLD,
        },
        "implementation_sources": implementation_sources | {
            "fecrni_recipe": {
                function.__name__: callable_source_sha256(function)
                for function in recipe_functions
            },
        },
    }


def _required_surrogate_artifact_members():
    """Return diagnostic member roles required by the current run settings."""
    roles = []
    if TC_CAPTURE_DIFFUSIVITY_DIAGNOSTICS or TC_CHECK_INTERFACE_BULK_DIFFUSIVITY_CONSISTENCY:
        roles.append("kinetics_diagnostics")
    if TC_CHECK_INTERFACE_BULK_DIFFUSIVITY_CONSISTENCY and BULK_DIFFUSIVITY_MODE != "phase_uniform":
        roles.append("interface_bulk_consistency")
    if TC_DROP_FAILED_BULK_CALCULATIONS:
        roles.append("failed_bulk_points")
    if TC_DROP_INVALID_BULK_MATRICES:
        roles.append("invalid_bulk_points")
    return tuple(roles)


def build_tieline_surrogate(source_thermodynamics, *, diffusivity_sampling=None, artifact_directory=None):
    """Sample the tie line and return its surrogate plus artifact sidecars.

    ``artifact_directory`` receives construction-only diagnostics before the
    verified bundle publisher copies them into content-addressed members.
    """
    pair_ab = TIELINE_PHASES
    if diffusivity_sampling is None:
        diffusivity_sampling = (
            {} if BULK_DIFFUSIVITY_MODE == "phase_uniform" else _surrogate_diffusivity_sampling_kwargs()
        )
    artifact_directory = OUTPUTS if artifact_directory is None else Path(artifact_directory)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    kinetics_path = artifact_directory / "kinetics.jsonl"
    members = {}
    capture = (
        _capture_surrogate_kinetics(source_thermodynamics, kinetics_path)
        if TC_CAPTURE_DIFFUSIVITY_DIAGNOSTICS or TC_CHECK_INTERFACE_BULK_DIFFUSIVITY_CONSISTENCY
        else nullcontext()
    )
    with _thermodynamics_context(source_thermodynamics), capture:
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
            skip_failed_bulk_calculations=TC_DROP_FAILED_BULK_CALCULATIONS,
            drop_invalid_bulk_matrices=TC_DROP_INVALID_BULK_MATRICES,
            **diffusivity_sampling,
            **kwargs_ab,
        )
    if TC_DROP_FAILED_BULK_CALCULATIONS:
        failures = surrogate_ab.metadata["failed_bulk_points"]
        report_path = artifact_directory / "failed_bulk_points.json"
        report_path.write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        members["failed_bulk_points"] = {"path": report_path, "schema_version": 1}
        print(f"Dropped {len(failures)} bulk phase samples; details: {report_path}")
    if TC_DROP_INVALID_BULK_MATRICES:
        invalid = surrogate_ab.metadata["invalid_bulk_points"]
        report_path = artifact_directory / "invalid_bulk_points.json"
        report_path.write_text(json.dumps(invalid, indent=2) + "\n", encoding="utf-8")
        members["invalid_bulk_points"] = {"path": report_path, "schema_version": 1}
        print(f"Dropped {len(invalid)} bulk samples with invalid matrices; details: {report_path}")
    if TC_CAPTURE_DIFFUSIVITY_DIAGNOSTICS:
        backend = _thermodynamics_backend(source_thermodynamics)
        wrapped_source = getattr(source_thermodynamics, "thermodynamics", source_thermodynamics)
        if backend == "tc_python":
            surrogate_ab.metadata["thermocalc_config"] = wrapped_source.config.to_metadata() | {
                "default_remove_cache": wrapped_source.default_remove_cache,
            }
        else:
            surrogate_ab.metadata["pycalphad_config"] = {
                "version": importlib.metadata.version("pycalphad"),
                "database": {"name": TDB_PATH.name, "sha256": file_sha256(TDB_PATH)},
                "source_type": type(source_thermodynamics).__name__,
                "phases": list(getattr(source_thermodynamics, "phases", wrapped_source.phases)),
                "g_offset": float(wrapped_source.gOffset),
                "equilibrium_sampling_density": int(wrapped_source.pDens),
                "local_sampling_density": int(wrapped_source.local_pDens),
            }
        surrogate_ab.metadata["kinetics_diagnostics_sidecar"] = "bundle:kinetics_diagnostics"
    if TC_CAPTURE_DIFFUSIVITY_DIAGNOSTICS or TC_CHECK_INTERFACE_BULK_DIFFUSIVITY_CONSISTENCY:
        members["kinetics_diagnostics"] = {"path": kinetics_path, "schema_version": 2}
    if TC_CHECK_INTERFACE_BULK_DIFFUSIVITY_CONSISTENCY and BULK_DIFFUSIVITY_MODE != "phase_uniform":
        consistency_path = artifact_directory / "interface_bulk_consistency.json"
        _record_interface_bulk_consistency(
            surrogate_ab, kinetics_path=kinetics_path, report_path=consistency_path
        )
        members["interface_bulk_consistency"] = {"path": consistency_path, "schema_version": 1}
    return surrogate_ab, members
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


def prepare_tieline_surrogate(source_thermodynamics):
    """Build by default, or strictly reload, the verified Fe-Cr-Ni artifact.

    Reload mode never invokes the builder. Missing, incompatible, or damaged
    bundles therefore fail before any Thermo-Calc calculation can start.
    """
    diffusivity_sampling = (
        {} if BULK_DIFFUSIVITY_MODE == "phase_uniform" else _surrogate_diffusivity_sampling_kwargs()
    )
    build_spec = _surrogate_build_spec(source_thermodynamics, diffusivity_sampling)
    bundle_path = OUTPUTS / f"{_diffusivity_artifact_stem()}.surrogate"

    with tempfile.TemporaryDirectory(prefix="fecrni-surrogate-", dir=OUTPUTS) as temporary:
        def builder():
            return build_tieline_surrogate(
                source_thermodynamics,
                diffusivity_sampling=diffusivity_sampling,
                artifact_directory=temporary,
            )

        artifact = prepare_surrogate_artifact(
            bundle_path,
            build_spec,
            builder,
            reload=RELOAD_TIELINE_SURROGATE,
            required_members=_required_surrogate_artifact_members(),
        )
    action = "Reloaded" if RELOAD_TIELINE_SURROGATE else "Built and published"
    print(f"{action} verified surrogate artifact: {artifact.path}")
    return artifact


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



def get_time_step_options():
    """Returns constructor options for the selected timestep schedule."""
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


def get_three_phase_molar_volumes():
    """Validates and returns the optional symmetric ``[VmA, VmB, VmA]`` tuple."""
    if PHASE_MOLAR_VOLUMES is None:
        return None
    values = np.asarray(PHASE_MOLAR_VOLUMES, dtype=np.float64).reshape(-1)
    if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("PHASE_MOLAR_VOLUMES must contain three positive finite values.")
    if values[0] != values[2]:
        raise ValueError(
            "The symmetric A|B|A example requires PHASE_MOLAR_VOLUMES[0] "
            "to equal PHASE_MOLAR_VOLUMES[2]."
        )
    return tuple(float(value) for value in values)


def build_two_phase_model(tieline_surrogate, bulk_thermodynamics):
    """Builds the two-phase model with the selected bulk-diffusivity mode."""
    time_options = get_time_step_options()
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=make_two_phase_mesh(),
        elements=ELEMENTS,
        phases=[PHASE_A, PHASE_B],
        thermodynamics=bulk_thermodynamics,
        temperature=TEMPERATURE,
        interfacePosition=INTERFACE_POSITION,
        interface_equilibrium=tieline_surrogate,
        initial_eta_method="instantaneous_balance",
        initial_eta_bracket=INITIAL_ETA_BRACKET,
        initial_eta_guess=INITIAL_ETA_GUESS,
        initial_velocity_guess=INITIAL_VELOCITY_GUESS_2PHASE,
        bulk_diffusivity_mode=BULK_DIFFUSIVITY_MODE,
        time_step=time_options["time_step"],
        dt_mode=time_options["dt_mode"],
        semiLog_dt=time_options["semiLog_dt"],
        semiLogT0=time_options["semiLogT0"],
        tolerance=TOLERANCE,
        residual_tolerance=RESIDUAL_TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        max_step_retries=MAX_STEP_RETRIES,
        terminal_thin_phase_width=TERMINAL_THIN_PHASE_WIDTH,
        terminal_thin_phase_policy=TERMINAL_THIN_PHASE_POLICY,
        record=True,
        record_pq_data=True,
        transformed_u_grid=U_A_2,
        transformed_v_grid=V_B_2,
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
    """Returns the legacy fixed-mesh independent-component profile."""
    return np.asarray(model.data.y(model.currentTime), dtype=np.float64)


def _physical_coordinates(model):
    """Returns the immutable coordinates paired with a legacy fixed-mesh profile."""
    return np.asarray(model.mesh.z, dtype=np.float64).reshape(-1)


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


def _load_lee_oh_fig9_curve(path):
    """Load and time-sort one finite two-column Lee and Oh Figure 9 curve."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Could not find Lee and Oh Figure 9 data at {path}.")
    values = np.loadtxt(path, delimiter=",", dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        if values.size != 2:
            raise ValueError(
                f"Lee and Oh Figure 9 data at {path} must have exactly two columns."
            )
        values = values.reshape(1, 2)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError(
            f"Lee and Oh Figure 9 data at {path} must be a non-empty two-column array."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Lee and Oh Figure 9 data at {path} must contain finite values.")
    return values[np.argsort(values[:, 0], kind="stable")]


def _lee_oh_fig9_curve_metrics(model_time_hours, model_normalized_alpha, reference):
    """Compare a model history with a reference curve by log-time interpolation.

    Only positive reference times inside the model's positive-time interval are
    compared. Interpolation in ``log10(time)`` matches the logarithmic Figure 9
    abscissa without extrapolating either history.
    """
    model_time_hours = np.asarray(model_time_hours, dtype=np.float64).reshape(-1)
    model_normalized_alpha = np.asarray(model_normalized_alpha, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64)
    positive_model = model_time_hours > 0.0
    comparison_time = model_time_hours[positive_model]
    comparison_values = model_normalized_alpha[positive_model]
    if comparison_time.size < 2 or np.any(np.diff(comparison_time) <= 0.0):
        raise ValueError(
            "The recorded model history must contain at least two strictly increasing positive times."
        )
    reference_time = reference[:, 0]
    overlap = (
        (reference_time > 0.0)
        & (reference_time >= comparison_time[0])
        & (reference_time <= comparison_time[-1])
    )
    if np.count_nonzero(overlap) < 2:
        raise ValueError(
            "Lee and Oh Figure 9 comparison requires at least two positive reference "
            "times inside the recorded model interval."
        )
    overlap_time = reference_time[overlap]
    reference_values = reference[overlap, 1]
    interpolated_model = np.interp(
        np.log10(overlap_time),
        np.log10(comparison_time),
        comparison_values,
    )
    error = interpolated_model - reference_values
    return {
        "overlap_count": int(error.size),
        "overlap_time_hours": [float(overlap_time[0]), float(overlap_time[-1])],
        "mean_absolute_error": float(np.mean(np.abs(error))),
        "root_mean_square_error": float(np.sqrt(np.mean(error**2))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }


def _print_lee_oh_fig9_metrics(metrics):
    """Print a compact summary of a Figure 9 comparison report."""
    model = metrics["model"]
    print("\nLee and Oh Figure 9 comparison:")
    print(
        "  model peak normalized alpha thickness = "
        f"{model['peak_normalized_alpha_thickness']:.6g} at "
        f"{model['peak_time_hours']:.6g} h"
    )
    print(
        "  model terminal normalized alpha thickness = "
        f"{model['terminal_normalized_alpha_thickness']:.6g} at "
        f"{model['terminal_time_hours']:.6g} h; deviation from "
        f"{metrics['equilibrium_normalized_alpha_thickness']:.6g} = "
        f"{model['terminal_equilibrium_deviation']:+.6g}"
    )
    for key in ("cr_correction", "ni_correction"):
        reference = metrics["references"][key]
        comparison = reference["comparison"]
        print(
            f"  {reference['label']}: peak={reference['peak_normalized_alpha_thickness']:.6g} "
            f"at {reference['peak_time_hours']:.6g} h; "
            f"overlap n={comparison['overlap_count']}, "
            f"MAE={comparison['mean_absolute_error']:.6g}, "
            f"RMSE={comparison['root_mean_square_error']:.6g}, "
            f"max|error|={comparison['maximum_absolute_error']:.6g}"
        )


def plot_lee_oh_fig9_comparison(
    model,
    *,
    cr_path=None,
    ni_path=None,
    xlims=None,
    show=True,
):
    """Plot and quantify the model against digitized Lee and Oh Figure 9.

    The left phase must be alpha/BCC. Its physical thickness is the recorded
    interface position and is normalized by its initial value. The coupled
    Illingworth trajectory is compared independently with the published Cr-
    and Ni-correction curves; it is not identified with either correction.

    Returns
    -------
    tuple
        ``(figure, axes, metrics)`` with log-time overlap errors calculated
        without extrapolation.
    """
    phases = [str(phase).split("#", 1)[0].upper() for phase in model.phases]
    if not phases or phases[0] != "BCC_A2":
        raise ValueError(
            "Lee and Oh Figure 9 compares the left alpha/BCC layer; "
            "the model's first phase must be BCC_A2."
        )
    n_history = int(model.interfaceData.N) + 1
    times_seconds = np.asarray(
        model.interfaceData._time[:n_history], dtype=np.float64
    ).reshape(-1)
    alpha_thickness = np.asarray(
        model.interfaceData._y[:n_history], dtype=np.float64
    ).reshape(-1)
    if (
        times_seconds.size != alpha_thickness.size
        or times_seconds.size < 3
        or not np.all(np.isfinite(times_seconds))
        or not np.all(np.isfinite(alpha_thickness))
    ):
        raise ValueError("The model must provide a finite recorded interface history.")
    initial_alpha_thickness = float(alpha_thickness[0])
    if initial_alpha_thickness <= 0.0:
        raise ValueError("Initial alpha/BCC thickness must be positive for normalization.")
    if np.any(np.diff(times_seconds) < 0.0):
        raise ValueError("Recorded model times must be nondecreasing.")

    time_hours = times_seconds / float(LEE_OH_FIG9_TIME_UNIT_SECONDS)
    normalized_alpha = alpha_thickness / initial_alpha_thickness
    cr_curve = _load_lee_oh_fig9_curve(
        LEE_OH_FIG9_LOWER_CR_PATH if cr_path is None else cr_path
    )
    ni_curve = _load_lee_oh_fig9_curve(
        LEE_OH_FIG9_UPPER_NI_PATH if ni_path is None else ni_path
    )
    peak_index = int(np.argmax(normalized_alpha))
    terminal_value = float(normalized_alpha[-1])
    equilibrium_value = float(LEE_OH_FIG9_EQUILIBRIUM_NORMALIZED_ALPHA)

    references = {}
    for key, label, curve in (
        ("cr_correction", "Lee & Oh Figure 9 lower curve (Cr correction)", cr_curve),
        ("ni_correction", "Lee & Oh Figure 9 upper curve (Ni correction)", ni_curve),
    ):
        reference_peak = int(np.argmax(curve[:, 1]))
        references[key] = {
            "label": label,
            "peak_normalized_alpha_thickness": float(curve[reference_peak, 1]),
            "peak_time_hours": float(curve[reference_peak, 0]),
            "comparison": _lee_oh_fig9_curve_metrics(
                time_hours, normalized_alpha, curve
            ),
        }
    metrics = {
        "time_unit": "hours",
        "equilibrium_normalized_alpha_thickness": equilibrium_value,
        "model": {
            "initial_alpha_thickness_m": initial_alpha_thickness,
            "peak_normalized_alpha_thickness": float(normalized_alpha[peak_index]),
            "peak_time_hours": float(time_hours[peak_index]),
            "terminal_normalized_alpha_thickness": terminal_value,
            "terminal_time_hours": float(time_hours[-1]),
            "terminal_equilibrium_deviation": terminal_value - equilibrium_value,
        },
        "references": references,
    }

    positive_time = time_hours > 0.0
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(
        time_hours[positive_time],
        normalized_alpha[positive_time],
        linewidth=1.8,
        label="Coupled Illingworth calculation",
    )
    ax.plot(
        cr_curve[:, 0], cr_curve[:, 1], color="dimgray", linewidth=1.5,
        label=references["cr_correction"]["label"],
    )
    ax.plot(
        ni_curve[:, 0], ni_curve[:, 1], color="black", linewidth=1.5,
        label=references["ni_correction"]["label"],
    )
    ax.axhline(equilibrium_value, color="0.6", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Normalized alpha/BCC thickness")
    ax.set_title("Fe-Cr-Ni comparison with Lee and Oh Figure 9")
    ax.set_xlim(*(LEE_OH_FIG9_XLIMS_HOURS if xlims is None else xlims))
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", frameon=True)
    _save_or_show(fig, "lee_oh_figure9_comparison.png")
    if show and plt.get_backend().lower() != "agg":
        plt.show(block=False)
    _print_lee_oh_fig9_metrics(metrics)
    return fig, ax, metrics


def plot_interface_comparison(arrays):
    """Plots interfaces, labeling the two-phase curve as a legacy reference when needed."""
    t = arrays["time"] / TIME_SCALE
    mask = _positive_time_mask(t)
    reduction_valid = arrays["reduction_comparison_valid"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    two_phase_label = "2-phase A|B" if reduction_valid else "2-phase A|B (fixed-domain reference)"
    ax.plot(t[mask], 1.0e6 * arrays["s2"][mask], label=two_phase_label)
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
    ax.set_title("Interface-position reduction check" if reduction_valid else "Moving-domain interface comparison")
    ax.legend()
    return _save_or_show(fig, "interface_comparison.png")


def plot_middle_width_comparison(arrays):
    """Plots middle-phase widths without claiming unequal-volume reduction equivalence."""
    t = arrays["time"] / TIME_SCALE
    mask = _positive_time_mask(t)
    reduction_valid = arrays["reduction_comparison_valid"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        t[mask],
        1.0e6 * arrays["b_half_2"][mask],
        label=(
            f"2-phase {PHASE_B} width"
            if reduction_valid
            else f"2-phase {PHASE_B} width (fixed-domain reference)"
        ),
    )
    ax.plot(
        t[mask],
        1.0e6 * arrays["b_half_3"][mask],
        "--",
        label=f"half of 3-phase {PHASE_B} width",
    )
    ax.set_xscale("log")
    ax.set_xlabel(f"time / {TIME_LABEL}")
    ax.set_ylabel(f"{PHASE_B} half-width / µm")
    ax.set_title("Middle-phase width reduction check" if reduction_valid else "Middle-phase width comparison")
    ax.legend()
    return _save_or_show(fig, "middle_width_comparison.png")


def plot_eta_comparison(arrays):
    """Plots interface etas and distinguishes a legacy unequal-volume reference."""
    t = arrays["time"] / TIME_SCALE
    mask = _positive_time_mask(t)
    reduction_valid = arrays["reduction_comparison_valid"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        t[mask],
        arrays["eta2"][mask],
        label="2-phase eta" if reduction_valid else "2-phase eta (fixed-domain reference)",
    )
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
    ax.set_title("Interface tie-line reduction check" if reduction_valid else "Interface tie-line comparison")
    ax.legend()
    return _save_or_show(fig, "eta_comparison.png")


def plot_final_profiles(arrays):
    """Plots complete physical profiles and only overlays a valid reduction profile."""
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
        if arrays["reduction_comparison_valid"]:
            ax.plot(
                x2_um,
                arrays["c3_on_x2"][:, j],
                "--",
                linewidth=1.5,
                label="3-phase left half",
            )
        ax.axvline(1.0e6 * arrays["three_phase_center"], linestyle=":", linewidth=1.0)
        ax.set_xlabel("position / µm")
        ax.set_ylabel(f"X({element})")
        ax.set_title(element)
        ax.legend()

    fig.suptitle(f"Final composition profiles at t = {SOLVE_TIME / 3600.0:g} h")
    return _save_or_show(fig, "final_profile_comparison.png")


def plot_discrepancies(arrays):
    """Plots valid reduction errors plus internal moving-center symmetry errors."""
    t = arrays["time"] / TIME_SCALE
    mask = _positive_time_mask(t)
    reduction_valid = arrays["reduction_comparison_valid"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.25))

    ax = axes[0]
    if reduction_valid:
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
    if reduction_valid:
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
    """Plots legacy or physical three-phase conservation without mixing units."""
    if arrays["three_phase_inventory_kind"] == "physical_moles_per_area":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.25))
        t2 = arrays["t_inv2"] / TIME_SCALE
        t3 = arrays["t_inv3"] / TIME_SCALE
        for j, element in enumerate(INDEPENDENT_ELEMENTS):
            axes[0].plot(t2, arrays["inv2_drift"][:, j], label=element)
        for j, element in enumerate(ELEMENTS):
            axes[1].plot(t3, arrays["inv3_all_drift"][:, j], label=element)
        axes[0].set_title("Two-phase fixed-domain reference")
        axes[0].set_ylabel("Legacy composition-length inventory drift")
        axes[1].set_title("Three-phase moving domain")
        axes[1].set_ylabel("Physical molar inventory drift (mol/m^2)")
        for ax in axes:
            ax.set_xlabel(f"time / {TIME_LABEL}")
            ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.legend()
        fig.suptitle("Conservation diagnostics (separate physical units)")
        return _save_or_show(fig, "inventory_drift.png")

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

# Build the shared thermodynamic data


validated_phase_molar_volumes = get_three_phase_molar_volumes()
if validated_phase_molar_volumes is not None and len(set(validated_phase_molar_volumes)) > 1:
    print(
        "Unequal PHASE_MOLAR_VOLUMES enabled. The three-phase model will use "
        "physical molar conservation and moving R(t); the fixed-domain two-phase "
        "model will be shown only as a legacy reference."
    )

source_thermodynamics = build_source_thermodynamics()
tieline_surrogate_artifact = prepare_tieline_surrogate(source_thermodynamics)
tieline_surrogate = tieline_surrogate_artifact.surrogate

print(f"Built tie-line surrogate with eta bounds {tieline_surrogate.eta_bounds}.")
tieline_surrogate.validity_policy='legacy'
# Construction-only dashboard: reads persisted build provenance and does not
# issue additional source-thermodynamics queries.
if not RELOAD_TIELINE_SURROGATE:
    construction_diagnostics = plot_surrogate_diagnostics(
        tieline_surrogate, renderer="browser"
    )
    construction_diagnostics["figures"]["construction"].show()

    fig = plot_selected_diffusivity_calculations_for_run(
        surrogate=tieline_surrogate,
        artifact=tieline_surrogate_artifact,
        renderer="browser",
    )
    fig.show()
if BULK_DIFFUSIVITY_MODE == "phase_uniform":
    fixed_diffusivity_matrices = select_fixed_diffusivity_matrices(
        source_thermodynamics,
        tieline_surrogate,
    )
    bulk_thermodynamics = FixedMatrixTernaryDiffusivity(
        fixed_diffusivity_matrices,
        phases=[PHASE_A, PHASE_B],
        temperature=TEMPERATURE,
    )
else:
    # The sampled surrogate evaluates phase-specific diffusivity from each
    # bulk composition; the model applies the requested lagged/implicit mode.
    bulk_thermodynamics = tieline_surrogate
tieline_surrogate.validity_policy='raise'

# %%
# Build matched models

two_phase_model = build_two_phase_model(
    tieline_surrogate,
    bulk_thermodynamics,
)


print("\nTwo-phase geometry:")
print(f"  domain = [0, {HALF_LENGTH:.8e}] m")
print(f"  initial interface = {INTERFACE_POSITION:.8e} m")
print(
    f"  initial {PHASE_B} width = "
    f"{HALF_LENGTH - INTERFACE_POSITION:.8e} m"
)



print("\nTransformed-grid counts:")
print(
    f"  2-phase: A={len(U_A_2)}, B={len(V_B_2)}; "
)


# %%
# Solve both cases
import importlib
import examples.debugInPlace as debug_module

importlib.invalidate_caches()
debug_module = importlib.reload(debug_module)

# Required if you previously used:
# from examples.debugInPlace import debugInPlace
debugInPlace = debug_module.debugInPlace

# debugInPlace()
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

print("\nSolve complete.")
print(
    f"  two-phase final time   = {two_phase_model.currentTime:.12g} s"
)

lee_oh_fig9_comparison = None
if PLOT_LEE_OH_FIG9_COMPARISON:
    comparison_figure, comparison_axes, comparison_metrics = plot_lee_oh_fig9_comparison(
        two_phase_model
    )
    lee_oh_fig9_comparison = {
        "figure": comparison_figure,
        "axes": comparison_axes,
        "metrics": comparison_metrics,
    }

# %%
# Plots
two_phase_model.plot_eta_vs_time()
two_phase_model.plot_latestCompProfile()
two_phase_model.plot_phaseWidths_vs_time()

# %%
indexOfMaxLiq = np.argmax(two_phase_model._R - two_phase_model.interfaceData._y)
widthOfMaxLiq = (two_phase_model._R - two_phase_model.interfaceData._y)[indexOfMaxLiq]
timeOfMaxLiq = two_phase_model.interfaceData._time[indexOfMaxLiq]
etaofMaxLiq = two_phase_model.etaData._y[np.where(two_phase_model.etaData._time==timeOfMaxLiq)[0]]
lastTime = two_phase_model.interfaceData._time[two_phase_model.interfaceData.N]
print(f"indexOfMaxLiq: {indexOfMaxLiq}")
print(f"widthOfMaxLiq: {widthOfMaxLiq/ 1e-6}  um")
print(f"timeOfMaxLiq: {timeOfMaxLiq}")
print(f"etaofMaxLiq: {etaofMaxLiq}")
print(f"lastTime: {lastTime}")
# %%
tieline_surrogate.validity_policy='legacy'
result = {
    "model": two_phase_model,
    "surrogate": tieline_surrogate,
    "surrogate_ab": tieline_surrogate,
    "therm_ab": source_thermodynamics,
    "phase_molar_volumes": get_three_phase_molar_volumes(),
    "lee_oh_fig9_comparison": lee_oh_fig9_comparison,
}
import importlib

from examples.ternaryExamples.IllingworthTernaryPlotly import plot_two_phase_composition_profile
ternaryPlotly_module = importlib.import_module(
    "examples.ternaryExamples.IllingworthTernaryPlotly"
)

importlib.invalidate_caches()
importlib.reload(ternaryPlotly_module)
plot_two_phase_composition_profile = ternaryPlotly_module.plot_two_phase_composition_profile

from pathlib import Path

plot_two_phase_composition_profile = (
    ternaryPlotly_module.plot_two_phase_composition_profile
)

results_plot = plot_two_phase_composition_profile(
    result,
    time_indices=np.arange(len(result['model'].interfaceData._time))[::100],
    show_tielines=True,
    tieline_eta_count=41,
    display_tieline_count=41,
    show_global_average=True,
    show_starting_phase_compositions=False,
    show_diffusivities=False,
    use_symmetric_log=True,
    renderer="browser",
    run_config={ "THERM_ENGINE": THERM_ENGINE,
    "TC_USE_DEFAULT_PHASES": TC_USE_DEFAULT_PHASES,
    "PYCALPHAD_USE_DEFAULT_PHASES": PYCALPHAD_USE_DEFAULT_PHASES,}
)
results_plot["fig"].show()
# import plotly
# plotly.offline.plot(results_plot["fig"], filename = fr"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\kawin\examples\ternaryExamples\examplesForZhaoxi\{''.join([el.capitalize() for el in ELEMENTS])}_{TEMPERATURE}K_200um_varSolidDiff_noDiffRes_results_plot.html", auto_open=False)

#%%


# %%
import importlib

import kawin.diffusion as diffusion
diagnostics_module = importlib.import_module(
    "kawin.diffusion.SurrogateDiagnostics"
)

importlib.invalidate_caches()
importlib.reload(diagnostics_module)
importlib.reload(diffusion)

# Refresh the alias used by plot_surrogate_diagnostics_for_run
plot_surrogate_diagnostics = diagnostics_module.plot_surrogate_diagnostics

# import importlib
# debugInPlace_module = importlib.import_module(
#     "examples.debugInPlace"
# )
# importlib.reload(debugInPlace_module)
# debugInPlace = debugInPlace_module.debugInPlace

import importlib
import examples.debugInPlace as debug_module

importlib.invalidate_caches()
debug_module = importlib.reload(debug_module)

# Required if you previously used:
# from examples.debugInPlace import debugInPlace
debugInPlace = debug_module.debugInPlace

def plot_surrogate_diagnostics_for_run(
    result,
    *,
    compare_ground_truth=False,
    hover_fields="diagnostic",
    **diagnostic_kwargs,
):
    """Build interactive diagnostics for this symmetric run's A|B surrogate.

    The symmetry example constructs one sampled ``A|B`` tie-line surrogate and
    reuses it through a reversed interface for the three-phase model. Therefore
    this helper returns one ``"ab"`` diagnostics entry rather than separate
    ``A|B`` and ``B|C`` entries. Ground-truth calls are made only when
    ``compare_ground_truth`` is true, using ``result["therm_ab"]``. The returned
    Plotly figures are not shown or saved; for example, a notebook can display
    ``diagnostics["ab"]["figures"]["thermodynamics"]`` directly.

    Parameters
    ----------
    result : dict
        Mapping containing the completed model, the ``"surrogate_ab"``
        surrogate, and, when ground-truth comparison is requested, the
        original ``"therm_ab"`` thermodynamics provider.
    compare_ground_truth : bool, optional
        Whether to compare the surrogate with the original A|B thermodynamics
        provider.
    hover_fields : str, sequence, or mapping, optional
        Hover preset or explicit field selection passed to the Plotly helper.
    **diagnostic_kwargs
        Additional sampling, color-scaling, and rendering arguments accepted by
        :func:`kawin.diffusion.plot_surrogate_diagnostics`.

    Returns
    -------
    dict
        Diagnostics under the single ``"ab"`` key.

    Raises
    ------
    KeyError
        If a required result entry is missing.
    """
    if "surrogate_ab" not in result:
        raise KeyError("run result is missing required key 'surrogate_ab'.")

    thermodynamics = None
    if compare_ground_truth:
        if "therm_ab" not in result:
            raise KeyError("Ground-truth comparison requires result['therm_ab'].")
        thermodynamics = result["therm_ab"]

    return {
        "ab": plot_surrogate_diagnostics(
            result["surrogate_ab"],
            thermodynamics=thermodynamics,
            hover_fields=hover_fields,
            **diagnostic_kwargs,
        )
    }

# Example use from an interactive session:
surrogate_diagnostics = plot_surrogate_diagnostics_for_run(
    result,
    compare_ground_truth=True,
    renderer="browser",
    on_truth_error="record",
)
surrogate_diagnostics["ab"]["figures"]["thermodynamics"].show()
bulk_diff_A_fig = surrogate_diagnostics["ab"]["figures"]["bulk_diffusivity"][PHASE_A]
bulk_diff_A_fig.show()
bulk_diff_B_fig = surrogate_diagnostics["ab"]["figures"]["bulk_diffusivity"][PHASE_B]
bulk_diff_B_fig.show()
interface_diff_fig = surrogate_diagnostics["ab"]["figures"]["interface_diffusivity"]
interface_diff_fig.show()
# import plotly
# plotly.offline.plot(bulk_diff_fig, filename = fr"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\kawin\examples\ternaryExamples\{''.join([el.capitalize() for el in ELEMENTS])}_{TEMPERATURE}K_bulk_diffusivity_ab_{PHASE_A}_surrDiag.html", auto_open=False)

# %%
from kawin.diffusion import evaluate_diffusivity_leave_one_out, plot_diffusivity_leave_one_out

report = evaluate_diffusivity_leave_one_out(bulk_thermodynamics, phases=PHASE_A)
fig = plot_diffusivity_leave_one_out(report, PHASE_A)
fig.show()

phase = report["phase_reports"][PHASE_A]

print(phase["summary"])
print(np.unique(phase["prediction_status"], return_counts=True))
print(np.unique(phase["refit_interpolation"], return_counts=True))
# %%
report = evaluate_diffusivity_leave_one_out(bulk_thermodynamics, phases=PHASE_B)
fig = plot_diffusivity_leave_one_out(report, PHASE_B)
fig.show()

phase = report["phase_reports"][PHASE_B]

print(phase["summary"])
print(np.unique(phase["prediction_status"], return_counts=True))
print(np.unique(phase["refit_interpolation"], return_counts=True))
# %%
