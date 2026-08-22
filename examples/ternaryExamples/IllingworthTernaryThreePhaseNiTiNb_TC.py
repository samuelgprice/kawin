# %%
"""
Selectable three-phase Illingworth ternary moving-boundary examples.

This cell script can run either the original ``BCC_A2 | LIQUID | BCC_A2``
Ni-Ti-Nb Thermo-Calc case or a ``FCC_A1 | LIQUID | BCC_A2`` Fe-Cr-Ni pycalphad
case. The Thermo-Calc case requires TC-Python, TCHEA5, MOBHEA4, and a license.
The Fe-Cr-Ni case uses kawin/pycalphad with the checked-in Lee-style TDB, but
requires an explicit fixed LIQUID diffusivity matrix because that TDB does not
define LIQUID mobility terms.
"""

# %%
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


def _find_repo_root(start: Path) -> Path:
    """Finds the repository root when the file is run from a notebook kernel."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "kawin").exists():
            return candidate
    return Path.cwd()


THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = _find_repo_root(THIS_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.ThermoCalc.tc_python_adapter import TCPythonThermodynamics, ThermoCalcConfig
from examples.ternaryExamples.pycalphad_default_phase_adapter import create_pycalphad_thermodynamics_source
from kawin.diffusion import (
    MergedPhaseDiffusivitySurrogate,
    MovingBoundaryIllingworthTernaryThreePhaseFD1DModel,
    TernaryMovingBoundaryThermodynamicsSurrogate,
    merge_phase_diffusivity_surrogates,
    plot_surrogate_diagnostics,
)
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder
from kawin.solver import explicitEulerIterator
from kawin.thermo import MulticomponentThermodynamics


OUTPUTS = REPO_ROOT / "examples" / "ThermoCalc" / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
EXAMPLES_DIR = REPO_ROOT / "examples"

if __name__ == "__main__":
    def debugInPlace():
        try:
            import debugpy
            # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
            debugpy.listen(5678)
            print("Waiting for debugger attach")
            debugpy.wait_for_client()
            debugpy.breakpoint()
            print('break on this line')
        except:
            pass
    

# %%
# Editable case configuration

CASE_NI_TI_NB_TC = "ni_ti_nb_tc"
CASE_FE_CR_NI_PYCALPHAD = "fe_cr_ni_pycalphad"
CASE_NAME = CASE_FE_CR_NI_PYCALPHAD = "fe_cr_ni_pycalphad"

ELEMENTS = ("NB", "NI", "TI")
INDEPENDENT_ELEMENTS = ("NI", "TI")
PHASE_BCC = "BCC_B2"
PHASE_FCC = None
PHASE_LIQUID = "LIQUID"
PHASES_FOR_MODEL = (PHASE_BCC, PHASE_LIQUID, PHASE_BCC)
TEMPERATURE = 1300.0
REFERENCE_ELEMENT = "NB"
TDB_PATH = None
PYCALPHAD_USE_DEFAULT_PHASES = True
PYCALPHAD_EQUILIBRIUM_PHASES = None

# Full mole fractions are listed in ELEMENTS order: [NB, NI, TI].
LEFT_BCC_FULL = np.array([0.899, 0.001, 0.100], dtype=np.float64)
LIQUID_FULL = np.array([0.100, 0.300, 0.600], dtype=np.float64)
RIGHT_BCC_FULL = np.array([0.01, 0.495, 0.495], dtype=np.float64)
INITIAL_PHASE_COMPOSITIONS = (
    np.array([0.001, 0.100], dtype=np.float64),
    np.array([0.300, 0.600], dtype=np.float64),
    np.array([0.495, 0.495], dtype=np.float64),
)

LEFT_WIDTH = 40.0e-6
LIQUID_WIDTH = 2.0e-6
RIGHT_WIDTH = 40.0e-6
LENGTH = LEFT_WIDTH + LIQUID_WIDTH + RIGHT_WIDTH
INTERFACE_POSITIONS = np.array([LEFT_WIDTH, LEFT_WIDTH + LIQUID_WIDTH], dtype=np.float64)

# Tie-line probes use independent mole fractions in [NI, TI] order. The
# defaults connect the requested nominal phase compositions at each interface.
TIELINE_SURROGATE_BUILD_MODE = "line"  # "line" or "seed_point"
ETA_SAMPLES = np.linspace(0.0, 1.0, 9)
AB_PROBE_START = np.array([0.16, 0.83], dtype=np.float64)
AB_PROBE_END = np.array([0.217, 0.394], dtype=np.float64)
BC_PROBE_START = np.array([0.39, 0.468], dtype=np.float64)
BC_PROBE_END = np.array([0.4099, 0.59], dtype=np.float64)
AB_PROBE_POINT = np.array([np.nan, np.nan], dtype=np.float64)
BC_PROBE_POINT = np.array([np.nan, np.nan], dtype=np.float64)
PROBE_SAMPLES_PER_SIDE = 8
PROBE_BOUNDARY_MARGIN = 1.0e-3
PROBE_BOUNDARY_SEARCH_STEP = 1.0e-2
PROBE_BOUNDARY_XTOL = 1.0e-6
PROBE_MAX_SEARCH_STEPS = 200
INITIAL_ETA_GUESS = (0.5, 0.5)

# ``simplex_linear`` samples a rectangular candidate grid but keeps only
# simplex-valid points before building a scattered linear diffusivity
# interpolator. This is better suited than ``continuous_grid`` for the present
# Ni-Ti-Nb paths, which sit close to Ni+Ti=1 where rectangular grids would have
# invalid upper-right corners.
DIFFUSIVITY_INTERPOLATION = "simplex_linear" # "nearest", "continuous_grid"
BULK_DIFFUSIVITY_MODE = "composition_dependent_lagged"
BULK_DIFFUSIVITY_POINTS = None
BULK_DIFFUSIVITY_GRIDS = None
GLOBAL_MINIMIZATION_MAX_GRID_POINTS = 2000
FECRNI_LIQUID_DIFFUSIVITY_MATRIX =  np.array([[1e-9, 0.0], [0.0, 1e-9]])

NODES = 165
PHASE_NODES = (50*2, 10*2, 50*2)

# Time stepping:
#   DT_MODE = "fixed"    -> advance by FIXED_TIME_STEP.
#   DT_MODE = "semi_log" -> advance to semi-log-spaced target times starting
#                           at SEMI_LOG_T0 with natural-log spacing SEMI_LOG_DT.
#                           SEMI_LOG_BASE_TIME_STEP remains a positive fallback
#                           scale used internally by the model.
DT_MODE = "semi_log"
FIXED_TIME_STEP = 1.0e-3
SEMI_LOG_BASE_TIME_STEP = 1.0e-3
SEMI_LOG_DT = 0.05
SEMI_LOG_T0 = 1.0e-6
SOLVE_TIME = 1e3
TOLERANCE = 1.0e-10
MAX_ITERATIONS = 100
MAX_STEP_RETRIES = 8
MIN_DT_FRAC = 1.0e-16
VERBOSE = True
VERBOSE_INTERVAL = 10
RUN_PREFLIGHT = True
RUN_SOLVE = True


_CASE_DEFAULTS = {
    CASE_NI_TI_NB_TC: {
        "ELEMENTS": ("NB", "NI", "TI"),
        "INDEPENDENT_ELEMENTS": ("NI", "TI"),
        "PHASE_BCC": "BCC_B2",
        "PHASE_FCC": None,
        "PHASE_LIQUID": "LIQUID",
        "PHASES_FOR_MODEL": ("BCC_B2", "LIQUID", "BCC_B2"),
        "TEMPERATURE": 1300.0,
        "REFERENCE_ELEMENT": "NB",
        "TDB_PATH": None,
        "INITIAL_PHASE_COMPOSITIONS": (
            np.array([0.001, 0.100], dtype=np.float64),
            np.array([0.300, 0.600], dtype=np.float64),
            np.array([0.495, 0.495], dtype=np.float64),
        ),
        "LEFT_WIDTH": 40.0e-6,
        "LIQUID_WIDTH": 2.0e-6,
        "RIGHT_WIDTH": 40.0e-6,
        "INTERFACE_POSITIONS": np.array([40.0e-6, 42.0e-6], dtype=np.float64),
        "TIELINE_SURROGATE_BUILD_MODE": "line",
        "AB_PROBE_START": np.array([0.16, 0.83], dtype=np.float64),
        "AB_PROBE_END": np.array([0.217, 0.394], dtype=np.float64),
        "BC_PROBE_START": np.array([0.39, 0.468], dtype=np.float64),
        "BC_PROBE_END": np.array([0.4099, 0.59], dtype=np.float64),
        "AB_PROBE_POINT": np.array([np.nan, np.nan], dtype=np.float64),
        "BC_PROBE_POINT": np.array([np.nan, np.nan], dtype=np.float64),
        "PROBE_SAMPLES_PER_SIDE": 8,
        "PROBE_BOUNDARY_MARGIN": 1.0e-3,
        "PROBE_BOUNDARY_SEARCH_STEP": 1.0e-2,
        "PROBE_BOUNDARY_XTOL": 1.0e-6,
        "PROBE_MAX_SEARCH_STEPS": 200,
        "RUN_PREFLIGHT": True,
    },
    CASE_FE_CR_NI_PYCALPHAD: {
        "ELEMENTS": ("FE", "CR", "NI"),
        "INDEPENDENT_ELEMENTS": ("CR", "NI"),
        "PHASE_BCC": "BCC_A2",
        "PHASE_FCC": "FCC_A1",
        "PHASE_LIQUID": "LIQUID",
        "PHASES_FOR_MODEL": ("FCC_A1", "LIQUID", "BCC_A2"),
        "TEMPERATURE": 1650.0,
        "REFERENCE_ELEMENT": "FE",
        "TDB_PATH": EXAMPLES_DIR / "FeCrNi_Lee1993_L_style_ternary_checked_withMobility.tdb",
        "PYCALPHAD_USE_DEFAULT_PHASES": True,
        "PYCALPHAD_EQUILIBRIUM_PHASES": None,
        "INITIAL_PHASE_COMPOSITIONS": (
            np.array([0.30, 0.34], dtype=np.float64),
            np.array([0.43, 0.37], dtype=np.float64),
            np.array([0.52, 0.14], dtype=np.float64),
        ),
        "LEFT_WIDTH": 50.0e-6,
        "LIQUID_WIDTH": 10.0e-6,
        "RIGHT_WIDTH": 50.0e-6,
        "INTERFACE_POSITIONS": np.array([50.0e-6, 60.0e-6], dtype=np.float64),
        "TIELINE_SURROGATE_BUILD_MODE": ["line", "seed_point"][1],
        "AB_PROBE_START": np.array([0.3815, 0.319], dtype=np.float64),
        "AB_PROBE_END": np.array([0.417, 0.58], dtype=np.float64),
        "BC_PROBE_START": np.array([0.446, 0.264], dtype=np.float64),
        "BC_PROBE_END": np.array([0.638, 0.36], dtype=np.float64),
        "AB_PROBE_POINT": np.array([0.4, 0.47], dtype=np.float64),
        "BC_PROBE_POINT": np.array([0.55, 0.31], dtype=np.float64),
        "PROBE_SAMPLES_PER_SIDE": 25,
        "PROBE_BOUNDARY_MARGIN": 1.0e-3,
        "PROBE_BOUNDARY_SEARCH_STEP": 1.0e-2,
        "PROBE_BOUNDARY_XTOL": 1.0e-6,
        "PROBE_MAX_SEARCH_STEPS": 10000,
        "RUN_PREFLIGHT": False,
    },
}

_CASE_CONFIG_KEYS = tuple(dict.fromkeys(key for config in _CASE_DEFAULTS.values() for key in config))
_APPLIED_CASE_NAME = None


# %%
# Helpers

_OVERRIDE_KEY_ALIASES = {
    "bulk_diffusivity_mode": "BULK_DIFFUSIVITY_MODE",
    "bulk_diffusivity_grids": "BULK_DIFFUSIVITY_GRIDS",
    "bulk_diffusivity_points": "BULK_DIFFUSIVITY_POINTS",
    "case_name": "CASE_NAME",
    "dt_mode": "DT_MODE",
    "fecrni_liquid_diffusivity_matrix": "FECRNI_LIQUID_DIFFUSIVITY_MATRIX",
    "fixed_time_step": "FIXED_TIME_STEP",
    "global_minimization_max_grid_points": "GLOBAL_MINIMIZATION_MAX_GRID_POINTS",
    "interface_positions": "INTERFACE_POSITIONS",
    "max_iterations": "MAX_ITERATIONS",
    "max_step_retries": "MAX_STEP_RETRIES",
    "min_dt_frac": "MIN_DT_FRAC",
    "nodes": "NODES",
    "phase_nodes": "PHASE_NODES",
    "run_preflight": "RUN_PREFLIGHT",
    "run_solve": "RUN_SOLVE",
    "semi_log_base_time_step": "SEMI_LOG_BASE_TIME_STEP",
    "semi_log_dt": "SEMI_LOG_DT",
    "semi_log_t0": "SEMI_LOG_T0",
    "solve_time": "SOLVE_TIME",
    "temperature": "TEMPERATURE",
    "tieline_surrogate_build_mode": "TIELINE_SURROGATE_BUILD_MODE",
    "ab_probe_point": "AB_PROBE_POINT",
    "bc_probe_point": "BC_PROBE_POINT",
    "probe_samples_per_side": "PROBE_SAMPLES_PER_SIDE",
    "probe_boundary_margin": "PROBE_BOUNDARY_MARGIN",
    "probe_boundary_search_step": "PROBE_BOUNDARY_SEARCH_STEP",
    "probe_boundary_xtol": "PROBE_BOUNDARY_XTOL",
    "probe_max_search_steps": "PROBE_MAX_SEARCH_STEPS",
    "pycalphad_equilibrium_phases": "PYCALPHAD_EQUILIBRIUM_PHASES",
    "pycalphad_use_default_phases": "PYCALPHAD_USE_DEFAULT_PHASES",
    "tolerance": "TOLERANCE",
    "verbose": "VERBOSE",
    "verbose_interval": "VERBOSE_INTERVAL",
}


def _copy_case_value(value):
    """Returns a mutable-safe copy of a case-default value."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        return tuple(_copy_case_value(item) for item in value)
    return value


def _refresh_derived_config():
    """Refresh derived geometry after case/default overrides are applied."""
    global LENGTH
    LENGTH = float(LEFT_WIDTH) + float(LIQUID_WIDTH) + float(RIGHT_WIDTH)


def select_case(case_name):
    """Applies one of the selectable three-phase example configurations."""
    global _APPLIED_CASE_NAME
    case_name = str(case_name)
    if case_name not in _CASE_DEFAULTS:
        raise ValueError(f"Unknown three-phase example CASE_NAME '{case_name}'.")
    globals()["CASE_NAME"] = case_name
    for key, value in _CASE_DEFAULTS[case_name].items():
        globals()[key] = _copy_case_value(value)
    _refresh_derived_config()
    _APPLIED_CASE_NAME = case_name


def _ensure_selected_case_defaults():
    """Applies case defaults when ``CASE_NAME`` was changed directly."""
    if CASE_NAME != _APPLIED_CASE_NAME:
        select_case(CASE_NAME)


def _normalize_override_key(key):
    key = str(key)
    if key in globals():
        return key
    upper_key = key.upper()
    if upper_key in globals():
        return upper_key
    if key in _OVERRIDE_KEY_ALIASES:
        return _OVERRIDE_KEY_ALIASES[key]
    raise KeyError(f"Unknown three-phase example override '{key}'.")


@contextmanager
def _temporary_config(overrides=None):
    """Temporarily applies module-level example configuration overrides."""
    if not overrides:
        _ensure_selected_case_defaults()
        yield
        return
    normalized = {_normalize_override_key(key): value for key, value in dict(overrides).items()}
    old_values = {
        key: _copy_case_value(globals()[key])
        for key in dict.fromkeys((*_CASE_CONFIG_KEYS, *normalized.keys()))
        if key in globals()
    }
    try:
        if "CASE_NAME" in normalized:
            select_case(normalized["CASE_NAME"])
        globals().update(normalized)
        _refresh_derived_config()
        yield
    finally:
        globals().update(old_values)
        _refresh_derived_config()
        globals()["_APPLIED_CASE_NAME"] = CASE_NAME


select_case(CASE_NAME)


class ThreePhaseStepProfile:
    """Piecewise-constant initial profile for the selected three-phase order."""

    def __init__(self, interface_positions, independent_values):
        self.interface_positions = np.asarray(interface_positions, dtype=np.float64).reshape(2)
        self.independent_values = tuple(np.asarray(v, dtype=np.float64).reshape(2) for v in independent_values)

    def __call__(self, z):
        x = np.asarray(z, dtype=np.float64).reshape((-1, 1))[:, 0]
        out = np.empty((len(x), 2), dtype=np.float64)
        out[x < self.interface_positions[0]] = self.independent_values[0]
        middle = (x >= self.interface_positions[0]) & (x < self.interface_positions[1])
        out[middle] = self.independent_values[1]
        out[x >= self.interface_positions[1]] = self.independent_values[2]
        return out


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
        self.diffusivity_matrix = _validate_liquid_diffusivity_matrix(diffusivity_matrix)
        self.elements = list(getattr(thermodynamics, "elements", ELEMENTS))
        self.phases = list(getattr(thermodynamics, "phases", ()))

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
            if values.ndim == 2:
                return np.broadcast_to(self.diffusivity_matrix, (values.shape[0], 2, 2)).copy()
            return self.diffusivity_matrix.copy()
        try:
            return self.thermodynamics.getInterdiffusivity(x, T, phase=phase, query_context=query_context, **kwargs)
        except TypeError:
            return self.thermodynamics.getInterdiffusivity(x, T, phase=phase, **kwargs)


class ThreePhaseDiffusivityProvider:
    """
    Routes model bulk-diffusivity queries to phase-specific providers.

    The three-phase solver asks one object for all phases. This adapter lets the
    Fe-Cr-Ni case use the FCC/liquid surrogate for FCC, the liquid/BCC surrogate
    for BCC, and a fixed matrix for LIQUID.
    """

    def __init__(self, phase_sources):
        self.phase_sources = {str(phase): source for phase, source in dict(phase_sources).items()}

    def clearCache(self):
        """Clears each unique source cache when available."""
        seen = set()
        for source in self.phase_sources.values():
            if id(source) in seen:
                continue
            seen.add(id(source))
            clear_cache = getattr(source, "clearCache", None)
            if clear_cache is not None:
                clear_cache()

    def getInterdiffusivity(self, x, T=None, phase=None, query_context=None, **kwargs):
        """Delegates a diffusivity query to the provider registered for ``phase``."""
        phase = str(phase)
        if phase not in self.phase_sources:
            raise ValueError(f"No diffusivity provider is registered for phase '{phase}'.")
        source = self.phase_sources[phase]
        try:
            return source.getInterdiffusivity(x, T, phase=phase, query_context=query_context, **kwargs)
        except TypeError:
            return source.getInterdiffusivity(x, T, phase=phase, **kwargs)


def _interface_phase_pairs():
    """Returns the ordered A|B and B|C phase pairs for the selected case."""
    return (
        (PHASES_FOR_MODEL[0], PHASES_FOR_MODEL[1]),
        (PHASES_FOR_MODEL[1], PHASES_FOR_MODEL[2]),
    )


def _validate_liquid_diffusivity_matrix(matrix):
    """Validates the explicit fixed LIQUID diffusivity matrix for Fe-Cr-Ni."""
    if matrix is None:
        raise ValueError(
            "The Fe-Cr-Ni pycalphad case requires FECRNI_LIQUID_DIFFUSIVITY_MATRIX because "
            "FeCrNi_Lee1993_L_style_ternary_checked_withMobility.tdb does not define LIQUID "
            "mobility terms. Set it to a finite 2x2 NumPy array, for example "
            "np.array([[D_CrCr, D_CrNi], [D_NiCr, D_NiNi]], dtype=np.float64)."
        )
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (2, 2) or not np.all(np.isfinite(values)):
        raise ValueError("FECRNI_LIQUID_DIFFUSIVITY_MATRIX must be a finite 2x2 matrix.")
    return values.copy()


def _make_tc_config(phases):
    """Returns a TC-Python config for one ordered two-phase interface."""
    return ThermoCalcConfig(
        thermodynamic_database="TCHEA5",
        kinetic_database="MOBHEA4",
        elements=ELEMENTS,
        phases=tuple(phases),
        reference_element=REFERENCE_ELEMENT,
        global_minimization_max_grid_points=GLOBAL_MINIMIZATION_MAX_GRID_POINTS,
        cache_dir=OUTPUTS / "tc_cache",
    )


def _make_default_bulk_points():
    """
    Returns simplex-valid independent-component points for bulk diffusivity.

    The list includes the three nominal phase compositions and small local
    perturbations clipped to the ternary simplex so the surrogate can answer
    bulk queries without sampling an invalid rectangular composition domain.
    """
    centers = np.asarray(INITIAL_PHASE_COMPOSITIONS, dtype=np.float64)
    offsets = np.asarray(
        [
            [0.0, 0.0],
            [0.01, 0.0],
            [-0.01, 0.0],
            [0.0, 0.01],
            [0.0, -0.01],
        ],
        dtype=np.float64,
    )
    points = []
    for center in centers:
        for offset in offsets:
            point = np.clip(center + offset, 1.0e-6, 1.0 - 1.0e-6)
            if np.sum(point) <= 1.0 - 1.0e-6:
                points.append(point)
    return np.unique(np.asarray(points, dtype=np.float64), axis=0)


def _make_default_bulk_grids():
    """
    Returns independent-composition axes whose simplex-valid subset covers the case path.

    The axes intentionally include nominal phase compositions and probe
    endpoints. ``simplex_linear`` discards invalid axis combinations where
    Ni+Ti exceeds one; ``continuous_grid`` requires the whole rectangle to be
    valid, so users may need narrower custom axes for that mode.
    """
    if CASE_NAME == CASE_FE_CR_NI_PYCALPHAD:
        cr_axis = np.unique(
            np.asarray(
                [
                    0.30,
                    0.36,
                    0.378,
                    0.404,
                    0.43,
                    0.448,
                    0.484,
                    0.52,
                ],
                dtype=np.float64,
            )
        )
        ni_axis = np.unique(
            np.asarray(
                [
                    0.14,
                    0.20,
                    0.232,
                    0.30,
                    0.324,
                    0.34,
                    0.358,
                    0.364,
                    0.37,
                ],
                dtype=np.float64,
            )
        )
        return cr_axis, ni_axis

    ni_axis = np.unique(
        np.asarray(
            [
                0.001,
                0.05,
                0.10,
                0.16,
                0.217,
                0.30,
                0.39,
                0.4099,
                0.495,
                0.55,
            ],
            dtype=np.float64,
        )
    )
    ti_axis = np.unique(
        np.asarray(
            [
                0.10,
                0.20,
                0.39,
                0.468,
                0.495,
                0.59,
                0.60,
                0.72,
                0.83,
                0.899,
            ],
            dtype=np.float64,
        )
    )
    return ni_axis, ti_axis


def _surrogate_diffusivity_sampling_kwargs():
    """Returns diffusivity sampling kwargs for the selected surrogate mode."""
    interpolation = str(DIFFUSIVITY_INTERPOLATION)
    if interpolation == "nearest":
        bulk_points = _make_default_bulk_points() if BULK_DIFFUSIVITY_POINTS is None else np.asarray(BULK_DIFFUSIVITY_POINTS, dtype=np.float64)
        return {
            "diffusivity_interpolation": interpolation,
            "diffusivity_bulk_points": bulk_points,
        }
    if interpolation in {"simplex_linear", "continuous_grid"}:
        grids = _make_default_bulk_grids() if BULK_DIFFUSIVITY_GRIDS is None else tuple(np.asarray(axis, dtype=np.float64) for axis in BULK_DIFFUSIVITY_GRIDS)
        kwargs = {
            "diffusivity_interpolation": interpolation,
            "diffusivity_bulk_grids": grids,
        }
        if interpolation == "simplex_linear" and BULK_DIFFUSIVITY_POINTS is not None:
            kwargs["diffusivity_bulk_points"] = np.asarray(BULK_DIFFUSIVITY_POINTS, dtype=np.float64)
        return kwargs
    raise ValueError("DIFFUSIVITY_INTERPOLATION must be 'nearest', 'continuous_grid', or 'simplex_linear'.")


def build_thermodynamics():
    """
    Builds thermodynamics facades for the two adjacent selected interfaces.

    TC-Python and kawin/pycalphad both treat the first configured phase as the
    matrix-side endpoint returned by ``getInterfacialComposition``. The Fe-Cr-Ni
    pycalphad case wraps LIQUID diffusivity with an explicit fixed matrix.
    """
    pair_ab, pair_bc = _interface_phase_pairs()
    if CASE_NAME == CASE_NI_TI_NB_TC:
        therm_ab = TCPythonThermodynamics(_make_tc_config(pair_ab))
        therm_bc = TCPythonThermodynamics(_make_tc_config(pair_bc))
        return therm_ab, therm_bc
    if CASE_NAME == CASE_FE_CR_NI_PYCALPHAD:
        liquid_matrix = _validate_liquid_diffusivity_matrix(FECRNI_LIQUID_DIFFUSIVITY_MATRIX)
        if TDB_PATH is None or not Path(TDB_PATH).exists():
            raise FileNotFoundError(f"Could not find Fe-Cr-Ni TDB at {TDB_PATH}.")
        therm_ab = FixedPhaseDiffusivityThermodynamics(
            create_pycalphad_thermodynamics_source(
                str(TDB_PATH),
                list(ELEMENTS),
                list(pair_ab),
                use_default_phases=PYCALPHAD_USE_DEFAULT_PHASES,
                equilibrium_phases=PYCALPHAD_EQUILIBRIUM_PHASES,
            ),
            PHASE_LIQUID,
            liquid_matrix,
        )
        therm_bc = FixedPhaseDiffusivityThermodynamics(
            create_pycalphad_thermodynamics_source(
                str(TDB_PATH),
                list(ELEMENTS),
                list(pair_bc),
                use_default_phases=PYCALPHAD_USE_DEFAULT_PHASES,
                equilibrium_phases=PYCALPHAD_EQUILIBRIUM_PHASES,
            ),
            PHASE_LIQUID,
            liquid_matrix,
        )
        return therm_ab, therm_bc
    raise ValueError(f"Unknown three-phase example CASE_NAME '{CASE_NAME}'.")


def _validate_tieline_probe_path(thermodynamics, label, probe_start, probe_end, tieline_phases):
    """
    Checks that each eta sample lies in the requested two-phase field.

    TC-Python can fail later with a low-level null-pointer exception when the
    surrogate builder asks for tie-line endpoints outside the intended
    two-phase region. This preflight keeps the error at the thermodynamic
    boundary: every sampled bulk composition must return both endpoint phases
    with positive phase amounts.
    """
    expected = tuple(str(phase).upper() for phase in tieline_phases)
    bad_samples = []
    for eta in np.asarray(ETA_SAMPLES, dtype=np.float64):
        point = (1.0 - eta) * np.asarray(probe_start, dtype=np.float64) + eta * np.asarray(probe_end, dtype=np.float64)
        equilibrium_data = getattr(thermodynamics, "getEquilibriumData", None)
        if equilibrium_data is not None:
            try:
                equilibrium = equilibrium_data(point, TEMPERATURE, removeCache=False)
            except AttributeError:
                equilibrium = None
        else:
            equilibrium = None
        if equilibrium is not None:
            stable_phases = tuple(str(phase).upper() for phase in equilibrium.get("stable_phases", ()))
            phase_amounts = {
                str(phase).upper(): float(amount)
                for phase, amount in equilibrium.get("phase_amounts", {}).items()
            }
            missing = [
                phase
                for phase in expected
                if phase not in stable_phases or phase_amounts.get(phase, 0.0) <= 1.0e-12
            ]
            extra = [
                phase
                for phase in stable_phases
                if phase not in expected and phase_amounts.get(phase, 1.0) > 1.0e-12
            ]
            if len(stable_phases) != len(set(stable_phases)):
                extra = [*extra, "DUPLICATE_PHASE_SET"]
        else:
            _, _, metadata = thermodynamics.getInterfacialComposition(
                point,
                TEMPERATURE,
                precPhase=tieline_phases[1],
                returnMeta=True,
            )
            stable_phases = tuple(str(phase).upper() for phase in metadata.get("endpoint_phases", ()))
            phase_amounts = {}
            missing = [phase for phase in expected if phase not in stable_phases]
            extra = [phase for phase in stable_phases if phase not in expected]
        if missing or extra:
            bad_samples.append((float(eta), point, stable_phases, phase_amounts, tuple(missing), tuple(extra)))

    if bad_samples:
        details = []
        for eta, point, stable_phases, phase_amounts, missing, extra in bad_samples[:5]:
            details.append(
                "eta={:.3g}, x{}={}, stable={}, amounts={}, missing={}, extra={}".format(
                    eta,
                    list(INDEPENDENT_ELEMENTS),
                    np.array2string(point, precision=6, separator=", "),
                    stable_phases,
                    phase_amounts,
                    missing,
                    extra,
                )
            )
        if len(bad_samples) > 5:
            details.append(f"... {len(bad_samples) - 5} more invalid eta samples")
        raise ValueError(
            f"{label} tie-line probe path is not in the {expected} two-phase field at "
            f"{TEMPERATURE:g} K.\n"
            + "\n".join(details)
            + "\nChoose probe endpoints inside the intended two-phase region, or check "
            "the selected phase name/database/temperature. This is the condition that "
            "can otherwise surface from TC-Python as "
            "'GeneralCalculationException: Null pointer exception'."
        )


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
            "probe_boundary_margin": PROBE_BOUNDARY_MARGIN,
            "probe_boundary_search_step": PROBE_BOUNDARY_SEARCH_STEP,
            "probe_boundary_xtol": PROBE_BOUNDARY_XTOL,
            "probe_max_search_steps": PROBE_MAX_SEARCH_STEPS,
        }
    raise ValueError("TIELINE_SURROGATE_BUILD_MODE must be 'line' or 'seed_point'.")


def build_interface_surrogates(therm_ab, therm_bc):
    """Samples the selected A|B and B|C tie-line families."""
    diffusivity_sampling = _surrogate_diffusivity_sampling_kwargs()
    pair_ab, pair_bc = _interface_phase_pairs()
    if str(TIELINE_SURROGATE_BUILD_MODE).lower() == "line":
        with therm_ab:
            _validate_tieline_probe_path(
                therm_ab,
                f"{pair_ab[0]}/{pair_ab[1]}",
                AB_PROBE_START,
                AB_PROBE_END,
                pair_ab,
            )
        with therm_bc:
            _validate_tieline_probe_path(
                therm_bc,
                f"{pair_bc[0]}/{pair_bc[1]}",
                BC_PROBE_START,
                BC_PROBE_END,
                pair_bc,
            )
    with therm_ab:
        kwargs_ab = {}
        if CASE_NAME == CASE_FE_CR_NI_PYCALPHAD:
            kwargs_ab["validation_database"] = TDB_PATH
        surrogate_ab = TernaryMovingBoundaryThermodynamicsSurrogate.from_database(
            thermodynamics=therm_ab,
            elements=ELEMENTS,
            phases=pair_ab,
            tieline_phases=pair_ab,
            temperature=TEMPERATURE,
            **_tieline_surrogate_probe_kwargs(AB_PROBE_START, AB_PROBE_END, AB_PROBE_POINT),
            precipitate_phase=pair_ab[1],
            **diffusivity_sampling,
            **kwargs_ab,
        )
    with therm_bc:
        kwargs_bc = {}
        if CASE_NAME == CASE_FE_CR_NI_PYCALPHAD:
            kwargs_bc["validation_database"] = TDB_PATH
        surrogate_bc = TernaryMovingBoundaryThermodynamicsSurrogate.from_database(
            thermodynamics=therm_bc,
            elements=ELEMENTS,
            phases=pair_bc,
            tieline_phases=pair_bc,
            temperature=TEMPERATURE,
            **_tieline_surrogate_probe_kwargs(BC_PROBE_START, BC_PROBE_END, BC_PROBE_POINT),
            precipitate_phase=pair_bc[1],
            **diffusivity_sampling,
            **kwargs_bc,
        )
    return surrogate_ab, surrogate_bc


def build_bulk_diffusivity_provider(surrogate_ab, surrogate_bc):
    """Returns the model thermodynamics/diffusivity provider for the selected case."""
    if CASE_NAME == CASE_FE_CR_NI_PYCALPHAD:
        liquid_source = FixedPhaseDiffusivityThermodynamics(
            surrogate_ab,
            PHASE_LIQUID,
            _validate_liquid_diffusivity_matrix(FECRNI_LIQUID_DIFFUSIVITY_MATRIX),
        )
        return ThreePhaseDiffusivityProvider(
            {
                PHASES_FOR_MODEL[0]: surrogate_ab,
                PHASES_FOR_MODEL[1]: liquid_source,
                PHASES_FOR_MODEL[2]: surrogate_bc,
            }
        )
    elif CASE_NAME == CASE_NI_TI_NB_TC:
        liquid_source = merge_phase_diffusivity_surrogates(
            surrogate_ab,
            surrogate_bc,
            PHASES_FOR_MODEL[1],
            diffusivity_interpolation=DIFFUSIVITY_INTERPOLATION,
        )
        return ThreePhaseDiffusivityProvider(
                    {
                        PHASES_FOR_MODEL[0]: surrogate_ab,
                        PHASES_FOR_MODEL[1]: liquid_source,
                        PHASES_FOR_MODEL[2]: surrogate_bc,
                    }
                )

    return surrogate_ab


def make_mesh():
    """Builds the initial three-phase independent-composition profile."""
    mesh = CartesianFD1D(INDEPENDENT_ELEMENTS, [0.0, LENGTH], NODES)
    profile = ProfileBuilder(
        [
            (
                ThreePhaseStepProfile(
                    INTERFACE_POSITIONS,
                    INITIAL_PHASE_COMPOSITIONS,
                ),
                INDEPENDENT_ELEMENTS,
            )
        ]
    )
    mesh.setResponseProfile(profile)
    return mesh


def get_time_step_options():
    """Returns constructor kwargs for the selected three-phase timestep mode."""
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


def build_model(surrogate_ab, surrogate_bc, bulk_thermodynamics=None):
    """Constructs the three-phase Illingworth model without starting the solve."""
    time_step_options = get_time_step_options()
    return MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=make_mesh(),
        elements=ELEMENTS,
        phases=PHASES_FOR_MODEL,
        thermodynamics=surrogate_ab if bulk_thermodynamics is None else bulk_thermodynamics,
        temperature=TEMPERATURE,
        interfacePositions=INTERFACE_POSITIONS,
        interface_equilibria=(surrogate_ab, surrogate_bc),
        initial_eta_guess=INITIAL_ETA_GUESS,
        bulk_diffusivity_mode=BULK_DIFFUSIVITY_MODE,
        time_step=time_step_options["time_step"],
        dt_mode=time_step_options["dt_mode"],
        semiLog_dt=time_step_options["semiLog_dt"],
        semiLogT0=time_step_options["semiLogT0"],
        phase_nodes=PHASE_NODES,
        tolerance=TOLERANCE,
        residual_tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        max_step_retries=MAX_STEP_RETRIES,
        record=True,
    )


def print_case_summary(model):
    """Prints initial/final widths and inventory drift for a solved model."""
    positions = model.getInterfacePositions()
    widths = np.array([positions[0], positions[1] - positions[0], LENGTH - positions[1]], dtype=np.float64)
    print("Interface positions (um):", positions * 1.0e6)
    print("Phase widths A|B|C (um):", widths * 1.0e6)
    if model.currentTime > 0.0:
        print(f"Inventory drift {list(INDEPENDENT_ELEMENTS)}:", model.checkConservation(TOLERANCE))


def plot_phase_widths(model):
    """Plots selected phase widths over time."""
    times = model.interfaceData._time[: model.interfaceData.N + 1]
    positions = model.interfaceData._y[: model.interfaceData.N + 1]
    widths = np.column_stack((positions[:, 0], positions[:, 1] - positions[:, 0], LENGTH - positions[:, 1]))
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, label in enumerate(PHASES_FOR_MODEL):
        ax.plot(times, widths[:, i] * 1.0e6, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Width (um)")
    ax.set_ylim(0, 60)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_independent_profiles(model, time=None):
    """Plots selected independent-component profiles on the physical mesh."""
    z_um = model._z * 1.0e6
    y = model.data.y(time)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, element in enumerate(INDEPENDENT_ELEMENTS):
        ax.plot(z_um, y[:, i], label=f"X({element})")
    for position in model.getInterfacePositions(time):
        ax.axvline(position * 1.0e6, color="0.35", linestyle="--", linewidth=1)
    ax.set_xlabel("Distance (um)")
    ax.set_ylabel("Mole fraction")
    ax.set_ylim(0.15, 0.55)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_inventory_drift(model):
    """Plots componentwise total-inventory drift from the initial state."""
    times = model.inventoryData._time[: model.inventoryData.N + 1]
    inventory = model.inventoryData._y[: model.inventoryData.N + 1]
    drift = inventory - inventory[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, element in enumerate(INDEPENDENT_ELEMENTS):
        ax.plot(times, drift[:, i], label=f"{element}")
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Inventory drift")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend()
    fig.tight_layout()
    return fig, ax

def plot_surrogate_diagnostics_for_run(
    result,
    *,
    compare_ground_truth=False,
    hover_fields="diagnostic",
    **diagnostic_kwargs,
):
    """Builds interactive surrogate diagnostics for a completed ``run_case`` result.

    Ground-truth calls are made only when ``compare_ground_truth`` is true.
    The returned Plotly figures are not shown or saved. In a notebook, display
    a figure directly, or persist one with, for example,
    ``diagnostics["ab"]["figures"]["thermodynamics"].write_html("ab.html")``.

    Parameters
    ----------
    result : dict
        Mapping returned by :func:`run_case`.
    compare_ground_truth : bool, optional
        Whether to compare A|B and B|C surrogates to their original
        thermodynamics providers.
    hover_fields : str, sequence, or mapping, optional
        Hover preset or explicit field selection passed to the Plotly helper.
    **diagnostic_kwargs
        Additional sampling and rendering arguments accepted by
        :func:`kawin.diffusion.plot_surrogate_diagnostics`.

    Returns
    -------
    dict
        Diagnostics keyed by ``"ab"``, ``"bc"``, and, when present,
        ``"merged_bulk"``.
    """
    required = ("surrogate_ab", "surrogate_bc")
    missing = [key for key in required if key not in result]
    if missing:
        raise KeyError(f"run_case result is missing required keys {missing}.")

    output = {}
    for label, surrogate_key, thermodynamics_key in (
        ("ab", "surrogate_ab", "therm_ab"),
        ("bc", "surrogate_bc", "therm_bc"),
    ):
        thermodynamics = None
        if compare_ground_truth:
            if thermodynamics_key not in result:
                raise KeyError(f"Ground-truth comparison requires result['{thermodynamics_key}'].")
            thermodynamics = result[thermodynamics_key]
        output[label] = plot_surrogate_diagnostics(
            result[surrogate_key],
            thermodynamics=thermodynamics,
            hover_fields=hover_fields,
            **diagnostic_kwargs,
        )

    bulk_provider = result.get("bulk_thermodynamics")
    phase_sources = getattr(bulk_provider, "phase_sources", {})
    merged = {}
    for phase, source in phase_sources.items():
        if isinstance(source, MergedPhaseDiffusivitySurrogate):
            merged_truth = result.get("therm_ab") if compare_ground_truth else None
            merged[phase] = plot_surrogate_diagnostics(
                source,
                thermodynamics=merged_truth,
                hover_fields=hover_fields,
                **diagnostic_kwargs,
            )
    if merged:
        output["merged_bulk"] = merged
    return output


def run_case(overrides=None, *, make_plots=True):
    """
    Builds selected interface surrogates, constructs the model, and optionally solves.

    ``overrides`` can temporarily replace module-level values, for example
    ``run_case({"case_name": "fe_cr_ni_pycalphad", "run_solve": False})``.
    """
    with _temporary_config(overrides):
        therm_ab, therm_bc = build_thermodynamics()
        if RUN_PREFLIGHT:
            pair_ab, pair_bc = _interface_phase_pairs()
            if hasattr(therm_ab, "preflight"):
                if str(TIELINE_SURROGATE_BUILD_MODE).lower() == "line":
                    print(f"{pair_ab[0]}/{pair_ab[1]} preflight:", therm_ab.preflight(x=AB_PROBE_START, T=TEMPERATURE))
                    print(f"{pair_bc[0]}/{pair_bc[1]} preflight:", therm_bc.preflight(x=BC_PROBE_START, T=TEMPERATURE))
            else:
                if str(TIELINE_SURROGATE_BUILD_MODE).lower() == "line":
                    _validate_tieline_probe_path(therm_ab, f"{pair_ab[0]}/{pair_ab[1]}", AB_PROBE_START, AB_PROBE_END, pair_ab)
                    _validate_tieline_probe_path(therm_bc, f"{pair_bc[0]}/{pair_bc[1]}", BC_PROBE_START, BC_PROBE_END, pair_bc)
                    print(f"Validated {pair_ab[0]}/{pair_ab[1]} and {pair_bc[0]}/{pair_bc[1]} tie-line probe paths.")

        start = time.perf_counter()
        surrogate_ab, surrogate_bc = build_interface_surrogates(therm_ab, therm_bc)
        print(f"Built interface surrogates in {time.perf_counter() - start:.1f} s.")

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
        bulk_thermodynamics = build_bulk_diffusivity_provider(surrogate_ab, surrogate_bc)
        model = build_model(surrogate_ab, surrogate_bc, bulk_thermodynamics=bulk_thermodynamics)
        if RUN_SOLVE:
            model.solve(
                SOLVE_TIME,
                iterator=explicitEulerIterator,
                verbose=VERBOSE,
                vIt=VERBOSE_INTERVAL,
                minDtFrac=MIN_DT_FRAC,
            )
        print_case_summary(model)

        figures = {}
        if make_plots and model.currentTime > 0.0:
            figures["phase_widths"] = plot_phase_widths(model)
            figures["profiles"] = plot_independent_profiles(model)
            figures["inventory_drift"] = plot_inventory_drift(model)
            plt.show()
        return {
            "model": model,
            "surrogate_ab": surrogate_ab,
            "surrogate_bc": surrogate_bc,
            "bulk_thermodynamics": bulk_thermodynamics,
            "therm_ab": therm_ab,
            "therm_bc": therm_bc,
            "figures": figures,
        }


# %%
if __name__ == "__main__":
    # debugInPlace()
    result = run_case(make_plots=True)
    surrogate_diagnostics = plot_surrogate_diagnostics_for_run(result, compare_ground_truth=True, renderer="browser")
    surrogate_diagnostics["ab"]["figures"]["thermodynamics"].show()
    surrogate_diagnostics["bc"]["figures"]["thermodynamics"].show()
    surrogate_diagnostics["ab"]["figures"]["interface_diffusivity"].show()
    surrogate_diagnostics["ab"]["figures"]["bulk_diffusivity"]['LIQUID'].show()
    surrogate_diagnostics["ab"]["figures"]["bulk_diffusivity"]['FCC_A1'].show()
    surrogate_diagnostics["ab"]["figures"]["bulk_diffusivity"]['LIQUID#1'].show()
    surrogate_diagnostics["ab"]["figures"]["bulk_diffusivity"]['FCC_L12#1'].show()

# %%
