"""Frozen Fe-Cr-Ni regression fixture for the three-phase example.

This module is deliberately independent of
``IllingworthTernaryThreePhaseNiTiNb_TC.py``, which is an editable interactive
cell script.  Its configuration and public helpers form the license-free
automated-test fixture; changes here require corresponding test updates.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np

from examples.ternaryExamples.pycalphad_default_phase_adapter import create_pycalphad_thermodynamics_source
from kawin.diffusion import (
    MergedPhaseDiffusivitySurrogate,
    MovingBoundaryIllingworthTernaryThreePhaseFD1DModel,
    TernaryMovingBoundaryThermodynamicsSurrogate,
    plot_surrogate_diagnostics,
)
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder


def _find_repo_root(start: Path) -> Path:
    """Returns the repository root when run from a notebook-like environment."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "kawin").exists():
            return candidate
    return Path.cwd()


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
EXAMPLES_DIR = REPO_ROOT / "examples"

CASE_FE_CR_NI_PYCALPHAD = "fe_cr_ni_pycalphad"
CASE_NAME = CASE_FE_CR_NI_PYCALPHAD
ELEMENTS = ("FE", "CR", "NI")
INDEPENDENT_ELEMENTS = ("CR", "NI")
PHASE_LIQUID = "LIQUID"
PHASES_FOR_MODEL = ("FCC_A1", PHASE_LIQUID, "BCC_A2")
TEMPERATURE = 1650.0
TDB_PATH = EXAMPLES_DIR / "FeCrNi_Lee1993_L_style_ternary_checked_withMobility.tdb"
PYCALPHAD_USE_DEFAULT_PHASES = False
PYCALPHAD_EQUILIBRIUM_PHASES = None
G_OFFSET = 0.0

INITIAL_PHASE_COMPOSITIONS = (
    np.array([0.30, 0.34], dtype=np.float64),
    np.array([0.43, 0.37], dtype=np.float64),
    np.array([0.52, 0.14], dtype=np.float64),
)
LEFT_WIDTH = 50.0e-6
LIQUID_WIDTH = 10.0e-6
RIGHT_WIDTH = 50.0e-6
INTERFACE_POSITIONS = np.array([LEFT_WIDTH, LEFT_WIDTH + LIQUID_WIDTH], dtype=np.float64)
LENGTH = LEFT_WIDTH + LIQUID_WIDTH + RIGHT_WIDTH

TIELINE_SURROGATE_BUILD_MODE = "seed_point"
ETA_SAMPLES = np.linspace(0.0, 1.0, 3)
AB_PROBE_POINT = np.array([0.4, 0.47], dtype=np.float64)
BC_PROBE_POINT = np.array([0.55, 0.31], dtype=np.float64)
PROBE_SAMPLES_PER_SIDE = 4
PROBE_BOUNDARY_SEARCH_STEP = 1.0e-2
PROBE_BOUNDARY_XTOL = 1.0e-10
PROBE_MAX_SEARCH_STEPS = 203
INITIAL_ETA_GUESS = (0.5, 0.5)

DIFFUSIVITY_INTERPOLATION = "nearest"
BULK_DIFFUSIVITY_POINTS = np.empty((0, 2), dtype=np.float64)
BULK_DIFFUSIVITY_MODE = "composition_dependent_lagged"
FECRNI_LIQUID_DIFFUSIVITY_MATRIX = np.array([[1.0e-8, 0.0], [0.0, 1.0e-8]], dtype=np.float64)
NODES = 31
PHASE_NODES = (5, 5, 5)
PHASE_MOLAR_VOLUMES = None
TIME_STEP = 1.0e-3
TOLERANCE = 1.0e-10
MAX_ITERATIONS = 100
MAX_STEP_RETRIES = 8
RUN_PREFLIGHT = False
RUN_SOLVE = False

_CONFIG_KEYS = (
    "CASE_NAME", "TEMPERATURE", "PYCALPHAD_USE_DEFAULT_PHASES",
    "PYCALPHAD_EQUILIBRIUM_PHASES", "G_OFFSET", "TIELINE_SURROGATE_BUILD_MODE",
    "ETA_SAMPLES", "AB_PROBE_POINT", "BC_PROBE_POINT", "PROBE_SAMPLES_PER_SIDE",
    "PROBE_BOUNDARY_SEARCH_STEP", "PROBE_BOUNDARY_XTOL", "PROBE_MAX_SEARCH_STEPS",
    "DIFFUSIVITY_INTERPOLATION", "BULK_DIFFUSIVITY_POINTS", "BULK_DIFFUSIVITY_MODE",
    "FECRNI_LIQUID_DIFFUSIVITY_MATRIX", "NODES", "PHASE_NODES", "PHASE_MOLAR_VOLUMES",
    "TIME_STEP", "TOLERANCE", "MAX_ITERATIONS", "MAX_STEP_RETRIES", "RUN_PREFLIGHT", "RUN_SOLVE",
)
_OVERRIDE_KEY_ALIASES = {
    "case_name": "CASE_NAME", "eta_samples": "ETA_SAMPLES",
    "diffusivity_interpolation": "DIFFUSIVITY_INTERPOLATION",
    "bulk_diffusivity_points": "BULK_DIFFUSIVITY_POINTS",
    "fecrni_liquid_diffusivity_matrix": "FECRNI_LIQUID_DIFFUSIVITY_MATRIX",
    "nodes": "NODES", "phase_nodes": "PHASE_NODES",
    "phase_molar_volumes": "PHASE_MOLAR_VOLUMES", "run_preflight": "RUN_PREFLIGHT",
    "run_solve": "RUN_SOLVE", "verbose": None,
}


def _copy_config_value(value):
    """Returns a copy suitable for restoring mutable fixture configuration."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        return tuple(_copy_config_value(item) for item in value)
    return value


def _normalize_override_key(key):
    """Maps public lower-case override names to fixture configuration names."""
    key = str(key)
    if key in _OVERRIDE_KEY_ALIASES:
        return _OVERRIDE_KEY_ALIASES[key]
    upper_key = key.upper()
    if upper_key in _CONFIG_KEYS:
        return upper_key
    raise KeyError(f"Unknown stable three-phase fixture override '{key}'.")


@contextmanager
def _temporary_config(overrides=None):
    """Applies overrides for one call and restores every fixture setting afterward."""
    normalized = {}
    for key, value in dict(overrides or {}).items():
        normalized_key = _normalize_override_key(key)
        if normalized_key is not None:
            normalized[normalized_key] = value
    if normalized.get("CASE_NAME", CASE_NAME) != CASE_FE_CR_NI_PYCALPHAD:
        raise ValueError(f"Unknown stable three-phase fixture CASE_NAME '{normalized['CASE_NAME']}'.")
    old_values = {key: _copy_config_value(globals()[key]) for key in _CONFIG_KEYS}
    try:
        globals().update(normalized)
        yield
    finally:
        globals().update(old_values)


class ThreePhaseStepProfile:
    """Piecewise-constant Fe-Cr-Ni profile with two fixed initial interfaces."""

    def __call__(self, z):
        x = np.asarray(z, dtype=np.float64).reshape((-1, 1))[:, 0]
        out = np.empty((len(x), 2), dtype=np.float64)
        out[x < INTERFACE_POSITIONS[0]] = INITIAL_PHASE_COMPOSITIONS[0]
        middle = (x >= INTERFACE_POSITIONS[0]) & (x < INTERFACE_POSITIONS[1])
        out[middle] = INITIAL_PHASE_COMPOSITIONS[1]
        out[x >= INTERFACE_POSITIONS[1]] = INITIAL_PHASE_COMPOSITIONS[2]
        return out


def _validate_liquid_diffusivity_matrix(matrix):
    """Validates the mandatory fixed LIQUID diffusivity for the checked-in TDB."""
    if matrix is None:
        raise ValueError(
            "The Fe-Cr-Ni pycalphad case requires FECRNI_LIQUID_DIFFUSIVITY_MATRIX because "
            "FeCrNi_Lee1993_L_style_ternary_checked_withMobility.tdb has no LIQUID mobility terms."
        )
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (2, 2) or not np.all(np.isfinite(values)):
        raise ValueError("FECRNI_LIQUID_DIFFUSIVITY_MATRIX must be a finite 2x2 matrix.")
    return values.copy()


class FixedPhaseDiffusivityThermodynamics:
    """Delegates thermodynamics while returning a fixed matrix for the LIQUID phase."""

    def __init__(self, thermodynamics, fixed_phase, diffusivity_matrix):
        self.thermodynamics = thermodynamics
        self.fixed_phase = str(fixed_phase)
        self.diffusivity_matrix = _validate_liquid_diffusivity_matrix(diffusivity_matrix)
        self.elements = list(getattr(thermodynamics, "elements", ELEMENTS))
        self.phases = list(getattr(thermodynamics, "phases", ()))

    def __enter__(self):
        enter = getattr(self.thermodynamics, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        exit_ = getattr(self.thermodynamics, "__exit__", None)
        return False if exit_ is None else exit_(exc_type, exc, traceback)

    def getInterfacialComposition(self, *args, **kwargs):
        """Delegates tie-line equilibrium queries."""
        return self.thermodynamics.getInterfacialComposition(*args, **kwargs)

    def getInterdiffusivity(self, x, T=None, phase=None, query_context=None, **kwargs):
        """Returns the fixed LIQUID matrix or delegates another phase query."""
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
    """Routes three-phase solver diffusivity requests to phase-specific sources."""

    def __init__(self, phase_sources):
        self.phase_sources = {str(phase): source for phase, source in dict(phase_sources).items()}

    def clearCache(self):
        """Clears each unique registered source cache when that source supports it."""
        seen = set()
        for source in self.phase_sources.values():
            if id(source) in seen:
                continue
            seen.add(id(source))
            clear_cache = getattr(source, "clearCache", None)
            if clear_cache is not None:
                clear_cache()

    def getInterdiffusivity(self, x, T=None, phase=None, query_context=None, **kwargs):
        """Delegates a diffusivity query to the source registered for ``phase``."""
        source = self.phase_sources[str(phase)]
        try:
            return source.getInterdiffusivity(x, T, phase=phase, query_context=query_context, **kwargs)
        except TypeError:
            return source.getInterdiffusivity(x, T, phase=phase, **kwargs)


def _interface_phase_pairs():
    """Returns the frozen FCC|LIQUID and LIQUID|BCC phase pairs."""
    return ((PHASES_FOR_MODEL[0], PHASES_FOR_MODEL[1]), (PHASES_FOR_MODEL[1], PHASES_FOR_MODEL[2]))


def build_thermodynamics():
    """Builds pycalphad facades for the two frozen Fe-Cr-Ni interfaces."""
    liquid_matrix = _validate_liquid_diffusivity_matrix(FECRNI_LIQUID_DIFFUSIVITY_MATRIX)
    if not TDB_PATH.exists():
        raise FileNotFoundError(f"Could not find Fe-Cr-Ni TDB at {TDB_PATH}.")
    sources = []
    for phases in _interface_phase_pairs():
        source = create_pycalphad_thermodynamics_source(
            str(TDB_PATH), list(ELEMENTS), list(phases),
            use_default_phases=PYCALPHAD_USE_DEFAULT_PHASES,
            equilibrium_phases=PYCALPHAD_EQUILIBRIUM_PHASES, g_offset=G_OFFSET,
        )
        sources.append(FixedPhaseDiffusivityThermodynamics(source, PHASE_LIQUID, liquid_matrix))
    return tuple(sources)


def _surrogate_probe_kwargs(point):
    """Returns the frozen seed-point sampling arguments for one interface."""
    if str(TIELINE_SURROGATE_BUILD_MODE).lower() != "seed_point":
        raise ValueError("The stable fixture supports only seed-point tie-line sampling.")
    return {
        "probe_point": np.asarray(point, dtype=np.float64),
        "probe_samples_per_side": PROBE_SAMPLES_PER_SIDE,
        "probe_boundary_search_step": PROBE_BOUNDARY_SEARCH_STEP,
        "probe_boundary_xtol": PROBE_BOUNDARY_XTOL,
        "probe_max_search_steps": PROBE_MAX_SEARCH_STEPS,
    }


def build_interface_surrogates(therm_ab, therm_bc):
    """Builds compact database-derived surrogates for both frozen interfaces."""
    surrogates = []
    for thermodynamics, phases, point in zip((therm_ab, therm_bc), _interface_phase_pairs(), (AB_PROBE_POINT, BC_PROBE_POINT)):
        with thermodynamics:
            surrogates.append(TernaryMovingBoundaryThermodynamicsSurrogate.from_database(
                thermodynamics=thermodynamics, elements=ELEMENTS, phases=phases, tieline_phases=phases,
                temperature=TEMPERATURE, precipitate_phase=phases[1], validation_database=TDB_PATH,
                diffusivity_interpolation=DIFFUSIVITY_INTERPOLATION,
                diffusivity_bulk_points=np.asarray(BULK_DIFFUSIVITY_POINTS, dtype=np.float64),
                **_surrogate_probe_kwargs(point),
            ))
    return tuple(surrogates)


def build_bulk_diffusivity_provider(surrogate_ab, surrogate_bc):
    """Returns the phase-routed provider with the fixed LIQUID diffusivity."""
    liquid_source = FixedPhaseDiffusivityThermodynamics(
        surrogate_ab, PHASE_LIQUID, _validate_liquid_diffusivity_matrix(FECRNI_LIQUID_DIFFUSIVITY_MATRIX)
    )
    return ThreePhaseDiffusivityProvider({
        PHASES_FOR_MODEL[0]: surrogate_ab, PHASES_FOR_MODEL[1]: liquid_source, PHASES_FOR_MODEL[2]: surrogate_bc,
    })


def make_mesh():
    """Builds the frozen three-phase initial profile on a compact mesh."""
    mesh = CartesianFD1D(INDEPENDENT_ELEMENTS, [0.0, LENGTH], NODES)
    mesh.setResponseProfile(ProfileBuilder([(ThreePhaseStepProfile(), INDEPENDENT_ELEMENTS)]))
    return mesh


def build_model(surrogate_ab, surrogate_bc, bulk_thermodynamics):
    """Constructs the frozen model without advancing time."""
    return MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=make_mesh(), elements=ELEMENTS, phases=PHASES_FOR_MODEL, thermodynamics=bulk_thermodynamics,
        temperature=TEMPERATURE, interfacePositions=INTERFACE_POSITIONS, interface_equilibria=(surrogate_ab, surrogate_bc),
        initial_eta_guess=INITIAL_ETA_GUESS, bulk_diffusivity_mode=BULK_DIFFUSIVITY_MODE,
        time_step=TIME_STEP, phase_nodes=PHASE_NODES, phase_molar_volumes=PHASE_MOLAR_VOLUMES,
        tolerance=TOLERANCE, residual_tolerance=TOLERANCE, max_iterations=MAX_ITERATIONS,
        max_step_retries=MAX_STEP_RETRIES, record=True,
    )


def run_case(overrides=None, *, make_plots=False):
    """Builds the frozen pycalphad fixture; solving and plotting are intentionally disabled."""
    if make_plots:
        raise ValueError("The stable three-phase fixture does not create plots.")
    with _temporary_config(overrides):
        therm_ab, therm_bc = build_thermodynamics()
        surrogate_ab, surrogate_bc = build_interface_surrogates(therm_ab, therm_bc)
        bulk_thermodynamics = build_bulk_diffusivity_provider(surrogate_ab, surrogate_bc)
        model = build_model(surrogate_ab, surrogate_bc, bulk_thermodynamics)
        return {
            "model": model, "surrogate_ab": surrogate_ab, "surrogate_bc": surrogate_bc,
            "bulk_thermodynamics": bulk_thermodynamics, "therm_ab": therm_ab, "therm_bc": therm_bc,
            "figures": {}, "overrides": {} if overrides is None else dict(overrides),
        }


def plot_surrogate_diagnostics_for_run(result, *, compare_ground_truth=False, hover_fields="diagnostic", **diagnostic_kwargs):
    """Builds diagnostics for a stable-fixture ``run_case`` result without displaying them."""
    required = ("surrogate_ab", "surrogate_bc")
    missing = [key for key in required if key not in result]
    if missing:
        raise KeyError(f"run_case result is missing required keys {missing}.")
    output = {}
    for label, surrogate_key, thermodynamics_key in (("ab", "surrogate_ab", "therm_ab"), ("bc", "surrogate_bc", "therm_bc")):
        thermodynamics = result.get(thermodynamics_key) if compare_ground_truth else None
        if compare_ground_truth and thermodynamics is None:
            raise KeyError(f"Ground-truth comparison requires result['{thermodynamics_key}'].")
        output[label] = plot_surrogate_diagnostics(
            result[surrogate_key], thermodynamics=thermodynamics, hover_fields=hover_fields, **diagnostic_kwargs
        )
    merged = {}
    for phase, source in getattr(result.get("bulk_thermodynamics"), "phase_sources", {}).items():
        if isinstance(source, MergedPhaseDiffusivitySurrogate):
            merged[phase] = plot_surrogate_diagnostics(
                source, thermodynamics=result.get("therm_ab") if compare_ground_truth else None,
                hover_fields=hover_fields, **diagnostic_kwargs,
            )
    if merged:
        output["merged_bulk"] = merged
    return output
