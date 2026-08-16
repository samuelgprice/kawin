# %%
"""
Ni-Ti-Nb three-phase Illingworth ternary moving-boundary example.

This cell script demonstrates the sequential ``BCC_A2 | LIQUID | BCC_A2``
three-phase model at 1300 K using Thermo-Calc TCHEA5 and MOBHEA4 through the
example-local TC-Python adapter. The two BCC regions use the same Thermo-Calc
phase model but are tracked as separate left and right spatial intervals.

Run cells top-to-bottom in VS Code/Jupyter, or run the file directly. A local
Thermo-Calc installation, TC-Python, and licensed access to TCHEA5/MOBHEA4 are
required.
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
from kawin.diffusion import (
    MovingBoundaryIllingworthTernaryThreePhaseFD1DModel,
    TernaryMovingBoundaryThermodynamicsSurrogate,
)
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder
from kawin.solver import explicitEulerIterator


OUTPUTS = REPO_ROOT / "examples" / "ThermoCalc" / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

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

ELEMENTS = ("NB", "NI", "TI")
INDEPENDENT_ELEMENTS = ("NI", "TI")
PHASE_BCC = "BCC_B2"
PHASE_LIQUID = "LIQUID"
PHASES_FOR_MODEL = (PHASE_BCC, PHASE_LIQUID, PHASE_BCC)
TEMPERATURE = 1300.0

# Full mole fractions are listed in ELEMENTS order: [NB, NI, TI].
LEFT_BCC_FULL = np.array([0.899, 0.001, 0.100], dtype=np.float64)
LIQUID_FULL = np.array([0.100, 0.300, 0.600], dtype=np.float64)
RIGHT_BCC_FULL = np.array([0.002, 0.499, 0.499], dtype=np.float64)

LEFT_WIDTH = 40.0e-6
LIQUID_WIDTH = 2.0e-6
RIGHT_WIDTH = 40.0e-6
LENGTH = LEFT_WIDTH + LIQUID_WIDTH + RIGHT_WIDTH
INTERFACE_POSITIONS = np.array([LEFT_WIDTH, LEFT_WIDTH + LIQUID_WIDTH], dtype=np.float64)

# Tie-line probes use independent mole fractions in [NI, TI] order. The
# defaults connect the requested nominal phase compositions at each interface.
ETA_SAMPLES = np.linspace(0.0, 1.0, 9)
AB_PROBE_START = np.array([0.16, 0.83], dtype=np.float64)
AB_PROBE_END = np.array([0.217, 0.394], dtype=np.float64)
BC_PROBE_START = np.array([0.39, 0.468], dtype=np.float64)
BC_PROBE_END = np.array([0.4099, 0.59], dtype=np.float64)
INITIAL_ETA_GUESS = (0.5, 0.5)

# ``nearest`` lets the bulk-diffusivity sample points follow the ternary
# simplex without requiring a rectangular grid that would include invalid
# Ni+Ti > 1 compositions.
DIFFUSIVITY_INTERPOLATION = "nearest" # "continuous_grid"
BULK_DIFFUSIVITY_MODE = "composition_dependent_lagged"
BULK_DIFFUSIVITY_POINTS = None

NODES = 165
PHASE_NODES = (81, 9, 81)
FIXED_TIME_STEP = 1.0e-3
SOLVE_TIME = 1.0
TOLERANCE = 1.0e-10
MAX_ITERATIONS = 25
MAX_STEP_RETRIES = 8
MIN_DT_FRAC = 1.0e-16
VERBOSE = True
VERBOSE_INTERVAL = 10
RUN_PREFLIGHT = True
RUN_SOLVE = True


# %%
# Helpers

_OVERRIDE_KEY_ALIASES = {
    "bulk_diffusivity_mode": "BULK_DIFFUSIVITY_MODE",
    "bulk_diffusivity_points": "BULK_DIFFUSIVITY_POINTS",
    "fixed_time_step": "FIXED_TIME_STEP",
    "interface_positions": "INTERFACE_POSITIONS",
    "max_iterations": "MAX_ITERATIONS",
    "max_step_retries": "MAX_STEP_RETRIES",
    "min_dt_frac": "MIN_DT_FRAC",
    "nodes": "NODES",
    "phase_nodes": "PHASE_NODES",
    "run_preflight": "RUN_PREFLIGHT",
    "run_solve": "RUN_SOLVE",
    "solve_time": "SOLVE_TIME",
    "temperature": "TEMPERATURE",
    "tolerance": "TOLERANCE",
    "verbose": "VERBOSE",
    "verbose_interval": "VERBOSE_INTERVAL",
}


def _normalize_override_key(key):
    key = str(key)
    if key in globals():
        return key
    upper_key = key.upper()
    if upper_key in globals():
        return upper_key
    if key in _OVERRIDE_KEY_ALIASES:
        return _OVERRIDE_KEY_ALIASES[key]
    raise KeyError(f"Unknown Ni-Ti-Nb three-phase example override '{key}'.")


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
        yield
    finally:
        globals().update(old_values)


class ThreePhaseStepProfile:
    """Piecewise-constant initial profile for ``BCC_A2 | LIQUID | BCC_A2``."""

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


def _make_tc_config(phases):
    """Returns a TC-Python config for one ordered two-phase interface."""
    return ThermoCalcConfig(
        thermodynamic_database="TCHEA5",
        kinetic_database="MOBHEA4",
        elements=ELEMENTS,
        phases=tuple(phases),
        reference_element="NB",
        cache_dir=OUTPUTS / "tc_cache",
    )


def _make_default_bulk_points():
    """
    Returns simplex-valid [NI, TI] points for nearest-neighbor bulk diffusivity.

    The list includes the three nominal phase compositions and small local
    perturbations clipped to the ternary simplex so the surrogate can answer
    bulk queries without sampling an invalid rectangular composition domain.
    """
    centers = np.asarray([LEFT_BCC_FULL[1:], LIQUID_FULL[1:], RIGHT_BCC_FULL[1:]], dtype=np.float64)
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


def build_thermodynamics():
    """
    Builds TC-Python thermodynamics facades for the two adjacent interfaces.

    Separate facades are used because the TC-Python adapter treats the first
    configured phase as the matrix-side endpoint returned by
    ``getInterfacialComposition``.
    """
    therm_ab = TCPythonThermodynamics(_make_tc_config((PHASE_BCC, PHASE_LIQUID)))
    therm_bc = TCPythonThermodynamics(_make_tc_config((PHASE_LIQUID, PHASE_BCC)))
    return therm_ab, therm_bc


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
        equilibrium = thermodynamics.getEquilibriumData(point, TEMPERATURE, removeCache=False)
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
        if missing or extra:
            bad_samples.append((float(eta), point, stable_phases, phase_amounts, tuple(missing), tuple(extra)))

    if bad_samples:
        details = []
        for eta, point, stable_phases, phase_amounts, missing, extra in bad_samples[:5]:
            details.append(
                "eta={:.3g}, x[NI,TI]={}, stable={}, amounts={}, missing={}, extra={}".format(
                    eta,
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


def build_interface_surrogates(therm_ab, therm_bc):
    """Samples BCC/liquid and liquid/BCC tie-line families from Thermo-Calc."""
    bulk_points = _make_default_bulk_points() if BULK_DIFFUSIVITY_POINTS is None else np.asarray(BULK_DIFFUSIVITY_POINTS, dtype=np.float64)
    with therm_ab:
        _validate_tieline_probe_path(
            therm_ab,
            "BCC/liquid",
            AB_PROBE_START,
            AB_PROBE_END,
            (PHASE_BCC, PHASE_LIQUID),
        )
    with therm_bc:
        _validate_tieline_probe_path(
            therm_bc,
            "Liquid/BCC",
            BC_PROBE_START,
            BC_PROBE_END,
            (PHASE_LIQUID, PHASE_BCC),
        )
    with therm_ab:
        surrogate_ab = TernaryMovingBoundaryThermodynamicsSurrogate.from_database(
            thermodynamics=therm_ab,
            elements=ELEMENTS,
            phases=(PHASE_BCC, PHASE_LIQUID),
            tieline_phases=(PHASE_BCC, PHASE_LIQUID),
            temperature=TEMPERATURE,
            probe_start=AB_PROBE_START,
            probe_end=AB_PROBE_END,
            eta_samples=ETA_SAMPLES,
            precipitate_phase=PHASE_LIQUID,
            diffusivity_bulk_points=bulk_points,
            diffusivity_interpolation=DIFFUSIVITY_INTERPOLATION,
        )
    with therm_bc:
        surrogate_bc = TernaryMovingBoundaryThermodynamicsSurrogate.from_database(
            thermodynamics=therm_bc,
            elements=ELEMENTS,
            phases=(PHASE_LIQUID, PHASE_BCC),
            tieline_phases=(PHASE_LIQUID, PHASE_BCC),
            temperature=TEMPERATURE,
            probe_start=BC_PROBE_START,
            probe_end=BC_PROBE_END,
            eta_samples=ETA_SAMPLES,
            precipitate_phase=PHASE_BCC,
            diffusivity_bulk_points=bulk_points,
            diffusivity_interpolation=DIFFUSIVITY_INTERPOLATION,
        )
    return surrogate_ab, surrogate_bc


def make_mesh():
    """Builds the initial three-phase [NI, TI] profile on a Cartesian FD mesh."""
    mesh = CartesianFD1D(INDEPENDENT_ELEMENTS, [0.0, LENGTH], NODES)
    profile = ProfileBuilder(
        [
            (
                ThreePhaseStepProfile(
                    INTERFACE_POSITIONS,
                    (LEFT_BCC_FULL[1:], LIQUID_FULL[1:], RIGHT_BCC_FULL[1:]),
                ),
                INDEPENDENT_ELEMENTS,
            )
        ]
    )
    mesh.setResponseProfile(profile)
    return mesh


def build_model(surrogate_ab, surrogate_bc):
    """Constructs the three-phase Illingworth model without starting the solve."""
    return MovingBoundaryIllingworthTernaryThreePhaseFD1DModel(
        mesh=make_mesh(),
        elements=ELEMENTS,
        phases=PHASES_FOR_MODEL,
        thermodynamics=surrogate_ab,
        temperature=TEMPERATURE,
        interfacePositions=INTERFACE_POSITIONS,
        interface_equilibria=(surrogate_ab, surrogate_bc),
        initial_eta_guess=INITIAL_ETA_GUESS,
        bulk_diffusivity_mode=BULK_DIFFUSIVITY_MODE,
        time_step=FIXED_TIME_STEP,
        dt_mode="fixed",
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
        print("Inventory drift [NI, TI]:", model.checkConservation(TOLERANCE))


def plot_phase_widths(model):
    """Plots BCC/liquid/BCC widths over time."""
    times = model.interfaceData._time[: model.interfaceData.N + 1]
    positions = model.interfaceData._y[: model.interfaceData.N + 1]
    widths = np.column_stack((positions[:, 0], positions[:, 1] - positions[:, 0], LENGTH - positions[:, 1]))
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, label in enumerate(("left BCC_A2", "LIQUID", "right BCC_A2")):
        ax.plot(times, widths[:, i] * 1.0e6, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Width (um)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_independent_profiles(model, time=None):
    """Plots independent [NI, TI] profiles on the physical mesh."""
    z_um = model._z * 1.0e6
    y = model.data.y(time)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, element in enumerate(INDEPENDENT_ELEMENTS):
        ax.plot(z_um, y[:, i], label=f"X({element})")
    for position in model.getInterfacePositions(time):
        ax.axvline(position * 1.0e6, color="0.35", linestyle="--", linewidth=1)
    ax.set_xlabel("Distance (um)")
    ax.set_ylabel("Mole fraction")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def run_case(overrides=None, *, make_plots=True):
    """
    Builds the TC-Python surrogates, constructs the model, and optionally solves.

    ``overrides`` can temporarily replace module-level values, for example
    ``run_case({"solve_time": 0.1, "fixed_time_step": 1e-4})``.
    """
    with _temporary_config(overrides):
        therm_ab, therm_bc = build_thermodynamics()
        if RUN_PREFLIGHT:
            print("BCC/liquid preflight:", therm_ab.preflight(x=AB_PROBE_START, T=TEMPERATURE))
            print("Liquid/BCC preflight:", therm_bc.preflight(x=BC_PROBE_START, T=TEMPERATURE))

        start = time.perf_counter()
        surrogate_ab, surrogate_bc = build_interface_surrogates(therm_ab, therm_bc)
        print(f"Built interface surrogates in {time.perf_counter() - start:.1f} s.")

        model = build_model(surrogate_ab, surrogate_bc)
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
            plt.show()
        return {
            "model": model,
            "surrogate_ab": surrogate_ab,
            "surrogate_bc": surrogate_bc,
            "therm_ab": therm_ab,
            "therm_bc": therm_bc,
            "figures": figures,
        }


# %%
if __name__ == "__main__":
    # debugInPlace()
    result = run_case(make_plots=True)

# %%
