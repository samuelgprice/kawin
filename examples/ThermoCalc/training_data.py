"""Dataset sampling and export helpers for the TC-Python kawin example."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .tc_python_adapter import (
    TCPythonThermodynamics,
    ThermoCalcCalculationError,
    ThermoCalcConfig,
    ThermoCalcError,
    validate_independent_composition,
)


DATASET_SCHEMA_VERSION = 1


def make_fecrni_demo_grid(
    cr_values: Any | None = None,
    ni_values: Any | None = None,
    *,
    config: ThermoCalcConfig | None = None,
) -> np.ndarray:
    """Create a small valid Fe-Cr-Ni composition grid.

    The default grid has 25 points and is intended for notebook smoke tests.
    Values are independent mole fractions in ``[x_CR, x_NI]`` order.
    """

    config = ThermoCalcConfig() if config is None else config
    if tuple(config.independent_elements) != ("CR", "NI"):
        raise ValueError("make_fecrni_demo_grid is specific to Fe-Cr-Ni independent order [CR, NI].")
    cr_values = np.linspace(0.05, 0.45, 5) if cr_values is None else np.asarray(cr_values, dtype=np.float64)
    ni_values = np.linspace(0.001, 0.20, 5) if ni_values is None else np.asarray(ni_values, dtype=np.float64)
    grid = np.array(np.meshgrid(cr_values, ni_values)).T.reshape(-1, 2)
    return np.array([point for point in grid if np.sum(point) <= 1.0], dtype=np.float64)


def initialize_dataset(
    compositions: Any,
    temperatures: Any,
    *,
    config: ThermoCalcConfig | None = None,
    phases: list[str] | tuple[str, ...] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Allocate dataset arrays filled with NaN and false validity masks."""

    config = ThermoCalcConfig() if config is None else config
    x, T = _broadcast_points(compositions, temperatures, config)
    phases = tuple(config.phases if phases is None else phases)
    n_points = len(x)
    n_elements = len(config.elements)
    n_independent = len(config.independent_elements)
    n_phases = len(phases)

    arrays: dict[str, np.ndarray] = {
        "compositions": x,
        "temperatures": T,
        "equilibrium_valid": np.zeros(n_points, dtype=bool),
        "equilibrium_phase_amounts": np.full((n_points, n_phases), np.nan, dtype=np.float64),
        "equilibrium_phase_compositions": np.full((n_points, n_phases, n_elements), np.nan, dtype=np.float64),
        "equilibrium_chemical_potentials": np.full((n_points, n_elements), np.nan, dtype=np.float64),
        "driving_force_valid": np.zeros(n_points, dtype=bool),
        "driving_force": np.full(n_points, np.nan, dtype=np.float64),
        "precipitate_compositions": np.full((n_points, n_independent), np.nan, dtype=np.float64),
    }
    for phase in phases:
        phase_key = _phase_key(phase)
        arrays[f"{phase_key}_kinetics_valid"] = np.zeros(n_points, dtype=bool)
        arrays[f"{phase_key}_interdiffusivity"] = np.full(
            (n_points, n_independent, n_independent),
            np.nan,
            dtype=np.float64,
        )
        arrays[f"{phase_key}_tracer_diffusivity"] = np.full((n_points, n_elements), np.nan, dtype=np.float64)

    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "complete": False,
        "config": config.to_metadata(),
        "tc_python_version": None,
        "elements": list(config.elements),
        "independent_elements": list(config.independent_elements),
        "phases": list(phases),
        "units": {
            "temperature": "K",
            "energy": "J/mol",
            "diffusivity": "m^2/s",
            "composition": "mole_fraction",
        },
        "failures": [],
        "metadata": {} if metadata is None else dict(metadata),
    }
    return {"arrays": arrays, "manifest": manifest}


def sample_training_data(
    thermodynamics: TCPythonThermodynamics,
    compositions: Any,
    temperatures: Any = 1373.0,
    *,
    phases: list[str] | tuple[str, ...] | None = None,
    output_prefix: str | Path | None = None,
    checkpoint_every: int = 10,
    resume: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sample equilibrium, driving-force, and phase-specific kinetic data.

    Each data group has its own validity mask.  A failed point records an entry
    in the manifest and does not stop the whole run.  Recoverable calculation
    failures get one TC-Python session restart before they are recorded.
    """

    config = thermodynamics.config
    phases = tuple(config.phases if phases is None else phases)
    output_prefix = None if output_prefix is None else Path(output_prefix)
    if output_prefix is not None and resume and _npz_path(output_prefix).exists():
        dataset = load_training_dataset(output_prefix)
    else:
        dataset = initialize_dataset(compositions, temperatures, config=config, phases=phases, metadata=metadata)

    arrays = dataset["arrays"]
    manifest = dataset["manifest"]
    manifest["tc_python_version"] = thermodynamics.getRuntimeVersion()
    processed_since_checkpoint = 0

    with thermodynamics:
        for index, (x, T) in enumerate(zip(arrays["compositions"], arrays["temperatures"])):
            if not arrays["equilibrium_valid"][index]:
                _sample_equilibrium(thermodynamics, dataset, index, x, T, phases)
            if not arrays["driving_force_valid"][index]:
                _sample_driving_force(thermodynamics, dataset, index, x, T)
            for phase in phases:
                valid_key = f"{_phase_key(phase)}_kinetics_valid"
                if not arrays[valid_key][index]:
                    _sample_kinetics(thermodynamics, dataset, index, x, T, phase)

            processed_since_checkpoint += 1
            if output_prefix is not None and checkpoint_every > 0 and processed_since_checkpoint >= checkpoint_every:
                save_training_dataset(dataset, output_prefix, complete=False)
                processed_since_checkpoint = 0

    manifest["complete"] = _is_complete(dataset, phases)
    manifest["updated_utc"] = _utc_now()
    if output_prefix is not None:
        save_training_dataset(dataset, output_prefix, complete=manifest["complete"])
    return dataset


def build_moving_boundary_surrogate(
    thermodynamics: TCPythonThermodynamics,
    *,
    temperature: float = 1373.0,
    probe_start: Any = (0.25, 0.068),
    probe_end: Any = (0.45, 0.242),
    eta_samples: Any | None = None,
    diffusivity_bulk_points: Any | None = None,
    diffusivity_bulk_grids: Any | None = None,
    diffusivity_interpolation: str = "nearest",
    output_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Build a moving-boundary thermodynamics surrogate from TC-Python.

    This wraps ``TernaryMovingBoundaryThermodynamicsSurrogate.from_database``
    using the example adapter as the thermodynamics source.  Tie-line probes
    are independent mole fractions in ``[x_CR, x_NI]`` order for the default
    Fe-Cr-Ni configuration.
    """

    from kawin.diffusion.MovingBoundarySurrogates import TernaryMovingBoundaryThermodynamicsSurrogate

    config = thermodynamics.config
    eta_samples = np.linspace(0.0, 1.0, 5) if eta_samples is None else np.asarray(eta_samples, dtype=np.float64)
    probe_start = validate_independent_composition(probe_start, config)
    probe_end = validate_independent_composition(probe_end, config)
    if diffusivity_bulk_points is None and diffusivity_bulk_grids is None:
        diffusivity_bulk_points = make_fecrni_demo_grid(config=config)

    with thermodynamics:
        surrogate = TernaryMovingBoundaryThermodynamicsSurrogate.from_database(
            thermodynamics=thermodynamics,
            elements=thermodynamics.elements,
            phases=thermodynamics.phases,
            tieline_phases=(config.matrix_phase, config.precipitate_phase),
            temperature=float(temperature),
            probe_start=probe_start,
            probe_end=probe_end,
            eta_samples=eta_samples,
            precipitate_phase=config.precipitate_phase,
            diffusivity_bulk_points=diffusivity_bulk_points,
            diffusivity_bulk_grids=diffusivity_bulk_grids,
            diffusivity_interpolation=diffusivity_interpolation,
        )

    surrogate.metadata.update(
        {
            "source": "tc_python_adapter",
            "tc_python_version": thermodynamics.getRuntimeVersion(),
            "thermocalc_config": config.to_metadata(),
            "dataset_schema_version": DATASET_SCHEMA_VERSION,
            **({} if metadata is None else dict(metadata)),
        }
    )
    if output_path is not None:
        surrogate.save(output_path)
    return surrogate


def save_training_dataset(dataset: dict[str, Any], output_prefix: str | Path, *, complete: bool | None = None):
    """Write dataset arrays to NPZ and metadata/failures to JSON."""

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(dataset["manifest"])
    if complete is not None:
        manifest["complete"] = bool(complete)
    manifest["updated_utc"] = _utc_now()
    np.savez_compressed(_npz_path(output_prefix), **dataset["arrays"])
    _manifest_path(output_prefix).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_training_dataset(output_prefix: str | Path) -> dict[str, Any]:
    """Load a dataset saved by ``save_training_dataset``."""

    output_prefix = Path(output_prefix)
    with np.load(_npz_path(output_prefix), allow_pickle=False) as arrays_file:
        arrays = {key: arrays_file[key] for key in arrays_file.files}
    manifest = json.loads(_manifest_path(output_prefix).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported dataset schema {manifest.get('schema_version')}; expected {DATASET_SCHEMA_VERSION}."
        )
    return {"arrays": arrays, "manifest": manifest}


def _sample_equilibrium(thermodynamics, dataset, index, x, T, phases):
    arrays = dataset["arrays"]

    def calculate():
        return thermodynamics.getEquilibriumData(x, T, removeCache=True)

    try:
        result = _with_one_restart(thermodynamics, calculate)
        phase_index = {phase: i for i, phase in enumerate(phases)}
        for phase, amount in result["phase_amounts"].items():
            if phase in phase_index:
                arrays["equilibrium_phase_amounts"][index, phase_index[phase]] = amount
        for phase, composition in result["phase_compositions"].items():
            if phase in phase_index:
                arrays["equilibrium_phase_compositions"][index, phase_index[phase], :] = composition
        arrays["equilibrium_chemical_potentials"][index, :] = [
            result["chemical_potentials"][element] for element in thermodynamics.elements
        ]
        arrays["equilibrium_valid"][index] = True
    except Exception as exc:
        _record_failure(dataset, index, "equilibrium", None, exc)


def _sample_driving_force(thermodynamics, dataset, index, x, T):
    arrays = dataset["arrays"]

    def calculate():
        return thermodynamics.getDrivingForce(x, T, removeCache=True)

    try:
        driving_force, precipitate_composition = _with_one_restart(thermodynamics, calculate)
        arrays["driving_force"][index] = driving_force
        arrays["precipitate_compositions"][index, :] = precipitate_composition
        arrays["driving_force_valid"][index] = True
    except Exception as exc:
        _record_failure(dataset, index, "driving_force", thermodynamics.config.precipitate_phase, exc)


def _sample_kinetics(thermodynamics, dataset, index, x, T, phase):
    arrays = dataset["arrays"]
    phase_key = _phase_key(phase)

    def calculate():
        interdiffusivity = thermodynamics.getInterdiffusivity(x, T, phase=phase, removeCache=True)
        tracer = thermodynamics.getTracerDiffusivity(x, T, phase=phase, removeCache=True)
        return interdiffusivity, tracer

    try:
        interdiffusivity, tracer = _with_one_restart(thermodynamics, calculate)
        arrays[f"{phase_key}_interdiffusivity"][index, :, :] = interdiffusivity
        arrays[f"{phase_key}_tracer_diffusivity"][index, :] = tracer
        arrays[f"{phase_key}_kinetics_valid"][index] = True
    except Exception as exc:
        _record_failure(dataset, index, "kinetics", phase, exc)


def _with_one_restart(thermodynamics, callback):
    try:
        return callback()
    except ThermoCalcCalculationError:
        thermodynamics.restartSession()
        return callback()


def _record_failure(dataset, index, group, phase, exc):
    dataset["manifest"]["failures"].append(
        {
            "index": int(index),
            "group": group,
            "phase": phase,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "time_utc": _utc_now(),
        }
    )


def _is_complete(dataset, phases):
    arrays = dataset["arrays"]
    masks = [arrays["equilibrium_valid"], arrays["driving_force_valid"]]
    masks.extend(arrays[f"{_phase_key(phase)}_kinetics_valid"] for phase in phases)
    return bool(all(np.all(mask) for mask in masks))


def _broadcast_points(compositions, temperatures, config):
    x = np.atleast_2d(np.asarray(compositions, dtype=np.float64))
    T = np.atleast_1d(np.asarray(temperatures, dtype=np.float64))
    if len(x) != len(T):
        if len(x) == 1:
            x = np.repeat(x, len(T), axis=0)
        elif len(T) == 1:
            T = np.repeat(T, len(x), axis=0)
        else:
            raise ValueError("compositions and temperatures must have the same length unless one has length 1.")
    for xi in x:
        validate_independent_composition(xi, config)
    if np.any(~np.isfinite(T)) or np.any(T <= 0):
        raise ValueError("temperatures must be positive finite values in K.")
    return x, T


def _npz_path(output_prefix):
    output_prefix = Path(output_prefix)
    return output_prefix if output_prefix.suffix == ".npz" else output_prefix.with_suffix(".npz")


def _manifest_path(output_prefix):
    output_prefix = Path(output_prefix)
    if output_prefix.suffix == ".npz":
        return output_prefix.with_suffix(".json")
    return output_prefix.with_suffix(".json")


def _phase_key(phase):
    return str(phase).lower()


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
