"""Real, local-only TC generation and the narrowly normalized O06 comparison.

The installed sampling call maps to examples/ThermoCalc/training_data.py:187-223
at 06dc7b4; arguments, conversions, call order and metadata are unchanged. The
reference executes that original builder and adapter. Research model assembly
follows the already approved O04/O05 shortened case, after the generation guard.
No TC input, database, generated surrogate or reference is distributed here.
"""
from contextlib import nullcontext
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import numpy as np


def load_configuration(artifacts, configuration):
    """Require the approved local input bytes; a corrupted present input must fail."""
    path = Path(artifacts) / configuration["input"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == configuration["input_sha256"], "O06 input changed"
    return json.loads(raw)


def normalize_metadata(array):
    """Replace only thermocalc_config/cache_dir's JSON value; retain all other text.

    Re-encoding the entire document could hide formatting, ordering or number
    changes. Locate just that configuration object and replace just its cache
    value. The scalar Unicode dtype is inferred again because this one permitted
    machine-path difference changes the original string length.
    """
    value = np.asarray(array)
    assert value.shape == () and value.dtype.kind == "U", "O06 metadata must be a JSON string scalar"
    text = value.item()
    marker = '"thermocalc_config": '
    assert text.count(marker) == 1
    start = text.index(marker) + len(marker)
    config, end = json.JSONDecoder().raw_decode(text[start:])
    block = text[start:start+end]
    needle = '"cache_dir": ' + json.dumps(config["cache_dir"])
    assert block.count(needle) == 1
    return np.asarray(text[:start] + block.replace(needle, '"cache_dir": null', 1) + text[start+end:])


def unavailable_reason(exc):
    """Classify missing TC resources without exposing vendor messages or masking bugs."""
    current = exc
    while current is not None:
        message = str(current).lower()
        if isinstance(current, ModuleNotFoundError) and (current.name or "").split(".")[0] == "tc_python":
            return "TC-Python is not installed"
        if "license" in message or "licence" in message:
            return "TC-Python licence unavailable"
        if any(term in message for term in ("database not found", "unknown database", "database does not exist")):
            return "configured TC database unavailable"
        if any(term in message for term in ("tc-python is not installed", "tc_python is not installed",
                                            "thermo-calc installation", "thermo-calc home", "tc installation")):
            return "TC-Python runtime unavailable"
        current = current.__cause__
    return None


def _source(inputs):
    """Resolve installed package code or the original frozen research module."""
    if inputs.role == "package":
        from moving_boundary.providers import thermocalc as source
        import moving_boundary
        assert Path(source.__file__).resolve().is_relative_to(Path(moving_boundary.__file__).resolve().parent)
    else:
        source = importlib.import_module("examples.ThermoCalc.tc_python_adapter")
        assert Path(source.__file__).resolve() == (inputs.root / "examples/ThermoCalc/tc_python_adapter.py").resolve()
    return source


def _fresh_directory(directory):
    """Allocate a separate ignored workspace and an empty cache for every worker."""
    work = Path(tempfile.mkdtemp(prefix="tc-O06-", dir=directory))
    cache = work / "cache"
    cache.mkdir()
    assert not any(cache.iterdir())
    return work, cache


def probe_availability(inputs, configuration, *, generation_guard, directory):
    """Start/close a real database session; only absent runtime/licence is skippable."""
    path = Path(inputs.artifacts) / configuration["input"]
    if not path.exists():
        return dict(available=False, reason="local O06 input missing")
    parameters = load_configuration(inputs.artifacts, configuration)
    work, cache = _fresh_directory(directory)
    with generation_guard() if inputs.role == "package" else nullcontext({}):
        source = _source(inputs)
        try:
            with source.TCPythonThermodynamics(source.ThermoCalcConfig(**parameters["thermocalc"], cache_dir=cache)):
                pass
        except Exception as exc:
            reason = unavailable_reason(exc)
            if reason is None:
                raise
            return dict(available=False, reason=reason)
    return dict(available=True)


def _installed_surrogate(therm, parameters, source):
    """Reproduce frozen training_data.py:187-223 using only installed generation APIs."""
    from moving_boundary.surrogates import TernaryMovingBoundaryThermodynamicsSurrogate
    config = therm.config
    eta_samples = np.asarray(parameters["eta_samples"], dtype=np.float64)
    probe_start = source.validate_independent_composition(parameters["probe_start"], config)
    probe_end = source.validate_independent_composition(parameters["probe_end"], config)
    with therm:
        surrogate = TernaryMovingBoundaryThermodynamicsSurrogate.from_database(
            thermodynamics=therm, elements=therm.elements, phases=therm.phases,
            tieline_phases=(config.matrix_phase, config.precipitate_phase),
            temperature=float(parameters["temperature"]), probe_start=probe_start, probe_end=probe_end,
            eta_samples=eta_samples, precipitate_phase=config.precipitate_phase,
            diffusivity_bulk_points=parameters["diffusivity_bulk_points"], diffusivity_bulk_grids=None,
            diffusivity_interpolation="nearest", skip_failed_bulk_calculations=False,
            drop_invalid_bulk_matrices=False)
    surrogate.metadata.update(dict(source="tc_python_adapter", tc_python_version=therm.getRuntimeVersion(),
                                  thermocalc_config=config.to_metadata(), dataset_schema_version=1))
    return surrogate


def build_case(inputs, configuration, *, generation_guard, directory):
    """Generate with real TC, then assemble the unchanged six-step reference/package solve."""
    parameters = load_configuration(inputs.artifacts, configuration)
    work, cache = _fresh_directory(directory)
    package = inputs.role == "package"
    with generation_guard() if package else nullcontext({}) as audit:
        source = _source(inputs)
        config = source.ThermoCalcConfig(**parameters["thermocalc"], cache_dir=cache)
        therm = source.TCPythonThermodynamics(config)
        if package:
            surrogate = _installed_surrogate(therm, parameters["sampling"], source)
        else:
            builder = importlib.import_module("examples.ThermoCalc.training_data")
            surrogate = builder.build_moving_boundary_surrogate(therm, **parameters["sampling"])
        saved = work / "surrogate.npz"
        surrogate.save(saved)
        audit.update(adapter_module=source.__name__, adapter_origin=source.__file__)
    inputs.generation_audit = audit
    inputs.thermocalc_surrogate = saved
    inputs.thermocalc_cache = cache
    root = inputs.root
    m = inputs.load(root / "examples/ternaryExamples/FeCrNi/IllingworthTernaryTwoPhase_FeCrNi_AnalyticComparison.py",
        ["build_comparison_model", "comparison_geometry", "interface_matrices_at_eta"],
        overrides=dict(surrogate=surrogate, PHASE_A=config.phases[0], PHASE_B=config.phases[1], PHYSICAL_NODES=401))
    matrices = m.interface_matrices_at_eta(surrogate, .5)
    s0, length = m.comparison_geometry(matrices)
    probe = m.build_comparison_model(surrogate, s0, length, 21, 41, .05)
    probe.setup()
    matrices = m.interface_matrices_at_eta(surrogate, probe.getInterfaceEta())
    s0, length = m.comparison_geometry(matrices)
    therm_cls = inputs.load(root / "examples/ternaryExamples/ternaryTwoPhaseSimilarity.py",
                            ["FixedMatrixTernaryDiffusivity"]).FixedMatrixTernaryDiffusivity
    model = m.build_comparison_model(therm_cls(matrices, list(config.phases)), s0, length, 21, 41, .05)
    model.dtMode, model.timeStep = "fixed", parameters["solve"]["time_step"]
    return model, parameters["solve"]["duration"], None
