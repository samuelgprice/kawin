"""Record a model's effective settings immediately before solving, for example acceptance.

Matching output is evidence of matching behaviour, but a setting can differ without
affecting a short run. So the example and the accepted case's settings are compared too.
The record is taken just before ``solve``, so it includes post-construction changes.
Scoped to the example configurations; it is not a general model serializer.
"""
import hashlib

import numpy as np

# Instance attributes holding objects handled separately, or run bookkeeping that isn't a setting.
SKIPPED_TYPES = (type(None),)

# Solver attributes that are run bookkeeping or diagnostics, not settings, so ``differences``
# ignores them on both sides. Most adopted settings files were captured before they existed (R07,
# R08). Keep this list narrow and named: genuine settings, private or not (``tolerance``,
# ``residualTolerance``, ``interfaceResidualCompositionScale``, ``timeStep``, ``dtMode``,
# ``semiLog_dt``, ...), stay compared.
BOOKKEEPING = {
    # R07 commit guard: the staged candidate and the "rejected, awaiting a new one" latch.
    "_pendingCandidate": "R07 commit-guard state, empty before solve",
    "_awaitingFreshCandidate": "R07 commit-guard latch, False before solve",
    # R07 outer timestep limits: copies of solve(minDtFrac, maxDtFrac), which the record keeps
    # under "solve", and the limits solve() derives from them; before solve() only placeholders.
    "_solveMinDtFrac": "copy of the solve() argument recorded under 'solve'",
    "_solveMaxDtFrac": "copy of the solve() argument recorded under 'solve'",
    "_outerMinTimeStep": "derived in solve() from minDtFrac and the solve duration",
    "_outerMaxTimeStep": "derived in solve() from maxDtFrac and the solve duration",
    # Last accepted step's diagnostics (NaN before solve).
    "_lastImplicitResidual": "diagnostic of the last accepted step",
    "_lastImplicitResidualScale": "R08 diagnostic of the last accepted step",
}


def digest(value):
    """Hash an array's shape, dtype and bytes, so settings can be compared exactly."""
    array = np.asarray(value)
    return f"{array.dtype.str}{array.shape}:{hashlib.sha256(array.tobytes()).hexdigest()}"


def plain(value):
    """Return a comparable form of a simple value, or ``None`` for an object handled elsewhere."""
    if value is None or isinstance(value, (bool, str)):
        return {"value": value}
    if isinstance(value, (int, float, np.integer, np.floating)):
        return {"value": repr(float(value))}
    if isinstance(value, np.ndarray):
        return {"array": digest(value)}
    if isinstance(value, (list, tuple)):
        items = [plain(item) for item in value]
        return {"items": items} if all(item is not None for item in items) else None
    return None


def provider_samples(model, compositions):
    """Sample the provider at each phase's interface compositions, ignoring its class name."""
    phases = list(model.phases)
    out = {}
    temperature = float(np.ravel(model.temperatureParameters(np.asarray([[0.0]]), 0.0))[0])
    for phase, composition in zip(phases, compositions):
        out[str(phase)] = digest(np.asarray(model.therm.getInterdiffusivity(composition, temperature, phase=phase), dtype=np.float64))
    out["temperature"] = repr(temperature)
    return out


def closure_samples(closure):
    """Sample one tie-line closure across its eta range."""
    low, high = closure.eta_bounds
    return [digest(np.concatenate([np.ravel(part) for part in closure.interface_compositions(float(eta))]))
            for eta in np.linspace(low, high, 7)]


def effective_settings(model, duration, solve_arguments):
    """Describe a built, unsolved model for an exact comparison between hosts."""
    record = {"duration": repr(float(duration)), "solve": dict(solve_arguments), "class": type(model).__mro__[0].__name__.replace("Tracked", "")}
    attributes = {}
    for name, value in sorted(vars(model).items()):
        described = plain(value)
        if described is not None:
            attributes[name] = described
    record["attributes"] = attributes
    record["mesh_z"] = digest(np.ravel(model.mesh.z))
    record["initial_response"] = digest(np.asarray(model.data.currentY))
    closures = []
    for name in ("interfaceEquilibrium", "interface_equilibrium", "interfaceEquilibria", "interface_equilibria"):
        value = getattr(model, name, None)
        if value is not None:
            closures = list(value) if isinstance(value, (list, tuple)) else [value]
            break
    record["closures"] = [closure_samples(closure) for closure in closures]
    if closures:
        left = closures[0].interface_compositions(float(np.mean(closures[0].eta_bounds)))
        record["provider"] = provider_samples(model, [left[0], left[1], left[1]][:len(model.phases)])
    else:
        record["provider"] = provider_samples(model, [np.asarray(0.3)] * len(model.phases))
    return record


def differences(reference, package, allowed=()):
    """List settings that differ, excluding named attributes approved with a reason.

    ``BOOKKEEPING`` attributes are never compared, on either side, so a field added to the solver
    after a reference was captured cannot fail the comparison. Everything else, including the
    solve arguments and every other attribute, must match exactly.
    """
    found = []
    for key in sorted(set(reference) | set(package)):
        if key == "attributes":
            continue
        if reference.get(key) != package.get(key):
            found.append(key)
    left, right = reference["attributes"], package["attributes"]
    for key in sorted(set(left) | set(right)):
        if key in BOOKKEEPING:
            continue
        if key not in allowed and left.get(key) != right.get(key):
            found.append("attributes." + key)
    return found
