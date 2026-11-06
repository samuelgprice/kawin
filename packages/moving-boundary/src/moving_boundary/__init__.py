"""Illingworth moving boundaries using unmodified public kawin services."""
import importlib

__version__ = "0.0.1.dev0"
PUBLIC_KAWIN_VERSION = "0.5.0"
REFERENCE_COMMIT = "06dc7b4ab56f67226ceabfde41ebf2690c088841"
_EXPORTS = {'MovingBoundaryEquilibriumClosure': 'equilibrium', 'FixedTernaryInterfaceEquilibrium': 'equilibrium', 'CallableTernaryInterfaceEquilibrium': 'equilibrium', 'ThermodynamicTernaryInterfaceEquilibrium': 'equilibrium', 'InitialEtaEstimate': 'shared', 'estimate_initial_eta_from_instantaneous_balance': 'shared', 'InterfaceDiffusivityConstructionError': 'surrogates', 'MergedPhaseDiffusivitySurrogate': 'surrogates', 'merge_phase_diffusivity_surrogates': 'surrogates', 'TernaryMovingBoundaryThermodynamicsSurrogate': 'surrogates', 'SurrogateArtifactError': 'surrogate_artifacts', 'SurrogateArtifactCompatibilityError': 'surrogate_artifacts', 'SurrogateArtifactIntegrityError': 'surrogate_artifacts', 'file_sha256': 'surrogate_artifacts', 'callable_source_sha256': 'surrogate_artifacts', 'canonicalize_build_spec': 'surrogate_artifacts', 'surrogate_build_fingerprint': 'surrogate_artifacts', 'SurrogateArtifactBundle': 'surrogate_artifacts', 'prepare_surrogate_artifact': 'surrogate_artifacts', 'MovingBoundaryIllingworthFD1DModel': 'solvers', 'MovingBoundaryIllingworthTernaryFD1DModel': 'solvers', 'MovingBoundaryIllingworthTernaryThreePhaseFD1DModel': 'solvers'}


def __getattr__(name):
    """Resolve local solver/provider APIs without loading kawin on namespace import."""
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(importlib.import_module('.' + _EXPORTS[name], __name__), name)
