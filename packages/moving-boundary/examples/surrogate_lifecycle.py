"""How do I save, verify, reload and solve with a surrogate?

Sampling thermodynamics is slow, so you normally build a surrogate once and reuse
it. This shows that workflow without needing a database: a tiny synthetic surrogate
is published as a verified bundle, reloaded, and used for a solve. For the same
workflow on real data, see ``ternary_fecrni_lee1996.py``.

**What happens on disk.** The bundle goes in ``--output-dir``, by default
``./moving_boundary_example_outputs/surrogate_lifecycle/`` under the current
directory, never in the package or source tree.

- No bundle yet: build and publish it.
- A compatible bundle: reload it. The builder is not called.
- An incompatible or damaged bundle: stop with an error. It is never rebuilt
  silently, so you always know which data a result came from.
- ``--rebuild``: build and atomically replace it.

A *build specification* records everything that defines the data. Reloading requires
an exact match, and every member file is checked against its SHA-256.

The surrogate is synthetic: a unit diffusivity matrix and linear tie lines. Elements
are ``[Z, X, Y]`` with Z dependent, ALPHA lies left of BETA, compositions are mole
fractions and units are nondimensional. So this checks the storage workflow, not
surrogate quality or physics: a reload that matches the original says the bundle
preserved the data, not that the data is right. With these arbitrary values the
interface barely moves over this short run, which doesn't matter for that purpose.

Run ``python surrogate_lifecycle.py`` (under a second).
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import shutil
import tempfile

import numpy as np

from moving_boundary.analytic import recorded_history
from moving_boundary.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from moving_boundary.solvers import MovingBoundaryIllingworthTernaryFD1DModel
from moving_boundary.surrogate_artifacts import (
    SurrogateArtifactBundle,
    SurrogateArtifactCompatibilityError,
    SurrogateArtifactIntegrityError,
    prepare_surrogate_artifact,
)
from moving_boundary.surrogates import TernaryMovingBoundaryThermodynamicsSurrogate

DEFAULT_OUTPUT = Path("moving_boundary_example_outputs") / "surrogate_lifecycle"
BUILD_SPEC = {"example": "surrogate_lifecycle", "surrogate": "synthetic unit-matrix", "revision": 1}


@dataclass(frozen=True)
class Config:
    """Numerical settings for the solve; the surrogate itself is fixed."""

    mesh_nodes: int = 21
    phase_nodes: int = 9
    time_step: float = 1.0e-4
    duration: float = 2.0e-3


@dataclass(frozen=True)
class ExampleResult:
    """A solved example: its settings, model and named metrics (no exact solution here)."""

    config: Config
    model: object
    exact: object
    metrics: dict


DEMO = Config()
FAST = DEMO  # already quick


def build_surrogate():
    """Make the surrogate directly from arrays; real workflows use ``from_database``."""
    eta = np.asarray([0.0, 1.0])
    tie_lines = {"ALPHA": np.asarray([[0.20, 0.10], [0.30, 0.10]]), "BETA": np.asarray([[0.10, 0.20], [0.15, 0.25]])}
    compositions = {context: {phase: values.copy() for phase, values in tie_lines.items()} for context in ("interface", "general")}
    matrices = {context: {phase: np.repeat(np.eye(2)[None, :, :], 2, axis=0) for phase in tie_lines} for context in ("interface", "general")}
    return TernaryMovingBoundaryThermodynamicsSurrogate(
        elements=("Z", "X", "Y"), phases=("ALPHA", "BETA"), tieline_phases=("ALPHA", "BETA"), temperature=1000.0,
        eta_samples=eta, tieline_compositions=tie_lines, diffusivity_compositions=compositions, diffusivities=matrices,
        validity_policy="legacy",
    )


def prepare_bundle(bundle_path, *, rebuild=False):
    """Build or strictly reload the bundle, as described in the module docstring.

    Returns ``(bundle, action)`` with ``action`` either ``"built"`` or ``"reloaded"``.
    """
    bundle_path = Path(bundle_path)

    def builder():
        return build_surrogate(), {}  # the second item lists extra member files to store

    if rebuild or not (bundle_path / "manifest.json").exists():
        return prepare_surrogate_artifact(bundle_path, BUILD_SPEC, builder), "built"
    try:
        return prepare_surrogate_artifact(bundle_path, BUILD_SPEC, builder, reload=True), "reloaded"
    except SurrogateArtifactCompatibilityError as exc:
        raise SurrogateArtifactCompatibilityError(f"{exc}\nUse --rebuild to replace {bundle_path}.") from exc


def build_model(config=DEMO, surrogate=None):
    """Build a two-phase solver from a surrogate; nothing is solved yet.

    The surrogate supplies both the diffusivities and the interface closure.
    """
    surrogate = build_surrogate() if surrogate is None else surrogate
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], config.mesh_nodes)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(0.45, [0.2, 0.1], [0.1, 0.2]), ["X", "Y"])]))
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh, ["Z", "X", "Y"], ["ALPHA", "BETA"], surrogate, 1000.0, 0.45,
        interface_equilibrium=surrogate,
        time_step=config.time_step,
        phase_a_nodes=config.phase_nodes,
        phase_b_nodes=config.phase_nodes,
        initial_eta_bracket=(0.0, 1.0),
        record=True,
        terminal_thin_phase_policy="raise",
    )


def solve(config, surrogate):
    """Solve with one surrogate and return the model."""
    model = build_model(config, surrogate)
    model.solve(config.duration, minDtFrac=1e-14)
    return model


def integrity_demo(bundle_path):
    """Show that damage and a changed specification are both refused, using a temporary copy."""
    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / "copy"
        shutil.copytree(bundle_path, copy)
        changed = dict(BUILD_SPEC, revision=BUILD_SPEC["revision"] + 1)
        try:
            SurrogateArtifactBundle.load(copy, changed)
        except SurrogateArtifactCompatibilityError:
            specification_refused = True
        else:
            specification_refused = False
        model_file = copy / SurrogateArtifactBundle.load(copy, BUILD_SPEC).manifest["members"]["model"]["path"]
        data = bytearray(model_file.read_bytes())
        data[len(data) // 2] ^= 0xFF
        model_file.write_bytes(bytes(data))
        try:
            SurrogateArtifactBundle.load(copy, BUILD_SPEC)
        except SurrogateArtifactIntegrityError:
            damage_refused = True
        else:
            damage_refused = False
    return {"changed_specification_refused": specification_refused, "damaged_member_refused": damage_refused}


def run(config=DEMO, *, output_dir=DEFAULT_OUTPUT, rebuild=False):
    """Run the whole lifecycle: prepare, solve, compare with a fresh surrogate, inspect."""
    bundle_path = Path(output_dir) / "bundle"
    bundle, action = prepare_bundle(bundle_path, rebuild=rebuild)
    reloaded = SurrogateArtifactBundle.load(bundle_path, BUILD_SPEC)
    model = solve(config, reloaded.surrogate)
    fresh = solve(config, build_surrogate())
    _, position = recorded_history(model.interfaceData)
    _, fresh_position = recorded_history(fresh.interfaceData)
    metrics = {
        "action": action,
        "bundle": str(bundle.path),
        "build_fingerprint": bundle.manifest["build_fingerprint"],
        "fresh_vs_reloaded_interface_abs": float(np.max(np.abs(position - fresh_position))),
        **integrity_demo(bundle_path),
    }
    return ExampleResult(config, model, None, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="where the bundle is kept")
    parser.add_argument("--rebuild", action="store_true", help="rebuild and replace an existing bundle")
    args = parser.parse_args()
    result = run(output_dir=args.output_dir, rebuild=args.rebuild)
    for name, value in result.metrics.items():
        print(f"{name}: {value}")
    print(f"final interface {result.model.getInterfacePosition():.6f}, eta {result.model.getInterfaceEta():.6f}")
