# %% [markdown]
# # How do I save, verify, reload and solve with a surrogate?
#
# Sampling thermodynamics is slow, so you normally build a surrogate once and reuse it. This
# shows that workflow without needing a database: a tiny synthetic surrogate is published as
# a verified bundle, reloaded, and used for a solve. For the same workflow on real data, see
# `ternary_fecrni_lee1996.py`.
#
# **What happens on disk.** The bundle goes in `OUTPUT_DIR`, by default
# `~/moving_boundary_example_outputs/surrogate_lifecycle/`, wherever you run from. It is never
# written into the package or source tree.
#
# - No bundle yet: build and publish it.
# - A compatible bundle: reload it. The builder is not called.
# - An incompatible or damaged bundle: stop with an error. It is never rebuilt silently, so
#   you always know which data a result came from.
# - `REBUILD = True` (or `--rebuild`): build and atomically replace it.
#
# A *build specification* records everything that defines the data. Reloading requires an
# exact match, and every member file is checked against its SHA-256.
#
# The surrogate is synthetic: a unit diffusivity matrix and linear tie lines. Elements are
# `[Z, X, Y]` with Z dependent, ALPHA lies left of BETA, compositions are mole fractions and
# units are nondimensional. So this checks the storage workflow, not surrogate quality or
# physics: a reload that matches the original says the bundle preserved the data, not that
# the data is right. With these arbitrary values the interface barely moves over this short
# run, which doesn't matter for that purpose.
#
# **Running.** Run the cells in order in a Jupyter or VS Code Interactive window; each workflow
# step is its own cell. From a terminal: `python surrogate_lifecycle.py [--output-dir DIR] [--rebuild]`
# (under a second).
"""How do I save, verify, reload and solve with a surrogate? See the notes above."""

# %%
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import shutil
import sys
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

DEFAULT_OUTPUT = Path.home() / "moving_boundary_example_outputs" / "surrogate_lifecycle"
BUILD_SPEC = {"example": "surrogate_lifecycle", "surrogate": "synthetic unit-matrix", "revision": 1}


# %% [markdown]
# ## Configuration

# %%
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

# %% [markdown]
# ## Settings

# %%
CONFIG = DEMO
OUTPUT_DIR = DEFAULT_OUTPUT
REBUILD = False


# %% [markdown]
# ## Building, storing and checking the surrogate

# %%
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
    """Build or strictly reload the bundle, as described in the notes above.

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
        raise SurrogateArtifactCompatibilityError(f"{exc}\nSet REBUILD = True (or use --rebuild) to replace {bundle_path}.") from exc


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


# %% [markdown]
# ## Model and solve

# %%
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


def interface_difference(model, other):
    """Largest difference between two runs' recorded interface positions."""
    return float(np.max(np.abs(recorded_history(model.interfaceData)[1] - recorded_history(other.interfaceData)[1])))


def run(config=DEMO, *, output_dir=DEFAULT_OUTPUT, rebuild=False):
    """Run the whole lifecycle in one call: prepare, solve, compare with a fresh surrogate, inspect."""
    bundle_path = Path(output_dir) / "bundle"
    bundle, action = prepare_bundle(bundle_path, rebuild=rebuild)
    reloaded = SurrogateArtifactBundle.load(bundle_path, BUILD_SPEC)
    model = solve(config, reloaded.surrogate)
    fresh = solve(config, build_surrogate())
    metrics = {
        "action": action,
        "bundle": str(bundle.path),
        "build_fingerprint": bundle.manifest["build_fingerprint"],
        "fresh_vs_reloaded_interface_abs": interface_difference(model, fresh),
        **integrity_demo(bundle_path),
    }
    return ExampleResult(config, model, None, metrics)


def settings_from_command_line(output_dir, rebuild):
    """Apply ``--output-dir`` and ``--rebuild`` from a terminal; inside a notebook kernel, change nothing.

    A Jupyter kernel's ``sys.argv`` holds the kernel's own arguments, so they are not parsed.
    """
    if "ipykernel" in sys.modules:
        return output_dir, rebuild
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, default=output_dir, help="where the bundle is kept")
    parser.add_argument("--rebuild", action="store_true", help="rebuild and replace an existing bundle")
    args = parser.parse_args()
    return args.output_dir, args.rebuild


# %% [markdown]
# ## Run, one step per cell
#
# 1. Build or reload the bundle, and look at what it records.

# %%
if __name__ == "__main__":
    OUTPUT_DIR, REBUILD = settings_from_command_line(OUTPUT_DIR, REBUILD)
    bundle, action = prepare_bundle(Path(OUTPUT_DIR) / "bundle", rebuild=REBUILD)
    print(f"{action}: {bundle.path}")
    print(json.dumps({role: entry["path"] for role, entry in bundle.manifest["members"].items()}, indent=2))
    print("build specification:", bundle.manifest["build_spec"])

# %% [markdown]
# 2. Reload it strictly and solve with the reloaded surrogate.

# %%
if __name__ == "__main__":
    reloaded = SurrogateArtifactBundle.load(bundle.path, BUILD_SPEC)
    model = solve(CONFIG, reloaded.surrogate)
    print(f"final interface {model.getInterfacePosition():.6f}, eta {model.getInterfaceEta():.6f}")

# %% [markdown]
# 3. Solve again with a freshly built surrogate; the results should be identical.

# %%
if __name__ == "__main__":
    fresh = solve(CONFIG, build_surrogate())
    print("largest interface difference, fresh vs reloaded:", interface_difference(model, fresh))

# %% [markdown]
# 4. Show that a changed specification and a damaged file are both refused.

# %%
if __name__ == "__main__":
    print(integrity_demo(bundle.path))
