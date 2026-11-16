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
# - An incompatible or damaged bundle, or a bundle folder with files but no `manifest.json`
#   (usually an interrupted build): stop with an error. It is never rebuilt silently, so you
#   always know which data a result came from.
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
# run, so the comparison covers the recorded times, eta, inventory and both phases' profiles,
# not just the interface.
#
# **Running.** Run the cells in order in a Jupyter or VS Code Interactive window; each workflow
# step is its own cell. From a terminal: `python surrogate_lifecycle.py [--output-dir DIR] [--rebuild]`
# (under a second).
"""How do I save, verify, reload and solve with a surrogate? See the notes above."""

# %%
from pathlib import Path
import sys

# Run notebook cells with the extraction repository root as the working directory.
sys.path.append(str(Path("packages/moving-boundary/examples").resolve()))

from advanced import surrogate_lifecycle as _advanced
from dataclasses import dataclass
import argparse
import json
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
    load_or_build_surrogate_artifact,
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
    """Build or strictly reuse a verified bundle; return ``(bundle, action)``.

    Only explicit ``rebuild`` replaces incompatible, damaged or incomplete data.
    """
    return load_or_build_surrogate_artifact(
        bundle_path, BUILD_SPEC, lambda: (build_surrogate(), {}), rebuild=rebuild)


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


def run_core(config=DEMO, *, output_dir=DEFAULT_OUTPUT, rebuild=False):
    """Build or reload one bundle and solve once, without corruption or twin-run checks."""
    bundle, action = prepare_bundle(Path(output_dir) / "bundle", rebuild=rebuild)
    model = solve(config, bundle.surrogate)
    return ExampleResult(config, model, None, {"action": action, "bundle": str(bundle.path)})


def solve(config, surrogate):
    """Solve with one surrogate and return the model."""
    model = build_model(config, surrogate)
    model.solve(config.duration, minDtFrac=1e-14)
    return model


# %% [markdown]
# ## Advanced checks and compatibility entry points
#
# The introductory path above needs only `run_core`. These wrappers retain
# the original detailed workflow and command-line behavior.

# %%
def integrity_demo(bundle_path):
    """Show that damage and a changed specification are both refused, using a temporary copy. See the advanced companion."""
    return _advanced.integrity_demo(bundle_path, case=_advanced_case())


def _advanced_case():
    """Supply current case inputs to the optional detailed workflow. See the advanced companion."""
    return _advanced.Case(
        BUILD_SPEC=BUILD_SPEC,
        ExampleResult=ExampleResult,
        description=__doc__,
        build_surrogate=build_surrogate,
        integrity_demo=integrity_demo,
        prepare_bundle=prepare_bundle,
        run_differences=run_differences,
        solve=solve,
    )


def run_differences(model, other):
    """Largest absolute differences between two two-phase runs, by quantity; all 0.0 if identical. See the advanced companion."""
    return _advanced.run_differences(model, other, case=_advanced_case())


def run(config=DEMO, *, output_dir=DEFAULT_OUTPUT, rebuild=False):
    """Run the whole lifecycle in one call: prepare, solve, compare with a fresh surrogate, inspect. See the advanced companion."""
    return _advanced.run(config, output_dir=output_dir, rebuild=rebuild, case=_advanced_case())


def settings_from_command_line(output_dir, rebuild):
    """Apply ``--output-dir`` and ``--rebuild`` from a terminal; inside a notebook kernel, change nothing. See the advanced companion."""
    return _advanced.settings_from_command_line(output_dir, rebuild, case=_advanced_case())


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
    print("largest differences, fresh vs reloaded:", run_differences(model, fresh))

# %% [markdown]
# 4. Show that a changed specification and a damaged file are both refused.

# %%
if __name__ == "__main__":
    print(integrity_demo(bundle.path))
