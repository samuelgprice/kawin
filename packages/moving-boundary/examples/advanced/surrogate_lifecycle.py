"""Advanced checks for ``surrogate_lifecycle``; see that example for assumptions and references.

The Case supplies explicit physical inputs and callbacks. No main script
is imported, and no solve or file operation runs during import.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import shutil
import sys
import tempfile
import numpy as np
from moving_boundary.analytic import recorded_history
from moving_boundary.surrogate_artifacts import SurrogateArtifactBundle, SurrogateArtifactCompatibilityError, SurrogateArtifactIntegrityError


@dataclass(frozen=True)
class Case:
    """Inputs and callbacks supplied by the main example at call time."""

    BUILD_SPEC: object
    ExampleResult: object
    description: str
    build_surrogate: object
    integrity_demo: object
    prepare_bundle: object
    run_differences: object
    solve: object


def integrity_demo(bundle_path, *, case):
    """Show that damage and a changed specification are both refused, using a temporary copy."""
    with tempfile.TemporaryDirectory() as directory:
        copy = Path(directory) / "copy"
        shutil.copytree(bundle_path, copy)
        changed = dict(case.BUILD_SPEC, revision=case.BUILD_SPEC["revision"] + 1)
        try:
            SurrogateArtifactBundle.load(copy, changed)
        except SurrogateArtifactCompatibilityError:
            specification_refused = True
        else:
            specification_refused = False
        model_file = copy / SurrogateArtifactBundle.load(copy, case.BUILD_SPEC).manifest["members"]["model"]["path"]
        data = bytearray(model_file.read_bytes())
        data[len(data) // 2] ^= 0xFF
        model_file.write_bytes(bytes(data))
        try:
            SurrogateArtifactBundle.load(copy, case.BUILD_SPEC)
        except SurrogateArtifactIntegrityError:
            damage_refused = True
        else:
            damage_refused = False
    return {"changed_specification_refused": specification_refused, "damaged_member_refused": damage_refused}

def run_differences(model, other, *, case):
    """Largest absolute differences between two two-phase runs, by quantity; all 0.0 if identical.

    Compares the recorded times, interface position, eta and solute inventory, and both
    phases' transformed states at every recorded time. Raises ``ValueError`` if the runs
    recorded different numbers of steps, since they can't then be compared pointwise.
    """
    times, other_times = recorded_history(model.interfaceData)[0], recorded_history(other.interfaceData)[0]
    if times.shape != other_times.shape:
        raise ValueError(f"The runs recorded {len(times)} and {len(other_times)} steps.")
    differences = {"times": float(np.max(np.abs(times - other_times)))}
    for name, history in (("interface", "interfaceData"), ("eta", "etaData"), ("inventory", "inventoryData")):
        values = np.asarray(recorded_history(getattr(model, history))[1], dtype=np.float64)
        other_values = np.asarray(recorded_history(getattr(other, history))[1], dtype=np.float64)
        differences[name] = float(np.max(np.abs(values - other_values)))
    differences["profiles"] = max(
        float(np.max(np.abs(np.asarray(state) - np.asarray(other_state))))
        for time, other_time in zip(times, other_times)
        for state, other_state in zip(model.getTransformedState(float(time)), other.getTransformedState(float(other_time)))
    )
    return differences

def run(config, *, output_dir, rebuild, case):
    """Run the whole lifecycle in one call: prepare, solve, compare with a fresh surrogate, inspect."""
    bundle_path = Path(output_dir) / "bundle"
    bundle, action = case.prepare_bundle(bundle_path, rebuild=rebuild)
    reloaded = SurrogateArtifactBundle.load(bundle_path, case.BUILD_SPEC)
    model = case.solve(config, reloaded.surrogate)
    fresh = case.solve(config, case.build_surrogate())
    metrics = {
        "action": action,
        "bundle": str(bundle.path),
        "build_fingerprint": bundle.manifest["build_fingerprint"],
        "fresh_vs_reloaded_max_abs": case.run_differences(model, fresh),
        **case.integrity_demo(bundle_path),
    }
    return case.ExampleResult(config, model, None, metrics)

def settings_from_command_line(output_dir, rebuild, *, case):
    """Apply ``--output-dir`` and ``--rebuild`` from a terminal; inside a notebook kernel, change nothing.

    A Jupyter kernel's ``sys.argv`` holds the kernel's own arguments, so they are not parsed.
    """
    if "ipykernel" in sys.modules:
        return output_dir, rebuild
    parser = argparse.ArgumentParser(description=case.description.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, default=output_dir, help="where the bundle is kept")
    parser.add_argument("--rebuild", action="store_true", help="rebuild and replace an existing bundle")
    args = parser.parse_args()
    return args.output_dir, args.rebuild
