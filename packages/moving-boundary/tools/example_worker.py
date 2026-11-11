"""One isolated example-acceptance worker; never executes from a checkout import path.

Package role: build an example through the installed API, record its effective settings,
solve, and save parity arrays and the settings separately. Reference role: build the
accepted case on the frozen source only to record its settings; existing fixtures are
never written. Package role with ``--settings-only``: build the accepted case from the
package-owned ``tools/definitions`` instead and record only its settings (regeneration
verification, plan §3a); seeded sampling cases (O05) then run under ``PYTHONHASHSEED=0``. Example metrics are not parity arrays: they would break the frozen schema.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import runpy
import site
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
_SAFETY = runpy.run_path(str(HERE / "batch_safety.py"))
_SAFETY["forbid_input"]()
_SAFETY["guard_independence"]()   # opt-in research/Git guards (MB_FORBID_RESEARCH, MB_FORBID_GIT)


def module(name, path):
    """Load a module by file without adding its directory to sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


class NoResearchImports:
    """Reject research and debugger modules so examples can't reach the research checkout."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"examples", "debugpy"}:
            raise AssertionError("Research or debugger import from an example: " + fullname)


def package_case(entry, args, cases, settings):
    """Run one example on the installed package and return its model, parity arrays and record."""
    import numpy as np
    example = module("mb_example_" + entry["example"], args.examples / (entry["example"] + ".py"))
    config = getattr(example, entry["config"])
    artifacts = Path(args.artifacts)
    source = entry.get("surrogate")
    if source == "lifecycle":
        with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
            surrogate = example.prepare_bundle(Path(directory) / "bundle")[0]
            surrogate = example.SurrogateArtifactBundle.load(surrogate.path, example.BUILD_SPEC).surrogate
        model = example.build_model(config, surrogate)
    elif source and source.endswith(".npz"):
        from moving_boundary.surrogates import TernaryMovingBoundaryThermodynamicsSurrogate as Surrogate
        model = example.build_model(config, Surrogate.load(artifacts / source))
    elif source:
        from moving_boundary.surrogate_artifacts import SurrogateArtifactBundle
        model = example.build_model(config, SurrogateArtifactBundle.load_published(artifacts / source).surrogate)
    else:
        model = example.build_model(config)
    solve_arguments = dict(minDtFrac=1e-14)
    record = settings.effective_settings(model, config.duration, solve_arguments)
    model.solve(config.duration, **solve_arguments)
    analytic = None
    if entry["parity_analytic"] == "binary":
        from moving_boundary.analytic import _planar_binary
        analytic = ("binary", _planar_binary, config.t_start)
    elif entry["parity_analytic"] == "three_phase":
        analytic = ("three_phase", example.exact_solution(config), example.T0)
    arrays = cases.collect(model, analytic)
    if entry.get("provider_arrays"):
        # A schema match only: these keys are copied from the same frozen file the
        # surrogate was loaded from, so they aren't evidence about generation.
        with np.load(artifacts / entry["provider_arrays"]) as saved:
            for key in saved.files:
                if saved[key].dtype.kind in "bifcu":
                    arrays["provider/" + key] = saved[key]
    return model, arrays, record


def reference_settings(entry, args, cases, settings):
    """Build the accepted case on the frozen reference and record its settings; never solve.

    O05's pycalphad sample is written beside ``--output``, never into ``--artifacts``, so a
    historical capture cannot rewrite the adopted ``pycalphad-reference.npz`` (E05's input).
    """
    inputs = module("acceptance_inputs", HERE / "acceptance_inputs.py").CaseInputs(args.root, "reference")
    return accepted_case_settings(inputs, entry, args, cases, settings)


def definitions_settings(entry, args, cases, settings):
    """Package role: build the accepted case from ``tools/definitions`` and record its settings.

    The package generation path for the adopted settings files (plan §3a): the same builder as
    ``reference_settings`` on the materialized definitions (``_bindings.Context``), never solved,
    without reading research files or importing ``acceptance_inputs``.
    """
    inputs = module("definitions_bindings", HERE / "definitions/_bindings.py").Context()
    record = accepted_case_settings(inputs, entry, args, cases, settings)
    assert "acceptance_inputs" not in sys.modules, "package-owned definitions imported the historical loader"
    return record


def accepted_case_settings(inputs, entry, args, cases, settings):
    """Build ``entry``'s accepted case with ``inputs`` and return its effective settings."""
    inputs.artifacts = args.artifacts
    inputs.pycalphad_sample = args.output.with_name(args.output.name + ".pycalphad.npz")
    case = entry["reference"]
    number = int(case[1:]) if case.startswith("C") else -1
    inputs.family = "binary" if case == "F04" or number in (4,) else ("ternary" if case in {"O04", "O05"} or number in (16, 24) else "three_phase")
    if case.startswith("O"):
        model, duration, _ = module("acceptance_optional", HERE / "acceptance_optional.py").build_optional(inputs, case)
    else:
        model, duration, _ = cases.build_case(inputs, case, args.output.parent)
    return settings.effective_settings(model, duration, dict(minDtFrac=1e-14))


def main():
    """Assert import origins before constructing anything, then run one role."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("reference", "package"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--root", type=Path, help="frozen research checkout; reference (historical) role only")
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--host", type=Path)
    parser.add_argument("--settings-only", action="store_true",
                        help="package role: record the accepted case's settings from tools/definitions; no solve")
    args = parser.parse_args()
    sys.path.extend(site.getsitepackages())
    manifest = json.loads((HERE / "example_manifest.json").read_text(encoding="utf-8"))
    entry = manifest["cases"][args.case]
    assert not args.settings_only or args.role == "package", "--settings-only is a package-role mode"
    assert (args.root is not None) == (args.role == "reference"), "--root is required by, and only by, the reference role"
    if args.role == "reference":
        sys.path.insert(0, str(args.root.resolve()))
    else:
        sys.path[:0] = [str(args.target.resolve()), str(args.host.resolve())]
        sys.meta_path.insert(0, NoResearchImports())
        if args.settings_only and entry["reference"] in {"O05", "O08"}:
            assert os.environ.get("PYTHONHASHSEED") == "0", "seeded sampling cases build under PYTHONHASHSEED=0"
        else:
            assert "PYTHONHASHSEED" not in os.environ, "Example solves must be unseeded"
    import kawin
    expected = args.root if args.role == "reference" else args.host
    assert Path(kawin.__file__).resolve() == (expected / "kawin/__init__.py").resolve(), kawin.__file__
    if args.role == "package":
        from importlib.metadata import version
        assert version("kawin") == "0.5.0"
        import moving_boundary
        assert Path(moving_boundary.__file__).resolve().is_relative_to(args.target.resolve())
    import numpy as np
    cases = module("acceptance_cases", HERE / "acceptance_cases.py")
    settings = module("example_settings", HERE / "example_settings.py")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if args.role == "reference" or args.settings_only:
        build = reference_settings if args.role == "reference" else definitions_settings
        record = build(entry, args, cases, settings)
        args.output.with_suffix(".settings.json").write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
    else:
        model, arrays, record = package_case(entry, args, cases, settings)
        solver_module = type(model).__module__
        assert solver_module.startswith("moving_boundary."), solver_module
        assert Path(sys.modules[solver_module].__file__).resolve().is_relative_to(args.target.resolve())
        assert not any(name.split(".")[0] == "examples" for name in sys.modules), "Research module imported"
        np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
        args.output.with_suffix(".settings.json").write_text(json.dumps(record, indent=1, sort_keys=True), encoding="utf-8")
        report = dict(case=args.case, solver_module=solver_module, final_time=float(model.currentTime),
                      stop_status="terminal" if getattr(model, "_terminalThinPhaseInfo", None) else "completed",
                      elapsed_seconds=time.monotonic() - started, hash_seed=os.environ.get("PYTHONHASHSEED"))
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
