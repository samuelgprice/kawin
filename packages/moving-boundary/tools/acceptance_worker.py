"""One isolated acceptance worker; never execute from a checkout import path."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import runpy
import site
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
runpy.run_path(str(HERE / "batch_safety.py"))["forbid_input"]()
# Cases whose package role builds from tools/definitions (materialized, hash-checked) instead of
# the historical acceptance_inputs AST loader; the reference role always keeps the loader.
PACKAGE_DEFINITIONS = ({f"C{n:02d}" for n in range(1, 33)} | {"LEGACY", "prepare", "F04", "F11", "F12"}
                       | {"O01", "O02", "O03a", "O03b", "O04", "O05", "O06", "O07", "O08"})


def module(name, path):
    """Load a tooling module by file without adding a checkout to sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def main():
    """Assert import origins before case construction and save local numeric output.

    The package role takes no ``--root``; every case it accepts is in ``PACKAGE_DEFINITIONS``
    and takes its inputs from ``tools/definitions/_bindings.Context``: ``acceptance_inputs`` is
    never imported (asserted after the run) and no research file is read. The reference role
    (historical only) needs ``--root``, the frozen research checkout. LEGACY's research baseline-npz assertion runs in
    the reference role only; the package role relies on ``compare()`` against the hashed
    ``reference/LEGACY.npz`` (decision F5) and keeps the recorded stop status label. Contracts
    (C25-C29) also report ``model_scales`` per ``model{i}/`` prefix, in both roles (item 7).

    ``--prepare`` builds the synthetic ``_surrogate`` and saves it to this worker's own
    ``--output`` npz, in both roles, never to ``--artifacts``: the adopted
    ``inputs/open-surrogate.npz`` stays the hashed C24 input (decision F7) and is only compared
    against, array-wise, by ``acceptance.regenerate_surrogate``. Likewise O05's pycalphad sample
    is saved beside ``--output`` (``<output>.pycalphad.npz``) in both roles, so no run rewrites
    ``inputs/pycalphad-<role>.npz``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("reference", "package"), required=True)
    parser.add_argument("--root", type=Path, help="frozen research checkout; reference (historical) role only")
    parser.add_argument("--target", type=Path)
    parser.add_argument("--host", type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--sample-pycalphad", action="store_true")
    parser.add_argument("--probe-thermocalc", action="store_true")
    args = parser.parse_args()
    if args.role == "reference":
        assert args.root is not None, "the reference role needs --root (historical)"
    else:
        assert args.root is None, "the package role takes no research root"
        assert args.case in PACKAGE_DEFINITIONS, args.case + " has no package-owned definitions"
    sys.path.extend(site.getsitepackages())
    if args.role == "reference":
        sys.path.insert(0, str(args.root.resolve()))
    else:
        sys.path[:0] = [str(args.target.resolve()), str(args.host.resolve())]
    import numpy as np
    if args.case not in {"O05","O08"}:
        assert "PYTHONHASHSEED" not in os.environ, "Unrelated cases must be unseeded"
    import kawin
    expected = args.root / "kawin/__init__.py" if args.role == "reference" else args.host / "kawin/__init__.py"
    assert Path(kawin.__file__).resolve() == expected.resolve(), kawin.__file__
    if args.role == "package":
        from importlib.metadata import version
        assert version("kawin") == "0.5.0"
        import moving_boundary
        assert Path(moving_boundary.__file__).resolve().is_relative_to(args.target.resolve())
    cases = module("acceptance_cases", HERE / "acceptance_cases.py")
    definitions = args.role == "package"  # every package-role case is in PACKAGE_DEFINITIONS (asserted)
    if definitions:
        inputs = module("definitions_bindings", HERE / "definitions/_bindings.py").Context()
    else:
        inputs_module = module("acceptance_inputs", HERE / "acceptance_inputs.py")
        inputs = inputs_module.CaseInputs(args.root, args.role)
    number = int(args.case[1:]) if args.case.startswith("C") else -1
    if args.case in {"LEGACY","F04","O07"} or number in list(range(1,5))+[13,14,15,25,30,31]:
        inputs.family = "binary"
    elif args.case in {"O01","O03a","O04","O05","O06"} or number in [5,6,7,8,16,17,18,19,24,26]:
        inputs.family = "ternary"
    else:
        inputs.family = "three_phase"
    inputs.artifacts = args.artifacts
    # O05's transient pycalphad sample goes next to this worker's output, never into --artifacts.
    inputs.pycalphad_sample = args.output.with_name(args.output.name + ".pycalphad.npz")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = dict(case=args.case, role=args.role, kawin_origin=kawin.__file__,
                  hash_seed=os.environ.get("PYTHONHASHSEED"), numpy_seed_override=False)
    try:
        import numpy as np
        if args.probe_thermocalc:
            optional = module("acceptance_optional", HERE / "acceptance_optional.py")
            tc = module("acceptance_thermocalc", HERE / "acceptance_thermocalc.py")
            configuration = json.loads((HERE / "acceptance_manifest.json").read_text(encoding="utf-8"))["thermocalc_sampling"]
            available = tc.probe_availability(inputs, configuration,
                generation_guard=optional.guard_package_generation, directory=args.output.parent)
            report.update(thermocalc_availability=available, stop_status="tc_availability_checked")
            arrays = {}
        elif args.prepare:
            source = inputs.load(inputs.root / "kawin/tests/test_surrogate_artifacts.py", ["_surrogate"])
            source._surrogate().save(args.output.with_suffix(".npz"))
            arrays = {}
        elif args.case == "O08":
            optional = module("acceptance_optional", HERE / "acceptance_optional.py")
            default = module("acceptance_default_phase", HERE / "acceptance_default_phase.py")
            configuration = json.loads((HERE / "acceptance_manifest.json").read_text(encoding="utf-8"))["default_phase_sampling"]
            arrays,contract,audit = default.sample_default_phase(inputs,configuration,
                generation_guard=optional.guard_package_generation)
            report.update(default_phase_contract=contract,pycalphad_generation=audit,
                          stop_status="default_phase_sampling_completed")
            if args.role == "package":
                inputs.generation_audit = audit
            else:
                report["invalid_branch_debugger_imported"] = "examples.debugInPlace" in sys.modules
            np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
        elif args.sample_pycalphad:
            optional = module("acceptance_optional", HERE / "acceptance_optional.py")
            optional.sample_pycalphad(inputs)
            arrays = {}
        elif args.case in {"C25", "C26", "C27", "C28", "C29"}:
            contracts = module("acceptance_contracts", HERE / "acceptance_contracts.py")
            arrays, completed = contracts.run_contract(inputs, args.case, cases.collect)
            report.update(contracts=completed, providers=inputs.providers, stop_status="contract_assertions_passed",
                          model_scales=contracts.model_scales(inputs.models))
            np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
        elif args.case == "LEGACY":
            source = inputs.load(inputs.root / "kawin/tests/illingworth_baseline_cases.py", ["_binary_model", "_collect"])
            arrays = {}
            for mode in ("phase_uniform", "composition_dependent_lagged", "composition_dependent_implicit"):
                model = source._binary_model(mode, mode != "phase_uniform")
                model.solve(.5, minDtFrac=1e-14)
                arrays.update(source._collect("binary_" + mode, model, ("interfaceData","concData","pData","qData")))
            if args.role == "reference":
                with np.load(args.root / "kawin/tests/data/illingworth_substitutional_baseline.npz") as fixture:
                    assert set(arrays) == {key for key in fixture.files if key.startswith("binary_")}
                    for key in arrays:
                        np.testing.assert_array_equal(arrays[key], fixture[key], err_msg=key)
            bulk = module("binary_bulk_cases", HERE / "binary_cases.py")
            arrays.update(bulk.run_bulk_cases(inputs.model("MovingBoundaryIllingworthFD1DModel")))
            report.update(providers=inputs.providers, stop_status="fixture_and_bulk_passed")
            np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
        else:
            if args.case.startswith("O"):
                optional = module("acceptance_optional", HERE / "acceptance_optional.py")
                if args.case == "O06":
                    tc = module("acceptance_thermocalc", HERE / "acceptance_thermocalc.py")
                    configuration = json.loads((HERE / "acceptance_manifest.json").read_text(encoding="utf-8"))["thermocalc_sampling"]
                    model, duration, analytic = tc.build_case(inputs, configuration,
                        generation_guard=optional.guard_package_generation, directory=args.output.parent)
                else:
                    model, duration, analytic = optional.build_optional(inputs, args.case)
            else:
                model, duration, analytic = cases.build_case(inputs, args.case, args.output.parent)
            length = float(np.ptp(np.ravel(model.mesh.z)))
            model.solve(duration, minDtFrac=1e-14)
            arrays = cases.collect(model, analytic)
            if args.case == "O05":
                with np.load(inputs.pycalphad_sample) as saved:
                    for key in saved.files:
                        if saved[key].dtype.kind in "bifcu":
                            arrays["provider/" + key] = saved[key]
            if args.case == "O06":
                with np.load(inputs.thermocalc_surrogate, allow_pickle=False) as saved:
                    arrays.update({"provider/" + key:saved[key] for key in saved.files})
                steps = len(arrays["interfaceData/time"]) - 1
                assert steps == 6, "O06 must complete six accepted steps"
                report.update(accepted_steps=steps, thermocalc_cache=str(inputs.thermocalc_cache))
            if args.case == "C30":
                arrays["temperature/provider_queries"] = np.array([float(np.ravel(item[2])[0]) for item in model.therm.calls])
            report.update(final_time=float(model.currentTime), requested_duration=duration,
                          length_scale=length, stop_status="terminal" if getattr(model, "_terminalThinPhaseInfo", None) else "completed",
                          providers=inputs.providers)
            volumes = getattr(model, "_phaseMolarVolumes", None)
            if volumes is not None:
                report["minimum_molar_volume"] = float(np.min(volumes))
            if args.role == "package":
                for name, provider in inputs.providers.items():
                    if provider.startswith("moving_boundary"):
                        mod = sys.modules[provider]
                        assert Path(mod.__file__).resolve().is_relative_to(args.target.resolve())
            np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
        if args.case in {"O05","O08","O06"} and args.role == "package" and not args.probe_thermocalc:
            audit = inputs.generation_audit
            assert audit["guarded"] and not audit["examples_imported"] and not audit["examples_read"]
            assert Path(audit["adapter_origin"]).resolve().is_relative_to(args.target.resolve())
            report["thermocalc_generation" if args.case == "O06" else "pycalphad_generation"] = audit
        if definitions:
            assert "acceptance_inputs" not in sys.modules, "package-owned definitions imported the historical loader"
            report["case_definitions"] = "tools/definitions"
        report.update(status="PASS", arrays=len(arrays))
    except Exception:
        report.update(status="FAIL", error=traceback.format_exc())
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "error"}))
    if report["status"] != "PASS":
        print(report["error"], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
