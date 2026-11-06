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


def module(name, path):
    """Load a tooling module by file without adding a checkout to sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def main():
    """Assert import origins before case construction and save local numeric output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("reference", "package"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--host", type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--sample-pycalphad", action="store_true")
    args = parser.parse_args()
    sys.path.extend(site.getsitepackages())
    if args.role == "reference":
        sys.path.insert(0, str(args.root.resolve()))
    else:
        sys.path[:0] = [str(args.target.resolve()), str(args.host.resolve())]
    import numpy as np
    if args.case != "O05":
        assert "PYTHONHASHSEED" not in os.environ, "Unrelated cases must be unseeded"
    import kawin
    expected = args.root / "kawin/__init__.py" if args.role == "reference" else args.host / "kawin/__init__.py"
    assert Path(kawin.__file__).resolve() == expected.resolve(), kawin.__file__
    if args.role == "package":
        from importlib.metadata import version
        assert version("kawin") == "0.5.0"
        import moving_boundary
        assert Path(moving_boundary.__file__).resolve().is_relative_to(args.target.resolve())
        assert not any(Path(p).resolve() == args.root.resolve() for p in sys.path if p)
    inputs_module = module("acceptance_inputs", HERE / "acceptance_inputs.py")
    cases = module("acceptance_cases", HERE / "acceptance_cases.py")
    inputs = inputs_module.CaseInputs(args.root, args.role)
    number = int(args.case[1:]) if args.case.startswith("C") else -1
    if args.case in {"LEGACY","F04","O07"} or number in list(range(1,5))+[13,14,15,25,30,31]:
        inputs.family = "binary"
    elif args.case in {"O01","O03a","O04","O05"} or number in [5,6,7,8,16,17,18,19,24,26]:
        inputs.family = "ternary"
    else:
        inputs.family = "three_phase"
    inputs.artifacts = args.artifacts
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = dict(case=args.case, role=args.role, kawin_origin=kawin.__file__,
                  hash_seed=os.environ.get("PYTHONHASHSEED"), numpy_seed_override=False)
    try:
        import numpy as np
        if args.prepare:
            source = inputs.load(args.root / "kawin/tests/test_surrogate_artifacts.py", ["_surrogate"])
            source._surrogate().save(args.artifacts / "open-surrogate.npz")
            arrays = {}
        elif args.sample_pycalphad:
            optional = module("acceptance_optional", HERE / "acceptance_optional.py")
            optional.sample_pycalphad(inputs)
            arrays = {}
        elif args.case in {"C25", "C26", "C27", "C28", "C29"}:
            contracts = module("acceptance_contracts", HERE / "acceptance_contracts.py")
            arrays, completed = contracts.run_contract(inputs, args.case, cases.collect)
            report.update(contracts=completed, providers=inputs.providers, stop_status="contract_assertions_passed")
            np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
        elif args.case == "LEGACY":
            source = inputs.load(args.root / "kawin/tests/illingworth_baseline_cases.py", ["_binary_model", "_collect"])
            arrays = {}
            for mode in ("phase_uniform", "composition_dependent_lagged", "composition_dependent_implicit"):
                model = source._binary_model(mode, mode != "phase_uniform")
                model.solve(.5, minDtFrac=1e-14)
                arrays.update(source._collect("binary_" + mode, model, ("interfaceData","concData","pData","qData")))
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
                model, duration, analytic = optional.build_optional(inputs, args.case)
            else:
                model, duration, analytic = cases.build_case(inputs, args.case, args.output.parent)
            length = float(np.ptp(np.ravel(model.mesh.z)))
            model.solve(duration, minDtFrac=1e-14)
            arrays = cases.collect(model, analytic)
            if args.case == "O05":
                sampled = Path(inputs.artifacts) / ("pycalphad-" + args.role + ".npz")
                with np.load(sampled) as saved:
                    for key in saved.files:
                        if saved[key].dtype.kind in "bifcu":
                            arrays["provider/" + key] = saved[key]
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
        if args.case == "O05" and args.role == "package":
            audit = inputs.generation_audit
            assert audit["guarded"] and not audit["examples_imported"] and not audit["examples_read"]
            assert Path(audit["adapter_origin"]).resolve().is_relative_to(args.target.resolve())
            report["pycalphad_generation"] = audit
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
