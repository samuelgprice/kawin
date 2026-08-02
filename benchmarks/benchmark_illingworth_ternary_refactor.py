"""Benchmark and compare the ternary Illingworth refactor.

Run from the repository root after activating the ``kawin`` conda
environment:

    python benchmarks/benchmark_illingworth_ternary_refactor.py
    python benchmarks/benchmark_illingworth_ternary_refactor.py --baseline-ref 66fa8fa

The script times complete accepted ``getdXdt`` timesteps. It is intentionally
optional and makes no performance assertions.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import time
import types

import numpy as np

from kawin.diffusion.MovingBoundaryIllingworthTernaryFDM import MovingBoundaryIllingworthTernaryFD1DModel
from kawin.diffusion.mesh import CartesianFD1D, MixedBoundary1D, ProfileBuilder, StepProfile1D
from kawin.diffusion.mesh.MovingBoundaryIllingworthTernaryFD1D import integrate_planar_transformed_profile_components


class _CoupledTernaryThermodynamics:
    def clearCache(self):
        pass

    def getInterdiffusivity(self, composition, temperature, phase=None, **kwargs):
        if phase == "ALPHA":
            return np.asarray([[1.0e-3, 2.0e-4], [1.0e-4, 8.0e-4]], dtype=np.float64)
        return np.asarray([[7.0e-4, -1.0e-4], [2.0e-4, 1.1e-3]], dtype=np.float64)


class _EtaVaryingInterfaceEquilibrium:
    eta_bounds = (0.0, 1.0)

    def interface_compositions(self, eta):
        eta = float(eta)
        bend = eta * (1.0 - eta)
        center = np.asarray([0.24 + 0.25 * eta, 0.15 - 0.015 * eta + 0.22 * bend], dtype=np.float64)
        center_derivative = np.asarray([0.25, -0.015 + 0.22 * (1.0 - 2.0 * eta)], dtype=np.float64)
        tangent = center_derivative / np.linalg.norm(center_derivative)
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        tilt = np.deg2rad(55.0 - 30.0 * eta)
        tie_direction = np.sin(tilt) * tangent + np.cos(tilt) * normal
        left = center - 0.042 * (1.0 + 0.12 * (2.0 * eta - 1.0)) * tie_direction
        right = center + 0.055 * (1.0 - 0.10 * (2.0 * eta - 1.0)) * tie_direction
        return left, right


def _load_model_from_ref(ref):
    code = subprocess.check_output(
        ["git", "show", f"{ref}:kawin/diffusion/MovingBoundaryIllingworthTernaryFDM.py"],
        text=True,
    )
    module_name = f"illingworth_ternary_{ref.replace('/', '_').replace('-', '_')}"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    exec(compile(code, f"{ref}:MovingBoundaryIllingworthTernaryFDM.py", "exec"), module.__dict__)
    return module.MovingBoundaryIllingworthTernaryFD1DModel


def _make_model(model_cls, n_nodes):
    equilibrium = _EtaVaryingInterfaceEquilibrium()
    mesh = CartesianFD1D(["X", "Y"], [0.0, 1.0], int(n_nodes))
    mesh.setResponseProfile(
        ProfileBuilder([(StepProfile1D(0.45, np.asarray([0.16, 0.06]), np.asarray([0.3261538461538461, 0.193])), ["X", "Y"])]),
        boundaryConditions=MixedBoundary1D(2),
    )
    return model_cls(
        mesh=mesh,
        elements=["Z", "X", "Y"],
        phases=["ALPHA", "BETA"],
        thermodynamics=_CoupledTernaryThermodynamics(),
        temperature=1000.0,
        interfacePosition=0.45,
        interface_equilibrium=equilibrium,
        initial_eta_method="instantaneous_balance",
        initial_eta_bracket=equilibrium.eta_bounds,
        time_step=1.0e-4,
        tolerance=1.0e-11,
        max_iterations=50,
        record=True,
    )


def _wrap_candidate_counter(model):
    counters = {"candidate_evaluations": 0}
    residual_original = model._interface_residual

    def residual_spy(*args, **kwargs):
        counters["candidate_evaluations"] += 1
        return residual_original(*args, **kwargs)

    model._interface_residual = residual_spy
    return counters


def _run_once(model_cls, n_nodes):
    model = _make_model(model_cls, n_nodes)
    with contextlib.redirect_stdout(io.StringIO()):
        model.setup()
    counters = _wrap_candidate_counter(model)
    x = model.getCurrentX()
    old_inventory = integrate_planar_transformed_profile_components(x[0], x[1], float(x[2]), model._R, model._u_grid, model._v_grid)
    start = time.perf_counter()
    dXdt = model.getdXdt(model.currentTime, x)
    elapsed = time.perf_counter() - start
    p_new = x[0] + dXdt[0] * model._currdt
    q_new = x[1] + dXdt[1] * model._currdt
    s_new = float(x[2] + dXdt[2] * model._currdt)
    eta_new = float(x[3] + dXdt[3] * model._currdt)
    new_inventory = integrate_planar_transformed_profile_components(p_new, q_new, s_new, model._R, model._u_grid, model._v_grid)
    c_left, c_right = model.interfaceEquilibrium.interface_compositions(eta_new)
    return {
        "time": elapsed,
        "s": s_new,
        "eta": eta_new,
        "p": p_new,
        "q": q_new,
        "c_left": np.asarray(c_left, dtype=np.float64),
        "c_right": np.asarray(c_right, dtype=np.float64),
        "scaled_residual": float(model._lastImplicitResidual),
        "physical_residual": float(model._lastImplicitPhysicalResidual),
        "inventory": new_inventory,
        "inventory_drift_inf": float(np.max(np.abs(new_inventory - old_inventory))),
        "iterations": int(model._lastImplicitIterations),
        "candidate_evaluations": int(counters["candidate_evaluations"]),
        "diagnostic_candidate_evaluations": int(getattr(model, "_lastImplicitCandidateEvaluations", counters["candidate_evaluations"])),
        "jacobian_evaluations": int(getattr(model, "_lastImplicitJacobianEvaluations", max(0, int(model._lastImplicitIterations) - 1))),
        "retries": int(model._lastStepRetries),
    }


def _summarize(model_cls, n_nodes, repetitions):
    runs = [_run_once(model_cls, n_nodes) for _ in range(repetitions)]
    times = np.asarray([run["time"] for run in runs], dtype=np.float64)
    median_run = runs[int(np.argsort(times)[len(times) // 2])]
    return {**median_run, "time": float(np.median(times)), "time_samples": times.tolist()}


def _public_summary(record):
    return {
        key: value
        for key, value in record.items()
        if key not in {"p", "q", "c_left", "c_right", "inventory"}
    }


def _compare(current, baseline):
    return {
        "time_ratio": current["time"] / baseline["time"],
        "s_abs_diff": abs(current["s"] - baseline["s"]),
        "eta_abs_diff": abs(current["eta"] - baseline["eta"]),
        "p_max_abs_diff": float(np.max(np.abs(current["p"] - baseline["p"]))),
        "q_max_abs_diff": float(np.max(np.abs(current["q"] - baseline["q"]))),
        "c_left_max_abs_diff": float(np.max(np.abs(current["c_left"] - baseline["c_left"]))),
        "c_right_max_abs_diff": float(np.max(np.abs(current["c_right"] - baseline["c_right"]))),
        "inventory_max_abs_diff": float(np.max(np.abs(current["inventory"] - baseline["inventory"]))),
        "scaled_residual_abs_diff": abs(current["scaled_residual"] - baseline["scaled_residual"]),
        "physical_residual_abs_diff": abs(current["physical_residual"] - baseline["physical_residual"]),
        "iterations_diff": current["iterations"] - baseline["iterations"],
        "candidate_evaluations_diff": current["candidate_evaluations"] - baseline["candidate_evaluations"],
        "jacobian_evaluations_diff": current["jacobian_evaluations"] - baseline["jacobian_evaluations"],
        "retries_diff": current["retries"] - baseline["retries"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ref", default=None)
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--nodes", type=int, nargs="+", default=[21, 81])
    args = parser.parse_args()

    baseline_cls = _load_model_from_ref(args.baseline_ref) if args.baseline_ref else None
    report = {"repetitions": args.repetitions, "cases": []}
    for n_nodes in args.nodes:
        current = _summarize(MovingBoundaryIllingworthTernaryFD1DModel, n_nodes, args.repetitions)
        case = {"nodes": n_nodes, "current": _public_summary(current)}
        if baseline_cls is not None:
            baseline = _summarize(baseline_cls, n_nodes, args.repetitions)
            case["baseline"] = _public_summary(baseline)
            case["comparison"] = _compare(current, baseline)
        report["cases"].append(case)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
