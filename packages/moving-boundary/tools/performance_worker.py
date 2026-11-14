"""Isolated installed-package measurement worker, launched only by performance.py."""
import contextlib
import builtins
import gc
import importlib.metadata as metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

from performance_results import SCHEMA, summarize

COUNTERS = ("_lastImplicitIterations", "_lastStepRetries", "_lastImplicitCandidateEvaluations",
            "_lastImplicitJacobianEvaluations", "_lastLeftPicardIterations", "_lastRightPicardIterations",
            "_lastBulkLeftPicardIterations", "_lastBulkRightPicardIterations",
            "_lastBulkDiffusivityEvaluations")


def forbid_input(*args, **kwargs):
    """Fail any unexpected prompt immediately in this noninteractive worker."""
    raise RuntimeError("Interactive input is forbidden in performance workers")


def json_value(value):
    """Normalize effective numerical controls without serializing provider objects."""
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str, list, tuple)):
        return value
    return repr(value)


def effective_settings(model):
    """Capture constructor defaults and resolved solver/constraint controls.

    Defaults are part of compatibility: a later solver-default change must be
    visible even when the benchmark's explicitly supplied arguments stay fixed.
    """
    defaults = {name: json_value(p.default) for name, p in inspect.signature(type(model)).parameters.items()
                if p.default is not inspect.Parameter.empty}
    controls = {key: json_value(value) for key, value in vars(model).items()
                if not key.startswith("_") and any(word in key.lower() for word in
                   ("tolerance", "picard", "iteration", "retry", "retries", "floor", "mode", "newton", "root"))}
    constraints = {key: json_value(value) for key, value in vars(model.constraints).items()}
    return dict(defaults=defaults, controls=controls, constraints=constraints)


def measure_once(case, build, validate, clock=time.perf_counter):
    """Measure one fresh trajectory and reset, validating outside timed regions.

    setup excludes construction and is nested inside the complete public solve.
    solve includes first setup and normal first finalization. finalize is the
    nested first postSolve call, never an
    already-finalized no-op. reset acts on the validated, completed model.
    The postProcess wrapper counts only commits that advance accepted time;
    binary/two-phase near-final no-op commits are counted separately and do
    not duplicate the previous step's nonlinear diagnostics.
    diagnostic sums reflect the exposed last-attempt counters, not all rejected
    nonlinear work. Unavailable counters are null, never implied zero.
    """
    gc.collect()  # Release previous instance wrappers/cycles outside timed regions.
    model = build(case)
    settings = effective_settings(model)
    work = {key: None for key in COUNTERS}
    work["accepted_steps"] = 0
    work["final_noop_commits"] = 0
    original_post, original_final, original_setup = model.postProcess, model.postSolve, model.setup
    final_times, setup_times = [], []

    def setup():
        start = clock()
        answer = original_setup()
        setup_times.append(clock()-start)
        return answer

    def post(time_, state):
        previous_time = model.currentTime
        noop = bool(getattr(model, "_nearFinalNoop", False))
        answer = original_post(time_, state)
        if model.currentTime > previous_time:
            if noop:
                work["final_noop_commits"] += 1
                return answer
            work["accepted_steps"] += 1
            for key in COUNTERS:
                if hasattr(model, key):
                    work[key] = (work[key] or 0) + int(getattr(model, key))
        return answer

    def finalize():
        start = clock()
        answer = original_final()
        final_times.append(clock()-start)
        return answer

    model.postProcess, model.postSolve, model.setup = post, finalize, setup
    with contextlib.redirect_stdout(io.StringIO()):
        start = clock()
        model.solve(case["dt"]*case["steps"], minDtFrac=1e-12, maxDtFrac=1.)
        solve_time = clock()-start
        validity = validate(model, case)
        if work["accepted_steps"] < case["steps"] or len(final_times) != 1 or not setup_times:
            raise ValueError("Missing accepted steps, setup or unexpected finalization count")
        start = clock()
        model.reset()
        reset_time = clock()-start
        if model.currentTime != 0.:
            raise ValueError("Reset did not restore initial time")
    return dict(setup=setup_times[0], solve=solve_time, finalize=final_times[0], reset=reset_time), work, validity, settings


def environment(target, host):
    """Assert installed origins and retain machine/library/BLAS/thread metadata."""
    import kawin
    import moving_boundary
    import numpy as np
    if not Path(moving_boundary.__file__).resolve().is_relative_to(target):
        raise RuntimeError("moving_boundary did not load from the fresh private target")
    if Path(kawin.__file__).resolve() != host / "kawin/__init__.py":
        raise RuntimeError("kawin did not load from the selected host")
    versions = {}
    for name in ("moving-boundary", "kawin", "numpy", "scipy", "numba", "llvmlite", "pycalphad", "symengine", "tqdm"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    if versions["kawin"] != "0.5.0":
        raise RuntimeError("Benchmarks require public kawin==0.5.0")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        np.show_config()
    return dict(python=platform.python_version(), implementation=platform.python_implementation(),
                platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                node=platform.node(), cpu_count=os.cpu_count(), versions=versions,
                numpy_config=output.getvalue(), threads={key: os.environ.get(key) for key in
                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")})


def main():
    """Run a fixed suite in one warm process; checkpoint each completed case."""
    builtins.input = forbid_input
    from performance_cases import build_model, validate_model
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    target, host = Path(request["target"]), Path(request["host"])
    result = dict(schema=SCHEMA, environment=environment(target, host),
                  identity=request["identity"], measurement=request["measurement"], cases={})
    start = time.perf_counter()
    failed = False
    for case in request["cases"]:
        name = case["id"]
        row = dict(status="running", configuration=dict(case=case), metrics={}, work=[], validation=[])
        result["cases"][name] = row
        try:
            samples = []
            for index in range(request["warmups"] + request["samples"]):
                metrics, work, valid, settings = measure_once(case, build_model, validate_model)
                if "effective" in row["configuration"] and row["configuration"]["effective"] != settings:
                    raise ValueError("Effective controls changed between samples")
                row["configuration"]["effective"] = settings
                if index >= request["warmups"]:
                    samples.append(metrics)
                    row["work"].append(work)
                    row["validation"].append(valid)
            row["metrics"] = {key: summarize([sample[key] for sample in samples]) for key in samples[0]}
            row["status"] = "ok"
            print(f"{name}: solve {row['metrics']['solve']['median']:.6f} s, "
                  f"MAD {row['metrics']['solve']['relative_mad']:.1%}, steps {row['work'][0]['accepted_steps']}; "
                  + ", ".join(f"{key} {row['metrics'][key]['median']:.6f} s" for key in ("setup", "finalize", "reset")), flush=True)
        except Exception:
            row.update(status="failed", error=traceback.format_exc())
            failed = True
            print(f"{name}: FAILED\n{row['error']}", flush=True)
        result["suite_seconds"] = time.perf_counter()-start
        result["status"] = "running"
        (request_path.parent / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    result["status"] = "failed" if failed else "ok"
    (request_path.parent / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Suite: {result['suite_seconds']:.2f} s; {request_path.parent / 'results.json'}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
