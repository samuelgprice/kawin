"""Benchmark two-phase W--Ti--Fe Illingworth surrogate inference costs.

This module deliberately instruments *only benchmark-owned objects*.  It does
not alter kawin's solver, surrogate, or example modules.  A saved surrogate is
required so Thermo-Calc construction time is not folded into simulation time.

Example (after activating the ``kawin`` conda environment)::

    python benchmarks/benchmark_wtife_surrogate_runtime.py --surrogate wtife.npz

The default delay backend is ``sleep``.  Requested and measured delay are both
reported because operating-system sleep granularity can exceed short requests.
``--delay-backend busy_wait`` exists only for controlled experiments and is
never the default.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Direct ``python benchmarks/...py`` execution otherwise places only the
# benchmarks directory on sys.path, while kawin retains imports from examples.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from kawin.diffusion import MovingBoundaryIllingworthTernaryFD1DModel
from kawin.diffusion.MovingBoundarySurrogates import TernaryMovingBoundaryThermodynamicsSurrogate
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.solver import explicitEulerIterator


WTIFE_ELEMENTS = ("W", "TI", "FE")
WTIFE_TEMPERATURE = 1973.0
WTIFE_A_BULK = np.asarray([0.02, 0.01])
WTIFE_B_BULK = np.asarray([0.59726, 0.37597])
WTIFE_HALF_LENGTH = 50.0e-6
WTIFE_INTERFACE = WTIFE_HALF_LENGTH - 2.0e-6 + 1.0e-12
VARIANTS = ("nearest", "simplex_linear", "simplex_positive_2x2", "continuous_grid")
MODES = ("phase_uniform", "composition_dependent_lagged", "composition_dependent_implicit")
OPTIMIZATION_PATHS = ("baseline", "batched_validation", "lagged_cache", "both")


class TimingCollector:
    """Nest-aware timer that retains inclusive and exclusive elapsed time."""

    def __init__(self):
        self.records = defaultdict(lambda: {"calls": 0, "inclusive_ns": 0, "child_ns": 0})
        self._stack: list[tuple[str, int, int]] = []

    @contextlib.contextmanager
    def region(self, name):
        start = time.perf_counter_ns()
        self._stack.append((name, start, 0))
        try:
            yield
        finally:
            _, _, child_ns = self._stack.pop()
            elapsed = time.perf_counter_ns() - start
            item = self.records[name]
            item["calls"] += 1
            item["inclusive_ns"] += elapsed
            item["child_ns"] += child_ns
            if self._stack:
                parent_name, parent_start, parent_child = self._stack[-1]
                self._stack[-1] = (parent_name, parent_start, parent_child + elapsed)

    def reset(self):
        self.records.clear()
        self._stack.clear()

    def summary(self):
        return {
            name: {
                "calls": value["calls"],
                "inclusive_ns": value["inclusive_ns"],
                "exclusive_ns": max(0, value["inclusive_ns"] - value["child_ns"]),
            }
            for name, value in self.records.items()
        }


class DelayInjector:
    """Inject and measure delay without using CPU by default."""

    def __init__(self, requested_us=0.0, backend="sleep"):
        self.requested_ns = int(round(float(requested_us) * 1000.0))
        self.backend = backend
        self.actual_ns = 0
        self.calls = 0

    def apply(self):
        if self.requested_ns <= 0:
            return
        start = time.perf_counter_ns()
        if self.backend == "sleep":
            time.sleep(self.requested_ns / 1.0e9)
        elif self.backend == "busy_wait":
            while time.perf_counter_ns() - start < self.requested_ns:
                pass
        else:
            raise ValueError("delay backend must be 'sleep' or 'busy_wait'.")
        self.actual_ns += time.perf_counter_ns() - start
        self.calls += 1

    def reset(self):
        """Discard setup-time delay evidence before a timed benchmark region."""
        self.actual_ns = 0
        self.calls = 0

    def record(self):
        """Return JSON-safe measured delay metadata for one benchmark region."""
        return {"requested_ns": self.requested_ns, "backend": self.backend,
                "actual_ns": self.actual_ns, "calls": self.calls}


class InstrumentedSurrogate:
    """Benchmark-local proxy that times surrogate API calls and their work size."""

    def __init__(self, source, collector, delays):
        self._source, self._collector, self._delays = source, collector, delays
        self.work = defaultdict(int)

    def __getattr__(self, name):
        return getattr(self._source, name)

    def interface_compositions(self, eta):
        with self._collector.region("inference.interface_compositions"):
            result = self._source.interface_compositions(eta)
            self._delays["interface_compositions"].apply()
        self.work["interface_endpoint_compositions"] += len(result)
        return result

    def getInterdiffusivity(self, x, *args, query_context=None, **kwargs):
        category = "interface_diffusivity" if query_context == "interface" else "bulk_diffusivity"
        with self._collector.region(f"inference.{category}"):
            result = self._source.getInterdiffusivity(x, *args, query_context=query_context, **kwargs)
            self._delays[category].apply()
        inputs = np.asarray(x)
        self.work[f"{category}_input_compositions"] += 1 if inputs.ndim == 1 else int(inputs.shape[0])
        matrices = np.asarray(result)
        self.work[f"{category}_returned_matrices"] += 1 if matrices.ndim == 2 else int(matrices.shape[0])
        return result


@contextlib.contextmanager
def _wrap_model_regions(model, collector):
    """Temporarily wrap selected methods on one model instance only."""
    names = {
        "getdXdt": "solver.getdXdt",
        "_evaluate_interface_candidate": "solver.interface_candidate",
        "_solve_concentration_left_planar": "solver.left_linear_bulk",
        "_solve_concentration_right_planar": "solver.right_linear_bulk",
        "_solve_concentration_left_picard": "solver.left_picard_bulk",
        "_solve_concentration_right_picard": "solver.right_picard_bulk",
    }
    originals = {}
    try:
        for attr, category in names.items():
            original = getattr(model, attr)
            originals[attr] = original

            def wrapped(*args, _original=original, _category=category, **kwargs):
                with collector.region(_category):
                    return _original(*args, **kwargs)

            setattr(model, attr, wrapped)
        yield
    finally:
        for attr, original in originals.items():
            setattr(model, attr, original)


def _validity_payload(source):
    """Copy archived validity/fit evidence without weakening its provenance."""
    out = {"interface": {}, "general": {}}
    for context in out:
        for phase in source.tieline_phases:
            domain = source.diffusivityValidity[context][phase]
            out[context][phase] = {
                "coordinates": domain.coordinates.copy(),
                "statuses": np.asarray([status.value for status in domain.statuses]),
                "reasons": domain.reasons.copy(),
                "fit_usable": domain.fit_usable.copy(),
            }
    return out


def reconstruct_variant(source, interpolation):
    """Rebuild one interpolation variant while retaining archive provenance."""
    grids = getattr(source, "diffusivityBulkGridAxes", None)
    if interpolation == "continuous_grid" and grids is None:
        raise ValueError("continuous_grid requires bulk-grid axes stored in the archive")
    return TernaryMovingBoundaryThermodynamicsSurrogate(
        elements=source.elements, phases=source.phases, tieline_phases=source.tieline_phases,
        temperature=source.temperature, eta_samples=source.eta_samples.copy(),
        tieline_compositions={p: source.tieline_compositions[p].copy() for p in source.tieline_phases},
        diffusivity_compositions={c: {p: source.diffusivity_compositions[c][p].copy() for p in source.tieline_phases}
                                  for c in ("interface", "general")},
        diffusivities={c: {p: source.diffusivities[c][p].copy() for p in source.tieline_phases}
                       for c in ("interface", "general")},
        min_composition=source.min_composition, metadata=dict(source.metadata),
        diffusivity_interpolation=interpolation, diffusivity_bulk_grids=grids,
        diffusivity_validity=_validity_payload(source), validity_policy=source.validity_policy,
        interface_query_tolerance=source.interface_query_tolerance,
    )


def _scalar_classify_many(domain, values, **_kwargs):
    """Benchmark-local scalar reference for the production batch classifier."""
    results = [domain._classify_scalar_reference(point) for point in np.asarray(values, dtype=np.float64)]
    return (
        np.asarray([status for status, _ in results], dtype=object),
        np.asarray([reason for _, reason in results], dtype=object),
    )


def _configure_optimization_path(surrogate, model, path):
    """Select benchmark-only reference/optimized paths without public toggles."""
    if path not in OPTIMIZATION_PATHS:
        raise ValueError(f"optimization path must be one of {OPTIMIZATION_PATHS}.")
    if path in {"baseline", "lagged_cache"}:
        for phase in surrogate.tieline_phases:
            domain = surrogate.diffusivityValidity["general"][phase]
            domain.classify_many = lambda values, _domain=domain, **kwargs: _scalar_classify_many(_domain, values, **kwargs)
        surrogate._has_general_fit_support_many = lambda phase, values, **kwargs: np.asarray(
            [surrogate._has_general_fit_support_scalar_reference(phase, point) for point in values], dtype=bool
        )
    if path in {"baseline", "batched_validation"}:
        model._prepare_lagged_face_diffusivity_cache = lambda *args, **kwargs: None


def validate_wtife_archive(source):
    """Reject archives that cannot represent the copied W--Ti--Fe benchmark."""
    if tuple(value.upper() for value in source.elements) != WTIFE_ELEMENTS:
        raise ValueError(f"Expected WTiFe elements {WTIFE_ELEMENTS}, got {source.elements}.")
    if not np.isclose(source.temperature, WTIFE_TEMPERATURE, rtol=0.0, atol=1e-8):
        raise ValueError(f"Expected {WTIFE_TEMPERATURE} K, got {source.temperature} K.")
    if len(source.tieline_phases) != 2:
        raise ValueError("The benchmark requires exactly two tie-line phases.")


def build_model(surrogate, mode):
    """Build the copied WTiFe two-phase configuration around a local proxy."""
    phases = list(surrogate.tieline_phases)
    mesh = CartesianFD1D(["TI", "FE"], [0.0, WTIFE_HALF_LENGTH], 201)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(WTIFE_INTERFACE, WTIFE_A_BULK, WTIFE_B_BULK), ["TI", "FE"])]))
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh, elements=list(surrogate.elements), phases=phases, thermodynamics=surrogate,
        temperature=WTIFE_TEMPERATURE, interfacePosition=WTIFE_INTERFACE,
        interface_equilibrium=surrogate, initial_eta_method="instantaneous_balance",
        initial_eta_bracket=(1.0e-3, 1.0 - 1.0e-3), initial_eta_guess=0.5,
        bulk_diffusivity_mode=mode, time_step=1.0, dt_mode="semi_log", semiLog_dt=0.25 / 20.0,
        semiLogT0=1.0e-6, tolerance=1.0e-12, max_iterations=25, max_step_retries=8,
        terminal_thin_phase_policy="continue", record=True, record_pq_data=True,
        transformed_u_grid=np.linspace(0.0, 1.0, 100), transformed_v_grid=np.linspace(0.0, 1.0, 50),
    )


def _clone_state(x):
    return [np.asarray(value).copy() if isinstance(value, np.ndarray) else float(value) for value in x]


def _states_match(first, second):
    """Compare the complete transformed state without sharing mutable arrays."""
    return all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(first, second))


def _solver_work(model):
    """Return solver diagnostics needed to interpret optimization speedups."""
    return {
        "candidate_evaluations": int(getattr(model, "_lastImplicitCandidateEvaluations", 0)),
        "bulk_provider_calls": int(getattr(model, "_lastBulkDiffusivityProviderCalls", 0)),
        "bulk_face_matrices_evaluated": int(getattr(model, "_lastBulkFaceMatricesEvaluated", 0)),
    }


def _delays(delay_us, delay_category, backend):
    return {
        name: DelayInjector(delay_us if delay_category in {"all", name} else 0.0, backend)
        for name in ("interface_compositions", "interface_diffusivity", "bulk_diffusivity")
    }


def measure_microbenchmark(base, variant, mode, optimization_path, repetitions, warmups, delay_us, delay_category, delay_backend):
    """Time identical initial numerical states; setup is outside the timed region."""
    samples = []
    reference_state = None
    for index in range(warmups + repetitions):
        collector = TimingCollector()
        delays = _delays(delay_us, delay_category, delay_backend)
        proxy = InstrumentedSurrogate(reconstruct_variant(base, variant), collector, delays)
        model = build_model(proxy, mode)
        with _wrap_model_regions(model, collector):
            model.setup()
            _configure_optimization_path(proxy._source, model, optimization_path)
            state = _clone_state(model.getCurrentX())
            if reference_state is None:
                reference_state = _clone_state(state)
            elif not _states_match(reference_state, state):
                raise RuntimeError("Repeated fixed-state setup produced a different numerical state.")
            collector.reset()  # setup is deliberately excluded.
            proxy.work.clear()
            for injector in delays.values():
                injector.reset()
            start = time.perf_counter_ns()
            model.getdXdt(0.0, _clone_state(state))
            elapsed = time.perf_counter_ns() - start
        if index >= warmups:
            samples.append({"elapsed_ns": elapsed, "timing": collector.summary(), "work": dict(proxy.work), "solver_work": _solver_work(model),
                            "delay": {k: v.record() for k, v in delays.items()}})
    return samples


def measure_end_to_end(base, variant, mode, optimization_path, end_time, delay_us, delay_category, delay_backend):
    """Time a complete solve; setup, loading, and output remain outside timing."""
    collector = TimingCollector()
    delays = _delays(delay_us, delay_category, delay_backend)
    proxy = InstrumentedSurrogate(reconstruct_variant(base, variant), collector, delays)
    model = build_model(proxy, mode)
    with _wrap_model_regions(model, collector):
        model.setup()
        _configure_optimization_path(proxy._source, model, optimization_path)
        collector.reset()
        proxy.work.clear()
        for injector in delays.values():
            injector.reset()
        start = time.perf_counter_ns()
        model.solve(float(end_time), iterator=explicitEulerIterator, verbose=False, minDtFrac=1e-16)
        elapsed = time.perf_counter_ns() - start
    return {
        "elapsed_ns": elapsed,
        "timing": collector.summary(),
        "work": dict(proxy.work),
        "delay": {name: value.record() for name, value in delays.items()},
        "final_time": float(model.currentTime),
    }


def measure_short_trajectory(base, variant, mode, optimization_path, steps):
    """Time a small, native trajectory without repeating delay experiments.

    This uses the same Euler update/post-processing sequence as the solver but
    caps accepted steps locally.  It is a representative state-evolution check,
    not a replacement for an explicitly requested full physical-horizon run.
    """
    collector = TimingCollector()
    delays = _delays(0.0, "all", "sleep")
    proxy = InstrumentedSurrogate(reconstruct_variant(base, variant), collector, delays)
    model = build_model(proxy, mode)
    with _wrap_model_regions(model, collector):
        model.setup()
        _configure_optimization_path(proxy._source, model, optimization_path)
        collector.reset()
        proxy.work.clear()
        state = _clone_state(model.getCurrentX())
        current_time = 0.0
        start = time.perf_counter_ns()
        accepted = 0
        for _ in range(int(steps)):
            derivative = model.getdXdt(current_time, state)
            dt = model.getDt(derivative)
            state = [np.asarray(value) + np.asarray(rate) * dt for value, rate in zip(state, derivative)]
            current_time += dt
            model.currentTime = current_time
            state, stop = model.postProcess(current_time, state)
            accepted += 1
            if stop:
                break
        elapsed = time.perf_counter_ns() - start
    return {"elapsed_ns": elapsed, "accepted_steps": accepted, "final_time": current_time,
            "timing": collector.summary(), "work": dict(proxy.work)}


def _flatten_samples(samples, variant, mode, optimization_path, delay_category, delay_us):
    rows = []
    for index, sample in enumerate(samples):
        for category, timing in sample["timing"].items():
            delay = sample["delay"]
            rows.append({"variant": variant, "mode": mode, "optimization_path": optimization_path, "delay_category": delay_category,
                         "delay_requested_us": delay_us, "sample": index, "category": category,
                         "step_elapsed_ns": sample["elapsed_ns"],
                         "realized_delay_ns": sum(item["actual_ns"] for item in delay.values()),
            "delay_calls": sum(item["calls"] for item in delay.values()), **timing, **sample["work"], **sample["solver_work"]})
    return rows


def _write_outputs(output, records):
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(records, indent=2, default=_json_default), encoding="utf-8")
    rows = [row for case in records["cases"] for row in case.get("rows", [])]
    if rows:
        keys = sorted({key for row in rows for key in row})
        with (output / "raw_samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader(); writer.writerows(rows)
    summaries = records["summary"]
    if summaries:
        keys = sorted({key for row in summaries for key in row})
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader(); writer.writerows(summaries)
        successful = [row for row in summaries if row.get("status") == "ok" and "median_step_ms" in row]
        if not successful:
            return
        fig, ax = plt.subplots(figsize=(9, 4.5))
        labels = [f"{row['variant']}\n{row['mode']}\n{row['optimization_path']}" for row in successful]
        ax.bar(labels, [row["median_step_ms"] for row in successful])
        ax.set_ylabel("Median fixed-state step time (ms)")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout(); fig.savefig(output / "step_runtime.png", dpi=160); plt.close(fig)
        sensitivity = [row for row in successful if "sensitivity_ns_per_realized_delay_ns" in row]
        if sensitivity:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            labels = [f"{row['variant']}\n{row['mode']}\n{row['delay_category']}" for row in sensitivity]
            ax.bar(labels, [row["sensitivity_ns_per_realized_delay_ns"] for row in sensitivity])
            ax.set_ylabel("Added simulation ns / realized injected ns")
            ax.tick_params(axis="x", labelrotation=45)
            fig.tight_layout(); fig.savefig(output / "latency_sensitivity.png", dpi=160); plt.close(fig)
        native_inference = [row for row in rows if row["delay_requested_us"] == 0.0 and row["category"].startswith("inference.")]
        if native_inference:
            grouped = defaultdict(list)
            for row in native_inference:
                key = (row["variant"], row["mode"])
                grouped[key].append(100.0 * row["exclusive_ns"] / row["step_elapsed_ns"])
            fig, ax = plt.subplots(figsize=(9, 4.5))
            labels = [f"{variant}\n{mode}" for variant, mode in grouped]
            ax.bar(labels, [float(np.median(values)) for values in grouped.values()])
            ax.set_ylabel("Direct surrogate exclusive share of step (%)")
            ax.tick_params(axis="x", labelrotation=45)
            fig.tight_layout(); fig.savefig(output / "inference_share.png", dpi=160); plt.close(fig)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surrogate", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results") / time.strftime("wtife_surrogate_%Y%m%d_%H%M%S"))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--optimization-paths", nargs="+", choices=OPTIMIZATION_PATHS, default=["baseline", "both"],
                        help="Benchmark-local reference/optimization configurations. Include baseline to report speedups.")
    parser.add_argument("--micro-repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--end-time", type=float, default=60.0)
    parser.add_argument("--full-end-to-end", action="store_true",
                        help="Run one native 60 s physical-horizon solve per variant/mode; disabled by default.")
    parser.add_argument("--skip-end-to-end", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--full-example", action="store_true", help="Use the WTiFe example's 5400 s horizon.")
    parser.add_argument("--trajectory-steps", type=int, default=5,
                        help="Native accepted steps per variant/mode; set 0 to disable.")
    parser.add_argument("--delay-us", type=float, nargs="+", default=[0.0],
                        help="Injected delay levels in microseconds; specify nonzero values for latency experiments.")
    parser.add_argument("--delay-categories", nargs="+", choices=("all", "interface_compositions", "interface_diffusivity", "bulk_diffusivity"),
                        default=["interface_compositions", "interface_diffusivity", "bulk_diffusivity"],
                        help="Categories to delay independently; 'all' is an explicitly combined experiment.")
    parser.add_argument("--delay-backend", choices=("sleep", "busy_wait"), default="sleep")
    args = parser.parse_args(argv)
    base = TernaryMovingBoundaryThermodynamicsSurrogate.load(args.surrogate)
    validate_wtife_archive(base)
    report: dict[str, Any] = {"archive": str(args.surrogate), "cases": [], "summary": [],
                               "limitations": ["Internal interpolation, validation, and allocation costs are included in public API timing but cannot be isolated without production instrumentation.",
                                               "The default sleep delay is OS-scheduled; realized elapsed delay is reported per call."]}
    for variant in args.variants:
        for mode in args.modes:
            for optimization_path in args.optimization_paths:
                for delay_category in args.delay_categories:
                    for delay_us in args.delay_us:
                        case = {"variant": variant, "mode": mode, "optimization_path": optimization_path,
                                "delay_category": delay_category, "delay_requested_us": delay_us}
                        try:
                            samples = measure_microbenchmark(base, variant, mode, optimization_path, args.micro_repetitions, args.warmups,
                                                             delay_us, delay_category, args.delay_backend)
                            case["rows"] = _flatten_samples(samples, variant, mode, optimization_path, delay_category, delay_us)
                            elapsed = np.asarray([sample["elapsed_ns"] for sample in samples], dtype=float)
                            realized = np.asarray(
                                [sum(item["actual_ns"] for item in sample["delay"].values()) for sample in samples], dtype=float
                            )
                            summary = {"variant": variant, "mode": mode, "optimization_path": optimization_path,
                                       "delay_requested_us": delay_us, "delay_category": delay_category, "status": "ok",
                                       "median_step_ms": float(np.median(elapsed) / 1e6), "min_step_ms": float(np.min(elapsed) / 1e6),
                                       "max_step_ms": float(np.max(elapsed) / 1e6), "std_step_ms": float(np.std(elapsed) / 1e6),
                                       "sample_count": int(elapsed.size), "median_realized_delay_ms": float(np.median(realized) / 1e6)}
                            summary.update(samples[int(np.argsort(elapsed)[len(elapsed) // 2])]["solver_work"])
                            native_baseline = delay_us == 0.0 and delay_category == args.delay_categories[0]
                            if native_baseline and args.trajectory_steps > 0:
                                trajectory = measure_short_trajectory(base, variant, mode, optimization_path, args.trajectory_steps)
                                summary.update({"trajectory_ms": trajectory["elapsed_ns"] / 1e6,
                                                "trajectory_steps": trajectory["accepted_steps"],
                                                "trajectory_final_time": trajectory["final_time"]})
                                case["trajectory"] = trajectory
                            if native_baseline and args.full_end_to_end and not args.skip_end_to_end:
                                end_time = 5400.0 if args.full_example else args.end_time
                                end = measure_end_to_end(base, variant, mode, optimization_path, end_time, 0.0, "all",
                                                         args.delay_backend)
                                summary.update({"end_to_end_ms": end["elapsed_ns"] / 1e6, "end_to_end_final_time": end["final_time"],
                                                "end_to_end_realized_delay_ms": sum(item["actual_ns"] for item in end["delay"].values()) / 1e6})
                                case["end_to_end"] = end
                            report["summary"].append(summary)
                        except Exception as exc:
                            case["status"] = "failed"; case["error"] = str(exc); case["traceback"] = traceback.format_exc()
                            report["summary"].append({"variant": variant, "mode": mode, "optimization_path": optimization_path,
                                                      "delay_requested_us": delay_us, "delay_category": delay_category,
                                                      "status": "failed", "error": str(exc)})
                        report["cases"].append(case)
    grouped = defaultdict(list)
    for row in report["summary"]:
        if row.get("status") == "ok":
            grouped[(row["variant"], row["mode"], row["optimization_path"], row["delay_category"])].append(row)
    for rows in grouped.values():
        x = np.asarray([row["median_realized_delay_ms"] for row in rows], dtype=float)
        y = np.asarray([row["median_step_ms"] for row in rows], dtype=float)
        if len(rows) >= 2 and np.ptp(x) > 0:
            slope, _ = np.polyfit(x, y, 1)
            for row in rows:
                row["sensitivity_ns_per_realized_delay_ns"] = float(slope)
    baseline_times = {
        (row["variant"], row["mode"], row["delay_category"], row["delay_requested_us"]): row["median_step_ms"]
        for row in report["summary"]
        if row.get("status") == "ok" and row.get("optimization_path") == "baseline"
    }
    for row in report["summary"]:
        key = (row.get("variant"), row.get("mode"), row.get("delay_category"), row.get("delay_requested_us"))
        if row.get("status") == "ok" and key in baseline_times:
            row["speedup_vs_baseline"] = float(baseline_times[key] / row["median_step_ms"])
    _write_outputs(args.output, report)
    for row in report["summary"]:
        if row.get("status") == "ok" and "speedup_vs_baseline" in row:
            print(f"{row['variant']} {row['mode']} {row['optimization_path']}: "
                  f"{row['median_step_ms']:.3f} ms, {row['speedup_vs_baseline']:.2f}x vs baseline")
    print(f"Wrote benchmark report to {args.output}")


if __name__ == "__main__":
    main()
