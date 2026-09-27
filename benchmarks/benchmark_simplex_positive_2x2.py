"""Compare scalar and vectorized positive-spectrum bulk interpolation.

Run after activating the ``kawin`` environment::

    python benchmarks/benchmark_simplex_positive_2x2.py --surrogate PATH.npz

The benchmark uses identical in-hull query arrays for both evaluators and
fails before timing if their matrices do not agree to the production tolerance.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.benchmark_wtife_surrogate_runtime import reconstruct_variant
from kawin.diffusion.MovingBoundarySurrogates import TernaryMovingBoundaryThermodynamicsSurrogate


def _queries(interpolator, count, seed):
    """Generate deterministic in-hull compositions from sampled simplices."""
    rng = np.random.default_rng(seed)
    simplices = interpolator._triangulation.simplices
    selected = simplices[rng.integers(0, len(simplices), size=count)]
    weights = rng.dirichlet((1.0, 1.0, 1.0), size=count)
    return np.einsum("ni,nij->nj", weights, interpolator.points[selected], optimize=True)


def _time(callable_, values, warmups, repetitions):
    for _ in range(warmups):
        callable_(values)
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        callable_(values)
        samples.append(time.perf_counter_ns() - start)
    return np.asarray(samples, dtype=np.int64)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surrogate", required=True, type=Path)
    parser.add_argument("--phase", action="append", default=None, help="Phase to benchmark; repeat for both phases.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[50, 200, 1000])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results") / time.strftime("simplex_positive_%Y%m%d_%H%M%S"))
    args = parser.parse_args(argv)
    if any(size <= 0 for size in args.batch_sizes) or args.warmups < 0 or args.repetitions <= 0:
        raise ValueError("batch sizes and repetitions must be positive; warmups must be nonnegative.")

    archive = TernaryMovingBoundaryThermodynamicsSurrogate.load(args.surrogate)
    surrogate = reconstruct_variant(archive, "simplex_positive_2x2")
    phases = tuple(args.phase or surrogate.tieline_phases)
    unknown = set(phases).difference(surrogate.tieline_phases)
    if unknown:
        raise ValueError(f"Requested phases are not tie-line phases: {sorted(unknown)}")

    records = []
    for phase_number, phase in enumerate(phases):
        interpolator = surrogate._bulkDiffusivityInterpolators[phase]
        for batch_size in args.batch_sizes:
            values = _queries(interpolator, batch_size, seed=12345 + phase_number * 1000 + batch_size)
            vector = interpolator.evaluate(values)
            scalar = interpolator._evaluate_scalar_reference(values)
            np.testing.assert_allclose(vector, scalar, rtol=4e-12, atol=1e-30)
            vector_samples = _time(interpolator.evaluate, values, args.warmups, args.repetitions)
            scalar_samples = _time(interpolator._evaluate_scalar_reference, values, args.warmups, args.repetitions)
            vector_median = float(np.median(vector_samples))
            scalar_median = float(np.median(scalar_samples))
            records.append({
                "phase": phase,
                "batch_size": int(batch_size),
                "equivalent": True,
                "vector_median_ms": vector_median / 1e6,
                "vector_min_ms": float(np.min(vector_samples)) / 1e6,
                "vector_max_ms": float(np.max(vector_samples)) / 1e6,
                "scalar_median_ms": scalar_median / 1e6,
                "scalar_min_ms": float(np.min(scalar_samples)) / 1e6,
                "scalar_max_ms": float(np.max(scalar_samples)) / 1e6,
                "speedup_scalar_over_vector": scalar_median / vector_median,
                "vector_matrices_per_second": batch_size * 1e9 / vector_median,
                "scalar_matrices_per_second": batch_size * 1e9 / scalar_median,
            })

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for row in records:
        print(
            f"{row['phase']:12s} batch={row['batch_size']:5d} "
            f"vector={row['vector_median_ms']:.3f} ms "
            f"scalar={row['scalar_median_ms']:.3f} ms "
            f"speedup={row['speedup_scalar_over_vector']:.2f}x"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
