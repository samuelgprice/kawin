"""Measure scalar versus batched strict ternary diffusivity-domain validation.

Run after activating the ``kawin`` Conda environment::

    python benchmarks/benchmark_diffusivity_domain.py --batch-sizes 32 256 2048
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

from kawin.diffusion._diffusivity_domain import DiffusivityDomain


def build_domain(seed=0, samples=256):
    """Build a deterministic triangular domain with valid and invalid evidence."""
    rng = np.random.default_rng(seed)
    interior = rng.dirichlet((1.0, 1.0, 1.0), size=max(0, samples - 3))[:, :2]
    points = np.vstack((np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))), interior))
    statuses = np.full(len(points), "VALID", dtype=object)
    statuses[1] = "KNOWN_INVALID"
    reasons = np.full(len(points), "", dtype=object)
    reasons[1] = "benchmark_invalid_vertex"
    return DiffusivityDomain(points, statuses, reasons=reasons)


def build_queries(domain, batch_size, seed=1):
    """Return deterministic in-hull, boundary, exact-invalid, and outside queries."""
    rng = np.random.default_rng(seed)
    values = rng.dirichlet((1.0, 1.0, 1.0), size=batch_size)[:, :2]
    if batch_size:
        values[0] = domain.coordinates[1]
    if batch_size > 1:
        values[1] = (0.5, 0.0)
    if batch_size > 2:
        values[2] = (1.1, 0.1)
    return values


def scalar_reference(domain, queries):
    """Run the retained one-point classifier for benchmark equivalence."""
    result = [domain._classify_scalar_reference(point) for point in queries]
    return np.asarray([status for status, _ in result], dtype=object), np.asarray([reason for _, reason in result], dtype=object)


def _time(callable_, repetitions):
    samples = []
    result = None
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        result = callable_()
        samples.append(time.perf_counter_ns() - start)
    return result, np.asarray(samples, dtype=np.int64)


def measure_case(domain, batch_size, repetitions):
    """Measure both implementations and verify their exact status/reason output."""
    queries = build_queries(domain, batch_size)
    reference, scalar_samples = _time(lambda: scalar_reference(domain, queries), repetitions)
    batched, batch_samples = _time(lambda: domain.classify_many(queries), repetitions)
    equal = bool(np.array_equal(reference[0], batched[0]) and np.array_equal(reference[1], batched[1]))
    scalar_median = float(np.median(scalar_samples))
    batch_median = float(np.median(batch_samples))
    return {
        "batch_size": int(batch_size), "repetitions": int(repetitions), "equivalent": equal,
        "scalar_median_ns": scalar_median, "batched_median_ns": batch_median,
        "scalar_queries_per_second": float(batch_size * 1e9 / scalar_median),
        "batched_queries_per_second": float(batch_size * 1e9 / batch_median),
        "speedup": float(scalar_median / batch_median),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 128, 512, 2048])
    parser.add_argument("--repetitions", type=int, default=21)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results") / "diffusivity_domain.json")
    args = parser.parse_args(argv)
    domain = build_domain(samples=args.samples)
    rows = [measure_case(domain, size, args.repetitions) for size in args.batch_sizes]
    if not all(row["equivalent"] for row in rows):
        raise RuntimeError("batched domain classification did not match the scalar reference.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"batch={row['batch_size']:4d}: {row['speedup']:.2f}x, "
              f"{row['batched_queries_per_second']:.0f} queries/s, equivalent={row['equivalent']}")
    print(f"Wrote {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
