"""Compare cached ternary diffusivity surrogate configurations without Thermo-Calc.

Run this file with a dataset created by ``DiffusivityDataset.save``.  The two
factories below are intentionally small placeholders: replace them with
factories that construct the existing nearest, simplex-linear, or future
surrogate configuration from the supplied training subset.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from kawin.diffusion import DiffusivityDataset, SurrogateScheme, make_loo_plan, run_paired_benchmark


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(value.real), "imaginary": float(value.imag)}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON.")


class _NearestPredictor:
    """Simple cached-data baseline used to make this example runnable."""

    def __init__(self, training):
        self.training = training

    def predict(self, compositions, *, phase, context, temperature):
        distances = np.linalg.norm(compositions[:, None, :] - self.training.compositions[None, :, :], axis=2)
        return self.training.matrices[np.argmin(distances, axis=1)]


def _nearest_factory(training):
    """Build a nearest-sample baseline from the supplied common training set."""
    return _NearestPredictor(training)


class _InverseDistancePredictor:
    """A second cached-data configuration for demonstrating paired reports."""

    def __init__(self, training, power=2.0):
        self.training = training
        self.power = float(power)

    def predict(self, compositions, *, phase, context, temperature):
        distances = np.linalg.norm(compositions[:, None, :] - self.training.compositions[None, :, :], axis=2)
        exact = distances == 0.0
        weights = 1.0 / np.maximum(distances, 1e-300) ** self.power
        values = np.einsum("ij,jkl->ikl", weights / weights.sum(axis=1, keepdims=True), self.training.matrices)
        if np.any(exact):
            values[np.any(exact, axis=1)] = self.training.matrices[np.argmax(exact[np.any(exact, axis=1)], axis=1)]
        return values


def _inverse_distance_factory(training):
    """Build an inverse-distance baseline from the supplied common training set."""
    return _InverseDistancePredictor(training)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Path to a DiffusivityDataset .npz archive")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args(argv)

    dataset = DiffusivityDataset.load(args.dataset)
    plan = make_loo_plan(dataset)
    schemes = [
        SurrogateScheme("nearest", _nearest_factory),
        SurrogateScheme("inverse_distance", _inverse_distance_factory),
    ]
    result = run_paired_benchmark(plan, schemes)
    summary = {name: report["summary"] for name, report in result.items()}
    print(json.dumps(summary, indent=2, default=_json_default))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, default=_json_default)


if __name__ == "__main__":
    main()
