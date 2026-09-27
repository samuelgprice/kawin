"""Unit tests for the self-contained WTiFe runtime benchmark helpers."""

import time

import numpy as np
import pytest

from benchmarks.benchmark_wtife_surrogate_runtime import (
    DelayInjector,
    InstrumentedSurrogate,
    TimingCollector,
    _write_outputs,
    reconstruct_variant,
    validate_wtife_archive,
)
from kawin.diffusion.MovingBoundarySurrogates import TernaryMovingBoundaryThermodynamicsSurrogate


def _surrogate():
    phases = ("BCC", "LIQUID")
    points = np.asarray([[0.02, 0.02], [0.20, 0.02], [0.02, 0.20], [0.20, 0.20]])
    matrices = np.repeat(np.asarray([[[1e-12, 0.0], [0.0, 2e-12]]]), len(points), axis=0)
    endpoints = {"BCC": np.asarray([[0.03, 0.02], [0.04, 0.02]]),
                 "LIQUID": np.asarray([[0.50, 0.30], [0.55, 0.30]])}
    interface_points = {phase: values.copy() for phase, values in endpoints.items()}
    interface_matrices = {phase: np.repeat(matrices[:1], 2, axis=0) for phase in phases}
    return TernaryMovingBoundaryThermodynamicsSurrogate(
        elements=("W", "TI", "FE"), phases=phases, tieline_phases=phases, temperature=1973.0,
        eta_samples=np.asarray([0.0, 1.0]), tieline_compositions=endpoints,
        diffusivity_compositions={"interface": interface_points, "general": {phase: points.copy() for phase in phases}},
        diffusivities={"interface": interface_matrices, "general": {phase: matrices.copy() for phase in phases}},
        diffusivity_interpolation="nearest", validity_policy="legacy",
    )


def test_collector_reports_nonnegative_exclusive_time():
    collector = TimingCollector()
    with collector.region("outer"):
        with collector.region("inner"):
            time.sleep(0.001)
    summary = collector.summary()
    assert summary["outer"]["inclusive_ns"] >= summary["inner"]["inclusive_ns"]
    assert summary["outer"]["exclusive_ns"] >= 0


def test_sleep_delay_reports_realized_duration():
    injector = DelayInjector(1000.0, "sleep")
    injector.apply()
    assert injector.calls == 1
    assert injector.actual_ns > 0
    assert injector.record()["actual_ns"] == injector.actual_ns


def test_proxy_counts_api_work_and_matrices():
    source = _surrogate()
    collector = TimingCollector()
    delays = {name: DelayInjector() for name in ("interface_compositions", "interface_diffusivity", "bulk_diffusivity")}
    proxy = InstrumentedSurrogate(source, collector, delays)
    proxy.interface_compositions(0.5)
    output = proxy.getInterdiffusivity(np.asarray([[0.03, 0.03], [0.04, 0.04]]), 1973.0, phase="BCC", query_context="general")
    assert output.shape == (2, 2, 2)
    assert proxy.work["interface_endpoint_compositions"] == 2
    assert proxy.work["bulk_diffusivity_input_compositions"] == 2
    assert proxy.work["bulk_diffusivity_returned_matrices"] == 2


def test_reconstruction_preserves_validity_provenance_and_validation():
    source = _surrogate()
    variant = reconstruct_variant(source, "simplex_linear")
    assert variant.validity_policy == source.validity_policy
    for context in ("interface", "general"):
        for phase in source.tieline_phases:
            assert np.array_equal(variant.diffusivityValidity[context][phase].coordinates,
                                  source.diffusivityValidity[context][phase].coordinates)
            assert np.array_equal(variant.diffusivityValidity[context][phase].fit_usable,
                                  source.diffusivityValidity[context][phase].fit_usable)
    validate_wtife_archive(source)
    source.temperature = 1500.0
    with pytest.raises(ValueError, match="1973"):
        validate_wtife_archive(source)


def test_output_writer_preserves_failed_cases_without_plotting_them(tmp_path):
    report = {
        "cases": [],
        "summary": [{"variant": "nearest", "mode": "phase_uniform", "delay_requested_us": 0.0,
                     "status": "failed", "error": "intentional test failure"}],
    }
    _write_outputs(tmp_path, report)
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "summary.csv").exists()
    assert not (tmp_path / "step_runtime.png").exists()
