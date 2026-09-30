import json

import numpy as np
import pytest

from kawin.diffusion import (
    SurrogateArtifactBundle,
    SurrogateArtifactIntegrityError,
    TernaryMovingBoundaryThermodynamicsSurrogate,
    compare_saved_surrogates,
    evaluate_saved_surrogate_comparison,
    plot_saved_surrogate_comparison,
)


def _surrogate(
    *, scale=1.0, shift=0.0, eta=None, elements=("Z", "X", "Y"),
    phases=("ALPHA", "BETA"), temperature=1000.0, validity_policy="legacy",
    bulk_points=None,
):
    eta = np.asarray([0.0, 0.5, 1.0] if eta is None else eta, dtype=np.float64)
    normalized = (eta - eta[0]) / (eta[-1] - eta[0])
    tielines = {
        phases[0]: np.column_stack((0.20 + 0.10 * normalized + shift, 0.10 + 0.02 * normalized)),
        phases[1]: np.column_stack((0.08 + 0.05 * normalized + shift, 0.24 + 0.04 * normalized)),
    }
    if bulk_points is None:
        bulk_points = np.asarray([[0.10, 0.10], [0.10, 0.35], [0.35, 0.10], [0.35, 0.35]])
    bulk_points = np.asarray(bulk_points, dtype=np.float64)
    interface_matrices = {
        phase: np.repeat((scale * (index + 1) * np.asarray([[2.0, -0.2], [0.1, 1.5]]))[None, :, :], len(eta), axis=0)
        for index, phase in enumerate(phases)
    }
    bulk_matrices = {
        phase: np.repeat((scale * (index + 1) * np.asarray([[3.0, -0.1], [0.2, 2.0]]))[None, :, :], len(bulk_points), axis=0)
        for index, phase in enumerate(phases)
    }
    return TernaryMovingBoundaryThermodynamicsSurrogate(
        elements=elements,
        phases=phases,
        tieline_phases=phases,
        temperature=temperature,
        eta_samples=eta,
        tieline_compositions=tielines,
        diffusivity_compositions={
            "interface": {phase: tielines[phase].copy() for phase in phases},
            "general": {phase: bulk_points.copy() for phase in phases},
        },
        diffusivities={"interface": interface_matrices, "general": bulk_matrices},
        validity_policy=validity_policy,
    )


def _bundle(tmp_path, name, surrogate, source="same", extra_spec=None):
    spec = {"system": "test", "implementation_sources": {"recipe": source}}
    if extra_spec:
        spec.update(extra_spec)
    return SurrogateArtifactBundle.publish(tmp_path / f"{name}.surrogate", surrogate, spec)


def _max_error(report, section, phase="ALPHA"):
    return report[section]["phases"][phase][
        "own_paths" if section == "interface" else "comparison"
    ]["summary"]["symmetric_relative_frobenius_error"]["maximum"]


def test_load_published_accepts_its_own_different_build_specs(tmp_path):
    first = _bundle(tmp_path, "first", _surrogate(), source="old")
    second = _bundle(tmp_path, "second", _surrogate(), source="new")

    loaded_first = SurrogateArtifactBundle.load_published(first.path)
    loaded_second = SurrogateArtifactBundle.load_published(second.path)

    assert loaded_first.manifest["build_fingerprint"] != loaded_second.manifest["build_fingerprint"]
    report = evaluate_saved_surrogate_comparison(first.path, second.path, interface_eta_count=9, bulk_grid_counts=(5, 5))
    assert report["only_implementation_sources_differ"] is True
    assert _max_error(report, "interface") == pytest.approx(0.0)
    assert _max_error(report, "bulk") == pytest.approx(0.0)


def test_load_published_rejects_raw_npz_and_corrupt_bundle(tmp_path):
    raw = tmp_path / "legacy.npz"
    _surrogate().save(raw)
    with pytest.raises(SurrogateArtifactIntegrityError, match="manifest does not exist"):
        SurrogateArtifactBundle.load_published(raw)

    bundle = _bundle(tmp_path, "corrupt", _surrogate())
    model_path = bundle.member_path("model")
    model_path.write_bytes(b"corrupt")
    with pytest.raises(SurrogateArtifactIntegrityError, match="unexpected size|SHA-256"):
        SurrogateArtifactBundle.load_published(bundle.path)


def test_scaled_matrices_have_expected_symmetric_error_and_are_order_independent(tmp_path):
    reference = _bundle(tmp_path, "reference", _surrogate(scale=1.0))
    candidate = _bundle(tmp_path, "candidate", _surrogate(scale=1.2), source="candidate")

    forward = evaluate_saved_surrogate_comparison(reference, candidate, interface_eta_count=7, bulk_grid_counts=4)
    reverse = evaluate_saved_surrogate_comparison(candidate, reference, interface_eta_count=7, bulk_grid_counts=4)

    assert _max_error(forward, "interface") == pytest.approx(1.0 / 6.0)
    assert _max_error(forward, "bulk") == pytest.approx(1.0 / 6.0)
    assert _max_error(reverse, "interface") == pytest.approx(_max_error(forward, "interface"))
    assert _max_error(reverse, "bulk") == pytest.approx(_max_error(forward, "bulk"))


def test_eta_density_change_with_same_predictions_compares_equal(tmp_path):
    reference = _bundle(tmp_path, "coarse", _surrogate(eta=[0.0, 0.5, 1.0]))
    candidate = _bundle(tmp_path, "fine", _surrogate(eta=[0.0, 0.25, 0.5, 0.75, 1.0]), source="fine")

    report = evaluate_saved_surrogate_comparison(reference, candidate, interface_eta_count=21, bulk_grid_counts=(4, 4))

    assert report["tieline"]["summary"]["maximum_endpoint_displacement"] == pytest.approx(0.0)
    assert _max_error(report, "interface") == pytest.approx(0.0)


def test_shifted_tieline_reports_displacement_and_cross_coverage_gaps(tmp_path):
    reference = _bundle(tmp_path, "reference", _surrogate(validity_policy="raise"))
    candidate = _bundle(tmp_path, "shifted", _surrogate(shift=0.01, validity_policy="raise"), source="shifted")

    report = evaluate_saved_surrogate_comparison(reference, candidate, interface_eta_count=11, bulk_grid_counts=4)

    assert report["tieline"]["summary"]["maximum_endpoint_displacement"] == pytest.approx(0.01)
    assert report["interface"]["summary"]["cross_shared_fraction"] < 1.0
    assert report["interface"]["phases"]["ALPHA"]["own_paths"]["summary"]["shared_fraction"] == 1.0


def test_bulk_report_tracks_partial_support_and_sample_origins(tmp_path):
    reference_points = np.asarray([[0.10, 0.10], [0.10, 0.30], [0.30, 0.10], [0.30, 0.30]])
    candidate_points = np.asarray([[0.20, 0.10], [0.20, 0.30], [0.40, 0.10], [0.40, 0.30]])
    reference = _bundle(tmp_path, "reference", _surrogate(validity_policy="raise", bulk_points=reference_points))
    candidate = _bundle(tmp_path, "candidate", _surrogate(validity_policy="raise", bulk_points=candidate_points), source="moved")

    report = evaluate_saved_surrogate_comparison(reference, candidate, interface_eta_count=5, bulk_grid_counts=(5, 5))
    phase = report["bulk"]["phases"]["ALPHA"]

    assert 0.0 < phase["comparison"]["summary"]["shared_fraction"] < 1.0
    assert {"reference training", "candidate training"} <= set(phase["sample_source"])
    assert {"shared", "reference only", "candidate only"} <= set(phase["coverage"])


@pytest.mark.parametrize(
    "candidate, message",
    [
        (_surrogate(elements=("Q", "X", "Y")), "element order"),
        (_surrogate(phases=("ALPHA", "GAMMA")), "phases"),
        (_surrogate(temperature=1001.0), "temperatures"),
    ],
)
def test_incompatible_model_identity_is_rejected(tmp_path, candidate, message):
    reference_bundle = _bundle(tmp_path, "reference", _surrogate())
    candidate_bundle = _bundle(tmp_path, "candidate", candidate, source="candidate")
    with pytest.raises(ValueError, match=message):
        evaluate_saved_surrogate_comparison(reference_bundle, candidate_bundle, interface_eta_count=5, bulk_grid_counts=3)


def test_optional_tolerances_produce_pass_fail_or_no_verdict(tmp_path):
    reference = _bundle(tmp_path, "reference", _surrogate())
    candidate = _bundle(tmp_path, "candidate", _surrogate(scale=1.01), source="candidate")

    no_verdict = evaluate_saved_surrogate_comparison(reference, candidate, interface_eta_count=5, bulk_grid_counts=3)
    passing = evaluate_saved_surrogate_comparison(
        reference, candidate, interface_eta_count=5, bulk_grid_counts=3,
        tolerances={
            "endpoint_composition_max": 1e-12,
            "interface_matrix_symmetric_relative_max": 0.02,
            "bulk_matrix_symmetric_relative_max": 0.02,
            "minimum_bulk_shared_fraction": 1.0,
        },
    )
    failing = evaluate_saved_surrogate_comparison(
        reference, candidate, interface_eta_count=5, bulk_grid_counts=3,
        tolerances={"interface_matrix_symmetric_relative_max": 1e-4},
    )

    assert no_verdict["verdict"]["equivalent"] is None
    assert passing["verdict"]["equivalent"] is True
    assert failing["verdict"]["equivalent"] is False


def test_plot_and_convenience_api_include_all_views_and_matrix_hover(tmp_path):
    reference = _bundle(tmp_path, "reference", _surrogate())
    candidate = _bundle(tmp_path, "candidate", _surrogate(scale=1.1), source="candidate")
    comparison = compare_saved_surrogates(
        reference.path, candidate.path, interface_eta_count=5, bulk_grid_counts=3, renderer=None
    )
    figure = comparison["figure"]
    labels = [button["label"] for button in figure.layout.updatemenus[0].buttons]

    assert labels == ["Overview", "Tie lines", "Interface: ALPHA", "Interface: BETA", "Bulk: ALPHA", "Bulk: BETA"]
    hover_text = " ".join(
        str(value)
        for trace in figure.data
        for value in (() if getattr(trace, "text", None) is None else trace.text)
    )
    assert "Reference:" in hover_text
    assert "Candidate:" in hover_text
    assert "Candidate - reference:" in hover_text
    assert plot_saved_surrogate_comparison(comparison["report"], renderer=None).data


def test_comparison_does_not_mutate_loaded_surrogates(tmp_path):
    reference = SurrogateArtifactBundle.load_published(
        _bundle(tmp_path, "reference", _surrogate()).path
    )
    candidate = SurrogateArtifactBundle.load_published(
        _bundle(tmp_path, "candidate", _surrogate(scale=1.1), source="candidate").path
    )
    before = {
        label: {
            context: {
                phase: values.copy()
                for phase, values in bundle.surrogate.diffusivities[context].items()
            }
            for context in ("interface", "general")
        }
        for label, bundle in (("reference", reference), ("candidate", candidate))
    }

    evaluate_saved_surrogate_comparison(
        reference, candidate, interface_eta_count=5, bulk_grid_counts=3
    )

    for label, bundle in (("reference", reference), ("candidate", candidate)):
        for context in ("interface", "general"):
            for phase, expected in before[label][context].items():
                np.testing.assert_array_equal(
                    bundle.surrogate.diffusivities[context][phase], expected
                )
