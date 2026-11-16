"""Hand-written tests for the ``phase_mapping`` extension of the saved-surrogate comparator.

The mapping is a package extension beyond the frozen port of
``kawin/diffusion/SurrogateComparison.py`` (see ``docs/relocation-map.json``); the generated port
``tests/test_surrogate_comparison.py`` stays untouched and covers the unmapped behaviour. These
tests check that a relabelled copy compares exactly like an identically labelled one, known
deviations survive the mapping, invalid mappings are refused, bundles are never modified, and the
report is unchanged when no mapping is supplied.
"""
import hashlib

import numpy as np
import pytest

from moving_boundary import (
    SurrogateArtifactBundle,
    TernaryMovingBoundaryThermodynamicsSurrogate,
    compare_saved_surrogates,
    evaluate_saved_surrogate_comparison,
)

PLAIN = ("ALPHA", "BETA")
SUFFIXED = ("ALPHA#1", "BETA#1")
MAPPING = {"ALPHA": "ALPHA#1", "BETA": "BETA#1"}
COUNTS = dict(interface_eta_count=7, bulk_grid_counts=(4, 4))


def _surrogate(*, phases=PLAIN, scale=1.0, shift=0.0, elements=("Z", "X", "Y"), temperature=1000.0,
               bulk_points=None):
    """Synthetic surrogate: linear tie lines and constant per-phase matrices (as in the generated port)."""
    eta = np.asarray([0.0, 0.5, 1.0])
    tielines = {
        phases[0]: np.column_stack((0.20 + 0.10 * eta + shift, 0.10 + 0.02 * eta)),
        phases[1]: np.column_stack((0.08 + 0.05 * eta + shift, 0.24 + 0.04 * eta)),
    }
    if bulk_points is None:
        bulk_points = np.asarray([[0.10, 0.10], [0.10, 0.35], [0.35, 0.10], [0.35, 0.35]])
    bulk_points = np.asarray(bulk_points, dtype=np.float64)
    interface = {p: np.repeat((scale * (i + 1) * np.asarray([[2.0, -0.2], [0.1, 1.5]]))[None], 3, axis=0)
                 for i, p in enumerate(phases)}
    general = {p: np.repeat((scale * (i + 1) * np.asarray([[3.0, -0.1], [0.2, 2.0]]))[None], len(bulk_points), axis=0)
               for i, p in enumerate(phases)}
    return TernaryMovingBoundaryThermodynamicsSurrogate(
        elements=elements, phases=phases, tieline_phases=phases, temperature=temperature, eta_samples=eta,
        tieline_compositions=tielines,
        diffusivity_compositions={"interface": {p: tielines[p].copy() for p in phases},
                                  "general": {p: bulk_points.copy() for p in phases}},
        diffusivities={"interface": interface, "general": general}, validity_policy="legacy",
    )


def _bundle(tmp_path, name, surrogate):
    return SurrogateArtifactBundle.publish(tmp_path / f"{name}.surrogate", surrogate,
                                           {"system": "test", "implementation_sources": {"recipe": "same"}})


def _tree_digest(path):
    return {str(f.relative_to(path)): (hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_mtime_ns)
            for f in sorted(path.rglob("*")) if f.is_file()}


def _assert_same(left, right, path="report", skip=()):
    """Recursive exact comparison; numeric arrays compared by dtype, shape and bit pattern."""
    if path in skip:
        return
    if isinstance(left, dict):
        assert isinstance(right, dict) and list(left) == list(right), path
        for key in left:
            _assert_same(left[key], right[key], f"{path}.{key}", skip)
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_same(a, b, f"{path}[{index}]", skip)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape, path
        if left.dtype == object:
            assert left.tolist() == right.tolist(), path
        else:
            assert left.tobytes() == right.tobytes(), path
    elif isinstance(left, float):
        assert type(left) is type(right) and np.asarray(left).tobytes() == np.asarray(right).tobytes(), path
    else:
        assert type(left) is type(right) and left == right, path


def _strip_mapping_entries(report):
    report = dict(report)
    report.pop("phase_labels")
    for section in ("interface", "bulk"):
        phases = {}
        for phase, entry in report[section]["phases"].items():
            entry = dict(entry)
            assert entry.pop("candidate_phase") == MAPPING[phase]
            phases[phase] = entry
        report[section] = {**report[section], "phases": phases}
    return report


@pytest.mark.parametrize("scale, shift", [(1.0, 0.0), (1.25, 0.0), (1.0, 0.01), (0.8, -0.005)])
def test_mapped_comparison_equals_identically_labelled_comparison(tmp_path, scale, shift):
    reference = _bundle(tmp_path, "reference", _surrogate())
    plain = _bundle(tmp_path, "plain", _surrogate(scale=scale, shift=shift))
    suffixed = _bundle(tmp_path, "suffixed", _surrogate(phases=SUFFIXED, scale=scale, shift=shift))

    unmapped = evaluate_saved_surrogate_comparison(reference.path, plain.path, **COUNTS)
    mapped = evaluate_saved_surrogate_comparison(reference.path, suffixed.path, phase_mapping=MAPPING, **COUNTS)

    assert mapped["phases"] == PLAIN
    assert mapped["phase_labels"] == {
        "mapping": MAPPING, "reference_phases": PLAIN, "candidate_phases": SUFFIXED,
        "reference_tieline_phases": PLAIN, "candidate_tieline_phases": SUFFIXED,
    }
    skip = {"report.candidate.path", "report.candidate.fingerprint", "report.candidate.created_utc",
            "report.reference.created_utc"}
    _assert_same(unmapped, _strip_mapping_entries(mapped), skip=skip)


def test_known_deviations_survive_the_mapping(tmp_path):
    reference = _bundle(tmp_path, "reference", _surrogate())
    scaled = _bundle(tmp_path, "scaled", _surrogate(phases=SUFFIXED, scale=1.25))
    shifted = _bundle(tmp_path, "shifted", _surrogate(phases=SUFFIXED, shift=0.01))

    identical = evaluate_saved_surrogate_comparison(
        reference.path, _bundle(tmp_path, "same", _surrogate(phases=SUFFIXED)).path, phase_mapping=MAPPING, **COUNTS)
    scale_report = evaluate_saved_surrogate_comparison(reference.path, scaled.path, phase_mapping=MAPPING, **COUNTS)
    shift_report = evaluate_saved_surrogate_comparison(reference.path, shifted.path, phase_mapping=MAPPING, **COUNTS)

    for phase in PLAIN:
        assert identical["interface"]["phases"][phase]["own_paths"]["summary"][
            "symmetric_relative_frobenius_error"]["maximum"] == 0.0
        for section, key in (("interface", "own_paths"), ("bulk", "comparison")):
            summary = scale_report[section]["phases"][phase][key]["summary"]["symmetric_relative_frobenius_error"]
            assert summary["maximum"] == pytest.approx(0.2, rel=1e-12)
        np.testing.assert_allclose(shift_report["tieline"]["endpoint_displacement"][phase], 0.01, rtol=1e-12)
    assert identical["tieline"]["summary"]["maximum_endpoint_displacement"] == 0.0


def test_coverage_is_computed_from_the_candidate_own_labels(tmp_path):
    reference = _bundle(tmp_path, "reference", _surrogate())
    narrow = [[0.10, 0.10], [0.10, 0.25], [0.25, 0.10], [0.25, 0.25]]
    plain = _bundle(tmp_path, "plain", _surrogate(bulk_points=narrow))
    suffixed = _bundle(tmp_path, "suffixed", _surrogate(phases=SUFFIXED, bulk_points=narrow))

    unmapped = evaluate_saved_surrogate_comparison(reference.path, plain.path, **COUNTS)
    mapped = evaluate_saved_surrogate_comparison(reference.path, suffixed.path, phase_mapping=MAPPING, **COUNTS)

    assert mapped["bulk"]["summary"] == unmapped["bulk"]["summary"]
    for phase in PLAIN:
        np.testing.assert_array_equal(mapped["bulk"]["phases"][phase]["candidate_training_compositions"], narrow)
        assert mapped["bulk"]["phases"][phase]["coverage"].tolist() == unmapped["bulk"]["phases"][phase]["coverage"].tolist()


@pytest.mark.parametrize("mapping, error, match", [
    ({"ALPHA": "ALPHA#1"}, ValueError, r"missing \['BETA'\]"),
    ({**MAPPING, "GAMMA": "GAMMA#1"}, ValueError, r"not in the reference \['GAMMA'\]"),
    ({"ALPHA": "ALPHA#1", "BETA": "ALPHA#1"}, ValueError, "one-to-one"),
    ({"ALPHA": "BETA#1", "BETA": "ALPHA#1"}, ValueError, "tie-line phases or phase order under phase_mapping"),
    ({"ALPHA": "ALPHA#2", "BETA": "BETA#1"}, ValueError, "tie-line phases or phase order under phase_mapping"),
    ({"ALPHA": "ALPHA", "BETA": "BETA"}, ValueError, "tie-line phases or phase order under phase_mapping"),
    ({"ALPHA": 1, "BETA": "BETA#1"}, TypeError, "strings"),
    ([("ALPHA", "ALPHA#1"), ("BETA", "BETA#1")], TypeError, "must be a mapping"),
])
def test_invalid_mappings_are_refused(tmp_path, mapping, error, match):
    reference = _bundle(tmp_path, "reference", _surrogate())
    candidate = _bundle(tmp_path, "candidate", _surrogate(phases=SUFFIXED))
    with pytest.raises(error, match=match):
        evaluate_saved_surrogate_comparison(reference.path, candidate.path, phase_mapping=mapping, **COUNTS)


def test_suffixes_are_never_stripped_without_a_mapping(tmp_path):
    reference = _bundle(tmp_path, "reference", _surrogate())
    candidate = _bundle(tmp_path, "candidate", _surrogate(phases=SUFFIXED))
    with pytest.raises(ValueError, match="^Cannot compare surrogates with different tie-line phases or phase order.$"):
        evaluate_saved_surrogate_comparison(reference.path, candidate.path, **COUNTS)


@pytest.mark.parametrize("candidate_kwargs, match", [
    (dict(elements=("Z", "Y", "X")), "different element order"),
    (dict(temperature=1001.0), "different temperatures"),
])
def test_element_order_and_temperature_checks_apply_with_a_mapping(tmp_path, candidate_kwargs, match):
    reference = _bundle(tmp_path, "reference", _surrogate())
    candidate = _bundle(tmp_path, "candidate", _surrogate(phases=SUFFIXED, **candidate_kwargs))
    with pytest.raises(ValueError, match=match):
        evaluate_saved_surrogate_comparison(reference.path, candidate.path, phase_mapping=MAPPING, **COUNTS)


def test_bundles_are_not_modified_or_republished(tmp_path):
    reference = _bundle(tmp_path, "reference", _surrogate())
    candidate = _bundle(tmp_path, "candidate", _surrogate(phases=SUFFIXED, scale=1.1))
    before = {path: _tree_digest(path) for path in (reference.path, candidate.path)}
    listing = sorted(p.name for p in tmp_path.iterdir())

    loaded = SurrogateArtifactBundle.load_published(candidate.path)
    evaluate_saved_surrogate_comparison(reference.path, loaded, phase_mapping=MAPPING, **COUNTS)
    with pytest.raises(ValueError):
        evaluate_saved_surrogate_comparison(reference.path, candidate.path, phase_mapping={"ALPHA": "ALPHA#1"})

    assert {path: _tree_digest(path) for path in (reference.path, candidate.path)} == before
    assert sorted(p.name for p in tmp_path.iterdir()) == listing
    assert tuple(loaded.surrogate.phases) == SUFFIXED and tuple(loaded.surrogate.tieline_phases) == SUFFIXED


def test_default_report_has_no_mapping_entries(tmp_path):
    reference = _bundle(tmp_path, "reference", _surrogate())
    candidate = _bundle(tmp_path, "candidate", _surrogate(scale=1.1))
    default = evaluate_saved_surrogate_comparison(reference.path, candidate.path, **COUNTS)
    explicit_none = evaluate_saved_surrogate_comparison(reference.path, candidate.path, phase_mapping=None, **COUNTS)

    assert "phase_labels" not in default
    assert all("candidate_phase" not in entry for section in ("interface", "bulk")
               for entry in default[section]["phases"].values())
    _assert_same(default, explicit_none)
    identity = evaluate_saved_surrogate_comparison(
        reference.path, candidate.path, phase_mapping={"ALPHA": "ALPHA", "BETA": "BETA"}, **COUNTS)
    _assert_same(default, _strip_identity(identity))


def _strip_identity(report):
    report = dict(report)
    assert report.pop("phase_labels")["mapping"] == {"ALPHA": "ALPHA", "BETA": "BETA"}
    for section in ("interface", "bulk"):
        report[section] = {**report[section], "phases": {
            phase: {k: v for k, v in entry.items() if k != "candidate_phase"}
            for phase, entry in report[section]["phases"].items()}}
    return report


def test_compare_saved_surrogates_forwards_the_mapping_and_labels_the_figure(tmp_path):
    pytest.importorskip("plotly")
    reference = _bundle(tmp_path, "reference", _surrogate())
    candidate = _bundle(tmp_path, "candidate", _surrogate(phases=SUFFIXED, scale=1.25))
    result = compare_saved_surrogates(reference.path, candidate.path, renderer=None, phase_mapping=MAPPING, **COUNTS)

    assert result["report"]["phase_labels"]["mapping"] == MAPPING
    table = result["figure"].data[0]
    assert table.cells.values[0][0] == "Phase labels (reference -> candidate)"
    assert table.cells.values[1][0] == "ALPHA -> ALPHA#1, BETA -> BETA#1"
    default = compare_saved_surrogates(
        reference.path, _bundle(tmp_path, "plain", _surrogate(scale=1.25)).path, renderer=None, **COUNTS)
    assert default["figure"].data[0].cells.values[0][0] == "Reference fingerprint"
