import numpy as np
import pytest

from kawin.diffusion import DiffusivityDomainError, DiffusivityDomainStatus, InterfaceDiffusivityConstructionError, TernaryMovingBoundaryThermodynamicsSurrogate, merge_phase_diffusivity_surrogates
from kawin.diffusion._diffusivity_domain import DiffusivityDomain, _same_coordinate, _same_coordinate_mask, classify_source_matrix


def _surrogate(validity=None, matrix=None, interface_matrix=None):
    points = np.asarray(((0.10, 0.10), (0.30, 0.10), (0.10, 0.30)))
    matrices = np.asarray([np.eye(2) if matrix is None else matrix for _ in points])
    interface = np.asarray(((0.10, 0.10), (0.20, 0.20), (0.30, 0.30)))
    interface_matrices = (np.asarray([np.eye(2) for _ in interface]) if interface_matrix is None
                          else np.asarray(interface_matrix if np.asarray(interface_matrix).shape == (3, 2, 2)
                                          else [interface_matrix for _ in interface]))
    return TernaryMovingBoundaryThermodynamicsSurrogate(
        elements=("A", "B", "C"), phases=("P", "Q"), tieline_phases=("P", "Q"), temperature=1000.0,
        eta_samples=(0.0, 0.5, 1.0), tieline_compositions={"P": interface, "Q": interface},
        diffusivity_compositions={"interface": {"P": interface, "Q": interface}, "general": {"P": points, "Q": points}},
        diffusivities={"interface": {"P": interface_matrices, "Q": interface_matrices}, "general": {"P": matrices, "Q": matrices}},
        diffusivity_validity=validity,
    )


def test_source_matrix_nonfinite_is_unknown_but_finite_bad_spectrum_is_known_invalid():
    assert classify_source_matrix(np.full((2, 2), np.nan))[0] is DiffusivityDomainStatus.UNKNOWN
    status, usable, reason = classify_source_matrix(np.asarray(((1.0, -2.0), (2.0, 1.0))))
    assert status is DiffusivityDomainStatus.KNOWN_INVALID
    assert not usable
    assert reason == "complex_spectrum"


def test_all_unknown_simplex_is_unknown_and_mixed_simplex_is_ambiguous():
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    unknown = DiffusivityDomain(points, ["UNKNOWN", "UNKNOWN", "UNKNOWN"])
    assert unknown.classify((0.2, 0.2))[0] is DiffusivityDomainStatus.UNKNOWN
    mixed = DiffusivityDomain(points, ["VALID", "UNKNOWN", "VALID"])
    assert mixed.classify((0.2, 0.2))[0] is DiffusivityDomainStatus.BOUNDARY_AMBIGUOUS


def test_valid_but_below_solver_threshold_is_insufficient_fit_support():
    tiny = np.asarray(((1.0, 0.0), (0.0, 1.0e-15)))
    records = {"interface": {}, "general": {}}
    points = np.asarray(((0.10, 0.10), (0.30, 0.10), (0.10, 0.30)))
    interface = np.asarray(((0.10, 0.10), (0.20, 0.20), (0.30, 0.30)))
    for phase in ("P", "Q"):
        records["interface"][phase] = {"coordinates": interface, "statuses": ("VALID",) * 3, "fit_usable": (True,) * 3}
        records["general"][phase] = {"coordinates": points, "statuses": ("VALID",) * 3, "fit_usable": (True,) * 3}
    surrogate = _surrogate(records, matrix=tiny)
    with pytest.raises(DiffusivityDomainError) as error:
        surrogate.getInterdiffusivity((0.10, 0.10), phase="P")
    assert error.value.status is DiffusivityDomainStatus.INSUFFICIENT_FIT_SUPPORT


@pytest.mark.parametrize(
    "status, usable, matrix, expected",
    [
        ("KNOWN_INVALID", False, np.eye(2), "KNOWN_INVALID"),
        ("UNKNOWN", False, np.eye(2), "UNKNOWN"),
        ("VALID", True, np.asarray(((1.0, 0.0), (0.0, 1.0e-15))), "VALID"),
        ("VALID", True, np.full((2, 2), np.nan), "UNKNOWN"),
    ],
)
def test_strict_interface_requires_complete_usable_trajectory(status, usable, matrix, expected):
    interface = np.asarray(((0.10, 0.10), (0.20, 0.20), (0.30, 0.30)))
    points = np.asarray(((0.10, 0.10), (0.30, 0.10), (0.10, 0.30)))
    records = {"interface": {}, "general": {}}
    for phase in ("P", "Q"):
        records["interface"][phase] = {"coordinates": interface, "statuses": ("VALID", status, "VALID"),
                                       "reasons": ("", "synthetic interface failure", ""), "fit_usable": (True, usable, True)}
        records["general"][phase] = {"coordinates": points, "statuses": ("VALID",) * 3, "fit_usable": (True,) * 3}
    matrices = np.asarray([np.eye(2), matrix, np.eye(2)])
    with pytest.raises(InterfaceDiffusivityConstructionError, match=r"phase=P.*index=1.*eta=0.5.*composition=.*reason") as error:
        _surrogate(records, interface_matrix=matrices)
    assert f"domain_status={expected}" in str(error.value)


def test_exact_labeled_invalid_takes_precedence_and_interface_uses_residual_gate():
    points = np.asarray(((0.10, 0.10), (0.30, 0.10), (0.10, 0.30)))
    records = {"interface": {}, "general": {}}
    for phase in ("P", "Q"):
        records["interface"][phase] = {"coordinates": np.asarray(((0.10, 0.10), (0.20, 0.20), (0.30, 0.30))),
                                       "statuses": ("VALID", "VALID", "VALID"), "fit_usable": (True, True, True)}
        records["general"][phase] = {"coordinates": points, "statuses": ("KNOWN_INVALID", "VALID", "VALID"),
                                     "reasons": ("nonpositive_spectrum", "", ""), "fit_usable": (False, True, True)}
    surrogate = _surrogate(records)
    with pytest.raises(DiffusivityDomainError) as error:
        surrogate.getInterdiffusivity((0.10, 0.10), phase="P")
    assert error.value.status is DiffusivityDomainStatus.KNOWN_INVALID
    assert np.allclose(surrogate.getInterdiffusivity((0.20, 0.20), phase="P", query_context="interface"), np.eye(2))
    with pytest.raises(DiffusivityDomainError) as error:
        surrogate.getInterdiffusivity((0.20, 0.201), phase="P", query_context="interface")
    assert error.value.status is DiffusivityDomainStatus.OUTSIDE_SUPPORT


def test_validity_records_round_trip_through_surrogate_archive(tmp_path):
    surrogate = _surrogate()
    path = tmp_path / "domain_surrogate"
    surrogate.save(path)
    loaded = TernaryMovingBoundaryThermodynamicsSurrogate.load(path)
    validity = loaded.diffusivityValidity["general"]["P"]
    assert loaded.validity_policy == "raise"
    assert [status.value for status in validity.statuses] == ["VALID", "VALID", "VALID"]
    assert validity.fit_usable.tolist() == [True, True, True]


def test_merge_preserves_invalid_validity_ledger_hole():
    points = np.asarray(((0.10, 0.10), (0.30, 0.10), (0.10, 0.30)))
    interface = np.asarray(((0.10, 0.10), (0.20, 0.20), (0.30, 0.30)))
    records = {"interface": {}, "general": {}}
    for phase in ("P", "Q"):
        records["interface"][phase] = {"coordinates": interface, "statuses": ("VALID",) * 3, "fit_usable": (True,) * 3}
        records["general"][phase] = {"coordinates": np.vstack((points, (0.18, 0.16))),
                                     "statuses": ("VALID", "VALID", "VALID", "KNOWN_INVALID"),
                                     "reasons": ("", "", "", "complex_spectrum"),
                                     "fit_usable": (True, True, True, False)}
    source = _surrogate(records)
    merged = merge_phase_diffusivity_surrogates(source, source, "P")
    with pytest.raises(DiffusivityDomainError) as error:
        merged.getInterdiffusivity((0.18, 0.16), phase="P")
    assert error.value.status is DiffusivityDomainStatus.KNOWN_INVALID
    assert error.value.reason == "complex_spectrum"


def test_strict_nearest_uses_cached_solver_usable_fit_subset():
    points = np.asarray(((0.05, 0.05), (0.35, 0.05), (0.05, 0.35), (0.35, 0.35), (0.20, 0.20)))
    matrices = np.asarray([np.eye(2), np.eye(2), np.eye(2), np.eye(2), np.diag((9.0, 9.0e-15))])
    interface = np.asarray(((0.10, 0.10), (0.20, 0.20), (0.30, 0.30)))
    interface_matrices = np.asarray([np.eye(2)] * 3)
    records = {"interface": {}, "general": {}}
    for phase in ("P", "Q"):
        records["interface"][phase] = {"coordinates": interface, "statuses": ("VALID",) * 3, "fit_usable": (True,) * 3}
        records["general"][phase] = {"coordinates": points, "statuses": ("VALID",) * 5, "fit_usable": (True,) * 5}
    strict = TernaryMovingBoundaryThermodynamicsSurrogate(
        elements=("A", "B", "C"), phases=("P", "Q"), tieline_phases=("P", "Q"), temperature=1000.0,
        eta_samples=(0.0, 0.5, 1.0), tieline_compositions={"P": interface, "Q": interface},
        diffusivity_compositions={"interface": {"P": interface, "Q": interface}, "general": {"P": points, "Q": points}},
        diffusivities={"interface": {"P": interface_matrices, "Q": interface_matrices}, "general": {"P": matrices, "Q": matrices}},
        diffusivity_validity=records,
    )
    first_geometry = strict._fitSupport["P"]["triangulation"]
    first_points = strict._fitSupport["P"]["points"]
    first_matrices = strict._fitSupport["P"]["matrices"]
    assert np.allclose(strict.getInterdiffusivity((0.21, 0.21), phase="P"), np.eye(2))
    strict._fit_diffusivity_samples = lambda *args: (_ for _ in ()).throw(AssertionError("query rebuilt fit subset"))
    strict.getInterdiffusivity((0.22, 0.22), phase="P")
    assert strict._fitSupport["P"]["triangulation"] is first_geometry
    assert strict._fitSupport["P"]["points"] is first_points
    assert strict._fitSupport["P"]["matrices"] is first_matrices


def test_ulp_duplicate_provenance_is_order_independent_and_conflicts_fail():
    point = np.asarray((0.2, 0.3))
    nearby = np.nextafter(point, np.asarray((1.0, 1.0)))
    first = DiffusivityDomain(np.asarray((point, nearby, (0.5, 0.2), (0.2, 0.5))), ("VALID",) * 4)
    second = DiffusivityDomain(np.asarray((nearby, point, (0.5, 0.2), (0.2, 0.5))), ("VALID",) * 4)
    assert first.coordinates.shape == second.coordinates.shape == (3, 2)
    assert np.array_equal(first.coordinates, second.coordinates)
    informative = DiffusivityDomain(np.asarray((nearby, point, (0.5, 0.2))), ("VALID",) * 3,
                                    reasons=("source_reason", "", ""))
    assert "source_reason" in informative.reasons
    with pytest.raises(ValueError, match="conflicting diffusivity validity labels"):
        DiffusivityDomain(np.asarray((point, nearby, (0.5, 0.2))), ("VALID", "UNKNOWN", "VALID"))


def test_vectorized_exact_lookup_matches_scalar_coordinate_equivalence():
    point = np.asarray((0.2, 0.3))
    coordinates = np.asarray((point, np.nextafter(point, (1.0, 1.0)), (0.2, 0.3001), (0.4, 0.1)))
    expected = np.asarray([_same_coordinate(candidate, point) for candidate in coordinates])
    assert np.array_equal(_same_coordinate_mask(coordinates, point), expected)


def test_batched_general_domain_classification_matches_scalar_reference():
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.4, 0.2)))
    domain = DiffusivityDomain(
        points,
        ("VALID", "KNOWN_INVALID", "VALID", "UNKNOWN"),
        reasons=("", "bad_vertex", "", "unknown_vertex"),
    )
    queries = np.asarray(
        (
            points[1],
            np.nextafter(points[0], (1.0, 1.0)),
            (0.2, 0.2),
            (0.8, 0.1),
            (1.1, 0.1),
            (np.nan, 0.2),
        )
    )
    expected = [domain._classify_scalar_reference(point) for point in queries]
    statuses, reasons = domain.classify_many(queries, chunk_size=2)

    assert list(statuses) == [status for status, _ in expected]
    assert list(reasons) == [reason for _, reason in expected]


def test_batched_strict_bulk_query_matches_scalar_results_and_first_error():
    points = np.asarray(((0.10, 0.10), (0.30, 0.10), (0.10, 0.30), (0.20, 0.20)))
    records = {"interface": {}, "general": {}}
    interface = np.asarray(((0.10, 0.10), (0.20, 0.20), (0.30, 0.30)))
    for phase in ("P", "Q"):
        records["interface"][phase] = {"coordinates": interface, "statuses": ("VALID",) * 3, "fit_usable": (True,) * 3}
        records["general"][phase] = {
            "coordinates": points,
            "statuses": ("VALID", "KNOWN_INVALID", "VALID", "VALID"),
            "reasons": ("", "bad_vertex", "", ""),
            "fit_usable": (True, False, True, True),
        }
    surrogate = _surrogate(records)
    valid = np.asarray(((0.10, 0.10), (0.10, 0.30)))
    expected_support = np.asarray([surrogate._has_general_fit_support_scalar_reference("P", point) for point in valid])
    assert np.array_equal(surrogate._has_general_fit_support_many("P", valid), expected_support)
    expected = np.asarray([surrogate.getInterdiffusivity(point, phase="P") for point in valid])
    actual = surrogate.getInterdiffusivity(valid, phase="P")
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    invalid = np.asarray(((0.12, 0.12), (0.30, 0.10), (1.1, 0.1)))
    with pytest.raises(DiffusivityDomainError) as batched:
        surrogate.getInterdiffusivity(invalid, phase="P")
    with pytest.raises(DiffusivityDomainError) as scalar:
        for point in invalid:
            surrogate.getInterdiffusivity(point, phase="P")
    assert batched.value.status is scalar.value.status
    assert batched.value.reason == scalar.value.reason
    assert np.array_equal(batched.value.composition, scalar.value.composition)


def test_chained_near_duplicate_cluster_fails_deterministically():
    def advance(value, steps):
        for _ in range(steps):
            value = np.nextafter(value, np.inf)
        return value

    a = np.asarray((0.2, 0.3))
    b = np.asarray((advance(a[0], 160), a[1]))
    c = np.asarray((advance(b[0], 160), a[1]))
    assert _same_coordinate(a, b) and _same_coordinate(b, c) and not _same_coordinate(a, c)
    for coordinates in (np.asarray((a, b, c)), np.asarray((c, b, a))):
        with pytest.raises(ValueError, match="ambiguous chained near-duplicate"):
            DiffusivityDomain(coordinates, ("VALID",) * 3)
