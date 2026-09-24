"""Synthetic regression coverage for positive-real 2x2 spectral interpolation."""

import numpy as np
import pytest

from kawin.diffusion.MovingBoundarySurrogates import (
    _BulkDiffusivitySimplexPositive2x2,
    _InterfaceDiffusivitySpectral2x2,
    _positive_2x2_spectral_parameters,
    _reconstruct_positive_2x2,
)
from kawin.diffusion import TernaryMovingBoundaryThermodynamicsSurrogate
from kawin.diffusion.MovingBoundaryIllingworthTernaryFDM import _validate_ternary_diffusivity_matrix


def _matrix(low, high, k=0.0, phi=0.0):
    """Build one admissible nonsymmetric matrix from its spectral fields."""
    m = 0.5 * low + 0.5 * high
    delta = 0.5 * high - 0.5 * low
    rho = np.hypot(delta, k)
    x, y = rho * np.cos(phi), rho * np.sin(phi)
    return np.asarray(((m + x, y + k), (y - k, m - x)), dtype=float)


@pytest.mark.parametrize("ratio", [6e7, 1e12, 1e15])
def test_spectral_round_trip_retains_extremely_separated_eigenvalues(ratio):
    matrix = _matrix(1e-18, ratio * 1e-18, k=0.3 * ratio * 1e-18, phi=1.1)
    parameters, direction = _positive_2x2_spectral_parameters(matrix, "separated")
    result = _reconstruct_positive_2x2(parameters, direction, "separated")
    assert np.allclose(result, matrix, rtol=3e-12, atol=1e-30)
    assert np.allclose(np.sort(np.linalg.eigvals(result).real), [1e-18, ratio * 1e-18], rtol=3e-12)


def test_tolerance_sized_negative_discriminant_is_repeated_root_but_complex_is_rejected():
    repeated = np.asarray(((1.0, 1e-13), (-1e-13, 1.0)), dtype=float)
    parameters, direction = _positive_2x2_spectral_parameters(repeated, "roundoff repeated")
    result = _reconstruct_positive_2x2(parameters, direction, "roundoff repeated")
    assert np.allclose(np.linalg.eigvals(result).imag, 0.0)
    with pytest.raises(ValueError, match="real eigenvalues"):
        _positive_2x2_spectral_parameters(np.asarray(((1.0, 1e-4), (-1e-4, 1.0))), "complex")


def test_barycentric_spectral_interpolation_avoids_entrywise_determinant_crossing():
    first = _matrix(1e-8, 4e-8, k=2e-8, phi=0.0)
    second = _matrix(1e-8, 4e-8, k=2e-8, phi=np.pi)
    third = _matrix(2e-8, 5e-8, k=1e-8, phi=0.8)
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    interpolator = _BulkDiffusivitySimplexPositive2x2(points, np.asarray((first, second, third)))
    result = interpolator.evaluate(np.asarray(((0.5, 0.0),)))[0]
    assert np.all(np.linalg.eigvals(result).real > 0.0)
    assert np.allclose(np.linalg.eigvals(result).imag, 0.0)


def test_scalar_vertices_have_zero_direction_and_interface_stays_positive():
    eta = np.asarray((0.0, 0.5, 1.0))
    endpoint = np.column_stack((eta + 0.1, eta * 0.0 + 0.1))
    matrices = np.asarray((_matrix(1e-8, 1e-8), _matrix(1e-8, 3e-8, 5e-9, 1.2), _matrix(2e-8, 2e-8)))
    interpolator = _InterfaceDiffusivitySpectral2x2(eta, endpoint, matrices)
    values = interpolator.evaluate_eta(np.linspace(0.0, 1.0, 17))
    eigenvalues = np.linalg.eigvals(values)
    assert np.all(np.abs(eigenvalues.imag) < 1e-12)
    assert np.all(eigenvalues.real > 0.0)
    assert np.allclose(interpolator.evaluate_eta(eta), matrices, rtol=2e-12, atol=1e-30)


def test_positive_representability_is_distinct_from_solver_usability():
    tiny_positive = np.diag((1.0, 1e-30))
    _positive_2x2_spectral_parameters(tiny_positive, "tiny positive")
    _validate_ternary_diffusivity_matrix(np.diag((1.0, 2e-14)), "ALPHA")
    for matrix in (np.diag((1.0, 5e-15)), tiny_positive, np.diag((1.0, -1e-30))):
        with pytest.raises(ValueError, match="strictly positive"):
            _validate_ternary_diffusivity_matrix(matrix, "ALPHA")
    with pytest.raises(ValueError, match="strictly positive"):
        _positive_2x2_spectral_parameters(np.diag((1.0, -1e-30)), "negative")


def test_spectral_mode_rejects_unsupported_matrix_shape():
    with pytest.raises(ValueError, match="shape \(n_points, 2, 2\)"):
        _BulkDiffusivitySimplexPositive2x2(np.asarray(((0., 0.), (1., 0.), (0., 1.))), np.ones((3, 3, 3)))


def test_bulk_vertices_random_barycentric_queries_and_global_tied_fallback():
    points = np.asarray(((0., 0.), (1., 0.), (0., 1.)))
    matrices = np.asarray((_matrix(1e-9, 4e-9, 2e-9, 0.), _matrix(1e-9, 4e-9, 2e-9, np.pi),
                           _matrix(2e-9, 8e-9, 6e-9, 1.2)))
    interpolator = _BulkDiffusivitySimplexPositive2x2(points, matrices)
    np.testing.assert_allclose(interpolator.evaluate(points), matrices, rtol=4e-12, atol=5e-24)
    rng = np.random.default_rng(12)
    weights = rng.dirichlet((1., 1., 1.), size=200)
    values = interpolator.evaluate(weights @ points)
    eigenvalues = np.linalg.eigvals(values)
    assert np.all(np.isfinite(values))
    assert np.all(np.abs(eigenvalues.imag) <= 1e-12)
    assert np.all(eigenvalues.real > 0.)
    # At this edge midpoint, source directions 0 and 1 cancel.  The fallback
    # must use global training index 0, not Delaunay's local ordering.
    midpoint = interpolator.evaluate(np.asarray(((.5, 0.),)))[0]
    parameters = .5 * interpolator._parameters[0] + .5 * interpolator._parameters[1]
    expected = _reconstruct_positive_2x2(parameters, interpolator._directions[0], "expected global fallback")
    np.testing.assert_allclose(midpoint, expected, rtol=4e-12, atol=1e-30)


def test_direction_fallback_uses_scale_aware_scores_and_global_ties():
    points = np.asarray(((0., 0.), (1., 0.), (0., 1.)))
    interpolator = _BulkDiffusivitySimplexPositive2x2(points, np.asarray((_matrix(1e-9, 2e-9),) * 3))
    interpolator._directions = np.asarray(((1e-30, 0.), (2e-30, 0.), (3e-30, 0.)))
    selected = interpolator._direction_fallback(np.asarray((0, 2, 1)), np.ones(3))
    np.testing.assert_allclose(selected, [3e-30, 0.])
    interpolator._directions = np.asarray(((0., 2e-30), (1e-30, 0.), (2e-30, 0.)))
    selected = interpolator._direction_fallback(np.asarray((2, 0, 1)), np.ones(3))
    np.testing.assert_allclose(selected, [0., 2e-30])


def test_repeated_defective_and_reconstruction_failure_guards():
    scalar = _matrix(2e-12, 2e-12)
    defective = np.asarray(((2e-12, 7e-12), (0., 2e-12)))
    for matrix in (scalar, defective):
        parameters, direction = _positive_2x2_spectral_parameters(matrix, "repeated")
        restored = _reconstruct_positive_2x2(parameters, direction, "repeated")
        np.testing.assert_allclose(restored, matrix, rtol=5e-12, atol=1e-30)
        assert np.all(np.linalg.eigvals(restored).real > 0.)
    for parameters in ((-1000., 0., 0.), (0., 1000., 0.), (0., 0., 1000.), (0., 0., 710.)):
        with pytest.raises(FloatingPointError):
            _reconstruct_positive_2x2(parameters, (1., 0.), "guard")
    with pytest.raises(FloatingPointError, match="non-finite"):
        _reconstruct_positive_2x2((0., 0., np.inf), (1., 0.), "nonfinite fields")


def _production_spectral_surrogate():
    eta = np.asarray((0., .5, 1.))
    interface_points = np.column_stack((.15 + .1 * eta, .10 + .02 * eta))
    general_points = np.asarray(((.15, .10), (.35, .10), (.15, .30)))
    interface = np.asarray((_matrix(1e-9, 1e-9), _matrix(1e-9, 4e-9, 2e-9, 0.),
                            _matrix(1e-9, 4e-9, 2e-9, np.pi)))
    general = np.asarray((_matrix(1e-9, 4e-9, 2e-9, .2), _matrix(2e-9, 5e-9, 3e-9, 1.),
                          _matrix(1e-9, 7e-9, 4e-9, 2.)))
    return TernaryMovingBoundaryThermodynamicsSurrogate(
        elements=("Z", "X", "Y"), phases=("ALPHA", "BETA"), tieline_phases=("ALPHA", "BETA"), temperature=1000.,
        eta_samples=eta, tieline_compositions={"ALPHA": interface_points, "BETA": interface_points},
        diffusivity_compositions={"interface": {"ALPHA": interface_points, "BETA": interface_points},
                                  "general": {"ALPHA": general_points, "BETA": general_points}},
        diffusivities={"interface": {"ALPHA": interface, "BETA": interface},
                       "general": {"ALPHA": general, "BETA": general}},
        diffusivity_interpolation="simplex_positive_2x2")


def test_public_interface_path_dense_positive_and_save_load_round_trip(tmp_path):
    surrogate = _production_spectral_surrogate()
    eta = np.linspace(0., 1., 51)
    points = np.column_stack((.15 + .1 * eta, .10 + .02 * eta))
    interface = surrogate.getInterdiffusivity(points, 1000., phase="ALPHA", query_context="interface")
    eigenvalues = np.linalg.eigvals(interface)
    assert np.all(np.isfinite(interface))
    assert np.all(np.abs(eigenvalues.imag) <= 1e-12)
    assert np.all(eigenvalues.real > 0.)
    raw_midpoint = .5 * (surrogate.diffusivities["interface"]["ALPHA"][1] + surrogate.diffusivities["interface"]["ALPHA"][2])
    assert np.any(np.abs(np.linalg.eigvals(raw_midpoint).imag) > 1e-12)
    assert np.all(np.linalg.eigvals(surrogate.getInterdiffusivity(points[[37]], 1000., phase="ALPHA", query_context="interface")).real > 0.)
    path = tmp_path / "spectral_surrogate.npz"
    surrogate.save(path)
    loaded = TernaryMovingBoundaryThermodynamicsSurrogate.load(path)
    assert loaded.diffusivityInterpolation == "simplex_positive_2x2"
    for context, query in (("interface", points[[0, 25, -1]]), ("general", np.asarray(((.20, .14),)) )):
        np.testing.assert_allclose(loaded.getInterdiffusivity(query, 1000., phase="ALPHA", query_context=context),
                                   surrogate.getInterdiffusivity(query, 1000., phase="ALPHA", query_context=context), rtol=4e-12, atol=1e-30)
