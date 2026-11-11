"""Ported from research ``kawin/tests/test_similarity_comparison_basis.py`` at ``7af24f4`` (LF-normalized sha256
``1cbeb810469bc5b0cfa92a109c8ca2ec656e92fa28a1815f476244c99f4fd941``) by
``tools/port_tests.py`` (batch 3); do not edit by hand. Imports resolve to the package as
``tools/definitions/_bindings.py`` does; the full transform list is in
``docs/test-port-provenance.json``. Excluded tests: none.


The similarity comparison helpers must use the solver state basis.

With interstitials the exact solution, closure and matrices are in u-fractions,
while ``getPhysicalPhaseProfiles`` returns mole fractions. These tests use a stub
model holding the exact state, so no solve is involved: a correct comparison gives
zero profile error, and subtracting mole fractions would leave ``u - x``.
"""

from types import SimpleNamespace

import numpy as np

from moving_boundary.analytic import compare_model_to_similarity, solve_similarity_roots
from moving_boundary import CallableTernaryInterfaceEquilibrium

S0 = 0.2
U_LEFT_FAR = np.array([0.002, 0.015])
U_RIGHT_FAR = np.array([0.010, 0.015])
D_LEFT = 1.0e-2 * np.array([[1.0, -0.02], [0.0, 1.0e-2]])
D_RIGHT = 1.0e-3 * np.array([[1.0, -0.05], [0.0, 1.0e-2]])


def _closure():
    return CallableTernaryInterfaceEquilibrium(
        lambda eta: (np.array([0.002, 0.010 + 0.02 * eta]), np.array([0.030 - 0.01 * eta, 0.020 + 0.03 * eta])),
        eta_bounds=(0.0, 1.0),
        composition_basis="u",
    )


def _history(times, values):
    values = np.asarray(values, dtype=np.float64)
    return SimpleNamespace(N=len(times) - 1, _time=np.asarray(times, dtype=np.float64), _y=values)


class _ExactStateModel:
    """Stub exposing the exact u state and its mole-fraction view, like the solver."""

    def __init__(self, solution, times, *, interstitial):
        self.solution = solution
        self.interstitial = interstitial
        self.u = np.linspace(0.0, 1.0, 21)
        self.v = np.linspace(0.0, 1.0, 21)
        self.interfaceData = _history(times, solution.interface_position(times))
        self.etaData = _history(times, np.full(len(times), solution.eta))
        self.inventoryData = _history(times, np.zeros((len(times), 2)))

    def _coordinates(self, time):
        s = float(self.solution.interface_position(time))
        return s * self.u, s + (0.4 - s) * self.v

    def getTransformedState(self, time):
        left, right = self._coordinates(time)
        return self.solution.profile(left, time, phase="left"), self.solution.profile(right, time, phase="right")

    def _full(self, state):
        if not self.interstitial:
            return np.column_stack((1.0 - np.sum(state, axis=1), state))
        x = state / (1.0 + state[:, :1])
        return np.column_stack((1.0 - np.sum(x, axis=1), x))

    def getPhysicalPhaseProfiles(self, time):
        left, right = self._coordinates(time)
        p, q = self.getTransformedState(time)
        return (left, self._full(p)), (right, self._full(q))


def _solution():
    roots = solve_similarity_roots(D_LEFT, D_RIGHT, U_LEFT_FAR, U_RIGHT_FAR, _closure(), s0=S0, k_bounds=(1.0e-6, 0.3), n_k=2001)
    assert len(roots) == 1
    return roots[0]


def test_comparison_uses_state_basis_with_interstitials():
    solution = _solution()
    model = _ExactStateModel(solution, np.geomspace(0.1, 1.0, 8), interstitial=True)
    metrics = compare_model_to_similarity(model, solution)
    assert np.max(metrics["profile_linf"]) < 1.0e-14

    # The mole-fraction view really does differ by u^2 / (1 + u) for carbon, so
    # this test would fail if the helper subtracted it from the exact solution.
    time = float(metrics["profile_times"][-1])
    (x_left, c_left), (x_right, c_right) = model.getPhysicalPhaseProfiles(time)
    exact = solution.profile(x_right, time, phase="right")
    np.testing.assert_allclose(exact[:, 0] - c_right[:, 1], exact[:, 0] ** 2 / (1.0 + exact[:, 0]), rtol=1.0e-12)
    assert np.max(np.abs(exact[:, 0] - c_right[:, 1])) > 1.0e-4


def test_comparison_is_unchanged_for_substitutional_state():
    solution = _solution()
    model = _ExactStateModel(solution, np.geomspace(0.1, 1.0, 8), interstitial=False)
    metrics = compare_model_to_similarity(model, solution)
    assert np.max(metrics["profile_linf"]) < 1.0e-14
    for entry in metrics["profiles"]:
        (_, c_left), (_, c_right) = model.getPhysicalPhaseProfiles(entry["t"])
        np.testing.assert_array_equal(entry["c_left"], c_left[:, 1:])
        np.testing.assert_array_equal(entry["c_right"], c_right[:, 1:])
