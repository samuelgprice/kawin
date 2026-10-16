"""Interstitial (u-fraction) support in the Illingworth moving-boundary solvers."""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from kawin.diffusion import MovingBoundaryIllingworthFD1DModel
from kawin.diffusion._illingworth_interstitials import (
    UFractionDiffusivityProvider,
    UFractionInterfaceEquilibrium,
    composition_bound_violation,
    full_mole_fractions,
    independent_interstitial_mask,
    resolve_diffusivity_provider,
    resolve_interface_equilibrium,
    u_to_x_independent,
    validate_interstitial_element_order,
    x_to_u_independent,
)
from kawin.diffusion.DiffusionParameters import TemperatureParameters
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.solver import explicitEulerIterator
from kawin.thermo.Mobility import expand_u_frac, expand_x_frac, interstitials, u_to_x_frac, x_to_u_frac


def _legacy_simplex_violation(profile, min_comp):
    """The ternary simplex test used before u-fraction support (reference copy)."""
    profile = np.asarray(profile, dtype=np.float64)
    dependent = 1.0 - np.sum(profile, axis=1)
    component_bad = np.argwhere(profile < min_comp)
    if component_bad.size:
        node, component = component_bad[0]
        return f"node {int(node)} component {int(component)} = {profile[node, component]:.6g}"
    dependent_bad = np.flatnonzero(dependent < min_comp)
    if dependent_bad.size:
        node = int(dependent_bad[0])
        return f"node {node} dependent component = {dependent[node]:.6g}"
    return None


# ---------------------------------------------------------------------------
# Shared conversion module
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("elements", [["FE", "C"], ["FE", "C", "MN"], ["FE", "MN", "C"], ["FE", "C", "N"], ["FE", "CR", "NI"]])
def test_u_x_conversion_roundtrip(elements):
    rng = np.random.default_rng(1234)
    n_independent = len(elements) - 1
    x = rng.uniform(0.001, 0.08, size=(50, n_independent))
    full_x = expand_x_frac(x)
    u_reference = x_to_u_frac(full_x, elements, interstitials)[:, 1:]
    u = x_to_u_independent(x if n_independent > 1 else x[:, 0], elements)
    u = np.asarray(u).reshape(50, n_independent)
    np.testing.assert_allclose(u, u_reference, rtol=1e-14, atol=0.0)

    x_reference = u_to_x_frac(expand_u_frac(u, elements, interstitials), elements, interstitials)[:, 1:]
    x_back = np.asarray(u_to_x_independent(u if n_independent > 1 else u[:, 0], elements)).reshape(50, n_independent)
    np.testing.assert_allclose(x_back, x_reference, rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(x_back, x, rtol=1e-14, atol=1e-16)

    np.testing.assert_allclose(full_mole_fractions(u, elements), full_x, rtol=1e-14, atol=1e-16)


def test_binary_conversion_accepts_scalars_and_node_arrays():
    elements = ["FE", "C"]
    assert u_to_x_independent(0.25, elements) == pytest.approx(0.2, rel=1e-15)
    assert x_to_u_independent(0.2, elements) == pytest.approx(0.25, rel=1e-15)
    nodes = np.asarray([0.01, 0.02, 0.05])
    assert u_to_x_independent(nodes, elements).shape == (3,)
    assert x_to_u_independent(nodes[:, None], elements).shape == (3, 1)


def test_bound_violation_matches_legacy_without_interstitials():
    rng = np.random.default_rng(7)
    mask = independent_interstitial_mask(["Z", "X", "Y"])
    assert not np.any(mask)
    for _ in range(200):
        profile = rng.uniform(-0.05, 0.7, size=(6, 2))
        assert composition_bound_violation(profile, mask, 1e-6) == _legacy_simplex_violation(profile, 1e-6)


def test_bound_violation_with_interstitials_allows_large_interstitial_u():
    mask = independent_interstitial_mask(["FE", "C", "MN"])
    assert mask.tolist() == [True, False]
    # u_C + u_Mn > 1 is legal: only substitutional u-fractions share the simplex.
    assert composition_bound_violation(np.asarray([[1.5, 0.3]]), mask, 1e-8) is None
    assert "component 0" in composition_bound_violation(np.asarray([[-1e-3, 0.3]]), mask, 1e-8)
    assert "dependent" in composition_bound_violation(np.asarray([[0.01, 1.0]]), mask, 1e-8)


def test_element_order_guard():
    validate_interstitial_element_order(["FE", "C"])
    validate_interstitial_element_order(["FE", "MN", "C"])
    with pytest.raises(ValueError, match="dependent"):
        validate_interstitial_element_order(["C", "FE"])


def test_interstitial_recognition_is_case_sensitive_like_kawin():
    """Matching agrees with ``DiffusionModel``/``Mobility`` (exact case), and lower-case interstitials are rejected."""
    from kawin.diffusion._illingworth_interstitials import has_interstitials

    assert has_interstitials(["FE", "C"])
    assert not has_interstitials(["FE", "c"])
    assert independent_interstitial_mask(["FE", "c", "MN"]).tolist() == [False, False]
    # The base class treats "c" as substitutional too, so the two never disagree...
    x = np.array([[0.9, 0.1]])
    np.testing.assert_allclose(x_to_u_frac(x, ["fe", "c"], interstitials), x[0])
    # ...but a lower-case interstitial symbol is almost certainly a mistake, so the models reject it.
    with pytest.raises(ValueError, match="upper case"):
        validate_interstitial_element_order(["FE", "c"])
    with pytest.raises(ValueError, match="upper case"):
        validate_interstitial_element_order(["FE", "MN", "n"])
    validate_interstitial_element_order(["fe", "cr"])  # lower-case substitutionals are not affected


def test_binary_model_rejects_lower_case_interstitial_symbol():
    therm = _ConstantBinaryThermodynamics({"ALPHA": 1.0, "BETA": 0.5})
    with pytest.raises(ValueError, match="upper case"):
        _binary_model(["FE", "c"], therm, StepProfile1D(1.0, 0.002, 0.03), (0.001, 0.02))


class _SpyThermodynamics:
    def __init__(self):
        self.calls = []

    def getInterdiffusivity(self, x, T, phase=None, query_context=None):
        self.calls.append((np.array(x, copy=True), query_context))
        return 1.0

    def clearCache(self):
        pass

    marker = "spy"


class _NoContextThermodynamics:
    def getInterdiffusivity(self, x, T, phase=None):
        return 2.0


def test_diffusivity_provider_wrapper_converts_and_forwards():
    spy = _SpyThermodynamics()
    assert resolve_diffusivity_provider(spy, ["NI", "CR"]) is spy
    wrapped = resolve_diffusivity_provider(spy, ["FE", "MN", "C"])
    assert isinstance(wrapped, UFractionDiffusivityProvider)
    wrapped.getInterdiffusivity(np.asarray([0.02, 0.05]), 1000.0, phase="A", query_context="general")
    x_sent, context = spy.calls[-1]
    np.testing.assert_allclose(x_sent, np.asarray([0.02, 0.05]) / 1.05, rtol=1e-15)
    assert context == "general"
    assert wrapped.marker == "spy"
    no_context = UFractionDiffusivityProvider(_NoContextThermodynamics(), ["FE", "C"])
    with pytest.raises(TypeError):
        no_context.getInterdiffusivity(0.01, 1000.0, phase="A", query_context="interface")
    assert no_context.getInterdiffusivity(0.01, 1000.0, phase="A") == 2.0


class _FixedClosure:
    eta_bounds = (0.0, 1.0)

    def __init__(self, left, right, basis=None):
        self.left = np.asarray(left, dtype=np.float64)
        self.right = np.asarray(right, dtype=np.float64)
        if basis is not None:
            self.composition_basis = basis

    def interface_compositions(self, eta):
        return self.left.copy(), self.right.copy()


def test_interface_equilibrium_wrapper_respects_basis():
    closure = _FixedClosure([0.02, 0.01], [0.004, 0.015])
    assert resolve_interface_equilibrium(closure, ["FE", "CR", "NI"]) is closure
    wrapped = resolve_interface_equilibrium(closure, ["FE", "C", "MN"])
    assert isinstance(wrapped, UFractionInterfaceEquilibrium)
    assert wrapped.eta_bounds == (0.0, 1.0)
    left, right = wrapped.interface_compositions(0.3)
    np.testing.assert_allclose(left, np.asarray([0.02, 0.01]) / 0.98, rtol=1e-15)
    np.testing.assert_allclose(right, np.asarray([0.004, 0.015]) / 0.996, rtol=1e-15)
    u_closure = _FixedClosure([0.02, 0.01], [0.004, 0.015], basis="u")
    left_u, _ = resolve_interface_equilibrium(u_closure, ["FE", "C", "MN"]).interface_compositions(0.3)
    np.testing.assert_array_equal(left_u, [0.02, 0.01])
    with pytest.raises(ValueError, match="composition_basis"):
        UFractionInterfaceEquilibrium(_FixedClosure([0.1, 0.1], [0.1, 0.1], basis="w"), ["FE", "C", "MN"])


# ---------------------------------------------------------------------------
# Binary Fe-C model
# ---------------------------------------------------------------------------


class _ConstantBinaryThermodynamics:
    def __init__(self, diffusivities):
        self.diffusivities = diffusivities
        self.calls = []

    def clearCache(self):
        return

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        self.calls.append((phase, np.array(x, dtype=np.float64, copy=True)))
        return np.squeeze(np.ones(np.atleast_1d(np.asarray(x)).shape) * self.diffusivities[phase])


class _VariableBinaryThermodynamics(_ConstantBinaryThermodynamics):
    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        self.calls.append((phase, np.array(x, dtype=np.float64, copy=True)))
        return np.squeeze(self.diffusivities[phase] * (1.0 + 2.0 * np.asarray(x, dtype=np.float64)))


def _binary_model(elements, therm, profile, interface_compositions, *, basis="x", R=5.0, s0=1.0, n_mesh=501, **kwargs):
    mesh = CartesianFD1D([elements[1]], [0.0, R], n_mesh)
    mesh.setResponseProfile(ProfileBuilder([(profile, elements[1])]))
    params = dict(time_step=0.05, phase_a_nodes=60, phase_b_nodes=120, tolerance=1e-10, record=True)
    params.update(kwargs)
    return MovingBoundaryIllingworthFD1DModel(
        mesh,
        elements,
        ["ALPHA", "BETA"],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000.0),
        interfacePosition=s0,
        interface_compositions=interface_compositions,
        interface_composition_basis=basis,
        **params,
    )


def test_binary_interstitial_initial_state_is_u_after_init_and_reset():
    therm = _ConstantBinaryThermodynamics({"ALPHA": 1.0, "BETA": 0.5})
    x_left, x_right = 0.002, 0.03
    model = _binary_model(["FE", "C"], therm, StepProfile1D(1.0, x_left, x_right), (0.001, 0.02))
    expected = np.where(np.ravel(model.mesh.z) < 1.0, x_left / (1 - x_left), x_right / (1 - x_right))
    np.testing.assert_allclose(np.ravel(model.data.currentY), expected, rtol=1e-15)
    np.testing.assert_allclose(model.interfaceCompositions, (0.001 / 0.999, 0.02 / 0.98), rtol=1e-15)
    model.reset()
    np.testing.assert_allclose(np.ravel(model.data.currentY), expected, rtol=1e-15)
    np.testing.assert_allclose(np.ravel(model.data.y(0)), expected, rtol=1e-15)


@pytest.mark.parametrize("mode", ["phase_uniform", "composition_dependent_lagged", "composition_dependent_implicit"])
def test_binary_interstitial_queries_receive_mole_fractions(mode):
    therm = _VariableBinaryThermodynamics({"ALPHA": 1.0, "BETA": 0.5})
    model = _binary_model(
        ["FE", "C"], therm, StepProfile1D(1.0, 0.002, 0.03), (0.001, 0.02), bulk_diffusivity_mode=mode
    )
    model.solve(0.1, iterator=explicitEulerIterator, minDtFrac=1e-14)
    assert therm.calls
    if mode == "phase_uniform":
        sent = {phase: float(np.ravel(x)[0]) for phase, x in therm.calls}
        assert sent["ALPHA"] == pytest.approx(0.001, rel=1e-14)
        assert sent["BETA"] == pytest.approx(0.02, rel=1e-14)
    else:
        therm.calls.clear()
        faces_u = model._left_face_compositions(model._p_curr)
        model._left_face_diffusivities(model._p_curr, model._s_curr, model.currentTime)
        phase, x_sent = therm.calls[-1]
        assert phase == "ALPHA"
        np.testing.assert_allclose(np.ravel(x_sent), faces_u / (1.0 + faces_u), rtol=1e-14)
    profile = np.ravel(model.data.currentY)
    assert np.all(profile > 0.0)
    x_full = model.getCompositions()
    np.testing.assert_allclose(x_full[:, 1], profile / (1.0 + profile), rtol=1e-12)
    np.testing.assert_allclose(np.sum(x_full, axis=1), 1.0, rtol=0.0, atol=1e-14)


def test_binary_label_invariance():
    """Fe-C (interstitial) and Ni-Cr (substitutional) give the same u-fraction run."""
    u_left, u_right = 0.01, 0.04
    interface = (0.005, 0.03)
    runs = {}
    for elements in (["NI", "CR"], ["FE", "C"]):
        therm = _ConstantBinaryThermodynamics({"ALPHA": 1.0, "BETA": 0.5})
        if elements[1] == "C":
            profile = StepProfile1D(1.0, u_left / (1 + u_left), u_right / (1 + u_right))
        else:
            profile = StepProfile1D(1.0, u_left, u_right)
        model = _binary_model(elements, therm, profile, interface, basis="u")
        model.solve(0.5, iterator=explicitEulerIterator, minDtFrac=1e-14)
        runs[elements[1]] = model
    sub, inter = runs["CR"], runs["C"]
    np.testing.assert_allclose(inter.interfaceData._y, sub.interfaceData._y, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(inter._p_curr, sub._p_curr, rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(inter._q_curr, sub._q_curr, rtol=1e-13, atol=1e-16)


def _zener_lambda(u_alpha, u_gamma, u_inf):
    """Solves sqrt(pi) * lam * exp(lam^2) * erfc(lam) = (u_gamma - u_inf) / (u_gamma - u_alpha)."""
    omega = (u_gamma - u_inf) / (u_gamma - u_alpha)

    def residual(lam):
        return math.sqrt(math.pi) * lam * math.exp(lam * lam) * math.erfc(lam) - omega

    return brentq(residual, 1e-8, 20.0, xtol=1e-15)


class _ZenerProfile:
    """Exact Zener u_C profile at time ``t0`` with a flat ferrite of composition ``u_alpha``."""

    def __init__(self, u_alpha, u_gamma, u_inf, lam, D, t0):
        self.u_alpha, self.u_gamma, self.u_inf = u_alpha, u_gamma, u_inf
        self.lam, self.D, self.t0 = lam, D, t0

    def u(self, z):
        z = np.asarray(z, dtype=np.float64)
        s = 2.0 * self.lam * math.sqrt(self.D * self.t0)
        eta = z / (2.0 * math.sqrt(self.D * self.t0))
        from scipy.special import erfc

        gamma = self.u_inf + (self.u_gamma - self.u_inf) * erfc(eta) / math.erfc(self.lam)
        return np.where(z < s, self.u_alpha, gamma)

    def __call__(self, z):
        u = self.u(np.ravel(z))
        return u / (1.0 + u)


def _run_zener(n_v, n_mesh):
    u_alpha, u_gamma, u_inf = 5.0e-4, 0.03, 0.007
    D_gamma, D_alpha = 1.0, 10.0
    lam = _zener_lambda(u_alpha, u_gamma, u_inf)
    t0, t_run, R = 1.0, 1.0, 40.0
    s0 = 2.0 * lam * math.sqrt(D_gamma * t0)
    therm = _ConstantBinaryThermodynamics({"ALPHA": D_alpha, "BETA": D_gamma})
    model = _binary_model(
        ["FE", "C"],
        therm,
        _ZenerProfile(u_alpha, u_gamma, u_inf, lam, D_gamma, t0),
        (u_alpha, u_gamma),
        basis="u",
        R=R,
        s0=s0,
        n_mesh=n_mesh,
        time_step=2.0e-3,
        phase_a_nodes=41,
        transformed_v_grid=np.linspace(0.0, 1.0, n_v) ** 1.5,
        phase_b_nodes=None,
    )
    model.solve(t_run, iterator=explicitEulerIterator, minDtFrac=1e-14)
    s_exact = 2.0 * lam * math.sqrt(D_gamma * (t0 + t_run))
    rel_error = (model.getInterfacePosition() - s_exact) / (s_exact - s0)
    drift = abs(model.getTotalInventory() - model.concData._y[0] * R)
    return rel_error, drift, R


def test_binary_interstitial_zener_parabolic_growth():
    """
    Constant-D Fe-C ferrite growth converges to the u-fraction Zener solution.

    Ferrite starts flat at its interface u_C (zero flux) and austenite at the
    exact erfc profile at t0, so s(t) = 2 lam sqrt(D t). The interface error
    is first order in the transformed spacing (it halves on refinement) and is
    insensitive to dt.
    """
    coarse, coarse_drift, R = _run_zener(801, 8001)
    fine, fine_drift, _ = _run_zener(1601, 16001)
    assert abs(fine) < 3e-3
    assert abs(coarse) / abs(fine) > 1.7
    assert coarse_drift < 1e-10 * R
    assert fine_drift < 1e-10 * R
