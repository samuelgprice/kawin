"""
Analytic similarity solution for a planar ternary three-phase layer A|B|C.

Exact reference for the three-phase ternary Illingworth solver
(``MovingBoundaryIllingworthTernaryThreePhaseFD1DModel``) on asymmetric
problems: a middle phase B grows (or shrinks) between two semi-infinite outer
phases A and C with different far-field compositions, different constant
diffusivity matrices and two independent tie-line closures. Each interface
has its own parabolic rate, so neither interface is a mirror image of the
other. The derivation is written out in
``examples/ternaryExamples/ternaryThreePhaseSimilarity.md``.

Assumptions (as in ``ternaryTwoPhaseSimilarity``)
-------------------------------------------------
* Planar, infinite domain; constant 2x2 diffusivity matrix per phase with real,
  positive, distinct eigenvalues; fluxes ``J = -D dC/dx``.
* Equal, constant molar volumes and no bulk flow, so the two independent
  mole fractions are conserved in a fixed frame.
* Local equilibrium at each interface through a closure
  ``eta -> (c_left(eta), c_right(eta))``: ``closure_ab`` gives ``(c_A, c_B)``
  at ``s_AB`` and ``closure_bc`` gives ``(c_B, c_C)`` at ``s_BC``.
* At ``t = 0`` the middle phase has zero width at ``s0``; afterwards
  ``s_AB = s0 + k_ab sqrt(t)`` and ``s_BC = s0 + k_bc sqrt(t)`` with
  ``k_ab < k_bc``.

Solution
--------
With ``z = (x - s0) / sqrt(t)`` and ``D_p = P_p diag(lam_p) P_p^-1`` each phase
decouples into scalar modes ``w = P_p^-1 C`` obeying ``-z/2 w' = lam w''``::

    A (z < k_ab):  C = c_A_far + P_A [a_A * erfc(-z / (2 sqrt(lam_A)))]
    B (between):   C = P_B [alpha + beta * erf(z / (2 sqrt(lam_B)))]
    C (z > k_bc):  C = c_C_far + P_C [a_C * erfc(+z / (2 sqrt(lam_C)))]

The outer amplitudes match the interface compositions as in the two-phase
solution; the middle phase matches both of its interface compositions, which
fixes ``alpha`` and ``beta`` mode by mode. The four unknowns
``(k_ab, k_bc, eta_ab, eta_bc)`` follow from the four Stefan conditions
``(k/2)(c_right - c_left) = sqrt(t) (J_right - J_left)`` (two solutes at two
interfaces), all of which are time independent.

The solution is exact only while both outer phases look semi-infinite; see
``semi_infinite_validity_time`` in ``ternaryTwoPhaseSimilarity``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf, erfc, erfcx

from examples.ternaryExamples.ternaryTwoPhaseSimilarity import (
    PhaseEigenData,
    _log_erfc,
    _recorded,
    eigen_decompose_diffusivity,
)


# ---------------------------------------------------------------------------
# Phase-wise building blocks
# ---------------------------------------------------------------------------

def _outer_flux(k, b, eig, side):
    """
    Returns ``sqrt(t) * J`` at the interface for an outer (semi-infinite) phase.

    ``b = P^-1 (c_interface - c_far)``. Same scaled-erfc form as the two-phase
    solution: for the left phase the argument is ``-k / (2 sqrt(lam))`` and the
    flux is ``-P g``; for the right phase ``+k / (2 sqrt(lam))`` and ``+P g``,
    with ``g = sqrt(lam / pi) b / erfcx(arg)``.
    """
    sign = -1.0 if side == "left" else 1.0
    arg = sign * float(k) / (2.0 * np.sqrt(eig.lam))
    g = np.sqrt(eig.lam / np.pi) * b / erfcx(arg)
    return sign * (eig.P @ g)


def _erf_difference(a_lo, a_hi):
    """
    Returns ``erf(a_hi) - erf(a_lo)`` (elementwise, ``a_lo < a_hi``) without cancellation.

    When both arguments are positive (or both negative) the difference is
    formed from complementary error functions, which keep their relative
    accuracy far from zero.
    """
    a_lo = np.asarray(a_lo, dtype=np.float64)
    a_hi = np.asarray(a_hi, dtype=np.float64)
    out = erf(a_hi) - erf(a_lo)
    positive = a_lo >= 0.0
    negative = a_hi <= 0.0
    out = np.where(positive, erfc(a_lo) - erfc(a_hi), out)
    out = np.where(negative, erfc(-a_hi) - erfc(-a_lo), out)
    return out


def _middle_terms(k_ab, k_bc, w_ab, w_bc, eig):
    """
    Returns ``sqrt(t) * J`` of the middle phase at both of its interfaces.

    In the eigenbasis ``w(z) = w_ab + (w_bc - w_ab) (E(z) - E_ab) / (E_bc - E_ab)``
    with ``E = erf(z / (2 sqrt(lam)))``, so
    ``sqrt(t) J = -P [sqrt(lam / pi) (w_bc - w_ab) exp(-a^2) / (E_bc - E_ab)]``
    at ``a = k / (2 sqrt(lam))``.
    """
    root_lam = np.sqrt(eig.lam)
    a_ab = float(k_ab) / (2.0 * root_lam)
    a_bc = float(k_bc) / (2.0 * root_lam)
    slope = np.sqrt(eig.lam / np.pi) * (w_bc - w_ab) / _erf_difference(a_ab, a_bc)
    j_ab = -(eig.P @ (slope * np.exp(-a_ab**2)))
    j_bc = -(eig.P @ (slope * np.exp(-a_bc**2)))
    return j_ab, j_bc


def _closure_pair(closure, eta):
    left, right = closure.interface_compositions(float(eta))
    return np.asarray(left, dtype=np.float64).reshape(2), np.asarray(right, dtype=np.float64).reshape(2)


def tie_line_crossing_etas(closure, n_eta=101):
    """
    Returns the sampled etas at which neighbouring tie lines of ``closure`` cross.

    Neighbouring tie lines stay apart when both phase-boundary branches move
    to the same side of the tie line: ``det(dc, c_left') * det(dc, c_right') > 0``
    with ``dc = c_right - c_left`` and ``'`` = d/d eta (central differences).
    Crossing tie lines are thermodynamically impossible, and they make the
    sharp-interface problem ill-posed. For a thin layer of the left phase,
    ``d(w c_left)/dt = v c_right - J_right`` gives
    ``d(delta eta)/dt ~ -det(dc, D_right c_right') / (l w det(dc, c_left')) delta eta``,
    where ``l`` is the length over which the right-phase flux responds. That
    rate grows as ``l`` shrinks, so a numerical model runs away faster on finer
    grids. The check ignores ``D`` (exact for diagonal, isotropic ``D``).
    """
    lo, hi = (float(v) for v in closure.eta_bounds)
    etas = np.linspace(lo, hi, int(n_eta))
    h = 1.0e-6 * (hi - lo)
    crossing = []
    for eta in etas:
        a, b = max(lo, eta - h), min(hi, eta + h)
        left_a, right_a = _closure_pair(closure, a)
        left_b, right_b = _closure_pair(closure, b)
        left, right = _closure_pair(closure, eta)
        dc = right - left
        d_left = (left_b - left_a) / (b - a)
        d_right = (right_b - right_a) / (b - a)
        det_left = dc[0] * d_left[1] - dc[1] * d_left[0]
        det_right = dc[0] * d_right[1] - dc[1] * d_right[0]
        if det_left * det_right < 0.0:
            crossing.append(float(eta))
    return np.asarray(crossing)


def three_phase_similarity_residual(k_ab, k_bc, eta_ab, eta_bc, eig_a, eig_b, eig_c, c_a_far, c_c_far, closure_ab, closure_bc):
    """
    Returns the four time-independent Stefan residuals ``(r_ab, r_bc)`` stacked.

    Each interface contributes ``(k/2)(c_right - c_left) - sqrt(t)(J_right - J_left)``,
    the same sign convention as ``similarity_flux_residual`` in the two-phase
    module. Units are ``sqrt(D)`` times mole fraction.
    """
    c_a, c_b_ab = _closure_pair(closure_ab, eta_ab)
    c_b_bc, c_c = _closure_pair(closure_bc, eta_bc)
    j_a = _outer_flux(k_ab, eig_a.P_inv @ (c_a - c_a_far), eig_a, "left")
    j_c = _outer_flux(k_bc, eig_c.P_inv @ (c_c - c_c_far), eig_c, "right")
    j_b_ab, j_b_bc = _middle_terms(k_ab, k_bc, eig_b.P_inv @ c_b_ab, eig_b.P_inv @ c_b_bc, eig_b)
    r_ab = 0.5 * float(k_ab) * (c_b_ab - c_a) - (j_b_ab - j_a)
    r_bc = 0.5 * float(k_bc) * (c_c - c_b_bc) - (j_c - j_b_bc)
    return np.concatenate([r_ab, r_bc])


# ---------------------------------------------------------------------------
# Solution object
# ---------------------------------------------------------------------------

@dataclass
class TernaryThreePhaseSimilaritySolution:
    """
    Exact planar similarity solution for one root ``(k_ab, k_bc, eta_ab, eta_bc)``.

    ``interface_positions(t)`` returns ``s0 + (k_ab, k_bc) sqrt(t)``; tie lines
    and interface compositions are constant in time. ``profile`` returns the
    two independent mole fractions. Valid for ``t > 0`` on a semi-infinite
    domain.
    """

    eig_a: PhaseEigenData
    eig_b: PhaseEigenData
    eig_c: PhaseEigenData
    c_a_far: np.ndarray
    c_c_far: np.ndarray
    closure_ab: object
    closure_bc: object
    k_ab: float
    k_bc: float
    eta_ab: float
    eta_bc: float
    s0: float = 0.0
    residual_norm: float = np.nan
    interface_compositions_ab: tuple = field(init=False)
    interface_compositions_bc: tuple = field(init=False)

    def __post_init__(self):
        self.c_a_far = np.asarray(self.c_a_far, dtype=np.float64).reshape(2)
        self.c_c_far = np.asarray(self.c_c_far, dtype=np.float64).reshape(2)
        self.interface_compositions_ab = _closure_pair(self.closure_ab, self.eta_ab)
        self.interface_compositions_bc = _closure_pair(self.closure_bc, self.eta_bc)
        c_a, c_b_ab = self.interface_compositions_ab
        c_b_bc, c_c = self.interface_compositions_bc
        self._b_a = self.eig_a.P_inv @ (c_a - self.c_a_far)
        self._b_c = self.eig_c.P_inv @ (c_c - self.c_c_far)
        self._w_b_ab = self.eig_b.P_inv @ c_b_ab
        self._w_b_bc = self.eig_b.P_inv @ c_b_bc

    @property
    def ks(self):
        return np.array([self.k_ab, self.k_bc])

    @property
    def etas(self):
        return np.array([self.eta_ab, self.eta_bc])

    def interface_positions(self, t):
        """Returns ``(n, 2)`` positions ``s0 + (k_ab, k_bc) sqrt(t)`` (shape ``(2,)`` for scalar ``t``)."""
        root_t = np.sqrt(np.asarray(t, dtype=np.float64))
        return self.s0 + root_t[..., None] * self.ks

    def flux_residual(self, ks=None, etas=None):
        """Returns the four Stefan residuals, defaulting to this root."""
        k_ab, k_bc = self.ks if ks is None else ks
        eta_ab, eta_bc = self.etas if etas is None else etas
        return three_phase_similarity_residual(
            k_ab, k_bc, eta_ab, eta_bc, self.eig_a, self.eig_b, self.eig_c,
            self.c_a_far, self.c_c_far, self.closure_ab, self.closure_bc,
        )

    def _phase_profile(self, x, t, phase):
        """Evaluates one phase's formula at all ``x`` (smooth extension past its interfaces)."""
        z = (np.asarray(x, dtype=np.float64).reshape(-1) - self.s0) / np.sqrt(float(t))
        if phase == 1:
            root_lam = np.sqrt(self.eig_b.lam)
            a = z[:, None] / (2.0 * root_lam[None, :])
            a_ab = self.k_ab / (2.0 * root_lam)
            a_bc = self.k_bc / (2.0 * root_lam)
            # (E(z) - E_ab) / (E_bc - E_ab), each difference formed without cancellation.
            fraction = _erf_difference(np.minimum(a, a_ab), np.maximum(a, a_ab)) * np.sign(a - a_ab)
            fraction = fraction / _erf_difference(a_ab, a_bc)[None, :]
            w = self._w_b_ab[None, :] + fraction * (self._w_b_bc - self._w_b_ab)[None, :]
            return w @ self.eig_b.P.T
        if phase == 0:
            eig, b, far, sign, k = self.eig_a, self._b_a, self.c_a_far, -1.0, self.k_ab
        else:
            eig, b, far, sign, k = self.eig_c, self._b_c, self.c_c_far, 1.0, self.k_bc
        root_lam = np.sqrt(eig.lam)
        arg = sign * z[:, None] / (2.0 * root_lam[None, :])
        arg_interface = sign * k / (2.0 * root_lam)
        ratio = np.exp(_log_erfc(arg) - _log_erfc(arg_interface)[None, :])
        return far[None, :] + (ratio * b[None, :]) @ eig.P.T

    def profile(self, x, t, phase=None):
        """
        Returns the ``(n, 2)`` independent mole fractions at positions ``x``.

        ``phase=None`` assigns each point to A (``x < s_AB``), B or C
        (``x >= s_BC``). ``phase`` 0, 1 or 2 evaluates that phase's formula
        everywhere, which is useful when the numerical interfaces differ
        slightly from the exact ones.
        """
        t = float(t)
        if t <= 0.0:
            raise ValueError("The similarity profile is defined for t > 0.")
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if phase in (0, 1, 2):
            return self._phase_profile(x, t, phase)
        if phase is not None:
            raise ValueError("phase must be None, 0, 1 or 2.")
        s_ab, s_bc = self.interface_positions(t)
        out = self._phase_profile(x, t, 1)
        left, right = x < s_ab, x >= s_bc
        if np.any(left):
            out[left] = self._phase_profile(x[left], t, 0)
        if np.any(right):
            out[right] = self._phase_profile(x[right], t, 2)
        return out


# ---------------------------------------------------------------------------
# Root finding
# ---------------------------------------------------------------------------

def solve_three_phase_similarity_roots(
    D_a,
    D_b,
    D_c,
    c_a_far,
    c_c_far,
    closure_ab,
    closure_bc,
    *,
    s0=0.0,
    k_hat_starts=(-3.0, -1.0, -0.3, 0.0, 0.3, 1.0, 3.0),
    width_hat_starts=(0.1, 0.5, 2.0),
    n_eta_starts=3,
    k_hat_max=1.0e2,
    residual_tol=1.0e-10,
    k_scale=None,
):
    """
    Finds similarity roots ``(k_ab, k_bc, eta_ab, eta_bc)`` and returns them as solutions.

    Variables are scaled by ``k_scale``, by default ``sqrt(lambda_ref)``
    (``lambda_ref`` = smallest eigenvalue of the three phases). Pass
    ``k_scale`` of the order of the expected rates when the eigenvalues are
    very disparate (e.g. ``sqrt`` of the fast eigenvalue for a fast
    interstitial next to a slow substitutional solute); the starts and
    ``k_hat_max`` would otherwise not reach the root. The scaled variables are
    ``k_ab_hat`` and the width rate ``k_bc_hat - k_ab_hat``,
    which is kept positive so the middle phase has positive width. The
    residual is scaled by ``sqrt(lambda_ref)`` times the largest composition
    difference among far fields and interface compositions. Every combination
    of ``k_hat_starts``, ``width_hat_starts`` and ``n_eta_starts`` evenly
    spaced interior etas per interface is refined with bounded least squares;
    converged, de-duplicated roots are returned sorted by ``(k_ab, k_bc)``.
    Several roots can exist, as in the two-phase problem.

    Raises ``ValueError`` when either closure has crossing tie lines (see
    :func:`tie_line_crossing_etas`): a similarity root still exists then, but
    it is not a usable reference because the time-dependent problem is
    ill-posed.
    """
    for name, closure in (("closure_ab", closure_ab), ("closure_bc", closure_bc)):
        crossing = tie_line_crossing_etas(closure)
        if crossing.size:
            raise ValueError(f"{name} has crossing tie lines near eta = {crossing[0]:.3g}..{crossing[-1]:.3g}.")
    eig_a = eigen_decompose_diffusivity(D_a, "D_a")
    eig_b = eigen_decompose_diffusivity(D_b, "D_b")
    eig_c = eigen_decompose_diffusivity(D_c, "D_c")
    c_a_far = np.asarray(c_a_far, dtype=np.float64).reshape(2)
    c_c_far = np.asarray(c_c_far, dtype=np.float64).reshape(2)
    bounds_ab = tuple(float(v) for v in closure_ab.eta_bounds)
    bounds_bc = tuple(float(v) for v in closure_bc.eta_bounds)
    root_ref = float(np.sqrt(min(eig_a.lam.min(), eig_b.lam.min(), eig_c.lam.min()))) if k_scale is None else float(k_scale)

    samples = [c_a_far, c_c_far]
    for closure, (lo, hi) in ((closure_ab, bounds_ab), (closure_bc, bounds_bc)):
        for eta in (lo, hi):
            samples.extend(_closure_pair(closure, eta))
    samples = np.asarray(samples)
    comp_scale = max(float(np.max(samples.max(axis=0) - samples.min(axis=0))), 1.0e-300)
    residual_scale = root_ref * comp_scale

    def unpack(p):
        k_ab = p[0] * root_ref
        return k_ab, k_ab + p[1] * root_ref, p[2], p[3]

    def scaled_residual(p):
        # Trial widths at the lower bound can make the middle-phase erf
        # difference underflow to zero; least squares rejects such points.
        with np.errstate(divide="ignore", invalid="ignore"):
            return three_phase_similarity_residual(
                *unpack(p), eig_a, eig_b, eig_c, c_a_far, c_c_far, closure_ab, closure_bc
            ) / residual_scale

    fractions = (np.arange(int(n_eta_starts)) + 0.5) / int(n_eta_starts)
    lower = [-k_hat_max, 1.0e-12, bounds_ab[0], bounds_bc[0]]
    upper = [k_hat_max, 2.0 * k_hat_max, bounds_ab[1], bounds_bc[1]]
    roots = []
    for k_hat in k_hat_starts:
        for width_hat in width_hat_starts:
            for f_ab in fractions:
                for f_bc in fractions:
                    start = np.array([
                        k_hat - 0.5 * width_hat,
                        width_hat,
                        bounds_ab[0] + f_ab * (bounds_ab[1] - bounds_ab[0]),
                        bounds_bc[0] + f_bc * (bounds_bc[1] - bounds_bc[0]),
                    ])
                    try:
                        result = least_squares(scaled_residual, start, bounds=(lower, upper), xtol=1.0e-15, ftol=1.0e-15, gtol=1.0e-15)
                    except (ValueError, FloatingPointError):
                        continue
                    norm = float(np.linalg.norm(result.fun))
                    if not norm < residual_tol:
                        continue
                    k_ab, k_bc, eta_ab, eta_bc = (float(v) for v in unpack(result.x))
                    candidate = np.array([k_ab / root_ref, k_bc / root_ref, eta_ab, eta_bc])
                    if any(np.all(np.abs(candidate - r[0]) <= 1.0e-6 * np.maximum(1.0, np.abs(r[0]))) for r in roots):
                        continue
                    roots.append((candidate, norm))

    roots.sort(key=lambda r: (r[0][0], r[0][1]))
    return [
        TernaryThreePhaseSimilaritySolution(
            eig_a=eig_a, eig_b=eig_b, eig_c=eig_c, c_a_far=c_a_far, c_c_far=c_c_far,
            closure_ab=closure_ab, closure_bc=closure_bc,
            k_ab=float(p[0] * root_ref), k_bc=float(p[1] * root_ref),
            eta_ab=float(p[2]), eta_bc=float(p[3]), s0=float(s0), residual_norm=norm,
        )
        for p, norm in roots
    ]


# ---------------------------------------------------------------------------
# Model-facing utilities
# ---------------------------------------------------------------------------

class SimilarityInitialProfile:
    """
    Callable initial profile: the exact solution at time ``t0`` on a mesh.

    Used with ``ProfileBuilder`` so a three-phase model starts from the exact
    similarity state at ``t0 > 0`` (model time 0 corresponds to similarity time
    ``t0``). The model interpolates the mesh values inside each phase only and
    pins the interface nodes to the tie-line values, so a fine mesh reproduces
    the exact profiles closely.
    """

    def __init__(self, solution, t0):
        self.solution = solution
        self.t0 = float(t0)

    def __call__(self, z):
        return self.solution.profile(np.asarray(z, dtype=np.float64).reshape(-1), self.t0)


def compare_three_phase_model_to_similarity(model, solution, t0, *, t_min=None, profile_times=None):
    """
    Compares a solved three-phase model, started from the exact state at ``t0``, with the solution.

    Model time ``tm`` corresponds to similarity time ``t0 + tm``. Returns a
    dict with, for each interface (axis ``-1`` = ``(AB, BC)``):

    - ``k_model``: least-squares fit of ``s - s0 = k sqrt(t)`` over recorded
      times ``t >= t_min`` (default: ``4 t0``, after the start has been left
      well behind), and ``k_rel_error`` against the exact rates;
    - ``interface_rel_error``: ``max |s - s_exact|`` over the window divided by
      the largest exact displacement from ``s0``;
    - ``eta_max_abs_error`` over the window;
    - ``profile_linf`` at ``profile_times`` (snapped to recorded model times),
      each phase compared with its own formula; requires ``record_pq_data=True``;
    - ``inventory_drift``: final minus initial component inventory.
    """
    tm, s = _recorded(model.interfaceData)
    _, eta = _recorded(model.etaData)
    t = tm + float(t0)
    t_min = 4.0 * float(t0) if t_min is None else float(t_min)
    window = t >= t_min
    if np.count_nonzero(window) < 2:
        raise ValueError("Comparison window contains fewer than two recorded times.")
    root_t = np.sqrt(t[window])
    ds = s[window] - solution.s0
    k_model = (root_t @ ds) / (root_t @ root_t)
    s_exact = solution.interface_positions(t[window])
    scale = max(float(np.max(np.abs(s_exact - solution.s0))), 1.0e-300)
    metrics = {
        "t_min": t_min,
        "t_max": float(t[-1]),
        "k_exact": solution.ks,
        "k_model": k_model,
        "k_rel_error": np.abs(k_model - solution.ks) / np.maximum(np.abs(solution.ks), 1.0e-300),
        "interface_rel_error": np.max(np.abs(s[window] - s_exact), axis=0) / scale,
        "eta_exact": solution.etas,
        "eta_max_abs_error": np.max(np.abs(eta[window] - solution.etas), axis=0),
        "steps": int(model.interfaceData.N),
    }
    if profile_times is None:
        profile_times = np.geomspace(t_min, t[-1], 4)
    linf = []
    for time in np.atleast_1d(profile_times):
        recorded = float(tm[np.argmin(np.abs(t - float(time)))])
        err = []
        for phase, (x, c) in enumerate(model.getPhysicalPhaseProfiles(recorded)):
            err.append(c[:, 1:] - solution.profile(x, recorded + float(t0), phase=phase))
        linf.append(np.max(np.abs(np.vstack(err)), axis=0))
    metrics["profile_linf"] = np.asarray(linf)
    _, inventory = _recorded(model.inventoryData)
    metrics["inventory_drift"] = inventory[-1] - inventory[0]
    return metrics
