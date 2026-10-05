"""
Analytic similarity solution for planar ternary two-phase diffusion couples.

This module provides an exact reference for the ternary Illingworth moving
boundary solver: the self-similar solution of a planar couple of two
semi-infinite phases with constant 2x2 interdiffusivity matrices and a
local-equilibrium tie-line closure ``eta -> (c_left(eta), c_right(eta))``
(Kirkaldy 1958; Coates & Kirkaldy 1971; Maugis et al., Acta Mater. 45 (1997);
Kajihara & Kikuchi, Acta Metall. Mater. 43 (1995) for the diagonal Fe-Cr-Ni
case).

Problem statement and assumptions
---------------------------------
Phase A ("left") occupies ``x < s(t)`` and phase B ("right") ``x > s(t)``.
Each phase obeys ``dC/dt = D_p d2C/dx2`` with constant ``D_p``; fluxes are
``J = -D_p dC/dx`` (row = flux component, column = gradient component, the
same orientation as the ternary Illingworth solver). Compositions are the two
independent mole fractions. Molar volumes are equal and constant and there is
no Kirkendall/bulk flow, so mole fractions are conserved in a fixed frame.
The initial condition is a step at ``s0`` with far-field values
``c_left_far`` and ``c_right_far``.

Diagonalizing ``D_p = P_p diag(lambda_p) P_p^-1`` (real, positive, distinct
eigenvalues are required) decouples each phase into scalar heat equations.
With ``z = (x - s0) / sqrt(t)`` and ``s = s0 + k sqrt(t)``::

    C_A = c_left_far  + P_A [a_A * erfc(-z / (2 sqrt(lambda_A)))]
    C_B = c_right_far + P_B [a_B * erfc(+z / (2 sqrt(lambda_B)))]

where ``a_p`` makes ``C_p(s) = c_p(eta)``. The two unknowns ``(k, eta)``
follow from the two solute Stefan conditions ``v (c_B - c_A) = J_B - J_A``
with ``v = k / (2 sqrt(t))``; every term scales with ``1/sqrt(t)``, leaving a
time-independent 2x2 nonlinear system. More than one root can exist for the
same end compositions (Maugis et al. 1997), so :func:`solve_similarity_roots`
returns all roots it finds.

The solution is exact only while the domain is effectively semi-infinite; see
:func:`semi_infinite_validity_time`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erfc, erfcx


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _log_erfc(x):
    """
    Returns ``log(erfc(x))`` without underflow for large positive ``x``.

    For ``x > 0`` the scaled form ``erfc(x) = erfcx(x) exp(-x^2)`` is used, so
    ratios of complementary error functions stay finite for any interface
    velocity.
    """
    x = np.asarray(x, dtype=np.float64)
    positive = x > 0.0
    out = np.empty_like(x)
    out[positive] = np.log(erfcx(x[positive])) - x[positive] ** 2
    out[~positive] = np.log(erfc(x[~positive]))
    return out


@dataclass(frozen=True)
class PhaseEigenData:
    """
    Eigen-decomposition ``D = P diag(lam) P^-1`` of one constant phase matrix.

    Construct with :func:`eigen_decompose_diffusivity`, which enforces real,
    positive and non-degenerate eigenvalues.
    """

    D: np.ndarray
    lam: np.ndarray
    P: np.ndarray
    P_inv: np.ndarray


def eigen_decompose_diffusivity(D, label="D", degeneracy_rtol=1.0e-6, imag_rtol=1.0e-10):
    """
    Returns the validated eigen-decomposition of a constant 2x2 interdiffusivity.

    The similarity solution diagonalizes each phase, so the eigenvalues must be
    real and strictly positive (well-posed diffusion), and distinct within
    ``degeneracy_rtol`` (otherwise ``P`` is ill-conditioned or defective).
    Raises ``ValueError`` when these assumptions fail.
    """
    D = np.asarray(D, dtype=np.float64)
    if D.shape != (2, 2) or not np.all(np.isfinite(D)):
        raise ValueError(f"{label} must be a finite 2x2 matrix.")
    lam, P = np.linalg.eig(D)
    scale = float(np.max(np.abs(lam)))
    if scale <= 0.0 or np.any(np.abs(np.imag(lam)) > imag_rtol * scale):
        raise ValueError(f"{label} has non-real eigenvalues {lam}; the similarity solution requires real eigenvalues.")
    lam = np.real(lam)
    P = np.real(P)
    if np.any(lam <= 0.0):
        raise ValueError(f"{label} has non-positive eigenvalues {lam}.")
    if abs(lam[0] - lam[1]) <= degeneracy_rtol * scale:
        raise ValueError(f"{label} has (nearly) degenerate eigenvalues {lam}; eigenvector basis is unreliable.")
    order = np.argsort(lam)
    lam = lam[order]
    P = P[:, order]
    return PhaseEigenData(D=D.copy(), lam=lam, P=P, P_inv=np.linalg.inv(P))


def semi_infinite_validity_time(D, distance, n_sigma=4.0):
    """
    Returns the time until a phase of width ``distance`` stops looking semi-infinite.

    Uses the slowest-decaying (largest-eigenvalue) mode: the result is the time
    ``t`` at which ``n_sigma * sqrt(lambda_max * t) == distance``. With
    ``n_sigma = 4`` the erfc tail at the wall is about ``erfc(2) ~ 5e-3`` of the
    interface perturbation. ``distance`` is measured from the interface to the
    zero-flux wall of that phase.
    """
    lam_max = float(np.max(np.real(np.linalg.eigvals(np.asarray(D, dtype=np.float64)))))
    return (float(distance) / float(n_sigma)) ** 2 / lam_max


# ---------------------------------------------------------------------------
# Similarity solution
# ---------------------------------------------------------------------------

def _interface_terms(k, b_left, b_right, eig_left, eig_right):
    """
    Returns ``sqrt(t) * J`` on each side of the interface.

    ``b_p = P_p^-1 (c_p(eta) - c_p_far)`` are the interface jumps in each
    phase's eigenbasis; ``b`` may carry leading batch dimensions with the
    component axis last. Scaled erfc (``erfcx``) keeps the factors
    ``exp(-x^2) / erfc(x)`` finite for any ``k``.
    """
    k = np.asarray(k, dtype=np.float64)[..., None]
    arg_left = -k / (2.0 * np.sqrt(eig_left.lam))
    arg_right = k / (2.0 * np.sqrt(eig_right.lam))
    g_left = np.sqrt(eig_left.lam / np.pi) * b_left / erfcx(arg_left)
    g_right = np.sqrt(eig_right.lam / np.pi) * b_right / erfcx(arg_right)
    j_left = -np.einsum("ij,...j->...i", eig_left.P, g_left)
    j_right = np.einsum("ij,...j->...i", eig_right.P, g_right)
    return j_left, j_right


def similarity_flux_residual(k, eta, eig_left, eig_right, c_left_far, c_right_far, interface_equilibrium):
    """
    Returns the time-independent Stefan residual ``(k/2)(c_B - c_A) - sqrt(t)(J_B - J_A)``.

    A root ``(k, eta)`` of this two-component residual defines the similarity
    solution. Units are those of ``sqrt(D)`` times mole fraction.
    """
    c_left, c_right = interface_equilibrium.interface_compositions(float(eta))
    c_left = np.asarray(c_left, dtype=np.float64).reshape(2)
    c_right = np.asarray(c_right, dtype=np.float64).reshape(2)
    b_left = eig_left.P_inv @ (c_left - np.asarray(c_left_far, dtype=np.float64))
    b_right = eig_right.P_inv @ (c_right - np.asarray(c_right_far, dtype=np.float64))
    j_left, j_right = _interface_terms(k, b_left, b_right, eig_left, eig_right)
    return 0.5 * float(k) * (c_right - c_left) - (j_right - j_left)


@dataclass
class TernaryTwoPhaseSimilaritySolution:
    """
    Exact planar similarity solution for one root ``(k, eta)``.

    ``interface_position(t) = s0 + k sqrt(t)``; the tie line ``eta`` and the
    interface compositions are constant in time. ``profile`` returns the two
    independent mole fractions. Valid for ``t > 0`` on a semi-infinite domain
    (see module docstring for all assumptions).
    """

    eig_left: PhaseEigenData
    eig_right: PhaseEigenData
    c_left_far: np.ndarray
    c_right_far: np.ndarray
    interface_equilibrium: object
    k: float
    eta: float
    s0: float = 0.0
    residual_norm: float = np.nan
    c_left_interface: np.ndarray = field(init=False)
    c_right_interface: np.ndarray = field(init=False)

    def __post_init__(self):
        self.c_left_far = np.asarray(self.c_left_far, dtype=np.float64).reshape(2)
        self.c_right_far = np.asarray(self.c_right_far, dtype=np.float64).reshape(2)
        c_left, c_right = self.interface_equilibrium.interface_compositions(float(self.eta))
        self.c_left_interface = np.asarray(c_left, dtype=np.float64).reshape(2).copy()
        self.c_right_interface = np.asarray(c_right, dtype=np.float64).reshape(2).copy()
        self._b_left = self.eig_left.P_inv @ (self.c_left_interface - self.c_left_far)
        self._b_right = self.eig_right.P_inv @ (self.c_right_interface - self.c_right_far)

    def interface_position(self, t):
        """Returns ``s0 + k sqrt(t)`` for scalar or array ``t >= 0``."""
        return self.s0 + self.k * np.sqrt(np.asarray(t, dtype=np.float64))

    def interface_compositions(self):
        """Returns the constant ``(c_left, c_right)`` interface compositions."""
        return self.c_left_interface.copy(), self.c_right_interface.copy()

    def flux_residual(self, k=None, eta=None):
        """Returns the Stefan residual at ``(k, eta)``, defaulting to this root."""
        return similarity_flux_residual(
            self.k if k is None else k,
            self.eta if eta is None else eta,
            self.eig_left,
            self.eig_right,
            self.c_left_far,
            self.c_right_far,
            self.interface_equilibrium,
        )

    def _phase_profile(self, x, t, side):
        """Evaluates one phase's formula at all ``x`` (smooth extension past ``s``)."""
        z = (np.asarray(x, dtype=np.float64).reshape(-1) - self.s0) / np.sqrt(float(t))
        if side == "left":
            eig, b, far, sign = self.eig_left, self._b_left, self.c_left_far, -1.0
        else:
            eig, b, far, sign = self.eig_right, self._b_right, self.c_right_far, 1.0
        root_lam = np.sqrt(eig.lam)
        arg = sign * z[:, None] / (2.0 * root_lam[None, :])
        arg_interface = sign * self.k / (2.0 * root_lam)
        ratio = np.exp(_log_erfc(arg) - _log_erfc(arg_interface)[None, :])
        return far[None, :] + (ratio * b[None, :]) @ eig.P.T

    def profile(self, x, t, phase=None):
        """
        Returns the ``(n, 2)`` independent mole fractions at positions ``x``.

        ``phase=None`` assigns each point by ``x < s(t)``. ``phase='left'`` or
        ``'right'`` evaluates that phase's formula everywhere, which is useful
        when comparing a numerical profile whose interface differs slightly
        from the exact one.
        """
        t = float(t)
        if t <= 0.0:
            raise ValueError("The similarity profile is defined for t > 0.")
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if phase in ("left", "right"):
            return self._phase_profile(x, t, phase)
        if phase is not None:
            raise ValueError("phase must be None, 'left', or 'right'.")
        s = float(self.interface_position(t))
        out = self._phase_profile(x, t, "right")
        left = x < s
        if np.any(left):
            out[left] = self._phase_profile(x[left], t, "left")
        return out


def solve_similarity_roots(
    D_left,
    D_right,
    c_left_far,
    c_right_far,
    interface_equilibrium,
    *,
    s0=0.0,
    n_eta=201,
    n_k=801,
    k_hat_max=1.0e3,
    k_bounds=None,
    residual_tol=1.0e-10,
):
    """
    Finds all similarity roots ``(k, eta)`` and returns them as solutions.

    The residual is nondimensionalized by ``sqrt(lambda_ref)`` (smallest
    eigenvalue of either phase) and the largest far-field/tie-line composition
    scale, and ``k_hat = k / sqrt(lambda_ref)``. Candidates are local minima of
    the residual norm on an ``eta x k_hat`` grid (``eta`` spans
    ``interface_equilibrium.eta_bounds``; ``k_hat`` is asinh-spaced on
    ``[-k_hat_max, k_hat_max]`` unless ``k_bounds`` gives physical limits), each
    refined with bounded least squares. Converged, de-duplicated roots are
    returned sorted by ``k``. Multiple roots are possible (Maugis et al. 1997);
    the caller decides which one a numerical model should follow. Roots on an
    ``eta`` bound may be artifacts of clipping in the closure.
    """
    eig_left = eigen_decompose_diffusivity(D_left, "D_left")
    eig_right = eigen_decompose_diffusivity(D_right, "D_right")
    c_left_far = np.asarray(c_left_far, dtype=np.float64).reshape(2)
    c_right_far = np.asarray(c_right_far, dtype=np.float64).reshape(2)
    eta_lo, eta_hi = (float(v) for v in interface_equilibrium.eta_bounds)
    if not eta_hi > eta_lo:
        raise ValueError("interface_equilibrium.eta_bounds must be an increasing interval.")

    root_ref = float(np.sqrt(min(eig_left.lam.min(), eig_right.lam.min())))
    eta_grid = np.linspace(eta_lo, eta_hi, int(n_eta))
    c_left_grid = np.empty((eta_grid.size, 2))
    c_right_grid = np.empty((eta_grid.size, 2))
    for i, eta in enumerate(eta_grid):
        c_l, c_r = interface_equilibrium.interface_compositions(float(eta))
        c_left_grid[i] = np.asarray(c_l, dtype=np.float64).reshape(2)
        c_right_grid[i] = np.asarray(c_r, dtype=np.float64).reshape(2)
    comp_scale = float(
        np.max(np.abs(np.concatenate([c_left_grid - c_right_grid, c_left_grid - c_left_far, c_right_grid - c_right_far])))
    )
    comp_scale = max(comp_scale, 1.0e-300)
    residual_scale = root_ref * comp_scale

    if k_bounds is None:
        w = np.linspace(-np.arcsinh(k_hat_max), np.arcsinh(k_hat_max), int(n_k))
        k_hat_grid = np.sinh(w)
        k_hat_lo, k_hat_hi = -float(k_hat_max), float(k_hat_max)
    else:
        k_hat_lo, k_hat_hi = (float(v) / root_ref for v in k_bounds)
        k_hat_grid = np.linspace(k_hat_lo, k_hat_hi, int(n_k))

    b_left = (c_left_grid - c_left_far) @ eig_left.P_inv.T
    b_right = (c_right_grid - c_right_far) @ eig_right.P_inv.T
    k_mesh = k_hat_grid[None, :] * root_ref
    j_left, j_right = _interface_terms(
        np.broadcast_to(k_mesh, (eta_grid.size, k_hat_grid.size)),
        b_left[:, None, :],
        b_right[:, None, :],
        eig_left,
        eig_right,
    )
    residual_grid = 0.5 * k_mesh[..., None] * (c_right_grid - c_left_grid)[:, None, :] - (j_right - j_left)
    norm_grid = np.linalg.norm(residual_grid, axis=-1) / residual_scale

    padded = np.pad(norm_grid, 1, mode="constant", constant_values=np.inf)
    is_min = np.ones_like(norm_grid, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            neighbor = padded[1 + di : 1 + di + norm_grid.shape[0], 1 + dj : 1 + dj + norm_grid.shape[1]]
            is_min &= norm_grid <= neighbor
    candidates = np.argwhere(is_min)

    def scaled_residual(p):
        return similarity_flux_residual(
            p[0] * root_ref, p[1], eig_left, eig_right, c_left_far, c_right_far, interface_equilibrium
        ) / residual_scale

    roots = []
    for i, j in candidates:
        start = np.array([k_hat_grid[j], eta_grid[i]])
        result = least_squares(
            scaled_residual,
            start,
            bounds=([k_hat_lo, eta_lo], [k_hat_hi, eta_hi]),
            xtol=1.0e-15,
            ftol=1.0e-15,
            gtol=1.0e-15,
        )
        norm = float(np.linalg.norm(result.fun))
        if not norm < residual_tol:
            continue
        k_hat, eta = (float(v) for v in result.x)
        duplicate = any(
            abs(k_hat - r[0]) <= 1.0e-6 * max(1.0, abs(r[0])) and abs(eta - r[1]) <= 1.0e-6 * (eta_hi - eta_lo)
            for r in roots
        )
        if not duplicate:
            roots.append((k_hat, eta, norm))

    roots.sort(key=lambda r: r[0])
    return [
        TernaryTwoPhaseSimilaritySolution(
            eig_left=eig_left,
            eig_right=eig_right,
            c_left_far=c_left_far,
            c_right_far=c_right_far,
            interface_equilibrium=interface_equilibrium,
            k=k_hat * root_ref,
            eta=eta,
            s0=float(s0),
            residual_norm=norm,
        )
        for k_hat, eta, norm in roots
    ]


# ---------------------------------------------------------------------------
# Model-facing utilities
# ---------------------------------------------------------------------------

class FixedMatrixTernaryDiffusivity:
    """
    Thermodynamics-like object returning one fixed 2x2 matrix per phase.

    Supplies ``getInterdiffusivity`` for the ternary Illingworth model so bulk
    and interface diffusivities are the same constant matrices the analytic
    solution uses. Composition and query context are ignored; an optional
    ``temperature`` makes the object reject other temperatures.
    """

    def __init__(self, diffusivity_matrices, phases, temperature=None):
        self.phases = list(phases)
        self.temperature = None if temperature is None else float(temperature)
        self.diffusivity_matrices = {}
        for phase in set(self.phases):
            if phase not in diffusivity_matrices:
                raise ValueError(f"Missing fixed diffusivity matrix for phase '{phase}'.")
            matrix = np.asarray(diffusivity_matrices[phase], dtype=np.float64)
            if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"Fixed diffusivity matrix for phase '{phase}' must be finite with shape (2, 2).")
            self.diffusivity_matrices[phase] = matrix.copy()

    def clearCache(self):
        return

    def getInterdiffusivity(self, x, T=None, phase=None, query_context=None, **kwargs):
        """Returns the fixed matrix (tiled for 2D composition input)."""
        if phase is None:
            phase = self.phases[0]
        if phase not in self.diffusivity_matrices:
            raise ValueError(f"Unknown phase '{phase}'. Expected one of {sorted(self.diffusivity_matrices)}.")
        if self.temperature is not None and T is not None:
            if not np.allclose(np.asarray(T, dtype=np.float64), self.temperature, rtol=0.0, atol=1.0e-8):
                raise ValueError(f"Fixed diffusivity object is isothermal at {self.temperature}; received T={T}.")
        matrix = self.diffusivity_matrices[phase]
        values = np.asarray(x, dtype=np.float64)
        if values.ndim <= 1:
            return matrix.copy()
        return np.tile(matrix, (values.shape[0], 1, 1))


def interface_resolved_start_time(model, solution, n_cells=5.0):
    """
    Returns the earliest time at which the near-interface grid resolves both phases.

    The similarity profile has width ``sqrt(lambda_min t)`` in each phase; this
    returns the largest time over the two phases at which that width equals
    ``n_cells`` interface-adjacent transformed cells (evaluated at the initial
    interface position). Earlier times are dominated by the unresolved step.
    Requires a model that has been set up (transformed grids available).
    """
    s0 = float(model.initialInterfacePosition)
    h_left = s0 * (1.0 - float(model._u_grid[-2]))
    h_right = (float(model._R) - s0) * float(model._v_grid[1])
    t_left = (n_cells * h_left) ** 2 / float(solution.eig_left.lam.min())
    t_right = (n_cells * h_right) ** 2 / float(solution.eig_right.lam.min())
    return max(t_left, t_right)


def _recorded(history):
    """Returns the recorded ``(time, values)`` arrays of a model history."""
    n = int(history.N) + 1
    return np.asarray(history._time[:n], dtype=np.float64), np.asarray(history._y[:n], dtype=np.float64)


def snap_to_recorded_times(model, times):
    """
    Returns recorded model times nearest (in log time) to the requested ``times``.

    Profile histories interpolate linearly between records; snapping avoids
    comparing interpolated profiles against the exact solution.
    """
    recorded, _ = _recorded(model.interfaceData)
    positive = recorded[recorded > 0.0]
    out = []
    for t in np.atleast_1d(np.asarray(times, dtype=np.float64)):
        out.append(float(positive[np.argmin(np.abs(np.log(positive) - np.log(t)))]))
    return np.unique(out)


def compare_model_to_similarity(model, solution, *, t_min=None, t_max=None, profile_times=None):
    """
    Compares a solved ternary Illingworth model with a similarity solution.

    Returns a dict with:

    - ``k_model``: least-squares fit of ``s - s0 = k sqrt(t)`` over the window
      ``t_min <= t <= t_max`` (defaults: first positive / last recorded time),
      and ``k_rel_error`` against ``solution.k``;
    - ``interface_rel_error``: ``max |s - s_exact| / max |s_exact - s0|`` over
      the window;
    - ``eta_max_abs_error`` over the window and the recorded ``eta`` history;
    - per-time profile errors (``profile_linf``, ``profile_rms``; shape
      ``(n_times, 2)`` for the two independent components), with each model
      phase compared against that phase's analytic formula so that a small
      interface offset does not produce spurious jumps;
    - ``inventory_drift``: final minus initial component inventory.

    ``profile_times`` are snapped to recorded times (see
    :func:`snap_to_recorded_times`); requires ``record_pq_data=True``.
    """
    t, s = _recorded(model.interfaceData)
    t_eta, eta = _recorded(model.etaData)
    t_min = float(t[t > 0.0][0]) if t_min is None else float(t_min)
    t_max = float(t[-1]) if t_max is None else float(t_max)
    window = (t > 0.0) & (t >= t_min) & (t <= t_max)
    if np.count_nonzero(window) < 2:
        raise ValueError("Comparison window contains fewer than two recorded times.")
    root_t = np.sqrt(t[window])
    ds = s[window] - solution.s0
    k_model = float(np.dot(root_t, ds) / np.dot(root_t, root_t))
    s_exact = solution.interface_position(t[window])
    displacement_scale = max(float(np.max(np.abs(s_exact - solution.s0))), 1.0e-300)
    eta_window = (t_eta > 0.0) & (t_eta >= t_min) & (t_eta <= t_max)

    metrics = {
        "t_min": t_min,
        "t_max": t_max,
        "k_exact": float(solution.k),
        "k_model": k_model,
        "k_rel_error": abs(k_model - solution.k) / max(abs(solution.k), 1.0e-300),
        "interface_rel_error": float(np.max(np.abs(s[window] - s_exact))) / displacement_scale,
        "eta_exact": float(solution.eta),
        "eta_initial_model": float(eta[0]),
        "eta_max_abs_error": float(np.max(np.abs(eta[eta_window] - solution.eta))) if np.any(eta_window) else np.nan,
        "history": {"t": t, "s": s, "t_eta": t_eta, "eta": eta},
    }

    if profile_times is None:
        profile_times = np.geomspace(t_min, t_max, 5)
    profile_times = snap_to_recorded_times(model, profile_times)
    linf, rms, profiles = [], [], []
    for time in profile_times:
        (x_left, c_left), (x_right, c_right) = model.getPhysicalPhaseProfiles(time)
        exact_left = solution.profile(x_left, time, phase="left")
        exact_right = solution.profile(x_right, time, phase="right")
        err = np.vstack([c_left[:, 1:] - exact_left, c_right[:, 1:] - exact_right])
        linf.append(np.max(np.abs(err), axis=0))
        rms.append(np.sqrt(np.mean(err**2, axis=0)))
        profiles.append(
            {"t": float(time), "x_left": x_left, "c_left": c_left[:, 1:], "x_right": x_right, "c_right": c_right[:, 1:]}
        )
    metrics["profile_times"] = profile_times
    metrics["profile_linf"] = np.asarray(linf)
    metrics["profile_rms"] = np.asarray(rms)
    metrics["profiles"] = profiles

    t_inv, inventory = _recorded(model.inventoryData)
    metrics["inventory_drift"] = inventory[-1] - inventory[0]
    metrics["inventory_rel_drift"] = metrics["inventory_drift"] / np.maximum(np.abs(inventory[0]), 1.0e-300)
    return metrics
