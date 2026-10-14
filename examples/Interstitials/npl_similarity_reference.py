"""Semi-analytical NPLE parabolic growth constant for Fe-C-X ferrite growth (reference for the benchmarks).

During no-partitioning local-equilibrium (NPLE) growth the spike of the slow
substitutional element X at the interface is far thinner (~D_X/v, sub-nm)
than the carbon diffusion length (~um). Following the classical thin-spike
analysis (Coates 1973; Kirkaldy 1958), the long-range problem then reduces to
a binary-like carbon similarity problem:

1. Operating tie-line: the ferrite end has the alloy X content (X is not
   partitioned into ferrite), u_X^alpha = u_X^alloy.
2. Across the thin spike the carbon flux is continuous and the spike holds
   negligible inventory, so carbon is (to leading order) at uniform chemical
   potential: the carbon content just beyond the spike, u_C*, is the austenite
   composition at the alloy X content with the same carbon activity as the
   interface austenite. As a cross-check u_C* is also computed by integrating
   du_C/du_X = -D_CX/D_CC across the spike with the tabulated DC matrices.
3. Beyond the spike X is uniform, and carbon obeys the 1-D similarity problem
   (eta = z/sqrt(t), s = k sqrt(t)):
       -(eta/2) du/deta = d/deta( D_CC(u, u_X^alloy) du/deta ),
       u(k) = u_C*,  u(inf) = u_C^alloy,
       (k/2) (u_C* - u_C^alpha) = -D_CC du/deta |_{eta=k}.
   The ferrite forms at uniform composition, so it carries no carbon flux.

This uses the same Thermo-Calc thermodynamics and DC tables as both kawin and
DICTRA, but has no spatial or temporal discretization error in the growth
law. Its only approximation is the thin-spike asymptotics, which is excellent
here (spike width / carbon diffusion length ~ 1e-7). It applies to the
parabolic regime (after start-up transients, before soft impingement).

Run (repo root, conda wrapper from AGENTS.md; needs the cached tc-inputs of
the case and Thermo-Calc for the iso-activity step):
    python -u -m examples.Interstitials.npl_similarity_reference ni
    python -u -m examples.Interstitials.npl_similarity_reference mn
"""

from __future__ import annotations

import json
import sys

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import brentq
from scipy.special import erfc

from examples.Interstitials import fe_mn_c_ferrite, fe_ni_c_exh2_le
from examples.Interstitials.ternary_ferrite_growth_le import OUTPUT_DIR, TabulatedTernaryDiffusivity, _tc_system, build_tc_inputs


def u_from_x(x):
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return x / (1.0 - x[:, :1])


def x_from_u(u):
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    return u / (1.0 + u[:, :1])


def operating_tie_line(data):
    """Returns (eta, ferrite u, austenite u) on the tie-line whose ferrite X content equals the alloy's."""
    tie = data["tie_lines"]
    eta = np.asarray(tie["eta"])
    left_u = u_from_x(tie["left"])
    right_u = u_from_x(tie["right"])
    u_alloy = u_from_x(data["x_alloy"])[0]
    mismatch = left_u[:, 1] - u_alloy[1]
    i = int(np.flatnonzero(np.diff(np.sign(mismatch)) != 0)[0])
    w = mismatch[i] / (mismatch[i] - mismatch[i + 1])
    lerp = lambda a: a[i] + w * (a[i + 1] - a[i])
    return float(lerp(eta)), lerp(left_u), lerp(right_u), u_alloy


def far_side_carbon_by_cross_term(D, u_gamma, u_x_alloy):
    """Integrates du_C/du_X = -D_CX/D_CC from the interface austenite to the alloy X content."""

    def rhs(u_x, u_c):
        M = D.getInterdiffusivity(x_from_u([[u_c[0], u_x]])[0], phase="FCC_A1")
        return [-M[0, 1] / M[0, 0]]

    sol = solve_ivp(rhs, (u_gamma[1], u_x_alloy), [u_gamma[0]], rtol=1e-10, atol=1e-14)
    return float(sol.y[0, -1])


def far_side_carbon_by_activity(case, u_gamma, u_x_alloy):
    """Austenite u_C at the alloy X content with the same carbon activity as the interface austenite."""
    from tc_python import ALL_PHASES, TCPython

    with TCPython() as session:
        calc = (
            _tc_system(session, case)
            .with_single_equilibrium_calculation()
            .set_condition("T", case.temperature)
            .set_phase_to_suspended(ALL_PHASES)
            .set_phase_to_entered("FCC_A1", 1.0)
        )

        def activity(u_c, u_x):
            x = x_from_u([[u_c, u_x]])[0]
            calc.set_condition("X(C)", float(x[0]))
            calc.set_condition(f"X({case.solute})", float(x[1]))
            return float(calc.calculate().get_value_of("ACR(C)"))

        target = activity(u_gamma[0], u_gamma[1])
        lo, hi = 0.2 * u_gamma[0], 3.0 * u_gamma[0]
        return float(brentq(lambda u: activity(u, u_x_alloy) - target, lo, hi, xtol=1e-14, rtol=1e-13))


def similarity_growth_constant_shooting(D, u_star, u_alpha_c, u_inf_c, u_x_alloy):
    """
    Solves the carbon similarity problem by shooting; returns k (m/sqrt(s)).

    For a trial k the ODE is integrated outward from the interface with
    u(k) = u_C* and the Stefan flux D_CC du/deta = -(k/2)(u_C* - u_C^alpha);
    k is then root-found so that u reaches u_C^alloy far from the interface.
    Both solution modes (constant and erfc-like decay) stay bounded, so the
    outward integration is stable.
    """

    def D_cc(u_c):
        x = x_from_u([[u_c, u_x_alloy]])
        return float(np.asarray(D.getInterdiffusivity(x[0], phase="FCC_A1"))[0, 0])

    D_ref = D_cc(u_inf_c)
    span = 14.0

    def far_field_mismatch(k_hat):
        def rhs(eta_hat, y):
            u, g = y  # g = (D/D_ref) du/deta_hat
            du = g / (D_cc(u) / D_ref)
            return [du, -0.5 * eta_hat * du]

        y0 = [u_star, -0.5 * k_hat * (u_star - u_alpha_c)]
        sol = solve_ivp(rhs, (k_hat, k_hat + span), y0, rtol=1e-11, atol=1e-15, method="DOP853")
        return float(sol.y[0, -1] - u_inf_c)

    omega = (u_star - u_inf_c) / (u_star - u_alpha_c)
    lam = brentq(lambda l: np.sqrt(np.pi) * l * np.exp(l * l) * erfc(l) - omega, 1e-8, 20.0)
    k0 = 2.0 * lam
    lo, hi = 0.5 * k0, 1.5 * k0
    k_hat = brentq(far_field_mismatch, lo, hi, xtol=1e-13, rtol=1e-13)
    return float(k_hat * np.sqrt(D_ref)), D_ref


def similarity_growth_constant(D, u_star, u_alpha_c, u_inf_c, u_x_alloy):
    """Solves the carbon similarity BVP by collocation with D_CC(u_C) at the alloy X content; returns k (m/sqrt(s))."""

    def D_cc(u_c):
        u_c = np.atleast_1d(u_c)
        x = x_from_u(np.column_stack((u_c, np.full_like(u_c, u_x_alloy))))
        return np.asarray(D.getInterdiffusivity(x, phase="FCC_A1"))[..., 0, 0].reshape(u_c.shape)

    D_ref = float(D_cc(np.asarray([u_inf_c]))[0])
    H = 14.0  # similarity domain length in units of sqrt(D_ref)

    def fun(xi, y, p):
        k_hat = p[0]
        eta_hat = k_hat + xi * H
        d = D_cc(y[0]) / D_ref
        du = H * y[1] / d
        return np.vstack((du, -0.5 * eta_hat * du))

    def bc(ya, yb, p):
        k_hat = p[0]
        return np.asarray([ya[0] - u_star, yb[0] - u_inf_c, 0.5 * k_hat * (u_star - u_alpha_c) + ya[1]])

    # Start from the constant-D Zener solution (D = D_ref): sqrt(pi) lam exp(lam^2) erfc(lam) = Omega,
    # k_hat = 2 lam, with the erfc profile in eta_hat measured from the moving interface.
    omega = (u_star - u_inf_c) / (u_star - u_alpha_c)
    lam = brentq(lambda l: np.sqrt(np.pi) * l * np.exp(l * l) * erfc(l) - omega, 1e-8, 20.0)
    k0 = 2.0 * lam
    xi = np.linspace(0.0, 1.0, 4001)
    eta_hat = k0 + xi * H
    profile = u_inf_c + (u_star - u_inf_c) * erfc(eta_hat / 2.0) / erfc(k0 / 2.0)
    grad = -(u_star - u_inf_c) * np.exp(-(eta_hat / 2.0) ** 2) / (np.sqrt(np.pi) * erfc(k0 / 2.0))
    sol = solve_bvp(fun, bc, xi, np.vstack((profile, grad)), p=[k0], tol=1e-10, max_nodes=500000)
    if not sol.success:
        raise RuntimeError(f"similarity BVP failed: {sol.message}")
    return float(sol.p[0] * np.sqrt(D_ref)), D_ref


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "ni"
    case = {"ni": fe_ni_c_exh2_le.CASE, "mn": fe_mn_c_ferrite.CASE}[key]
    data = build_tc_inputs(case)
    D = TabulatedTernaryDiffusivity(data["tables"])
    eta, u_alpha, u_gamma, u_alloy = operating_tie_line(data)
    u_star_cross = far_side_carbon_by_cross_term(D, u_gamma, u_alloy[1])
    u_star_act = far_side_carbon_by_activity(case, u_gamma, u_alloy[1])
    report = {
        "case": case.name,
        "operating_eta": eta,
        "u_ferrite": u_alpha.tolist(),
        "u_austenite_interface": u_gamma.tolist(),
        "u_alloy": u_alloy.tolist(),
        "u_C_far_side_iso_activity": u_star_act,
        "u_C_far_side_cross_term": u_star_cross,
    }
    for label, u_star in (("iso_activity", u_star_act), ("cross_term", u_star_cross)):
        k, D_ref = similarity_growth_constant_shooting(D, u_star, u_alpha[0], u_alloy[0], u_alloy[1])
        report[f"k_um_per_sqrt_s_{label}"] = k * 1e6
        try:
            k_bvp, _ = similarity_growth_constant(D, u_star, u_alpha[0], u_alloy[0], u_alloy[1])
            report[f"k_um_per_sqrt_s_{label}_collocation_check"] = k_bvp * 1e6
        except RuntimeError as exc:
            report[f"k_um_per_sqrt_s_{label}_collocation_check"] = f"collocation did not converge: {exc}"
    report["D_CC_far_field"] = D_ref
    out = OUTPUT_DIR / f"{case.name}_npl_similarity_reference.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
