# %%
"""
Compare the ternary Illingworth two-phase solver with the exact similarity solution.

For constant 2x2 interdiffusivity matrices, a planar couple of two
semi-infinite phases has an exact self-similar solution (interface at
``s0 + k sqrt(t)``, constant tie line ``eta*``); see
``examples/ternaryExamples/ternaryTwoPhaseSimilarity.py``. This script runs
``MovingBoundaryIllingworthTernaryFD1DModel`` on a Lee & Oh (1996) Fe-Cr-Ni
case (Kajihara, Lim & Kikuchi 1993 couples) with frozen matrices and the real
Fe-Cr-Ni tie-line surrogate, and compares interface motion, tie line,
profiles and inventory against that solution.

The tie-line surrogate is reloaded from the bundle written by
``IllingworthTernaryTwoPhase_FeCrNi_Lee1996Cases.py``; no Thermo-Calc or
pycalphad session is started. The model and the analytic solution always use
identical matrices and the same closure, so any disagreement is numerical.

Default matrices (``MATRIX_SOURCE = "initial_eta"``): the model's own
instantaneous-balance initial tie line ``eta0`` is computed on the comparison
grid with the surrogate's eta-dependent interface diffusivities, and the
interface matrices at ``c_A(eta0)``, ``c_B(eta0)`` are then frozen. ``eta0``
is the solver's discrete t = 0 estimate and generally differs from the exact
``eta*``; the eta(t) plot shows the relaxation between them.

Run the cells in order. Outside ``GEOMETRY = "lee1996_case"`` the domain is
sized so both phases stay effectively semi-infinite up to ``T_END``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "kawin").exists():
            return candidate
    return Path.cwd()


THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
REPO_ROOT = _find_repo_root(THIS_DIR)
EXAMPLES_DIR = REPO_ROOT / "examples"

# Keep this directory off sys.path for the same pycalphad plugin-discovery
# reason documented in IllingworthTernaryTwoPhase_FeCrNi_Lee1996Cases.py.
_this_dir_resolved = THIS_DIR.resolve()


def _is_this_dir_sys_path_entry(entry):
    path = Path.cwd() if entry == "" else Path(entry)
    return path.resolve() == _this_dir_resolved


sys.path[:] = [entry for entry in sys.path if not _is_this_dir_sys_path_entry(entry)]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kawin.diffusion import MovingBoundaryIllingworthTernaryFD1DModel
from kawin.diffusion.SurrogateArtifacts import SurrogateArtifactBundle
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.diffusion.mesh.TransformedGrids import two_phase_landau_grids
from kawin.solver import explicitEulerIterator
from examples.ternaryExamples.ternaryTwoPhaseSimilarity import (
    FixedMatrixTernaryDiffusivity,
    compare_model_to_similarity,
    eigen_decompose_diffusivity,
    interface_resolved_start_time,
    semi_infinite_validity_time,
    solve_similarity_roots,
)

# %%
# Configuration


def _weight_percent_fe_cr_ni_to_atomic_fraction(weight_percent):
    """Convert Fe-Cr-Ni weight percentages to atomic fractions in [Fe, Cr, Ni] order.

    Atomic masses match the Lee Fe-Cr-Ni TDB (copied from the Lee1996 script).
    """
    weights = np.asarray(weight_percent, dtype=np.float64)
    atomic_masses = np.array([55.847, 51.996, 58.69], dtype=np.float64)
    moles = weights / atomic_masses
    return moles / np.sum(moles, axis=-1, keepdims=True)


# Compositions [X(CR), X(NI)] and geometry copied from case_dict in
# IllingworthTernaryTwoPhase_FeCrNi_Lee1996Cases.py (keep in sync manually).
# Geometry: alpha half-layer on [0, HALF_LENGTH - B_PHASE_LENGTH], gamma beyond.
CASE_DICT = {
    "fig9": {"HALF_LENGTH": 30e-6, "B_PHASE_LENGTH": 18e-6, "A_BULK": np.array([0.38, 0.001]), "B_BULK": np.array([0.13, 0.15])},
    "A": {"HALF_LENGTH": 2000e-6 + (186.9e-6 / 2), "B_PHASE_LENGTH": 2000e-6, "A_BULK": np.array([0.39696, 0.0001]), "B_BULK": np.array([0.2881427, 0.264718])},
    "B": {"HALF_LENGTH": 2000e-6 + (124.1e-6 / 2), "B_PHASE_LENGTH": 2000e-6, "A_BULK": np.array([0.39696, 0.0001]), "B_BULK": np.array([0.139297, 0.142387])},
    "C": {"HALF_LENGTH": 2000e-6 + (130.6e-6 / 2), "B_PHASE_LENGTH": 2000e-6, "A_BULK": _weight_percent_fe_cr_ni_to_atomic_fraction([54, 37, 9])[1:], "B_BULK": _weight_percent_fe_cr_ni_to_atomic_fraction([44, 24, 32])[1:]},
    "D": {"HALF_LENGTH": 2000e-6 + (71.6e-6 / 2), "B_PHASE_LENGTH": 2000e-6, "A_BULK": _weight_percent_fe_cr_ni_to_atomic_fraction([76, 23, 1])[1:], "B_BULK": _weight_percent_fe_cr_ni_to_atomic_fraction([72, 13, 15])[1:]},
}
caseStr = "B"
CASE = CASE_DICT[caseStr]
A_BULK = np.asarray(CASE["A_BULK"], dtype=np.float64)
B_BULK = np.asarray(CASE["B_BULK"], dtype=np.float64)

ELEMENTS = ["FE", "CR", "NI"]
INDEPENDENT_ELEMENTS = ["CR", "NI"]
PHASE_A = "BCC_A2"  # left phase (alpha)
PHASE_B = "FCC_A1"  # right phase (gamma)
TEMPERATURE = 1373.0

# Tie-line surrogate written by the Lee1996 script (pycalphad, 0.01 grid).
SURROGATE_BUNDLE_PATH = (
    EXAMPLES_DIR / "ThermoCalc" / "outputs" / "FeCrNi_pycalphad_1373K_default_phases_0_g_offset_0p0_bulk_0p01.surrogate"
)

# Matrix selection: "initial_eta" | "interface_eta" | "bulk_far_field" | "manual".
MATRIX_SOURCE = "initial_eta"
DIFFUSIVITY_SAMPLE_ETA = 0.5  # used by "interface_eta" and by the semi-infinite sizing pass
ZERO_OFF_DIAGONAL = False  # True mimics Kajihara & Kikuchi (1995): no Cr-Ni coupling
MANUAL_MATRICES = None  # {PHASE_A: 2x2, PHASE_B: 2x2} in m^2/s for "manual"

# Geometry: "semi_infinite" sizes each phase as DOMAIN_SAFETY*sqrt(lambda_max*T_END);
# "lee1996_case" uses the real case geometry (comparison limited to its validity time).
GEOMETRY = "semi_infinite"
DOMAIN_SAFETY = 8.0
VALIDITY_N_SIGMA = 4.0
INTERFACE_OFFSET = 1.0e-12  # keeps the initial step off a physical mesh node
PHYSICAL_NODES = 61

# Numerics.
T_END = 100.0 * 3600.0
PHASE_A_NODES = 100
PHASE_B_NODES = 400
GRID_METHOD = "linear"
GRID_SPACING_RATIO = 0.1
SEMI_LOG_DT = 0.02
SEMI_LOG_T0 = 1.0e-6
TOLERANCE = 1.0e-12
MAX_ITERATIONS = 25
MAX_STEP_RETRIES = 8
INITIAL_ETA_BRACKET = (1.0e-3, 1.0 - 1.0e-3)
INITIAL_ETA_GUESS = 0.5
VERBOSE = False
VERBOSE_INTERVAL = 100

# Comparison.
ROOT_INDEX = None  # None: the unique root, else the root closest to eta0
RESOLVED_N_CELLS = 5.0  # comparison starts once the profile spans this many interface cells
COMPARE_T_MIN = None  # None: use interface_resolved_start_time
N_COMPARE_TIMES = 5
TIME_UNIT_SECONDS = 3600.0
TIME_LABEL = "h"
SAVE_FIGURES = False
FIGURE_DIR = THIS_DIR / "analytic_comparison_figures"

# Optional refinement sweep (geometry and matrices stay fixed across levels).
RUN_REFINEMENT = False
REFINEMENT_LEVELS = [(50, 200, 0.04), (100, 400, 0.02), (200, 800, 0.01)]  # (nodes A, nodes B, semiLog_dt)

# %%
# Surrogate, geometry and model helpers


@contextmanager
def _surrogate_validity_policy(surrogate, policy):
    """Temporarily sets the surrogate validity policy (as the Lee1996 script does)."""
    previous = getattr(surrogate, "validity_policy", None)
    surrogate.validity_policy = policy
    try:
        yield surrogate
    finally:
        if previous is not None:
            surrogate.validity_policy = previous


def interface_matrices_at_eta(surrogate, eta):
    """Returns surrogate interface diffusivities at the tie-line ends ``c_A(eta)``, ``c_B(eta)``."""
    c_a, c_b = surrogate.interface_compositions(float(eta))
    with _surrogate_validity_policy(surrogate, "legacy"):
        return {
            PHASE_A: np.asarray(surrogate.getInterdiffusivity(c_a, TEMPERATURE, phase=PHASE_A, query_context="interface"), dtype=np.float64).reshape(2, 2),
            PHASE_B: np.asarray(surrogate.getInterdiffusivity(c_b, TEMPERATURE, phase=PHASE_B, query_context="interface"), dtype=np.float64).reshape(2, 2),
        }


def far_field_matrices(surrogate):
    """Returns surrogate bulk diffusivities at the far-field compositions ``A_BULK``, ``B_BULK``."""
    with _surrogate_validity_policy(surrogate, "legacy"):
        return {
            PHASE_A: np.asarray(surrogate.getInterdiffusivity(A_BULK, TEMPERATURE, phase=PHASE_A, query_context="general"), dtype=np.float64).reshape(2, 2),
            PHASE_B: np.asarray(surrogate.getInterdiffusivity(B_BULK, TEMPERATURE, phase=PHASE_B, query_context="general"), dtype=np.float64).reshape(2, 2),
        }


def _lambda_max(D):
    return float(np.max(np.real(np.linalg.eigvals(D))))


def comparison_geometry(matrices):
    """
    Returns ``(s0, domain_length)`` in metres for the selected ``GEOMETRY``.

    ``semi_infinite`` sizes each phase as ``DOMAIN_SAFETY * sqrt(lambda_max T_END)``
    using the supplied matrices; ``lee1996_case`` reproduces the Lee1996
    script's geometry.
    """
    if GEOMETRY == "lee1996_case":
        return CASE["HALF_LENGTH"] - CASE["B_PHASE_LENGTH"] + INTERFACE_OFFSET, CASE["HALF_LENGTH"]
    if GEOMETRY != "semi_infinite":
        raise ValueError("GEOMETRY must be 'semi_infinite' or 'lee1996_case'.")
    width_a = DOMAIN_SAFETY * np.sqrt(_lambda_max(matrices[PHASE_A]) * T_END)
    width_b = DOMAIN_SAFETY * np.sqrt(_lambda_max(matrices[PHASE_B]) * T_END)
    return width_a + INTERFACE_OFFSET, width_a + width_b


def build_comparison_model(thermodynamics, s0, domain_length, nodes_a, nodes_b, semi_log_dt):
    """
    Builds a phase-uniform two-phase model on ``[0, domain_length]`` with a step at ``s0``.

    ``thermodynamics`` supplies diffusivities (the surrogate for the eta0 probe,
    ``FixedMatrixTernaryDiffusivity`` for the comparison run); the surrogate is
    always the interface closure.
    """
    mesh = CartesianFD1D(INDEPENDENT_ELEMENTS, [0.0, domain_length], PHYSICAL_NODES)
    if np.min(np.abs(np.asarray(mesh.z).reshape(-1) - s0)) < 1.0e-3 * INTERFACE_OFFSET:
        raise ValueError("Initial interface lies on a physical mesh node; change PHYSICAL_NODES or INTERFACE_OFFSET.")
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(s0, A_BULK, B_BULK), INDEPENDENT_ELEMENTS)]))
    u_grid, v_grid = two_phase_landau_grids(nodes_a, nodes_b, method=GRID_METHOD, spacing_ratio=GRID_SPACING_RATIO)
    return MovingBoundaryIllingworthTernaryFD1DModel(
        mesh=mesh,
        elements=ELEMENTS,
        phases=[PHASE_A, PHASE_B],
        thermodynamics=thermodynamics,
        temperature=TEMPERATURE,
        interfacePosition=s0,
        interface_equilibrium=surrogate,
        initial_eta_method="instantaneous_balance",
        initial_eta_bracket=INITIAL_ETA_BRACKET,
        initial_eta_guess=INITIAL_ETA_GUESS,
        bulk_diffusivity_mode="phase_uniform",
        time_step=1.0,
        dt_mode="semi_log",
        semiLog_dt=float(semi_log_dt),
        semiLogT0=float(SEMI_LOG_T0),
        tolerance=TOLERANCE,
        max_iterations=MAX_ITERATIONS,
        max_step_retries=MAX_STEP_RETRIES,
        terminal_thin_phase_width=1.0e-9,
        terminal_thin_phase_policy="continue",
        record=True,
        record_pq_data=True,
        transformed_u_grid=u_grid,
        transformed_v_grid=v_grid,
    )


def estimated_steps(semi_log_dt, t_end=None):
    """Returns the approximate number of semi-log steps from ``SEMI_LOG_T0`` to ``t_end``."""
    t_end = T_END if t_end is None else t_end
    return int(np.ceil(np.log(t_end / SEMI_LOG_T0) / semi_log_dt)) + 1


def _save_or_show(fig, name):
    fig.tight_layout()
    if SAVE_FIGURES:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_DIR / f"{caseStr}_{name}.png", dpi=200)
    plt.show()


# %%
# Load the surrogate and select the frozen matrices

surrogate = SurrogateArtifactBundle.load_published(SURROGATE_BUNDLE_PATH).surrogate
print(f"Loaded tie-line surrogate {SURROGATE_BUNDLE_PATH.name}; eta bounds {surrogate.eta_bounds}, tie-line phases {surrogate.tieline_phases}.")

eta0_probe = None
if MATRIX_SOURCE == "initial_eta":
    # Size the domain with eta = DIFFUSIVITY_SAMPLE_ETA matrices, then keep that
    # geometry: eta0 depends on the interface-adjacent spacing, so resizing after
    # freezing the matrices would change the problem being compared.
    s0, domain_length = comparison_geometry(interface_matrices_at_eta(surrogate, DIFFUSIVITY_SAMPLE_ETA))
    probe = build_comparison_model(surrogate, s0, domain_length, PHASE_A_NODES, PHASE_B_NODES, SEMI_LOG_DT)
    with _surrogate_validity_policy(surrogate, "legacy"):
        probe.setup()
    eta0_probe = float(probe.initialEta)
    matrices = interface_matrices_at_eta(surrogate, eta0_probe)
    print(f"Model instantaneous-balance eta0 (eta-dependent interface D) = {eta0_probe:.8f}")
elif MATRIX_SOURCE == "interface_eta":
    matrices = interface_matrices_at_eta(surrogate, DIFFUSIVITY_SAMPLE_ETA)
elif MATRIX_SOURCE == "bulk_far_field":
    matrices = far_field_matrices(surrogate)
elif MATRIX_SOURCE == "manual":
    if MANUAL_MATRICES is None:
        raise ValueError("Set MANUAL_MATRICES for MATRIX_SOURCE='manual'.")
    matrices = {phase: np.asarray(MANUAL_MATRICES[phase], dtype=np.float64).reshape(2, 2) for phase in (PHASE_A, PHASE_B)}
else:
    raise ValueError("MATRIX_SOURCE must be 'initial_eta', 'interface_eta', 'bulk_far_field', or 'manual'.")

if ZERO_OFF_DIAGONAL:
    matrices = {phase: np.diag(np.diag(matrix)) for phase, matrix in matrices.items()}
if MATRIX_SOURCE != "initial_eta":
    s0, domain_length = comparison_geometry(matrices)

for phase, matrix in matrices.items():
    eig = eigen_decompose_diffusivity(matrix, f"D[{phase}]")
    print(f"\nD[{phase}] (m^2/s):\n{matrix}")
    print(f"  eigenvalues {eig.lam}, cond(P) = {np.linalg.cond(eig.P):.3g}")
    print(f"  eigenvectors (columns):\n{eig.P}")

print(f"\nGeometry ({GEOMETRY}): s0 = {s0 * 1e6:.3f} um, domain = {domain_length * 1e6:.3f} um")

# %%
# Analytic similarity roots and validity window

roots = solve_similarity_roots(matrices[PHASE_A], matrices[PHASE_B], A_BULK, B_BULK, surrogate, s0=s0)
if not roots:
    raise RuntimeError("No similarity root found within the surrogate eta bounds.")
for i, root in enumerate(roots):
    c_a, c_b = root.interface_compositions()
    print(f"root {i}: k = {root.k:.6e} m/s^0.5, eta* = {root.eta:.8f}, c_A = {c_a}, c_B = {c_b}, |res| = {root.residual_norm:.1e}")

if ROOT_INDEX is not None:
    solution = roots[ROOT_INDEX]
elif len(roots) == 1:
    solution = roots[0]
else:
    reference_eta = eta0_probe if eta0_probe is not None else DIFFUSIVITY_SAMPLE_ETA
    solution = min(roots, key=lambda r: abs(r.eta - reference_eta))
    print(f"Multiple roots (Maugis et al. 1997); using eta* = {solution.eta:.6f} closest to {reference_eta:.6f}. Set ROOT_INDEX to override.")


def _validity_time(s_initial, length):
    """Semi-infinite validity for a domain, allowing for the interface displacement at T_END."""
    displacement = abs(solution.k) * np.sqrt(T_END)
    t_a = semi_infinite_validity_time(matrices[PHASE_A], max(s_initial - displacement, 0.0), VALIDITY_N_SIGMA)
    t_b = semi_infinite_validity_time(matrices[PHASE_B], max(length - s_initial - displacement, 0.0), VALIDITY_N_SIGMA)
    return min(t_a, t_b)


t_valid = _validity_time(s0, domain_length)
lee_s0 = CASE["HALF_LENGTH"] - CASE["B_PHASE_LENGTH"]
t_valid_lee = min(
    semi_infinite_validity_time(matrices[PHASE_A], lee_s0, VALIDITY_N_SIGMA),
    semi_infinite_validity_time(matrices[PHASE_B], CASE["B_PHASE_LENGTH"], VALIDITY_N_SIGMA),
)
print(f"\nSemi-infinite validity in comparison domain: {t_valid / TIME_UNIT_SECONDS:.4g} {TIME_LABEL} (T_END = {T_END / TIME_UNIT_SECONDS:.4g} {TIME_LABEL})")
print(f"Semi-infinite validity in real case-{caseStr} geometry: {t_valid_lee / TIME_UNIT_SECONDS:.4g} {TIME_LABEL}")
if t_valid < T_END:
    warnings.warn("Comparison will be truncated at the validity time; increase DOMAIN_SAFETY or use GEOMETRY='semi_infinite'.")
COMPARE_T_MAX = min(T_END, t_valid)

# %%
# Solve the fixed-matrix model

fixed_thermodynamics = FixedMatrixTernaryDiffusivity(matrices, phases=[PHASE_A, PHASE_B], temperature=TEMPERATURE)
model = build_comparison_model(fixed_thermodynamics, s0, domain_length, PHASE_A_NODES, PHASE_B_NODES, SEMI_LOG_DT)
print(f"Estimated number of time-steps: {estimated_steps(SEMI_LOG_DT)}")
start = time.perf_counter()
model.solve(T_END, iterator=explicitEulerIterator, verbose=VERBOSE, vIt=VERBOSE_INTERVAL, minDtFrac=1.0e-16)
base_runtime = time.perf_counter() - start
print(f"Solve complete in {base_runtime:.1f} s (final time {model.currentTime:.6g} s).")
if eta0_probe is not None:
    print(f"Fixed-matrix model eta0 = {model.initialEta:.8f} vs probe eta0 = {eta0_probe:.8f} (should agree; a mismatch means a non-unique initial root).")

# %%
# Compare with the analytic solution

t_min = COMPARE_T_MIN if COMPARE_T_MIN is not None else interface_resolved_start_time(model, solution, RESOLVED_N_CELLS)
if t_min >= COMPARE_T_MAX:
    raise RuntimeError(f"Resolved start time {t_min:.3g} s exceeds the comparison end {COMPARE_T_MAX:.3g} s; refine the interface grid.")
metrics = compare_model_to_similarity(
    model, solution, t_min=t_min, t_max=COMPARE_T_MAX, profile_times=np.geomspace(t_min, COMPARE_T_MAX, N_COMPARE_TIMES)
)

print(f"Comparison window: {t_min / TIME_UNIT_SECONDS:.4g} - {COMPARE_T_MAX / TIME_UNIT_SECONDS:.4g} {TIME_LABEL}")
print(f"  k exact / model      : {metrics['k_exact']:.6e} / {metrics['k_model']:.6e}  (rel. err {metrics['k_rel_error']:.3e})")
print(f"  interface rel. error : {metrics['interface_rel_error']:.3e}")
print(f"  eta* / model eta0    : {metrics['eta_exact']:.6f} / {metrics['eta_initial_model']:.6f}; max |eta - eta*| in window {metrics['eta_max_abs_error']:.3e}")
for time_value, linf, rms in zip(metrics["profile_times"], metrics["profile_linf"], metrics["profile_rms"]):
    print(f"  t = {time_value / TIME_UNIT_SECONDS:10.4g} {TIME_LABEL}: profile Linf (Cr, Ni) = {linf}, RMS = {rms}")
print(f"  inventory drift      : {metrics['inventory_drift']} (relative {metrics['inventory_rel_drift']})")

# %%
# Plots

history = metrics["history"]
t_hist, s_hist = history["t"], history["s"]
positive = t_hist > 0.0
in_window = positive & (t_hist >= t_min) & (t_hist <= COMPARE_T_MAX)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
root_t_hours = np.sqrt(t_hist[positive] / TIME_UNIT_SECONDS)
axes[0].plot(root_t_hours, (s_hist[positive] - s0) * 1e6, label="model")
axes[0].plot(root_t_hours, (solution.interface_position(t_hist[positive]) - s0) * 1e6, "--", label=r"analytic $k\sqrt{t}$")
axes[0].set_xlabel(rf"$\sqrt{{t}}$ ($\sqrt{{\mathrm{{{TIME_LABEL}}}}}$)")
axes[0].set_ylabel(r"$s - s_0$ ($\mu$m)")
axes[0].legend()
displacement_scale = np.max(np.abs(solution.interface_position(t_hist[in_window]) - s0))
axes[1].semilogx(t_hist[positive] / TIME_UNIT_SECONDS, np.abs(s_hist[positive] - solution.interface_position(t_hist[positive])) / displacement_scale)
axes[1].axvspan(t_min / TIME_UNIT_SECONDS, COMPARE_T_MAX / TIME_UNIT_SECONDS, color="0.9", label="comparison window")
axes[1].set_xlabel(f"t ({TIME_LABEL})")
axes[1].set_ylabel(r"$|s - s_{exact}|$ / max displacement")
axes[1].legend()
fig.suptitle(f"Case {caseStr}: interface position")
_save_or_show(fig, "interface")

fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogx(history["t_eta"][history["t_eta"] > 0] / TIME_UNIT_SECONDS, history["eta"][history["t_eta"] > 0], label="model")
ax.axhline(solution.eta, color="k", ls="--", label=r"analytic $\eta^*$")
ax.axhline(metrics["eta_initial_model"], color="C3", ls=":", label=r"model $\eta_0$")
ax.axvspan(t_min / TIME_UNIT_SECONDS, COMPARE_T_MAX / TIME_UNIT_SECONDS, color="0.9")
ax.set_xlabel(f"t ({TIME_LABEL})")
ax.set_ylabel(r"$\eta$")
ax.legend()
ax.set_title(f"Case {caseStr}: interface tie line")
_save_or_show(fig, "eta")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
lam_a = solution.eig_left.lam.max()
lam_b = solution.eig_right.lam.max()
for i, entry in enumerate(metrics["profiles"]):
    color = f"C{i}"
    x_dense = np.linspace(max(0.0, s0 - 6.0 * np.sqrt(lam_a * entry["t"])), min(domain_length, s0 + 6.0 * np.sqrt(lam_b * entry["t"])), 1500)
    exact = solution.profile(x_dense, entry["t"])
    for j, element in enumerate(INDEPENDENT_ELEMENTS):
        label = f"{entry['t'] / TIME_UNIT_SECONDS:.3g} {TIME_LABEL}" if j == 0 else None
        axes[j].plot(x_dense * 1e6, exact[:, j], color=color, lw=1.0, label=label)
        axes[j].plot(entry["x_left"] * 1e6, entry["c_left"][:, j], ".", color=color, ms=3)
        axes[j].plot(entry["x_right"] * 1e6, entry["c_right"][:, j], ".", color=color, ms=3)
x_hi = min(domain_length, s0 + 6.0 * np.sqrt(lam_b * COMPARE_T_MAX))
x_lo = max(0.0, s0 - 6.0 * np.sqrt(lam_a * COMPARE_T_MAX))
for j, element in enumerate(INDEPENDENT_ELEMENTS):
    axes[j].set_xlim(x_lo * 1e6, x_hi * 1e6)
    axes[j].set_xlabel(r"x ($\mu$m)")
    axes[j].set_ylabel(f"X({element})")
axes[0].legend(title="lines: analytic, dots: model", fontsize=8)
fig.suptitle(f"Case {caseStr}: profiles")
_save_or_show(fig, "profiles")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for i, entry in enumerate(metrics["profiles"]):
    root_t = np.sqrt(entry["t"])
    for j in range(2):
        axes[j].plot((entry["x_left"] - s0) / root_t, entry["c_left"][:, j], ".", color=f"C{i}", ms=3)
        axes[j].plot((entry["x_right"] - s0) / root_t, entry["c_right"][:, j], ".", color=f"C{i}", ms=3)
z_dense = np.linspace(-6.0 * np.sqrt(lam_a), 6.0 * np.sqrt(lam_b), 2000)
exact_collapse = solution.profile(s0 + z_dense, 1.0)
for j, element in enumerate(INDEPENDENT_ELEMENTS):
    axes[j].plot(z_dense, exact_collapse[:, j], "k-", lw=1.0, label="analytic")
    axes[j].set_xlim(z_dense[0], z_dense[-1])
    axes[j].set_xlabel(r"$(x - s_0)/\sqrt{t}$ (m/s$^{1/2}$)")
    axes[j].set_ylabel(f"X({element})")
axes[0].legend()
fig.suptitle(f"Case {caseStr}: similarity collapse (all comparison times)")
_save_or_show(fig, "collapse")

# %%
# Optional refinement sweep (geometry and matrices held fixed)

if RUN_REFINEMENT:
    base_work = (PHASE_A_NODES + PHASE_B_NODES) * estimated_steps(SEMI_LOG_DT)
    estimate = sum(base_runtime * (na + nb) * estimated_steps(dt) / base_work for na, nb, dt in REFINEMENT_LEVELS)
    print(f"Estimated refinement runtime: {estimate / 60.0:.1f} min (scaled from the base run).")
    refinement = []
    for nodes_a, nodes_b, semi_log_dt in REFINEMENT_LEVELS:
        level_model = build_comparison_model(fixed_thermodynamics, s0, domain_length, nodes_a, nodes_b, semi_log_dt)
        start = time.perf_counter()
        level_model.solve(T_END, iterator=explicitEulerIterator, verbose=False, minDtFrac=1.0e-16)
        runtime = time.perf_counter() - start
        level_metrics = compare_model_to_similarity(
            level_model, solution, t_min=t_min, t_max=COMPARE_T_MAX, profile_times=metrics["profile_times"]
        )
        h_interface = max(s0 * (1.0 - level_model._u_grid[-2]), (domain_length - s0) * level_model._v_grid[1])
        refinement.append((nodes_a, nodes_b, semi_log_dt, h_interface, level_metrics["k_rel_error"], level_metrics["profile_rms"].max(), runtime))
        print(f"  A={nodes_a}, B={nodes_b}, dlnt={semi_log_dt}: k err {level_metrics['k_rel_error']:.3e}, profile RMS {level_metrics['profile_rms'].max():.3e}, {runtime:.1f} s")
    refinement = np.asarray(refinement, dtype=np.float64)
    for i in range(1, len(refinement)):
        ratio = refinement[i - 1, 3] / refinement[i, 3]
        print(
            f"  observed order {i - 1}->{i}: k {np.log(refinement[i - 1, 4] / refinement[i, 4]) / np.log(ratio):.2f}, "
            f"profile {np.log(refinement[i - 1, 5] / refinement[i, 5]) / np.log(ratio):.2f}"
        )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(refinement[:, 3] * 1e6, refinement[:, 4], "o-", label="k relative error")
    ax.loglog(refinement[:, 3] * 1e6, refinement[:, 5], "s-", label="max profile RMS")
    ax.set_xlabel(r"interface-adjacent spacing ($\mu$m)")
    ax.set_ylabel("error")
    ax.legend()
    ax.set_title(f"Case {caseStr}: refinement (space and dt refined together)")
    _save_or_show(fig, "refinement")
