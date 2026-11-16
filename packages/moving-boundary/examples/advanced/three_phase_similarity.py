"""Advanced checks for ``three_phase_similarity``; see that example for assumptions and references.

The Case supplies explicit physical inputs and callbacks. No main script
is imported, and no solve or file operation runs during import.
"""

from dataclasses import dataclass
from moving_boundary.plotting import plot_interface_history
import argparse
import sys
import numpy as np
from moving_boundary.analytic import compare_three_phase_model_to_similarity, recorded_history, semi_infinite_validity_time


# Numerical accuracy limits at DEMO settings only: 81 nodes/phase, semi_log_dt=0.025, mesh
# 6001, t to 1, window t >= 4*T0. Each applies to both interfaces. k_rel: |k-k_exact|/|k_exact|.
# eta_abs: max|eta-eta_exact|. profile_linf: max over phases of |c-c_exact| at four recorded
# times. inventory_abs: largest component |final-initial| in composition x length.
DEMO_LIMITS = {"k_rel": 2.5e-2, "eta_abs": 1.5e-2, "profile_linf": 1.0e-2, "inventory_abs": 1.0e-9}


@dataclass(frozen=True)
class Case:
    """Inputs and callbacks supplied by the main example at call time."""

    DEMO: object
    DEMO_DRIFT: object
    DEMO_LIMITS: object
    DIFFUSIVITIES: object
    DOMAIN_LENGTH: object
    FAR_FIELDS: object
    FAST: object
    FAST_DRIFT: object
    PHASES: object
    T0: object
    T_FINAL: object
    description: str


def evaluate(model, config, exact, *, case):
    """Return per-interface accuracy metrics; a short run has too few points in the window."""
    s_ab, s_bc = exact.interface_positions(config.t_end)
    metrics = {
        "validity_time_a": semi_infinite_validity_time(case.DIFFUSIVITIES["A"], s_ab),
        "validity_time_c": semi_infinite_validity_time(case.DIFFUSIVITIES["C"], case.DOMAIN_LENGTH - s_bc),
    }
    if config.t_end < case.T_FINAL:
        return metrics
    comparison = compare_three_phase_model_to_similarity(model, exact, case.T0)
    metrics.update(
        k_rel=np.asarray(comparison["k_rel_error"]).tolist(),
        eta_abs=np.asarray(comparison["eta_max_abs_error"]).tolist(),
        profile_linf=float(np.max(comparison["profile_linf"])),
        inventory_abs=float(np.max(np.abs(comparison["inventory_drift"]))),
    )
    return metrics

def report(result, *, case):
    """Print the final state and the metrics, with limits where they apply."""
    print(f"interfaces {result.model.getInterfacePositions()}, etas {result.model.getInterfaceEtas()}; exact rates {result.exact.ks}")
    limits = case.DEMO_LIMITS if result.config in (case.DEMO, case.DEMO_DRIFT) else {}
    if not limits:
        print("These settings have no accuracy limits; DEMO and DEMO_DRIFT are the configurations to judge accuracy by.")
    for name, value in result.metrics.items():
        print(f"{name}: {value}" + (f"  (limit {limits[name]:g})" if name in limits else ""))

def plot(result, *, case):
    """Plot both interfaces against the exact solution, and the final profiles; returns the figure."""
    import matplotlib.pyplot as plt

    model, exact = result.model, result.exact
    times, interfaces = recorded_history(model.interfaceData)
    similarity_time = times + case.T0
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    plot_interface_history(model, ax=left, time_offset=case.T0, label=["s_AB, model", "s_BC, model"])
    left.plot(similarity_time, exact.interface_positions(similarity_time), "--", label=["s_AB, exact", "s_BC, exact"])
    left.set(xlabel="similarity time", ylabel="interface position", title="Interfaces")
    left.legend(fontsize="small")
    time = float(times[-1])
    for phase, (x, c) in enumerate(model.getPhysicalPhaseProfiles(time)):
        right.plot(x, c[:, 1], color=f"C{phase}", label=f"x(X), {case.PHASES[phase]}")
        right.plot(x, exact.profile(x, time + case.T0, phase=phase)[:, 0], "--", color=f"C{phase}")
    right.set(xlabel="position", ylabel="x(X)", title="Final profiles (dashed: exact)")
    right.legend(fontsize="small")
    figure.tight_layout()
    return figure

def settings_from_command_line(config, plot_figures, *, case):
    """Apply ``--fast``, ``--case`` and ``--plot`` from a terminal; inside a notebook kernel, change nothing.

    A Jupyter kernel's ``sys.argv`` holds the kernel's own arguments, so they are not parsed.
    """
    if "ipykernel" in sys.modules:
        return config, plot_figures
    parser = argparse.ArgumentParser(description=case.description.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--fast", action="store_true", help="use quick check settings")
    parser.add_argument("--case", choices=sorted(case.FAR_FIELDS), default="opposite")
    parser.add_argument("--plot", action="store_true", help="show plots")
    args = parser.parse_args()
    choices = {("opposite", False): case.DEMO, ("drift", False): case.DEMO_DRIFT, ("opposite", True): case.FAST, ("drift", True): case.FAST_DRIFT}
    return choices[(args.case, args.fast)], args.plot
