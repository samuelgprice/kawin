#%%

%matplotlib inline
"""
Replicate Olaye & Ojo (2020) Figure 5 with the Olaye moving-boundary FD model.

This script runs the binary planar moving-interface solver and plots liquid
half-width versus time for a Ni-P TLP-style setup.

Notes
-----
- The model implementation currently supports planar geometry only.
- By default this script uses Table-2-style Ni-P parameters from the paper
  (converted from percent to mole-fraction-like units and micrometers to SI).
- If you have digitized Figure 5 experimental points, pass them with
  ``--exp-csv`` (columns: ``time_h`` and ``half_width_um``).
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kawin.diffusion import MovingBoundaryOlayeFD1DModel, TemperatureParameters
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D
from kawin.solver import explicitEulerIterator


class ConstantBinaryThermodynamics:
    """Minimal binary thermodynamics interface for fixed diffusivity runs."""

    def __init__(self, phases, diffusivities):
        self.phases = list(phases)
        self.diffusivities = dict(diffusivities)

    def clearCache(self):
        return

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None, query_context=None):
        values = np.atleast_1d(T).astype(np.float64)
        return np.squeeze(np.ones(values.shape, dtype=np.float64) * self.diffusivities[phase])


def _load_experimental_csv(path: pathlib.Path):
    """
    Loads an experimental overlay CSV.

    Expected columns (with header): ``time_h,half_width_um``.
    """
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if "time_s" not in data.dtype.names or "half_width_um" not in data.dtype.names:
        raise ValueError("Experimental CSV must contain columns: time_s, half_width_um")
    return np.asarray(data["time_s"], dtype=np.float64), np.asarray(data["half_width_um"], dtype=np.float64)


def run_case(
    *,
    R_um: float,
    s0_um: float,
    c_liquid0_pct: float,
    c_solid0_pct: float,
    c_liquid_int_pct: float,
    c_solid_int_pct: float,
    D_liquid_base: float,
    D_solid_base: float,
    D_scale: float,
    n_nodes: int,
    t_end_h: float,
    dt_mode: str,
):
    """Runs one Figure-5-style simulation and returns time and liquid half-width."""
    R_m = float(R_um) * 1e-6
    s0_m = float(s0_um) * 1e-6
    interface_position_m = R_m - s0_m

    # Percent inputs from the paper -> fraction-like internal values.
    c_liquid0 = float(c_liquid0_pct) / 100.0
    c_solid0 = float(c_solid0_pct) / 100.0
    c_liquid_int = float(c_liquid_int_pct) / 100.0
    c_solid_int = float(c_solid_int_pct) / 100.0

    # Map to model convention (left phase A, right phase B, and C_BA > C_AB):
    # left = solid (low solute), right = liquid (high solute).
    profile = ProfileBuilder([(StepProfile1D(interface_position_m, c_solid0, c_liquid0), "P")])
    mesh = CartesianFD1D(["P"], [0.0, R_m], int(n_nodes))
    mesh.setResponseProfile(profile)

    therm = ConstantBinaryThermodynamics(
        phases=["SOLID", "LIQUID"],
        diffusivities={
            "SOLID": float(D_solid_base) * float(D_scale),
            "LIQUID": float(D_liquid_base) * float(D_scale),
        },
    )

    model = MovingBoundaryOlayeFD1DModel(
        mesh,
        ["NI", "P"],
        ["SOLID", "LIQUID"],
        thermodynamics=therm,
        temperature=TemperatureParameters(1000.0),
        interfacePosition=interface_position_m,
        interface_compositions=(c_solid_int, c_liquid_int),
        first_step_mode="classical_explicit",
        main_step_mode="leapfrog_dufort_frankel",
        dt_mode=dt_mode,
        geometry="planar",
        semi_log_points=250,
        record=True,
    )

    model.solve(float(t_end_h) * 3600.0, iterator=explicitEulerIterator, verbose=True, vIt=100)

    t_s = np.array(model.interfaceData._time[: model.interfaceData.N + 1], dtype=np.float64)
    s_m = np.array(model.interfaceData._y[: model.interfaceData.N + 1], dtype=np.float64)
    liquid_half_width_um = (R_m - s_m) * 1e6
    t_h = t_s / 3600.0
    return t_h, liquid_half_width_um, model


def build_parser():
    parser = argparse.ArgumentParser(description="Replicate Olaye & Ojo 2020 Figure 5 using the Olaye FD model.")
    parser.add_argument("--R-um", type=float, default=3012.5, help="Half-domain size R in micrometers.")
    parser.add_argument("--s0-um", type=float, default=12.5, help="Initial liquid half-width s0 in micrometers.")
    parser.add_argument("--c-liquid0-pct", type=float, default=19.0, help="Initial liquid composition (percent).")
    parser.add_argument("--c-solid0-pct", type=float, default=0.0, help="Initial solid composition (percent).")
    parser.add_argument("--c-liquid-int-pct", type=float, default=10.223, help="Liquid-side interface composition (percent).")
    parser.add_argument("--c-solid-int-pct", type=float, default=0.166, help="Solid-side interface composition (percent).")
    parser.add_argument("--D-liquid-base", type=float, default=500.0, help="Liquid diffusivity base value from table.")
    parser.add_argument("--D-solid-base", type=float, default=18.0, help="Solid diffusivity base value from table.")
    parser.add_argument("--D-scale", type=float, default=1e-12, help="Scale to convert table diffusivities to m^2/s.")
    parser.add_argument("--n-nodes", type=int, default=3013+1, help="Number of FD nodes.")
    parser.add_argument("--t-end-h", type=float, default=20.0, help="Simulation end time (hours).")
    parser.add_argument(
        "--dt-mode",
        type=str,
        choices=["cfl", "semi_log_optional"],
        default="semi_log_optional",
        help="Time-step policy for Olaye model.",
    )
    parser.add_argument("--exp-csv", type=str, default=None, help="Optional CSV with experimental points: time_h,half_width_um.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output figure path. Default: examples/Olaye2020/olaye2020_fig5_replication.png",
    )
    parser.add_argument("--show", action="store_true", help="Show the matplotlib window.")
    return parser


def main():
    # args = build_parser().parse_args()
    DEFAULT_FIG5_ARGS = {
    "R_um": 3012.5,
    "s0_um": 12.5,
    "c_liquid0_pct": 19.0,
    "c_solid0_pct": 0.0,
    "c_liquid_int_pct": 10.223,
    "c_solid_int_pct": 0.166,
    "D_liquid_base": 500.0,
    "D_solid_base": 18.0,
    "D_scale": 1e-12,
    "n_nodes": 3013+1,
    "t_end_h": 0.5,
    "dt_mode": "semi_log_optional",
    "exp_csv": r"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\kawin\examples\Olaye2020\Olaye2020_fig5_PresentModel_curve.csv",
    "out": None,
    "show": True,
}
    args=DEFAULT_FIG5_ARGS.copy()

    script_dir = pathlib.Path(__file__).resolve().parent
    out_path = (
        pathlib.Path(args['out']).resolve()
        if args['out'] is not None
        # pathlib.Path(args.out).resolve()
        # if args.out is not None
        else script_dir / "olaye2020_fig5_replication.png"
    )

    t_h, width_um, model = run_case(
        R_um=args["R_um"],
        s0_um=args["s0_um"],
        c_liquid0_pct=args["c_liquid0_pct"],
        c_solid0_pct=args["c_solid0_pct"],
        c_liquid_int_pct=args["c_liquid_int_pct"],
        c_solid_int_pct=args["c_solid_int_pct"],
        D_liquid_base=args["D_liquid_base"],
        D_solid_base=args["D_solid_base"],
        D_scale=args["D_scale"],
        n_nodes=args["n_nodes"],
        t_end_h=args["t_end_h"],
        dt_mode=args["dt_mode"],
    )
    # t_h, width_um, model = run_case(
    #     R_um=args.R_um,
    #     s0_um=args.s0_um,
    #     c_liquid0_pct=args.c_liquid0_pct,
    #     c_solid0_pct=args.c_solid0_pct,
    #     c_liquid_int_pct=args.c_liquid_int_pct,
    #     c_solid_int_pct=args.c_solid_int_pct,
    #     D_liquid_base=args.D_liquid_base,
    #     D_solid_base=args.D_solid_base,
    #     D_scale=args.D_scale,
    #     n_nodes=args.n_nodes,
    #     t_end_h=args.t_end_h,
    #     dt_mode=args.dt_mode,
    # )

    mask = np.isfinite(t_h) & np.isfinite(width_um)
    t_h_plot = t_h[mask]
    width_um_plot = width_um[mask]
    if t_h_plot.size < 2:
        raise ValueError("Simulation did not produce enough finite points to plot.")

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=140)
    ax.plot(t_h_plot*3600, width_um_plot, lw=2.0, color="tab:blue", label="Olaye model (this work)")

    exp_csv = args["exp_csv"]
    # exp_csv = args.exp_csv
    if exp_csv is not None:
        exp_t_h, exp_w_um = _load_experimental_csv(pathlib.Path(exp_csv))
        exp_mask = np.isfinite(exp_t_h) & np.isfinite(exp_w_um)
        ax.scatter(
            exp_t_h[exp_mask],
            exp_w_um[exp_mask],
            s=26,
            color="tab:red",
            marker="o",
            label="Experimental (digitized)",
        )

    # ax.set_xlabel("Time (h)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Liquid half-width (um)")
    ax.set_title("Olaye & Ojo (2020) Figure 5 Replication")

    x_min = float(np.min(t_h_plot)*3600)
    x_max = float(np.max(t_h_plot)*3600)
    y_min = float(np.min(width_um_plot))
    y_max = float(np.max(width_um_plot))
    if not np.isfinite(x_min + x_max + y_min + y_max):
        raise ValueError("Plot limits are not finite.")
    if x_max <= x_min:
        pad_x = max(1e-6, abs(x_min) * 1e-3 + 1e-6)
        x_min -= pad_x
        x_max += pad_x
    if y_max <= y_min:
        pad_y = max(1e-6, abs(y_min) * 1e-3 + 1e-6)
        y_min -= pad_y
        y_max += pad_y
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.grid(True, alpha=0.25)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved: {out_path}")

    # if args.show:
    if args["show"]:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

# %%
