"""Fe-Ni-C ferrite growth under local equilibrium: ternary Illingworth (u-fractions) vs DICTRA.

Setup follows DICTRA example exh2 (FEDEMO + MFEDEMO, Fe-Ni-C, BCC_A2 and
FCC_A1 only, 973 K, closed planar system, 1e-9 m ferrite next to 50e-6 m of
austenite, both Fe-2.02 wt% Ni-0.0885 wt% C, 50 s), but with **local
equilibrium** at the interface instead of the para-equilibrium used by exh2.
The experimental exh2 data (para-equilibrium) are plotted only as a reference.
See ``ternary_ferrite_growth_le.py`` for the method, stages and diagnostics.

Run (repo root, through the conda activation wrapper in AGENTS.md):
    python -m examples.Interstitials.fe_ni_c_exh2_le --stage all
"""

from examples.Interstitials.ternary_ferrite_growth_le import FerriteGrowthCase, main

CASE = FerriteGrowthCase(
    name="fe_ni_c_exh2_le",
    solute="NI",
    temperature=973.0,
    w_c=0.0885,
    w_x=2.02,
    sim_time=50.0,
    experiment={"time": [6.16, 10.27, 20.55], "half_thickness": [4.06e-6, 5.94e-6, 1.0e-5]},
    experiment_label="exh2 data (para-eq. reference)",
)


if __name__ == "__main__":
    main(CASE, __doc__.splitlines()[0])
