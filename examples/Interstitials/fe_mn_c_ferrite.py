"""Fe-Mn-C ferrite growth under local equilibrium: ternary Illingworth (u-fractions) vs DICTRA.

Fe-1.5 wt% Mn-0.1 wt% C held in the intercritical (ferrite + austenite) range
(FEDEMO + MFEDEMO, BCC_A2 and FCC_A1 only, closed planar system, 1e-9 m
ferrite seed next to 50e-6 m of austenite of the alloy composition), with
local equilibrium at the interface. Mn diffuses about 1e6-1e8 times more slowly
than C, so growth starts carbon controlled with a sub-nanometre Mn spike
(no-partitioning local equilibrium). See ``ternary_ferrite_growth_le.py`` for
the method, stages and diagnostics.

Run (repo root, through the conda activation wrapper in AGENTS.md):
    python -m examples.Interstitials.fe_mn_c_ferrite --stage all
"""

from examples.Interstitials.ternary_ferrite_growth_le import FerriteGrowthCase, main

CASE = FerriteGrowthCase(
    name="fe_mn_c_ferrite_le",
    solute="MN",
    temperature=1000.0,
    w_c=0.1,
    w_x=1.5,
    sim_time=100.0,
    report_times=(0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0),
)


if __name__ == "__main__":
    main(CASE, __doc__.splitlines()[0])
