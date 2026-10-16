# Digitised figure data (Illingworth & Golosnoy 2005 benchmarks)

These CSV files hold curves digitised with WebPlotDigitizer. Each row is an
`x, y` pair in the axis units of the source figure. They are used by
`examples/Illingworth2005/compare_illingworth2005_planar.py`,
`replicate_illingworth2005_fig4_5_6_planar.py` and the Illingworth tests in
`kawin/tests/test_diffusion.py`.

Only the CSVs are tracked. The WebPlotDigitizer projects (`*.tar`) and the
figure images used for digitising (`*.png`) stay local, because the images are
copies of the published figures.

**`fig4a_*`, `fig5a_*`, `fig6a_*`:** Figures 4a, 5a and 6a of T.C. Illingworth,
I.O. Golosnoy, "Numerical solutions of diffusion-controlled moving boundary
problems which conserve solute", J. Comput. Phys. 209 (2005) 207–225,
doi:10.1016/j.jcp.2005.02.031.

**`otherPaper_fig3_*`:** Figure 3 (thin/thick β layers, D_α = 1.4e-8 and
2.5e-8) of T.C. Illingworth, I.O. Golosnoy, V. Gergely, T.W. Clyne, "Numerical
modelling of transient liquid phase bonding and other diffusion controlled
phase changes", J. Mater. Sci. 40 (2005) 2505–2511,
doi:10.1007/s10853-005-1983-y.
