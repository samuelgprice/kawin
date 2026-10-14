# Interstitial (C, N, O, H, B) support in the Illingworth moving-boundary solvers

Status as of 2026-10-07 (afternoon). The changes are not committed. All
benchmark runs described here have finished.

## 1. What changed and why

Before this work, all three Illingworth solvers raised an error if any element
of the system was an interstitial:

- binary `MovingBoundaryIllingworthFDM.py`
- ternary two-phase `MovingBoundaryIllingworthTernaryFDM.py`
- ternary three-phase `MovingBoundaryIllingworthTernaryThreePhaseFDM.py`

The goal was to benchmark against DICTRA on Fe-C and Fe-C-X (X = Mn, Ni, Cr).

### Composition convention

With interstitials present, the solver state is **u-fractions**:

    u_k = x_k / sum(substitutional x)

Diffusion is in DICTRA's **volume-fixed frame**:

- only substitutional atoms carry volume;
- the molar volume per mole of substitutional atoms is constant and equal in
  all phases;
- `du_k/dt = -dJ_k/dz` with `J_k = -sum_j D_kj du_j/dz`;
- the interface balance is `v [u_k] = [J_k]`.

In u-fractions the transformed-coordinate PDE, the finite-volume fluxes, the
block-tridiagonal solves (including the Numba kernels) and the interface
balance all keep exactly the same form. That made a thin conversion layer
possible, instead of a rewrite.

### Design: convert at the boundary only

Everything outside the solvers stays in **mole fractions**: kawin
`Thermodynamics`, the Thermo-Calc adapter, the surrogate/interpolation tables
and the tie-line closures. One new private module does the conversion:

**`kawin/diffusion/_illingworth_interstitials.py`**

| Function / class | Purpose |
|---|---|
| `u_to_x_independent`, `x_to_u_independent` | Closed forms `x_k = u_k/(1+Σ_I u_I)` and `u_k = x_k/(1-Σ_I x_I)`. They match `Mobility.u_to_x_frac`/`x_to_u_frac` to 1e-14 without a per-point matrix inverse. |
| `full_mole_fractions` | `(n, e)` mole fractions for output. |
| `composition_bound_violation` | Substitutional u (and the dependent `1-Σ_sub u`) must be ≥ `minComposition`. Interstitial u has a lower bound only, with no site-ratio cap. With no interstitials it runs exactly the old simplex arithmetic. |
| `UFractionDiffusivityProvider` | Wraps a thermodynamics object: converts u→x before `getInterdiffusivity` and forwards `query_context` unchanged, so the solvers' `TypeError` fallbacks still work. The returned D is already the u-basis matrix and is not modified. |
| `UFractionInterfaceEquilibrium` | Wraps a tie-line closure: converts endpoints x→u unless the closure declares `composition_basis = "u"`. |
| `resolve_diffusivity_provider`, `resolve_interface_equilibrium` | Return the **original object** when there are no interstitials. This is what keeps substitutional runs bit-identical. |
| `validate_interstitial_element_order` | The dependent (first) element must be substitutional. The independent elements may be in any order, because kawin's `Thermodynamics` and the TC adapter already return D in the caller's element order. |
| `record_initial_u_fractions` | Fixes a latent bug: `DiffusionModel.reset()` re-recorded the raw mole-fraction profile, so `reset()` followed by `setup()` would have started from x instead of u. |

### File-by-file changes

**`MovingBoundaryEquilibrium.py`**

- `Fixed…`, `Callable…` and `ThermodynamicTernaryInterfaceEquilibrium` take an optional `composition_basis` (`"x"` by default, or `"u"`).
- The docstrings now say the closures omit the dependent component because the models enforce the *substitutional* sum.

**`MovingBoundaryIllingworthFDM.py` (binary)**

- The interstitial guard is replaced by the element-order check.
- New argument `interface_composition_basis="x"`. The fixed interface compositions are converted to u once and stored in `interfaceCompositions`; the caller's values are kept in `interfaceCompositionsInput`.
- All diffusivity queries (`_phase_diffusivity`, `_bulk_face_diffusivities`) go through `_diffusivity_source()`.
- `reset()` re-records u when interstitials are present.
- The class docstring states the convention.

**`MovingBoundaryIllingworthTernaryFDM.py` (two-phase)**

- Guard, `reset()`, diffusivity query sites and tie-line closure as in the binary model. The initial-η estimator receives the wrapped provider and closure.
- Composition validators delegate to `composition_bound_violation`, keeping the "ternary composition bounds" message.
- `getCompositions` and `getPhysicalPhaseProfiles` return mole fractions; the stored state stays in u.
- New optional `interface_residual_composition_scale` (per component, default `1.0`). It tightens the interface balance of a slow or dilute component.
- New optional `initial_eta_componentwise_scaling`: per-row scaling of the startup η balance.
  - The scales are frozen within each Newton iterate.
  - An earlier version re-evaluated them at perturbed points; the scaled residual then saturated as |V| grew and Newton stalled. That is fixed.
  - Each component scale is floored at `64·eps/root_rtol` times the largest one, so a component whose balance terms nearly vanish (a non-partitioning slow solute at t = 0) is required to balance only to round-off of the dominant terms. Without the floor the test was unattainable and the option failed on strongly graded Fe-Ni-C grids.
- Initial-η branch selection now picks the best *converged* motion branch, compared in the active scaling's norm. Previously the branch with the smallest raw residual was taken and then rejected if unconverged, which could discard a converged branch. That logic predates this work.
- Interstitial symbols must be upper case. Matching is exact, as in the rest of kawin, and lower-case interstitial symbols such as `"c"` are rejected instead of silently being treated as substitutional.
- New diagnostics after each accepted step, all computed in closed form for 2×2 matrices without LAPACK:
  - `_lastImplicitResidualComponents`
  - `_lastImplicitJacobianCond`
  - `_lastInterfacePeclet = |v|·dz_interface/λ_min(D)` per phase
- The plot label switches to "u-fraction" when interstitials are present.

**`MovingBoundaryIllingworthTernaryThreePhaseFDM.py`**

- Same guard, diffusivity query, closure, bounds, `reset()` and output changes as the two-phase model.
- `phase_molar_volumes` are now interpreted as volumes per mole of substitutional atoms.
- The total-moles check and `_dependent_mole_closure_error` sum only the substitutional components.
- It does **not** have the per-component initial-η option.

**`mesh/MovingBoundaryIllingworthTernaryFD1D.py`**

- `integrate_planar_transformed_molar_inventories` takes an optional `substitutional_mask`.

**`MovingBoundarySurrogates.py`**

- Docstring note only. The tables stay in x-space; the D values are the u-basis matrices; conversion happens at the query boundary.

**`examples/ThermoCalc/tc_python_adapter.py`**

- Removed a stray `print(x)`. The file has CRLF line endings, which were preserved even though `.gitattributes` asks for LF.

**`pyproject.toml`**

- Registers a `slow` pytest marker.

**Surrogate and interpolation tables were deliberately not changed.** Their
sampling, simplex masks and Delaunay/KD-tree validity domains all assume the
x-simplex. The D values don't depend on the composition basis, and u↔x is
smooth, so converting at the query boundary costs essentially nothing.

## 2. Is Thermo-Calc's `DC` in the same frame as kawin? (yes)

Script: `examples/Interstitials/check_tc_interstitial_frame.py` (~1 min).

**Setup**
- A DICTRA single-phase austenite Fe-C diffusion problem at 1373 K, with a high-carbon cosine start (u_C 0.08 → 0.005) and 1000 s simulated.
- The same problem is re-solved by an independent implicit finite-volume code using TC's tabulated `DC(FCC_A1,C,C,FE)` in two candidate frames.
- Step and linear starts were tried first and rejected: DICTRA's trapezoidal integration rings at a step, and a linear profile barely evolves.

| Comparison | Max abs. error in u_C |
|---|---|
| u-frame (`J = -D du/dz`) vs DICTRA | **1.6e-6** (2.1e-5 of the span) |
| x-frame (`J = -D dx/dz`) vs DICTRA | 1.7e-5 (10× worse) |
| FV self-convergence (2× refinement) | 1.3e-7 |

**Cross-database check.** kawin `FECRC_DB` D_CC ÷ TC FEDEMO D_CC = **1.0011,
constant** from x_C = 0.0025 to 0.078. A frame difference of the (1 − x) kind
would drift by about 8% over that range.

**Conclusion:** TC's `DC` and kawin's `interdiffusivity` use the same u-basis,
volume-fixed frame, so **no correction is needed in the adapter**.

## 3. Tests

All Python runs go through the conda activation wrapper (see AGENTS.md).

| Test file | What it checks | Runtime |
|---|---|---|
| `test_illingworth_substitutional_baseline.py` (+ `illingworth_baseline_cases.py`, `make_illingworth_substitutional_baseline.py`, `data/illingworth_substitutional_baseline.npz`) | Binary in all three D modes, two-phase ternary (uniform/lagged/implicit, both motion directions) and three-phase (uniform/lagged). Every recorded history is **bitwise equal** (NaN-aware) to the pre-change code. | ~7 s |
| `test_illingworth_interstitials.py` | Shared module: u↔x round trips vs `Mobility` (1e-14), bounds identical to the legacy simplex, wrappers. Exact-case interstitial matching and rejection of lower-case interstitial symbols. Binary Fe-C: u after `__init__` and after `reset()`, providers receive x in all three D modes, Fe-C vs Ni-Cr label invariance, and an analytic u-fraction **Zener** test. | ~5 s |
| `test_illingworth_interstitials_ternary.py` | Fe-C-Mn-like plumbing (x and u closures, queries in x, mole-fraction outputs, diagnostics, bounds, dependent-element guard). **Disparate-eigenvalue similarity tests** at Mn/C ratios 1e-2, 1e-4, 1e-6. A 1e-8 case is gated by `KAWIN_RUN_SLOW=1` and has not been run. Initial-η branch selection (converged branch preferred over a smaller raw residual) and the per-component scale floor (ratio 1e-10, with a floor-disabled negative control). | ~50 s |
| `test_illingworth_interstitials_three_phase.py` | Masked molar inventory vs hand integration. Fe-C-Mn three-phase: u state, queries in x, conservation of component moles, substitutional-only total-moles closure, mole-fraction outputs. | ~6 s |

**Full suite:** 1020 passed, 2 skipped (6 min 49 s). The existing Illingworth
suites (`test_illingworth_ternary*.py`, `test_illingworth_ternary_three_phase.py`,
`test_three_phase_examples.py`, the binary tests in `test_diffusion.py`) pass
unchanged.

### Binary Zener (u-fractions, constant D)

The ferrite starts flat at its interface u_C and the austenite starts at the
exact erfc profile at t0, so `s = 2λ√(Dt)`.

- The interface error is first order in the transformed spacing: 0.54% → 0.26% when the grid is doubled, and insensitive to dt.
- The test asserts < 0.3% on the fine grid and a convergence ratio > 1.7.
- Inventory drift is < 1e-10·R.

### Ternary similarity with disparate eigenvalues

Setup:
- u-basis matrices `D = D_C·[[1, -c],[0, r]]` (carbon fast, Mn slow by r), weak C–Mn coupling.
- A linear no-partitioning-like tie-line family in u.
- Linear-spacing transformed grids refined at the interface.
- Roots from `solve_similarity_roots(..., k_bounds=...)`; explicit bounds are needed because the default k range scales with the smallest eigenvalue.

| r = λ_Mn/λ_C | Grid (nodes / interface spacing ratio) | k error | η error | Inventory drift | Time |
|---|---|---|---|---|---|
| 1e-2 | 41 / 0.1 | 0.53% | 7e-5 | < 1e-10 | 2 s |
| 1e-4 | 161 / 0.001 | 0.51% | 8e-5 | < 5e-10 | 10 s |
| 1e-6 | 161 / 0.001 | 0.57% | 1e-5 | < 1e-9 | 31 s |
| 1e-6 | 321 / 1e-4 | 0.27% | 2e-5 | 1.3e-9 | 67 s |

The 1e-4 case converges with the interface Péclet number: k error 1.8% → 1.0% →
0.51% as Pe falls 18 → 2.3 → 0.23. Mn profiles are not compared point by point,
because the Mn spike (width D_Mn/v) is narrower than the remaining interface
offset. The Jacobian condition number rises to about 1e4 at r = 1e-6.

## 4. Benchmarks against DICTRA

### 4.1 Binary Fe-C — DICTRA exb1a / pyex_D_02

Script: `examples/Interstitials/fe_c_exb1a.py --case exb1a`

**Setup**
- FEDEMO/MFEDEMO, 1050 K.
- 1e-9 m ferrite (0.01 wt% C) next to 2 mm austenite (0.15 wt% C), run to 1e9 s.
- kawin uses TC's LE compositions (x_BCC = 6.58e-4, x_FCC = 0.0206) and `composition_dependent_lagged` D from a 1-D DC table.

**DICTRA's default time-step control is not converged for this case.** It takes
about 44 steps to 1e4 s and gives a parabolic constant about 15% low. Refining
the austenite grid (100 → 800 points) changed nothing; tightening the time-step
control did. The reference therefore uses `TimestepControl(max_relative_error=1e-3,
max_timestep_increase_factor=1.2, max 1% of sim time, check_interface_position)`.
Settings of 1e-4 / 1.05 gave the same answer.

DICTRA also **jumps to about 0.6 µm within its first steps** (t ~ 1e-7 s)
whatever the time-step settings, then follows the parabolic law. kawin grows
smoothly from the 1e-9 m seed. The start-up offset in DICTRA decays slowly:
its s/√t rises from 1.93e-6 at 100 s to 1.99e-6 at 1e4 s.

| t (s) | DICTRA (tight) | DICTRA (default) | kawin | kawin vs tight |
|---|---|---|---|---|
| 1e2 | 19.39 µm | 16.75 µm | 19.85 µm | +2.4% |
| 1e3 | 62.01 µm | 52.78 µm | 62.70 µm | +1.1% |
| 1e4 | 198.08 µm | 166.47 µm | 198.14 µm | +0.03% |
| 1e5 | 624.6 µm | 530.1 µm | 626.4 µm | +0.30% |
| 1e9 (final) | 1.37898 mm | 1.37642 mm | 1.37957 mm | +0.04% |

**Other results**
- **Carbon profiles away from the interface:** max |Δu_C| = 2.3e-4 at 1e3 s, 7.8e-5 at 1e5 s, 2.3e-6 at 1e9 s.
- **kawin cost:** about 3 s, 1743 steps; inventory drift 1e-12.
- **kawin convergence:** k(1e3 s) = 1.981–1.987e-6 m/√s across dt/2, 2× austenite nodes, 4× ferrite nodes and lagged/implicit D.
- **`phase_uniform` mode:** reproduces the analytic Zener k for D at the interface composition exactly (2.140e-6).
- **pyex_D_02 (30 h, 50-point grid):** this was only run against the default-time-step DICTRA, where kawin was 7% thicker at the end. It has not been rerun with the tight reference.

Output: `outputs/fe_c_exb1a_comparison.json`, `outputs/fe_c_exb1a_thickness.png`.

### 4.2 Ternary Fe-C-X ferrite growth (local equilibrium)

Shared driver: `examples/Interstitials/ternary_ferrite_growth_le.py`. Case
scripts: `fe_ni_c_exh2_le.py`, `fe_mn_c_ferrite.py`.

**Stages** (`--stage`):
- `tc-inputs`: tie-lines and DC tables, under 30 s.
- `dictra`: the reference, run to `--dictra-sim-time`, default 2 s.
- `kawin`: requires the DICTRA cache.
- `kawin-only`: no comparison.
- `all`.

**Other kawin options:** `--n-right`, `--power`, `--semi-log-dt`,
`--semi-log-t0`, `--x-residual-scale`, `--eta-componentwise` and `--tag`
(suffix for the output files).

**DICTRA reference settings:** a 100-point geometric austenite grid (×1.05) and
`TimestepControl(rel 1e-3, step growth 1.2, max 1% of sim time)`, with no
interface-position check.

**Tie-lines.** Ordinary equilibria are computed on a 16 × 27 grid of global
(x_C, x_X). Every two-phase result gives one tie-line; they are sorted by
η = ferrite x_X and interpolated linearly in η, in mole fractions.
(Fixing the BCC phase amount instead failed in TC with "NO PHASE WITH VARIABLE
AMOUNT".)

**Diffusivities.** TC `DC` matrices on a (x_C, x_X) grid per phase,
interpolated bilinearly.

**kawin settings**
- Austenite grid: power-law, 401 nodes, exponent 3, so the first cell is about 1e-12 m.
- `composition_dependent_lagged` D, semi-log dt (T0 = 1e-7 s, Δln t = 0.02).
- `minDtFrac=1e-14` is required with such a small first step.

**The t = 0 η solve depends on the grid.** It starts from a step profile:
- The 401/p3 grid picks a negative-velocity root (η = 0.0082, V = −0.9 m/s).
- Finer grids pick the no-partitioning root (η ≈ 0.0191).

In every run, the first steps move η to the no-partitioning value within about
1e-5 s, so growth after that is unaffected.

#### Fe-Ni-C (exh2 geometry, local equilibrium instead of para-equilibrium)

973 K, Fe-2.02Ni-0.0885C wt%, 1e-9 m ferrite + 50 µm austenite, 50 s. 432
tie-lines, η ∈ [0.0033, 0.0245].

**kawin-only result (401/p3):** 267 s, 3222 steps. η goes from 0.0082 to
0.01925; the alloy x_Ni is 0.01918, i.e. no-partitioning LE. Inventory drift is
about 1e-12. The austenite interface Péclet number falls from ~5000 at 1e-7 s
to 3.2 at 50 s, so the Ni spike is **under-resolved**.

| t (s) | kawin LE half-thickness | exh2 experiment (para-eq., reference only) |
|---|---|---|
| 6.16 | 4.15 µm | 4.06 µm |
| 10.27 | 5.35 µm | 5.94 µm |
| 20.55 | 7.56 µm | 10.0 µm |
| 50 | 11.79 µm | — |

**DICTRA cost and accuracy study (1 s simulated)**

The first attempt used tight settings (interface checking plus a 200-point
geometric grid with a ~4 nm first cell). dt stuck near 4e-5 s, about two weeks
to 50 s, so it was stopped. A short study followed:

| DICTRA setting (1 s) | k(0.01) | k(0.1) | k(1) (µm/√s) | Wall time |
|---|---|---|---|---|
| default dt, geometric 100 × 1.05 | 1.415 | 1.291 | 1.459 | 30 s |
| rel 1e-3, growth 1.2, 100 × 1.05 | 1.592 | 1.505 | 1.614 | 18 min |
| rel 1e-3, growth 1.2, 200 × 1.03 | 1.646 | 1.585 | 1.588 | 9.7 h |
| **kawin 401/p3** | 1.689 | 1.680 | **1.674** | ~4 min (to 2 s) |

DICTRA's default time stepping is again about 10% low; the tightened DICTRA
runs give 1.59–1.61.

**DICTRA reference to 2 s** (100 × 1.05 grid, rel 1e-3; 1973 steps, 1 h 17 min)
compared with kawin 401/p3 (`--stage kawin`; 3222 steps, about 4.5 min alone,
11 min while sharing the CPU):

| t (s) | DICTRA k (µm/√s) | kawin k (µm/√s) | Thickness difference |
|---|---|---|---|
| 0.01 | 1.592 | 1.689 | +6.0% |
| 0.1 | 1.502 | 1.680 | +11.8% |
| 0.5 | 1.592 | 1.676 | +5.3% |
| 1 | 1.613 | 1.674 | +3.8% |
| 2 | 1.624 | 1.673 | +3.0% |

kawin's k is essentially constant (1.689 → 1.668 from 0.01 s to 50 s). DICTRA's
k is still rising at 2 s, as in Fe-C, where DICTRA's start-up transient decayed
over several decades of time. The gap shrinks from 5.3% at 0.5 s to 3.0% at
2 s, but we cannot yet tell how much of the remaining 3% is DICTRA's transient
and how much is kawin error.

**kawin refinement for Fe-Ni-C did not converge on finer grids.** Every variant
finer than 401/p3 (801/p3.5, 1601/p3.7, X residual 1e-3, dt/2, 161 ferrite
nodes, first step 1e-5 or 1e-4 s) failed in the first steps with "step failed
after timestep retries". Two mechanisms:

1. **Non-monotone coupled ferrite solve.** The ferrite seed starts strongly
   supersaturated in carbon (x_C 0.0041 vs 0.0007 at the interface, across
   1 nm). The C→Ni coupling (D_NiC/D_NiNi ≈ 0.14) combined with the implicit
   block solve overshoots Ni strongly negative (down to −0.18).
2. **Block-pivot conditioning (risk R3 in the plan).** With a larger first step
   the block-tridiagonal system reaches a condition number of about 1.9e20,
   beyond the 1e12 guard. Fe-Ni-C is the hardest case here: the austenite
   eigenvalue ratio is λ_Ni/λ_C ≈ 5e-8, against 9e-7 for Fe-Mn-C and 1e-6 in
   the hardest similarity test.

The default 401/p3 grid runs (with about one retry per step) but leaves the
austenite interface Péclet number at about 20 at 2 s. A grid-converged kawin
Fe-Ni-C result therefore needs solver work; see section 6.

#### Fe-Mn-C

Fe-1.5Mn-0.1C wt% at 1000 K (two-phase; equilibrium ferrite fraction 0.765),
1e-9 m ferrite + 50 µm austenite, 100 s. 424 tie-lines, η ∈ [0.0015, 0.018].

**kawin-only result (401/p3):** 177 s, 5833 steps.
- Growth is cleanly parabolic: s = 0.706 µm/√s · √t from 0.1 s to 100 s (7.06 µm at 100 s).
- η goes from 0.0043 to 0.01526; the alloy x_Mn is 0.01519, i.e. no-partitioning LE.
- Inventory drift is about 1e-11.

**kawin refinement (to 2 s): grid-converged.** 801/p3.5 with
`--x-residual-scale 1e-3` ran in 16 s (847 steps, 22 retries). Its k(1 s) =
0.70516 µm/√s agrees with 401/p3 (0.70513) to 0.004%. Without the tightened Mn
residual, 801/p3.5 hit the 15-minute cap, so the tightened residual matters for
both robustness and cost. 1601/p3.7 failed (first-step conditioning, as in
Fe-Ni-C).

**DICTRA reference to 2 s** (same settings as Fe-Ni-C; 2 h 2 min) compared
with kawin 801/p3.5 and Mn residual 1e-3 (`--tag n801`; 1085 steps, 41 s for the
full 100 s, inventory drift about 1e-13):

| t (s) | DICTRA k (µm/√s) | kawin k (µm/√s) | Thickness difference |
|---|---|---|---|
| 0.01 | 0.822 | 0.706 | −14% (DICTRA start-up) |
| 0.1 | 0.715 | 0.706 | −1.4% |
| 0.5 | 0.7055 | 0.7052 | −0.04% |
| 1 | 0.7045 | 0.7052 | +0.09% |
| 2 | 0.7018 | 0.7051 | +0.47% |

**Fe-Mn-C agrees with DICTRA to better than 0.5% from 0.2 s to 2 s.** DICTRA's
small downward drift in k by 2 s may be the start of soft impingement or
residual DICTRA time error; kawin's k stays at 0.705 to 100 s.

Outputs: `outputs/fe_mn_c_ferrite_le_n801_comparison.json` and `_thickness.png`.

### Summary of the ternary benchmarks

| System | kawin vs converged DICTRA (local eq., to 2 s) | kawin grid convergence |
|---|---|---|
| Fe-Mn-C, 1000 K | within 0.5% from 0.2 s to 2 s | converged (0.004%) |
| Fe-Ni-C, 973 K | +3.0% at 2 s, gap shrinking (+5.3% at 0.5 s) | finer grids fail in the first steps |

## 5. Problems found along the way

1. **`reset()` returned the state to mole fractions** (all Illingworth models). It was harmless before because u = x for substitutional systems. Fixed.
2. **Per-component η scaling** first saturated (fixed by freezing the scales per iterate), then failed when one component's balance terms nearly vanished (fixed with a round-off scale floor). Both scalings now find the same root on the Fe-Ni-C 401/p3.5 and 801/p3.5 grids. It remains opt-in in the examples (`--eta-componentwise`). These fixes only affect the t = 0 η estimate; the first-step failures of the finer Fe-Ni-C grids (item 8) remain.
3. **DICTRA's default time-step control under-predicts growth** by 10–15% in Fe-C and Fe-Ni-C. Use tightened `TimestepControl` for references.
4. **DICTRA's start-up jump** to about 0.6 µm from a 1e-9 m seed. Compare growth rates from about 100 s (Fe-C), or compare k(t), not early thicknesses.
5. **Strict DICTRA settings can make a run impractical** (interface checking with nm-scale first cells pins dt).
6. **Running scripts:**
   - Start parallel conda activations a few seconds apart (they collide on conda's temp file).
   - Thermo-Calc failures can leave the Python process hanging (its TC Java server keeps it alive) and holding its log file open.
   - Thermo-Calc is unreachable while the "vEthernet (WSL (Hyper-V firewall))" adapter is enabled.
7. **`landau_grid` `min_interval`** (1e-12) limits power-law exponents: 1601 nodes allows p ≤ 3.74.
8. **First steps of the step-start ternary problems on very fine grids.**
   - The t = 0 η balance can give velocities of hundreds of m/s.
   - The coupled ferrite solve can overshoot the slow component negative.
   - Block pivots can reach a condition number of about 1e20.
   - Tightening the slow component's interface residual (`--x-residual-scale 1e-3`) fixed this for Fe-Mn-C at 801/p3.5; a larger first step made it worse.

## 6. Limitations and open items

- The Fe-Ni-C/Fe-Mn-C DICTRA comparison is limited to short times (2 s), because a converged 50/100 s DICTRA LE run is projected at many hours to days.
- **Fe-Ni-C kawin is not grid-converged**: finer grids fail in the first steps (section 4.2). Possible fixes, none implemented yet:
  - Row/column equilibration of the 2×2 block pivots when the conditioning guard trips (plan risk R3).
  - A smoother start, e.g. a ferrite seed at its interface composition instead of the supersaturated alloy, or a short analytic carbon-only start-up.
  - A positivity-preserving treatment of the C→X coupling in the first steps.
- The tie-line closure is linear in η. Near the no-partitioning end the austenite composition changes quickly with η, and the sampling could be refined there.
- The three-phase model has no per-component η option and has not been benchmarked with real thermodynamics.
- The 1e-8 ratio similarity test (`KAWIN_RUN_SLOW=1`) has not been run.
- pyex_D_02 has not been rerun against the tight DICTRA reference.
- Para-equilibrium is out of scope (local equilibrium only).

## 7. How to run

```powershell
# from the repo root; every command through the conda wrapper in AGENTS.md
python -m pytest kawin/tests/test_illingworth_substitutional_baseline.py kawin/tests/test_illingworth_interstitials.py kawin/tests/test_illingworth_interstitials_ternary.py kawin/tests/test_illingworth_interstitials_three_phase.py --basetemp .pytest_tmp\inter

python -u -m examples.Interstitials.check_tc_interstitial_frame
python -u -m examples.Interstitials.fe_c_exb1a --case exb1a
python -u -m examples.Interstitials.fe_ni_c_exh2_le --stage tc-inputs
python -u -m examples.Interstitials.fe_ni_c_exh2_le --stage dictra --dictra-sim-time 2
python -u -m examples.Interstitials.fe_ni_c_exh2_le --stage kawin --dictra-sim-time 2
python -u -m examples.Interstitials.fe_mn_c_ferrite --stage dictra --dictra-sim-time 2
python -u -m examples.Interstitials.fe_mn_c_ferrite --stage kawin --dictra-sim-time 2 --n-right 801 --power 3.5 --x-residual-scale 1e-3 --tag n801
```

Start parallel runs a few seconds apart, because simultaneous conda activations
collide on conda's temp file.

Outputs (JSON, PNG, logs) go to `examples/Interstitials/outputs/`; Thermo-Calc
inputs and DICTRA references are cached there as JSON.
