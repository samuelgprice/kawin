# Interstitial (C, N, O, H, B) support in the Illingworth moving-boundary solvers

Status as of 2026-10-15. Everything here is committed on
`feature/illingworth-interstitial`. The benchmark numbers in section 4 come from
`convergence_study.py`, which extrapolates kawin to zero time step and zero grid
spacing. The diagnosis behind the solver fixes in section 1 is in
`startup_and_retry_diagnosis.md`.

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
- **Interface Newton fixes**, needed for sharp Fe-C-X fronts. They apply to all
  ternary two-phase runs; see `startup_and_retry_diagnosis.md`.
  - **Upwind branch per trial.** Every trial is solved with the upwind direction
    of its own interface displacement. Before, the direction was fixed per Newton
    iteration, so trials on the other side of zero displacement were solved
    downwind. With nm-scale interface cells that gave oscillating, negative Ni
    profiles and expensive dense fallback solves.
  - **Kink-aware Newton.** Each iteration stays on one side of zero
    displacement, and switches sides only at it, after re-measuring that side's
    Jacobian column.
  - **`interface_newton_start` (default `"extrapolated"`).** Each step's Newton
    starts from the last accepted step extrapolated to the new `dt`.
    `"old_state"` restores the previous start.
  - **η finite-difference step (`_INTERFACE_ETA_FD_STEP = 1e-6`).** The η step
    in the scaled tie-line coordinate used to be sqrt(eps) ≈ 1.5e-8. On
    interface cells of about 1e-17 m the carbon flux loses about 7 digits to
    cancellation, and that noise divided by the tiny step swamped the carbon
    row's weak η dependence. On Fe-Mn-C at 3201 nodes this gave a wrong Newton
    direction and hundreds of retries. The s step is unchanged.

  Together these took the 801-node Fe-Ni-C run to 2 s from 34 min (thousands of
  retries) to about 4 s (no retries). They made the 1601-node grids practical.

**`MovingBoundaryIllingworthTernaryThreePhaseFDM.py`**

- Same guard, diffusivity query, closure, bounds, `reset()` and output changes as the two-phase model.
- `phase_molar_volumes` are now interpreted as volumes per mole of substitutional atoms.
- The total-moles check and `_dependent_mole_closure_error` sum only the substitutional components.
- It does **not** have the per-component initial-η option.

**`mesh/MovingBoundaryIllingworthTernaryFD1D.py`**

- `integrate_planar_transformed_molar_inventories` takes an optional `substitutional_mask`.
- **Row equilibration after a pivot failure.** When the block-Thomas pivot check
  fails, the system is row-equilibrated, an exact rescaling, and solved again.
  If that also fails, the dense equilibrated system is solved. Systems that
  pass the first check are solved exactly as before.

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

**Ternary check, including the cross term.**
`check_tc_interstitial_frame_ternary.py` (~3 min) repeats the test for the full
2×2 `DC` matrix in Fe-Ni-C austenite: 973 K, 10 µm, cosine Ni and C starts,
60 s. The C–Ni cross term drives uphill carbon diffusion, and it controls how
carbon responds to the Ni spike in the ferrite-growth benchmark.

| Finite-volume variant vs DICTRA | Max abs. error in u_C |
|---|---|
| u-frame | **2.4e-7** |
| x-frame | 2.9e-6 |
| cross terms removed | 1.7e-4 |
| D_CNi sign flipped | 3.3e-4 |

**Conclusion:** TC's `DC` and kawin's `interdiffusivity` use the same u-basis,
volume-fixed frame, including the sign and size of the cross terms, so **no
correction is needed in the adapter**.

## 3. Tests

All Python runs go through the conda activation wrapper (see AGENTS.md).

| Test file | What it checks | Runtime |
|---|---|---|
| `test_illingworth_substitutional_baseline.py` (+ `illingworth_baseline_cases.py`, `make_illingworth_substitutional_baseline.py`, `data/illingworth_substitutional_baseline.npz`) | Binary in all three D modes, two-phase ternary (uniform/lagged/implicit, both motion directions) and three-phase (uniform/lagged). Every recorded history is **bitwise equal** (NaN-aware) to the stored baseline. That baseline is the pre-interstitial code, except for the two documented ternary regenerations below. | ~7 s |
| `test_illingworth_interstitials.py` | Shared module: u↔x round trips vs `Mobility` (1e-14), bounds identical to the legacy simplex, wrappers. Exact-case interstitial matching and rejection of lower-case interstitial symbols. Binary Fe-C: u after `__init__` and after `reset()`, providers receive x in all three D modes, Fe-C vs Ni-Cr label invariance, and an analytic u-fraction **Zener** test. | ~5 s |
| `test_illingworth_interstitials_ternary.py` | Fe-C-Mn-like plumbing (x and u closures, queries in x, mole-fraction outputs, diagnostics, bounds, dependent-element guard). **Disparate-eigenvalue similarity tests** at Mn/C ratios 1e-2, 1e-4, 1e-6. A 1e-8 case is gated by `KAWIN_RUN_SLOW=1` and has not been run. Initial-η branch selection (converged branch preferred over a smaller raw residual) and the per-component scale floor (ratio 1e-10, with a floor-disabled negative control). | ~50 s |
| `test_illingworth_interstitials_three_phase.py` | Masked molar inventory vs hand integration. Fe-C-Mn three-phase: u state, queries in x, conservation of component moles, substitutional-only total-moles closure, mole-fraction outputs. | ~6 s |

**Interface-Newton tests** (in `test_illingworth_ternary.py`):
- every trial uses the upwind branch of its own displacement on a receding
  interface;
- finite-difference perturbations stay on the iterate's side of zero
  displacement;
- the two `interface_newton_start` modes agree;
- an unknown start option is rejected.

The block-solver row equilibration has its own tests.

**Baseline regenerations.** The substitutional baseline was regenerated three
times, each time with the reason recorded in
`make_illingworth_substitutional_baseline.py`:
- for the upwind-branch fix: only the receding ternary case moved, by 1e-10;
- for the extrapolated-start default: the four ternary cases moved by at most
  2e-10;
- for the η finite-difference step: the four ternary cases moved by at most
  3e-13.

Binary and three-phase stay bitwise identical.

**Full suite:** 1031 passed, 2 skipped (about 5 min).

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

**kawin numbers.** These come from `convergence_study.py fe-c`. It uses
austenite grids of 801, 1601, 3201 and 6401 nodes (one v² mapping) and
Δln t = 0.02, 0.01, 0.005 and 0.0025, and extrapolates to zero. Each run takes
3–33 s.
- **Time-step error:** first order (observed ratios 1.97–2.01) and small:
  k(1e3 s) changes by 0.06% per halving.
- **Grid error:** first order. The observed ratios are 1.975–1.993 for the
  coarser triple and 1.988–1.997 for the finer one, so they approach 2.
- **Extrapolation:** the first-order and observed-order extrapolations agree to
  0.001 µm. The extrapolated value differs from the finest run by at most 0.07%
  (at 100 s).

| t (s) | DICTRA (tight) | DICTRA (default) | kawin (dt, h → 0) | kawin vs tight |
|---|---|---|---|---|
| 1e2 | 19.39 µm | 16.75 µm | 19.84 µm | +2.3% |
| 1e3 | 62.01 µm | 52.78 µm | 62.75 µm | +1.2% |
| 1e4 | 198.08 µm | 166.47 µm | 198.42 µm | +0.17% |
| 1e5 | 624.6 µm | 530.1 µm | 627.4 µm | +0.46% |
| 1e9 (final) | 1.37898 mm | 1.37642 mm | 1.37957 mm | +0.043% |

**The final state checks conservation exactly.** At equilibrium the ferrite
thickness follows from a mass balance alone (the lever rule).
- **u-fraction lever rule:** 1.379570 mm, which kawin reproduces to 1e-6 on
  every grid.
- **Mole-fraction lever rule:** would give 1.3709 mm, another confirmation of
  the u frame.
- **DICTRA:** settles 0.043% short of the lever rule. That is a DICTRA-side
  mass-balance or interface-composition error, not a growth-rate difference.

**Other results**
- **Carbon profiles away from the interface:** max |Δu_C| = 2.3e-4 at 1e3 s, 7.8e-5 at 1e5 s, 2.3e-6 at 1e9 s.
- **kawin cost:** 3 s at the script's default Δln t = 0.02 (1743 steps); inventory drift 1e-12.
- **kawin convergence:** the lagged and implicit D modes and 4× ferrite nodes all
  agree, so the grid and time-step study above captures the remaining error.
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
`--semi-log-t0`, `--x-residual-scale`, `--eta-componentwise`, `--newton-start`
and `--tag` (suffix for the output files).

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
- **Austenite grid:** power-law. The driver default is 401 nodes with exponent 3
  (first cell about 1e-12 m). The convergence study uses 401/p3, 801/p3.5,
  1601/p3.7 (first cell about 7e-17 m) and 3201/p3.7 (about 5e-18 m, needs
  `grid_min_interval=1e-14`). The last step is a clean doubling at a fixed
  exponent.
- **Diffusivity:** `composition_dependent_lagged` D.
- **Time steps:** semi-log dt, with T0 = 1e-7 s, and the extrapolated Newton
  start (the default).
- **Step size matters.** The script's default Δln t = 0.02 carries about −0.26%
  time error in k. Use Δln t ≤ 0.005 (about −0.07%) or Richardson extrapolation
  for benchmark numbers.
- **`minDtFrac=1e-14`** is required with such a small first step.
- **Cost:** with the extrapolated start, a run to 2 s takes about 3–30 s with no
  retries.

**The t = 0 η solve depends on the grid.** It starts from a step profile:
- The 401/p3 grid picks a negative-velocity root (η = 0.0082, V = −0.9 m/s).
- Finer grids pick the no-partitioning root (η ≈ 0.0191).

In every run, the first steps move η to the no-partitioning value within about
1e-5 s, so growth after that is unaffected.

**Semi-analytical reference (no discretization error).**
`npl_similarity_reference.py` computes the parabolic constant for
no-partitioning local-equilibrium growth from the thin-spike asymptotics
(Coates; Kirkaldy). The slow-solute spike is about 1e-7 of the carbon diffusion
length, so carbon reduces to a binary-like similarity problem:
- D_CC(u_C) is taken at the alloy X content;
- the ferrite end of the operating tie-line has the alloy X content;
- the carbon content just beyond the spike comes from TC iso-activity, or,
  cross-checked, from integrating du_C/du_X = −D_CX/D_CC across the spike.

The problem is solved by shooting, with `solve_bvp` as a check. It uses the same
thermodynamics and DC tables as kawin and DICTRA, so it is an independent test
of both.

#### Fe-Ni-C (exh2 geometry, local equilibrium instead of para-equilibrium)

973 K, Fe-2.02Ni-0.0885C wt%, 1e-9 m ferrite + 50 µm austenite, 50 s. 432
tie-lines, η ∈ [0.0033, 0.0245].

**kawin to 50 s (1601/p3.7, Δln t = 0.005):**
- 18 s, 4033 steps, no retries.
- η stays at 0.019249, the no-partitioning value (alloy x_Ni = 0.01918).
- k = 1.6649 µm/√s at 0.1 s and 1.6628 at 50 s.

| t (s) | kawin LE half-thickness | exh2 experiment (para-eq., reference only) |
|---|---|---|
| 6.16 | 4.13 µm | 4.06 µm |
| 10.27 | 5.33 µm | 5.94 µm |
| 20.55 | 7.54 µm | 10.0 µm |
| 50 | 11.76 µm | — |

**DICTRA cost and accuracy study (1 s simulated)**

The first attempt used tight settings (interface checking plus a 200-point
geometric grid with a ~4 nm first cell). dt stuck near 4e-5 s, about two weeks
to 50 s, so it was stopped. A short study followed:

| DICTRA setting (1 s) | k(0.01) | k(0.1) | k(1) (µm/√s) | Wall time |
|---|---|---|---|---|
| default dt, geometric 100 × 1.05 | 1.415 | 1.291 | 1.459 | 30 s |
| rel 1e-3, growth 1.2, 100 × 1.05 | 1.592 | 1.505 | 1.614 | 18 min |
| rel 1e-3, growth 1.2, 200 × 1.03 | 1.646 | 1.585 | 1.588 | 9.7 h |
| **kawin (dt, h → 0)** | 1.6625 | 1.6625 | **1.6625** | about 13 min for the whole study |

DICTRA's default time stepping is again about 10% low; the tightened DICTRA
runs give 1.59–1.61.

**Convergence study** (`convergence_study.py fe-ni-c`, to 2 s, Δln t = 0.01,
0.005, 0.0025 and 0.00125; about 13 min). There were no retries except 6 on the
3201 grid at Δln t = 0.01.

| Grid | k(2 s), dt → 0 (µm/√s) | vs reference (iso-activity) |
|---|---|---|
| 401/p3 | 1.67317 | +0.65% |
| 801/p3.5 | 1.66725 | +0.30% |
| 1601/p3.7 | 1.66479 | +0.15% |
| 3201/p3.7 | 1.66363 | +0.080% |
| **dt and h → 0, first order** | **1.66247** | **+0.010%** (−0.001% vs cross term) |
| dt and h → 0, observed order | 1.66259 | +0.017% (+0.006% vs cross term) |
| semi-analytical reference | 1.66231 (iso-activity) / 1.66249 (cross term) | — |
| DICTRA (tight, still rising) | 1.6243 | −2.3% |

**Observed orders.**
- **Time step:** the ratio of successive differences is 1.91–2.07, i.e. first
  order.
- **Grid:** the mixed-exponent triple (401/801/1601) gives 2.4–2.9. The clean
  fixed-exponent doubling (801/1601/3201) gives 2.12–2.27, close to first
  order. The error halves at each level (0.65 → 0.30 → 0.15 → 0.08%).

**Results against time.** Extrapolated to zero step and spacing, kawin's k is
1.66247–1.66249 µm/√s at every time from 0.01 s to 2 s. DICTRA is lower by 4.4%
at 0.01 s, 10.7% at 0.1 s, 4.4% at 0.5 s, 3.0% at 1 s and 2.3% at 2 s, and it
is still rising.

**The semi-analytical reference decides between them: kawin is correct, and
DICTRA's remaining gap is its start-up transient plus its own discretization
error.** Earlier versions of this README reported kawin +3% from DICTRA with
finer grids failing. That was a solver problem, since fixed (section 1); the
diagnosis is in `startup_and_retry_diagnosis.md`.

#### Fe-Mn-C

Fe-1.5Mn-0.1C wt% at 1000 K (two-phase; equilibrium ferrite fraction 0.765),
1e-9 m ferrite + 50 µm austenite, 100 s. 424 tie-lines, η ∈ [0.0015, 0.018].

**kawin to 100 s (1601/p3.7, Δln t = 0.005):**
- 27 s, 4623 steps, no retries.
- Growth is cleanly parabolic: k = 0.7055–0.7058 µm/√s from 0.1 s to 100 s
  (7.055 µm at 100 s).
- η stays at 0.015256, the no-partitioning value (alloy x_Mn = 0.01519).
- Inventory drift is below 1e-13.

**Convergence study** (`convergence_study.py fe-mn-c`, to 2 s, same grids and
steps):

| Grid | k(2 s), dt → 0 (µm/√s) | vs reference (iso-activity) |
|---|---|---|
| 401/p3 | 0.70729 | +0.24% |
| 801/p3.5 | 0.70630 | +0.098% |
| 1601/p3.7 | 0.70587 | +0.036% |
| 3201/p3.7 | 0.70565 | +0.006% |
| **dt and h → 0, first order** | **0.70544** | −0.024% (+0.002% vs cross term) |
| dt and h → 0, observed order | 0.70545 | −0.022% (+0.003% vs cross term) |
| semi-analytical reference | 0.70561 (iso-activity) / 0.70543 (cross term) | — |

**Observed orders.**
- **Time step:** 1.79–1.98 over the three finest steps.
- **Grid:** 2.06–2.16 for the clean 801/1601/3201 doubling; 2.25–2.83 for the
  mixed-exponent triple.

**Retries at the coarsest step only.** There were no retries at Δln t ≤ 0.005
on any grid. At Δln t = 0.01 there was 1 retry on the 801 grid, 476 on 1601
and 2967 on 3201. Those runs hit the solver's resolution limit (section 6),
which corrupts the observed time ratios that involve them. The extrapolation
uses only the two finest steps, so it is unaffected.

**DICTRA reference to 2 s** (same settings as Fe-Ni-C; 2 h 2 min) against the
extrapolated kawin result:

| t (s) | DICTRA k (µm/√s) | kawin k (dt, h → 0) | kawin vs DICTRA |
|---|---|---|---|
| 0.01 | 0.822 | 0.7055 | −14% (DICTRA start-up) |
| 0.1 | 0.715 | 0.7054 | −1.4% |
| 0.5 | 0.7055 | 0.7054 | −0.01% |
| 1 | 0.7045 | 0.7054 | +0.13% |
| 2 | 0.7018 | 0.7054 | +0.52% |

**Fe-Mn-C:** kawin agrees with the semi-analytical reference to within 0.024%,
inside the spread of its two variants. DICTRA agrees with both to 0.15% from
0.5 s to 1 s and then drifts about 0.5% low by 2 s. That drift is DICTRA's
residual time error or the start of soft impingement on its side; kawin's k
stays constant to 100 s.

Outputs: `outputs/*_convergence.json` and `outputs/*_kawin_only_n1601_dt0p005_comparison.json`.

### Summary of the benchmarks

| System | kawin (dt, h → 0) vs semi-analytical reference | kawin vs tight DICTRA |
|---|---|---|
| Fe-C exb1a, 1050 K | final state equals the u-fraction lever rule | +1.2% at 1e3 s (DICTRA start-up), +0.17% at 1e4 s, +0.04% final |
| Fe-Mn-C, 1000 K | −0.024% / +0.002% (iso-activity / cross term) | within 0.15% from 0.5 s to 1 s, +0.5% at 2 s |
| Fe-Ni-C, 973 K | +0.010% / −0.001% (iso-activity / cross term) | +2.3% at 2 s, DICTRA still rising |

Where an independent check exists (the reference for the ternaries, the lever
rule for Fe-C), kawin matches it to within about 0.02%, which is inside the
spread between the reference's two variants. The kawin–DICTRA differences are
therefore DICTRA's start-up transient and discretization error.

## 5. Problems found along the way

1. **`reset()` returned the state to mole fractions** (all Illingworth models). It was harmless before because u = x for substitutional systems. Fixed.
2. **Per-component η scaling** first saturated (fixed by freezing the scales per iterate), then failed when one component's balance terms nearly vanished (fixed with a round-off scale floor). Both scalings now find the same root on the Fe-Ni-C 401/p3.5 and 801/p3.5 grids. It remains opt-in in the examples (`--eta-componentwise`). These fixes only affect the t = 0 η estimate.
3. **DICTRA's default time-step control under-predicts growth** by 10–15% in Fe-C and Fe-Ni-C. Use tightened `TimestepControl` for references.
4. **DICTRA's start-up jump** to about 0.6 µm from a 1e-9 m seed. Compare growth rates from about 100 s (Fe-C), or compare k(t), not early thicknesses.
5. **Strict DICTRA settings can make a run impractical** (interface checking with nm-scale first cells pins dt).
6. **Running scripts:**
   - Start parallel conda activations a few seconds apart (they collide on conda's temp file).
   - Thermo-Calc failures can leave the Python process hanging (its TC Java server keeps it alive) and holding its log file open.
   - Thermo-Calc is unreachable while the "vEthernet (WSL (Hyper-V firewall))" adapter is enabled.
7. **`landau_grid` `min_interval`** (1e-12) limits power-law exponents: 1601 nodes allows p ≤ 3.74, and 3201 nodes p ≤ 3.42. The example driver's `run_kawin(..., grid_min_interval=...)` lowers it, for example to 1e-14 for 3201/p3.7.
8. **Fine-grid ternary runs failed in their first steps and were slow** (fixed).
   - **Cause:** the interface Newton loop held the upwind branch fixed per
     iteration, so trials on the other side of zero displacement were solved
     downwind. That produced negative, sign-alternating Ni next to the
     interface, and pivot condition numbers up to 1e20. The negative values
     looked like a coupled-solve overshoot but are not one.
   - **Fixes** (section 1):
     - per-trial upwind branch;
     - kink-aware Newton;
     - extrapolated Newton start;
     - row equilibration after a pivot failure.
   - **Result:** all grids up to 1601/p3.7 now run with no retries. Full
     analysis: `startup_and_retry_diagnosis.md`.
9. **Retries had been hiding time-step error.** Before the fixes, retries split
   most steps into smaller ones. With full-size steps, k at Δln t = 0.02 is
   about 0.26% low. Earlier grid-convergence numbers mixed time and space error
   of opposite sign. Use Δln t ≤ 0.005 or Richardson extrapolation
   (`convergence_study.py`).
10. **Fe-Mn-C stalled on the 3201/p3.7 grid** (fixed). The interface Newton
    gave up after one iteration at a residual just above the 1e-10 tolerance,
    with 168–340 retries per run and a non-monotone time-step sequence.
    - **Ruled out:** the residual floor. A careful Newton reaches about 1e-13,
      because the carbon residual's round-off noise is about 5e-13.
    - **Cause:** the noise divided by the old sqrt(eps) η step swamped the
      carbon row's weak η dependence. The Jacobian entry came out 8–8000×
      wrong, sometimes with the wrong sign, in a nearly singular (condition
      number 1e7) system.
    - **Fix:** a 1e-6 η step (`_INTERFACE_ETA_FD_STEP`) removed every retry at
      Δln t ≤ 0.005, and the grid ratio became 2.06–2.16.

## 6. Limitations and open items

- **Short DICTRA comparison.** The Fe-Ni-C and Fe-Mn-C DICTRA comparison stops
  at 2 s, because a converged 50/100 s DICTRA LE run is projected at many hours
  to days. The semi-analytical reference covers the parabolic regime instead.
- **Long-time Fe-Mn-C not studied.** Fe-Mn-C beyond 100 s, including any switch
  to Mn-partitioning growth, has not been run.
- **First step from 1e-4 s fails.** A first semi-log step of 1e-4 s still fails
  on fine grids: the first step has no motion history for the extrapolated start
  and has to cross zero displacement. A first step of 1e-7 s works and costs
  nothing. Steps 2a/2b in `startup_and_retry_diagnosis.md` would address it.
- **Resolution limit on extremely small interface cells.** The interface fluxes
  are difference quotients across the first cell, so cells of about 1e-16 m
  and below lose digits to cancellation.
  - **Fe-Mn-C at 3201/p3.7 (about 5e-18 m):** the scaled carbon residual's
    noise is about 5e-13 at Δln t ≤ 0.005 but about 1e-10, the tolerance
    itself, at Δln t = 0.01. The 1601 grid (about 7e-17 m) also reaches it at
    Δln t = 0.01.
  - **Effect:** no Newton variant can converge there, so steps are retried at
    smaller dt. A careful central-difference Newton also stalls at
    2e-11–1.6e-10.
  - **How to avoid it:** use smaller steps, or interface cells above about
    1e-16 m. 1601/p3.7 at Δln t ≤ 0.005 already reaches the references to
    within 0.04%.
  - **Removing the limit** would mean solving the bulk profiles as deviations
    from the interface value. That is not planned.
- **Three-phase model:**
  - it starts each Newton iteration from the old state (it already chooses the
    upwind direction per face from each trial's geometry);
  - it has no per-component η option;
  - it has not been benchmarked with real thermodynamics or a sharp-front case.
- **Tie-line closure.** It is linear in η. Near the no-partitioning end the
  austenite composition changes quickly with η, and the sampling could be
  refined there.
- **Untested items:**
  - the 1e-8 ratio similarity test (`KAWIN_RUN_SLOW=1`) has not been run;
  - pyex_D_02 has not been rerun against the tight DICTRA reference (only its
    old v1 TC-input cache exists).
- **Out of scope / not enforced:**
  - para-equilibrium (local equilibrium only);
  - a site-fraction cap on interstitials;
  - unequal or varying molar volumes per substitutional mole.

## 7. How to run

```powershell
# from the repo root; every command through the conda wrapper in AGENTS.md
python -m pytest kawin/tests/test_illingworth_substitutional_baseline.py kawin/tests/test_illingworth_interstitials.py kawin/tests/test_illingworth_interstitials_ternary.py kawin/tests/test_illingworth_interstitials_three_phase.py --basetemp .pytest_tmp\inter

# Thermo-Calc frame checks (need a TC license)
python -u -m examples.Interstitials.check_tc_interstitial_frame
python -u -m examples.Interstitials.check_tc_interstitial_frame_ternary

# Fe-C exb1a (TC inputs and DICTRA references are cached on the first run)
python -u -m examples.Interstitials.fe_c_exb1a --case exb1a

# Fe-C-X: TC inputs, DICTRA reference, then kawin against it
python -u -m examples.Interstitials.fe_ni_c_exh2_le --stage tc-inputs
python -u -m examples.Interstitials.fe_ni_c_exh2_le --stage dictra --dictra-sim-time 2
python -u -m examples.Interstitials.fe_ni_c_exh2_le --stage kawin --dictra-sim-time 2 --n-right 1601 --power 3.7 --semi-log-dt 0.005 --tag n1601
python -u -m examples.Interstitials.fe_mn_c_ferrite --stage dictra --dictra-sim-time 2
python -u -m examples.Interstitials.fe_mn_c_ferrite --stage kawin --dictra-sim-time 2 --n-right 1601 --power 3.7 --semi-log-dt 0.005 --tag n1601

# Semi-analytical NPLE references (TC needed for the iso-activity step)
python -u -m examples.Interstitials.npl_similarity_reference ni
python -u -m examples.Interstitials.npl_similarity_reference mn

# Time-step and grid convergence with Richardson extrapolation (cached inputs only, a few minutes each)
python -u -m examples.Interstitials.convergence_study fe-c
python -u -m examples.Interstitials.convergence_study fe-ni-c
python -u -m examples.Interstitials.convergence_study fe-mn-c
```

Start parallel runs a few seconds apart, because simultaneous conda activations
collide on conda's temp file.

Outputs (JSON, PNG, logs) go to `examples/Interstitials/outputs/`; Thermo-Calc
inputs and DICTRA references are cached there as JSON.
