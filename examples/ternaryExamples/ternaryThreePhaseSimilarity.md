# Asymmetric three-phase similarity solution (A | B | C)

This is the exact reference used to validate
`MovingBoundaryIllingworthTernaryThreePhaseFD1DModel` on problems where the
two interfaces behave differently. Code:
[`ternaryThreePhaseSimilarity.py`](ternaryThreePhaseSimilarity.py). Test:
`kawin/tests/test_illingworth_ternary_three_phase_similarity.py`.

The symmetric γ|α|γ check (`examples/Interstitials/three_phase_symmetric_check.py`)
cannot detect a whole class of errors. With mirror symmetry the middle phase
carries no net flux, both interfaces have the same speed, and the
phase-frame terms cancel. The solution below has none of that symmetry.

## 1. Problem

The domain is planar and infinite. The middle phase B lies between two outer
phases, A on the left and C on the right:

```
   A (semi-infinite)  | B (layer) |  C (semi-infinite)
                    s_AB        s_BC
```

**Bulk equations.**
- Each phase p obeys ∂C/∂t = D_p ∂²C/∂x², where D_p is a constant 2×2 matrix
  with real, positive, distinct eigenvalues. C holds the two independent
  mole fractions.
- The flux is J = −D_p ∂C/∂x. Rows are flux components and columns are
  gradient components, as in the solver.
- Molar volumes are equal and constant, and there is no bulk flow.

**Far fields.** C → c_A^∞ as x → −∞ and C → c_C^∞ as x → +∞. These may
differ.

**Local equilibrium.** Each interface has its own closure:
- at s_AB: (c_A, c_B) = Φ_AB(η_AB);
- at s_BC: (c_B, c_C) = Φ_BC(η_BC).

**Stefan conditions.** There are two per interface (one per solute):

  v (c_right − c_left) = J_right − J_left

This sign convention comes from conservation: d/dt of the integral of C
across a moving interface must vanish.

**Initial state.** At t = 0, B has zero width at s0, so both interfaces start
from the same point.

## 2. Similarity form

With z = (x − s0)/√t, the interfaces are s_AB = s0 + k_AB √t and
s_BC = s0 + k_BC √t, with k_AB < k_BC (B has positive width). Their velocities
are v = k/(2√t).

**Decoupling each phase.** Write D_p = P_p diag(λ_p) P_p⁻¹ and w = P_p⁻¹ C.
Each mode then satisfies the scalar equation −(z/2) w′ = λ w″, whose general
solution is w = α + β erf(z / 2√λ).

**Choosing α and β in each phase:**

| Phase | Conditions | Solution (per mode) |
|---|---|---|
| A | w → w^∞ as z → −∞; w = w_A at z = k_AB | C = c_A^∞ + P_A [ a_A ⊙ erfc(−z/2√λ_A) ] |
| C | w → w^∞ as z → +∞; w = w_C at z = k_BC | C = c_C^∞ + P_C [ a_C ⊙ erfc(+z/2√λ_C) ] |
| B | w = w_B,AB at z = k_AB and w = w_B,BC at z = k_BC | w = w_B,AB + (w_B,BC − w_B,AB) · (E(z) − E_AB)/(E_BC − E_AB), where E(z) = erf(z/2√λ_B) |

Here a_A = P_A⁻¹(c_A − c_A^∞) / erfc(−k_AB/2√λ_A), and a_C is defined in the
same way. The outer phases are exactly the two-phase solution
(`ternaryTwoPhaseSimilarity.py`). The middle phase is the new part: it has
two Dirichlet ends instead of one end and one far field.

**Fluxes at the interfaces**, multiplied by √t, with a = k/(2√λ) for each
mode:

- A at s_AB: √t J = −P_A [ √(λ_A/π) b_A / erfcx(−a) ], with b_A = P_A⁻¹(c_A − c_A^∞).
- C at s_BC: √t J = +P_C [ √(λ_C/π) b_C / erfcx(+a) ], with b_C = P_C⁻¹(c_C − c_C^∞).
- B at either interface: √t J = −P_B [ √(λ_B/π) (w_B,BC − w_B,AB) e^{−a²} / (E_BC − E_AB) ].

**Equations to solve.** Every Stefan term scales as 1/√t, so the four
conditions become time independent:

  (k_AB/2)(c_B − c_A) − √t (J_B − J_A) = 0   at s_AB (2 equations)
  (k_BC/2)(c_C − c_B) − √t (J_C − J_B) = 0   at s_BC (2 equations)

That is four equations for (k_AB, k_BC, η_AB, η_BC).

## 3. Numerics in the reference

- **Outer phases:** the scaled complementary error function `erfcx` keeps
  e^{−a²}/erfc(a) finite for any k, as in the two-phase reference.
- **Middle phase:** E_BC − E_AB is formed from erfc differences when both
  arguments have the same sign (`_erf_difference`), so a layer moving fast in
  one direction does not lose digits to cancellation.
- **Root finding:**
  - Bounded least squares in the variables k_AB/√λ_ref, the width rate
    (k_BC − k_AB)/√λ_ref (kept > 0), and η_AB, η_BC.
  - It starts from 7 × 3 × 3 × 3 = 189 starting points.
  - Converged roots are de-duplicated and sorted, as in the two-phase solver.
  - Several roots can exist. The `drift` test case has two: the right-moving
    layer used in the test, and a thin layer drifting left.

**Checks on the reference itself** (`test_three_phase_similarity_reference_self_consistency`):
- the Stefan residual is about 1e-19;
- each phase formula reproduces its interface compositions exactly;
- each phase satisfies its diffusion equation (finite differences);
- the total solute over ±1 around s0 equals the initial step's to 1e-12,
  which checks the signs of all four Stefan conditions independently;
- the mirror image x → 2s0 − x (A↔C, closures flipped and swapped) gives
  (−k_BC, −k_AB) and swapped η values.

## 4. A finding: crossing tie-lines make the problem ill-posed

My first B|C closure was linear, but its tie-lines crossed. A similarity root
existed, yet both the three-phase model and the two-phase model drove η_BC
exponentially to its bound. The two-phase model was run on the same
thin-B | C configuration and failed at exactly the same step. The failure
came sooner on finer grids, and changing dt had no effect.

**Why.** Consider a thin, nearly uniform layer of the left phase (width w)
against the right phase. Conservation of the layer gives

  d(w c_left)/dt = v c_right − J_right,   with v = dw/dt.

If the flux into the right phase responds over a length ℓ to a change in its
interface composition, then J_right ≈ (D_right/ℓ) c_right′(η) δη. Solving the
two components for v and dη/dt gives

  d(δη)/dt ≈ −det(Δc, D_right c_right′) / (ℓ w det(Δc, c_left′)) · δη,

where Δc = c_right − c_left and ′ = d/dη.

**When it is stable.** For an isotropic D the system is stable exactly when
c_left′ and c_right′ lie on the same side of the tie-line, i.e. when
neighbouring tie-lines don't cross. Crossing tie-lines are
thermodynamically impossible.

**Why finer grids fail sooner.** ℓ shrinks with the grid, and in the
continuous problem a fresh perturbation has ℓ → 0. The growth rate therefore
has no bound, which is ill-posedness, not a numerical instability. The
original closure had det(Δc, c_B′) = +0.0085 and det(Δc, c_C′) = −0.0027.

**Guard.** `tie_line_crossing_etas` checks this condition, and
`solve_three_phase_similarity_roots` refuses closures that fail it. Closures
built from real phase diagrams satisfy it automatically. Hand-made
closures for tests should be checked.

## 5. Test design and results

**Setup.**
- The start is at T0 = 0.01 from the exact state.
  `SimilarityInitialProfile` supplies it on a 6001-node mesh; the model
  interpolates each phase separately and pins the interface nodes. A
  zero-width start would be singular.
- Rates are fitted over [4 T0, T_END = 1], domain [0, 0.6], s0 = 0.3. Both
  outer phases stay semi-infinite to T_END.
- Grids: linear clustering, outer spacing_ratio 0.1, middle 0.2, with
  21/41/81 nodes per phase. Time step: semi-log, Δln t = 0.025 (time error
  well below the spatial error).

**k error, relative.** Each cell lists A|B, B|C.

| Case | k (exact) | 21 | 41 | 81 | Richardson 2·k₈₁ − k₄₁ |
|---|---|---|---|---|---|
| opposite | −0.02702, 0.05071 | 2.8%, 6.3% | 1.4%, 3.3% | 0.62%, 1.6% | −0.13%, −0.02% |
| drift | 0.00773, 0.05045 | 2.1%, 6.4% | 1.3%, 3.4% | 0.68%, 1.7% | 0.08%, 0.00% |

**Other results.**
- **η errors** at 81 nodes are at most 0.009 (opposite) and 0.010 (drift).
- **Profile errors** at 81 nodes are at most 0.008.
- **Solute drift** is about 1e-10 in every run.
- **Convergence order** is 0.7–1.0, so the scheme is first order in the grid
  spacing. The two-phase model is also about first order on its own
  similarity problem with these grids (0.54% → 0.26% → 0.10%), but with a
  smaller error constant. The problems differ, so this does not compare
  the two discretisations directly.
- **Runtime:** about 5 s per run, and about 45 s for the whole test file.
