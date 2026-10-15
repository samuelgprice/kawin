# Fe-Ni-C fine-grid start-up failures and retry cost: diagnosis and proposed fixes

This note explains two problems seen in the fine-grid Fe-Ni-C local-equilibrium
benchmark (`ternary_ferrite_growth_le.py`, case `fe_ni_c_exh2_le`, austenite grid
801 points with power 3.5):

1. **Retry cost.** The run needs dt retries on most steps and takes about 34 min to reach 2 s.
2. **Start-up overshoot.** The run fails in its first steps unless the first time
   step is made very small (`semi_log_t0=1e-10`).

The diagnosis below comes from instrumented runs. The probe scripts live in the
session scratchpad and are not part of the repo. The main finding is that **both
problems have one root cause**: the upwind direction used for the interface-motion
term. A one-line experimental change to that choice already removes all of the
negative-composition failures and makes the 801-point run about 28× faster, with
the same answer. What remains is a smaller Newton-convergence issue, and a
proposed fix for it is in Part 2.

Some of what I said earlier was imprecise and is corrected here. I had described
Part 1 as "dt doesn't grow back after retries." The time step does go straight
back to the schedule. The real costs are that the step size has no memory, and
that each failed attempt is expensive. Part 1 explains both.

---

## Background: how one implicit step works

Each time step solves for two interface unknowns: the new interface position
`s_new` and the tie-line coordinate `eta`. It uses a damped Newton iteration in
`_solve_interface_planar`:

- **Start point:** every step starts Newton from the old position, so the
  displacement is `ds = s_new - s = 0`, and from the old `eta`.
- **Candidates:** every trial `(s_new, eta)` is one *candidate*. Evaluating it
  means two implicit bulk solves, one per phase, on the front-fixed (Landau)
  grids, and then the two interface balance residuals (C and Ni).
- **Grid-motion term:** when the interface moves, the transformed grid moves
  with it. This appears in each bulk solve as an advection-like term, which is
  upwinded. The upwind direction is the *motion branch*: `positive` when the
  ferrite grows (`ds > 0`) and `negative` when it shrinks.
- **How the branch is chosen today:** the branch is fixed **once per Newton
  iteration**, from the current iterate's `ds`
  (`_select_interface_motion_branch`). It is then reused for that iteration's
  finite-difference Jacobian perturbations and for every line-search trial
  (`MovingBoundaryIllingworthTernaryFDM.py:2697-2768`).
- **When a candidate fails:** if any candidate fails, because a profile leaves
  the composition simplex or a linear solve is unusable, the **whole time step
  is abandoned**. It is not treated as a bad line-search point. The only
  exception is when the unmoved start already met the tolerance. The step is
  then retried with `dt` halved, up to 12 times.
- **Time schedule:** each attempt targets the next time on the semi-log schedule
  (spacing `Δln t = 0.02`, so each step is about 2% of `t`). The schedule has no
  memory of earlier failures.

---

## Part 1: retry cost

### What the data show

Baseline run: 801/p3.5, `semi_log_t0=1e-10`, to 2 s.

| Quantity | Value |
|---|---|
| Schedule points between 1e-10 s and 2 s | about 1,185 |
| Accepted steps | 3,716 (3.1 per schedule interval) |
| Failed attempts | 4,582 |
| Retries per accepted step | 0: 1,185 steps; 1: 788; 2: 1,414; 3: 288; 4–5: 25 |
| Wall time | 2,048 s (34 min) |

The failures start at about 1e-6 s and continue at a steady rate until the end
of the run. They are not a start-up effect.

A detailed probe of the window t ≈ 1.0–1.2e-5 s logged 507 attempts, of which
250 failed:

- **The failing step size is retried every step.** The first attempt at the full
  schedule gap failed on almost every step. Accepted steps used the full gap
  (87), half of it (92) or a quarter of it (78). Each new step starts again at
  the full remaining gap, even though the step before just showed that size
  fails. So the failure is paid again on every step. When a halved step is
  accepted, the rest of the interval needs further steps, which is why there
  are about 3 accepted steps per schedule interval.
- **Failed attempts are very expensive.** In that window only about 19% of
  candidate evaluations were in failed attempts. The wall time is far more
  lopsided, because a failing candidate's bulk system is badly conditioned (see
  Part 2):
  1. the 2×2 block Thomas sweep rejects a pivot;
  2. the fallback row-equilibrates and retries the sweep;
  3. it then solves the full **dense 1602×1602** system and estimates its
     condition number.

  That fallback costs on the order of a second, against microseconds for a
  normal sweep.

  **Evidence:** with the Part 2 change, which makes those candidates
  well-posed, the run reached 0.01 s in **27 s** with about the same number of
  accepted steps (2,107) and still about one failed attempt per step. The
  baseline took **930 s** to reach the same time.

### Fix A: give the step-size controller a memory (safety net)

After a step needs retries, start the next attempt at

    dt_next = min(schedule_gap, g * dt_last_accepted),   g ≈ 1.5–2

The step then grows back to the schedule gradually. Once an attempt at the full
gap succeeds, the memory is cleared.

**Why it helps:**
- **Standard practice.** It is how most adaptive integrators work, and how
  DICTRA's `TimestepControl` works (growth factor 1.2, which we already use for
  the tight DICTRA references).
- **No repeated failures.** It stops the solver from re-paying a known failure
  on every step.
- **Bitwise safe.** Runs that never retry never touch the memory, so the
  substitutional baseline stays bitwise identical.

**Why it is only a safety net:** Fix A treats the symptom. Once the Part 2
changes are in, the failures should nearly disappear and Fix A will rarely
trigger. If failures remain at about one per step, Fix A would trade failed
attempts for smaller accepted steps, which is roughly a wash. So I'd implement
Part 2 first and add Fix A only if retries remain.

---

## Part 2: the start-up overshoot

### What happens: a concrete example

**Run:** 801/p3.5, a run started at `t0 = 1e-10`, at its second step:

| Quantity | Value |
|---|---|
| Time | t = 1.25e-11 s |
| Step | dt = 8.75e-11 s |
| Ferrite half-thickness | s = 0.998 nm |

**Which candidate failed:** the finite-difference Jacobian perturbation. It moves
the interface by `ds = +7.45e-13 m`, so the ferrite *grows*. But it was evaluated
on the **`negative`** branch, because the current iterate was the unmoved start
and the previous accepted step had receded.

The austenite Ni profile from that bulk solve, next to the interface (u-fractions):

| node | distance from interface | old u_Ni | new u_Ni | new u_Ni, cross terms removed |
|---:|---:|---:|---:|---:|
| 4 | 4.4e-13 m | 0.0252 | 0.0320 | 0.0320 |
| 5 | 9.7e-13 m | 0.0236 | **0.1427** | 0.1427 |
| 6 | 1.8e-12 m | 0.0219 | **−0.2434** | −0.2434 |
| 7 | 3.1e-12 m | 0.0207 | **+0.2545** | +0.2545 |
| 8 | 5.0e-12 m | 0.0199 | **−0.0995** | −0.0995 |
| 9 | 7.6e-12 m | 0.0195 | +0.0595 | +0.0595 |

Three checks on the same candidate:
- **Linear algebra:** a dense `numpy.linalg.solve` of the same system agrees with
  the block solver to 6e-13. The oscillation is the real solution of the
  discrete equations, not a solver error.
- **Cross-diffusion:** zeroing the cross terms (D_CNi, D_NiC) gives the same Ni
  profile. Carbon goes negative in other cases (down to u_C = −7.5) only because
  the D_CNi cross term couples it to this Ni oscillation; that is a consequence,
  not the cause.
- **Interface motion:** the same solve with the interface held fixed (`ds = 0`)
  stays positive (minimum u_Ni = 0.0193). The oscillation comes from the
  interface-motion term.

**How common this is:** every bound or conditioning failure I classified had the
same signature, a trial whose `ds` has the opposite sign to the branch it was
evaluated on:
- all failures in the first steps at `t0 = 1e-4`, `1e-7` and `1e-10`;
- all 250 failures in the 1e-5 s window.

### Why a branch mismatch produces this

In the front-fixed frame, moving the interface by `ds` slides the grid nodes
through the material. The upwind discretization takes each node's new value
from the side the material is coming from. In implicit form, every node becomes
a positive weighted average of its old value and its upstream neighbor, so no
new maxima or minima can appear. For a single component, the matrix is an
M-matrix and positivity is guaranteed.

With the branch reversed, the scheme takes information from the *downstream*
side. Each node's new value is then an extrapolation from its neighbor, not an
average. Any mismatch is amplified from node to node with alternating sign.
That is the +, −, +, − pattern in the table.

How strong this is depends on the cell Courant number, `|ds| / Δz`:

| Grid | First austenite cells next to the interface | Courant number for a trial `ds` of ~1e-12 to 1e-11 m |
|---|---|---|
| 401/p3 | 7.8e-13 m | about 1–10 |
| 801/p3.5 | 3.5e-15, 3.6e-14, 1.2e-13 m | hundreds to thousands |
| 1601/p3.7 | 7e-17 m | far larger |

Ni barely diffuses in a step (D_NiNi ≈ 2e-20 m²/s in austenite), so nothing damps
the oscillation. **This explains the puzzle that the coarser grid worked while
the finer one failed**: a finer interface spacing means a larger Courant number
for the same trial move, and so a more violent downwind oscillation.

### Why Newton keeps proposing trials on the wrong side of `ds = 0`

**1. At start-up the true `ds` really is near zero and changes sign.** The ferrite
seed copies DICTRA's setup: 1 nm at the *alloy* composition. Its carbon
(u_C = 4.1e-3) is about 12× the ferrite equilibrium value (3.5e-4). In the first
~1e-10 s that excess carbon drains through the interface, and the interface
recedes slightly (s goes from 1.000 to 0.996 nm). Only later does it advance.
Newton therefore works right around `ds = 0`, and its trials straddle it.

**2. Later, Newton starts every step at `ds = 0` and overshoots.** Here is one
step at t = 1.007e-5 s. The converged answer is `ds = +2.64e-11 m` with
`eta = 0.019225`.

| Newton iterate | ds (m) | eta | result |
|---|---|---|---|
| start | 0 | 0.019223 | residual 1.9e-8 |
| iteration 1 | +8.3e-12 | 0.020583 | residual 9e-10, accepted |
| iteration 2 | **−3.0e-11** | 0.021863 | evaluated on the `positive` branch, so the profile goes negative and the **whole step is aborted** |

Iteration 1 already moves `eta` the wrong way, by 7% of its table range.
Iteration 2 lands on the wrong side of `ds = 0`. The residual is strongly
nonlinear near `ds = 0`:
- the Ni spike ahead of the interface (width ~D_Ni/v, about 1e-16 m here) is
  unresolved;
- the advective Ni flux switches on with `ds`;
- the tie-line closure is piecewise linear in `eta`.

All of this makes the start point the worst place to linearize.

### Why `t0 = 1e-10` "sort of" fixes it

With a tiny first step, two things save the start-up:

1. **Frozen steps are accepted.** The unmoved start already meets the tolerance,
   because the residual scales with `dt` and the tolerance does not. When a
   Newton trial then fails, the solver accepts the unmoved start instead of
   failing the step (the `forced_update_skipped` path). In the probe, almost
   every other early step was accepted this way: the interface was held still
   for that step.
2. **Trial moves are tiny**, so the downwind oscillations are milder.

**Limitations:**
- **No a-priori rule for `t0`.** On the 801-point grid, `1e-4` and `1e-7` fail
  and `1e-10` works. A finer grid would probably need a smaller value. Each grid
  needs trial and error.
- **Frozen steps are a tolerance loophole, not a solution.** A frozen step
  leaves the step's flux imbalance as a small inventory defect. It is within
  tolerance, but it is not a converged step.
- **It does nothing for the later failures.** Those continue at 1–3 retries per
  step all the way to 2 s, and they are what costs 34 min.
- **It moves further from the DICTRA setup in spirit.** DICTRA's own first step
  is about 1e-4 s or larger.

### The proposed solution

**Step 1, tested: choose the upwind branch per candidate, not per Newton iteration.**

Each candidate, including the finite-difference perturbations and line-search
trials, is evaluated on the branch that matches its *own* displacement. The
consistency the code needs is that the two bulk solves and the residual all use
the same branch, so the face fluxes cancel. That holds per candidate, so
nothing is lost. The upwind matrices stay M-matrices for each component, which
rules out the downwind oscillation.

I tested this as a scratch monkeypatch with no repo changes:

| Run (801/p3.5 unless noted) | Wall time | Accepted steps | Failed attempts | k at 2 s (µm/√s) |
|---|---|---|---|---|
| baseline, t0 = 1e-10 | 2,048 s | 3,716 | 4,582 (bounds / conditioning) | 1.6667 |
| per-candidate branch, t0 = 1e-10 | **74 s** | 3,606 | 4,376 (all cheap no-convergence) | 1.6665 |
| per-candidate branch, t0 = 1e-7 (failed before the change) | **78 s** | 3,277 | 4,441 (all cheap no-convergence) | 1.6665 |
| per-candidate branch, 1601/p3.7, t0 = 1e-7 (not feasible before) | **243 s** | 6,268 | 18,588 (all cheap no-convergence) | 1.6639 |
| semi-analytical reference | – | – | – | 1.6623 |

**A bonus result:** the 1601-point run was impractical before and now takes
4 min. It gives the first three-grid sequence measured against the reference:

| Grid | Error in k vs the reference |
|---|---|
| 401/p3 | +0.6% |
| 801/p3.5 | +0.25% |
| 1601/p3.7 | +0.10% |

kawin converges monotonically toward the semi-analytical value. On the
1601-point grid, though, the failed attempts rise to about 3 per accepted step.
Step 2 below matters more as the grid is refined.

What Step 1 achieves:
- **Negative compositions are gone.** No candidate is rejected for bounds or
  conditioning anymore.
- **The dense fallback no longer runs**, which is where the speed-up comes from.
- **`t0 = 1e-7` now works on the 801-point grid**, which failed before.
- **The answer is unchanged.** At 2 s it agrees with the baseline to 0.02%.

What Step 1 leaves:
- **About one failed attempt per step**, now as cheap Newton non-convergence
  (best residual just above the 1e-10 tolerance).
- **`t0 = 1e-4` still fails.** At start-up the root sits near `ds = 0`, where the
  per-candidate residual has a kink: the two branches agree at `ds = 0` but have
  different slopes on each side. Newton then zig-zags across the kink.

**Step 2, proposed: make Newton kink-aware and start it near the answer.** This
has three parts.

- **(a) Branch-constrained Newton.** Solve on one branch at a time, with `ds`
  bounded to that branch's side of zero. The existing bounded-Newton machinery
  (`_bounded_scaled_newton_step`) already handles bounds. On each side the
  residual is smooth, so Newton behaves.
  - Try the branch of the predicted motion first.
  - If Newton converges to the `ds = 0` bound with a nonzero residual, the root
    is on the other side, so switch.
  - This mirrors what the initial-η estimator already does with its motion
    branches (`_select_converged_branch`).
- **(b) Treat an infeasible trial as a rejected line-search point.** Keep halving
  it, rather than aborting the whole time step. One bad trial then costs one
  evaluation instead of a full retry at half `dt`.
- **(c) Predictor start.** Start Newton from the previous step's motion, scaled
  to the new `dt`. For parabolic growth, `ds ≈ ds_prev · dt/dt_prev` to within
  about 1% per step. Start `eta` at `eta_prev`. Newton then begins about 1%
  from the answer instead of at `ds = 0`, which is the most nonlinear point. It
  should converge in 1–2 iterations instead of about 4–10, and its trials should
  not come near `ds = 0` after start-up.
  - The existing "at least one Newton update" rule stays, so a predicted start
    is never accepted unexamined.
  - The predictor only sets where Newton starts, not what it converges to.

**Why I expect this to work:**
- **Step 1 is already demonstrated.** Every classified failure was a branch
  mismatch. Removing the mismatch removed every bounds and conditioning failure,
  cut wall time 28×, and left k unchanged.
- **Step 2 targets the one failure mode left.** That is Newton non-convergence
  near the start point and near the `ds = 0` kink:
  - (a) removes the kink from each solve;
  - (c) keeps later steps away from it entirely;
  - (b) makes any leftover bad trial cheap instead of fatal.
- **The pieces already exist.** Bounded Newton and branch selection are both in
  the code, so this is reuse rather than new numerics.

**Risks and checks:**
- **Bitwise baseline.** If any baseline run evaluated a trial on the mismatched
  branch, Step 1 changes its results slightly, and arguably makes them more
  correct. The substitutional baseline test will show this. If it does, either
  regenerate the baseline with a documented reason, or gate the change behind
  an option, default off, until it is validated.
- **Root very close to `ds = 0`** at start-up: (a) has to handle a root lying
  exactly on the boundary, where both branches agree. This needs a unit test.
- **Predictor in the shrinking regime** (start-up, or a dissolving phase):
  extrapolating `ds_prev` is still the right guess there. Fall back to `ds = 0`
  when the previous step was frozen or retried.
- **Order of work:**
  1. Step 1.
  2. Step 2b (cheapest).
  3. Step 2c.
  4. Step 2a.
  5. Fix A from Part 1, only if retries remain.

  After each item, check bitwise baselines, the existing ternary tests, and the
  801-point Fe-Ni-C run, comparing wall time, failed attempts and k at 2 s.

---

## Update: Step 1 implemented, and the time-step fix reassessed

### What was implemented

All changes are in `_solve_interface_planar`
(`MovingBoundaryIllingworthTernaryFDM.py`).

- **Per-candidate branch.** Every candidate is evaluated on the motion branch of
  its own displacement. This covers the iterate, the finite-difference
  perturbations and the line-search trials.
- **Minimal kink handling, which turned out to be necessary.** With per-candidate
  branches alone, the bitwise baseline case `ternary_uniform_left` (a receding
  interface) failed at its first step:
  1. Newton starts at the kink, `ds = 0`.
  2. The previous motion is zero, so the iterate's side defaults to positive,
     and the Jacobian is measured on the positive side.
  3. The step heads into negative `ds`.
  4. Those trials are now (correctly) solved on the negative branch, which the
     positive-side Jacobian does not describe, so the line search fails.

  The old code got through this only because it evaluated those trials downwind.

  Each Newton iteration now works on the iterate's side of the kink:
  - its finite-difference perturbations and bounded step stay on that side;
  - at the kink, a step that points into the other side is recomputed with an
    `s` Jacobian column measured there, and the iteration switches sides only if
    the recomputed step still points there.

  An iteration whose steps never approach the kink follows exactly the old path.
  That is why every growing-interface case stays bitwise identical.
- **Baseline.** The substitutional baseline was regenerated. Only
  `ternary_uniform_left` changed, by at most 1e-10 relative, which is
  Newton-tolerance level: it is the same solution reached by a correct Newton
  path. The reason is recorded in `make_illingworth_substitutional_baseline.py`.
- **Regression test.**
  `test_interface_newton_evaluates_every_trial_on_its_own_motion_branch`
  (`test_illingworth_ternary.py`) checks that, on the receding-interface case,
  every evaluated trial's branch matches the sign of its displacement.
- **Two existing tests updated to the new contract.** Both encoded the old
  per-iteration branch freeze from commit 79f1ea9. That commit froze the branch
  so finite-difference perturbations would not switch stencils inside one
  Jacobian, and that goal is kept: perturbations now stay on the iterate's side
  of `ds = 0` instead of being solved on a mismatched stencil.
  - The freeze test is now
    `test_ternary_solver_keeps_finite_difference_jacobian_on_iterate_side_near_zero_motion`.
    It asserts that the `s` perturbation after a receding step goes backward,
    on the negative branch, and that every trial matches its own branch.
  - `test_ternary_interface_reuses_accepted_trial_only_on_same_branch` still
    forces a branch mismatch, and still checks that the reuse guard
    re-evaluates the trial. It now flips the selection made right after the
    accepted trial, rather than the second selector call overall.
- **Full suite:** 1,028 passed, 2 skipped.

### Fe-Ni-C with the implemented fix, to 2 s

| Grid, t0 | Wall time | Accepted steps (schedule points) | Time in accepted / failed attempts | k at 2 s (µm/√s) | k before the fix |
|---|---|---|---|---|---|
| 401/p3, 1e-7 | 60 s | 2,328 (841) | 21 s / 36 s | 1.6726 | 1.6726 (about 460 s) |
| 801/p3.5, 1e-7 | 78 s | 2,855 (841) | 27 s / 48 s | 1.6661 | start-up failure |
| 801/p3.5, 1e-10 | 87 s | 3,271 (1,186) | 31 s / 54 s | 1.6661 | 1.6667 (2,048 s) |
| 801/p3.5, 1e-4 | start-up failure (10 retries, then non-convergence) | | | | start-up failure |

These four runs ran concurrently, so their wall times include some CPU contention.

On the 801-point grid, k agrees with the pre-fix result to 0.04%. That
difference is consistent with the different accepted-dt sequence, which changes
the time-discretization error. The DICTRA-like `t0 = 1e-4` start still fails, as
expected; Step 2 is aimed at that.

### Is the time-step fix (Fix A) still worthwhile?

**The retries are still the largest cost.** All remaining failures are Newton
non-convergence. They take about 62% of wall time, because a failed attempt
runs Newton to its 50-iteration limit. They are spread evenly over every time
decade from 1e-6 to 2 s, so this is a steady-state problem, not a start-up one.
Accepted steps outnumber schedule points by 3.4×.

**Why late-time attempts fail.** I traced one step at t = 0.5 s:
1. Newton starts at `ds = 0` with `eta = 0.019249`.
2. Its first update jumps `eta` to 0.0215, a 10% overshoot across the tie-line
   range, while the converged answer stays at 0.019249.
3. It then wanders for 50 iterations at best residual 9e-8.
4. Retrying at half the step also fails.
5. At a quarter of the step it converges in 6 iterations.

So the full schedule step is not too large for the physics. Newton simply starts
too far from the answer, at zero displacement.

**What Fix A would gain:**
- Starting each step at the dt that last succeeded would avoid most failed
  attempts.
- It would not reduce the step count, which stays at about 3.4× the schedule,
  because it never gets back to full schedule steps.
- Estimated wall time: about 30 s instead of about 78 s on 801, roughly 2.5×.

**What the predictor start (Step 2c) would gain instead:**
- It attacks the cause directly. The previous step's displacement and `eta`,
  scaled to the new `dt`, put Newton within about 1% of the answer.
- That should remove most failures *and* allow full schedule steps: about 841
  steps instead of 2,855, each needing fewer iterations.
- Plausibly a further 5–10× beyond Fix A.

**Assessment.** Fix A is still worthwhile only as a cheap safety net, not as the
next step:
- It is a small, low-risk change.
- Runs that never retry are untouched, so it is bitwise safe.
- It would protect against cases where a large `dt` genuinely fails.

The bigger and more targeted gain is the predictor start. Recommended order:
1. Step 2c (predictor).
2. Re-measure.
3. Step 2a/2b if start-up from `t0 = 1e-4` still fails.
4. Fix A only if a meaningful fraction of attempts still fail after that.

---

## Update 2: extrapolated Newton start (Step 2c) implemented and re-measured

### What was implemented

There is a new constructor option, `interface_newton_start`, with two values:
- `'old_state'` (the default when this update was written; changed in Update 3)
  keeps the existing behavior exactly.
- `'extrapolated'` starts each step's interface Newton iteration from a linear
  extrapolation of the last accepted step, scaled to the new `dt`:
  `s + (s - s_old) * dt / dt_last` and `eta + (eta - eta_old) * dt / dt_last`.

How it behaves:
- **Start only.** It changes where Newton starts, not what a step converges to.
  The residual, the tolerance, the kink handling (which still pivots on the
  unmoved position) and the at-least-one-Newton-update rule are all unchanged.
- **Fallback.** If the extrapolated point cannot be evaluated, Newton starts from
  the old state as before.
- **First step.** The first step has no history, so it always starts from the
  old state.
- **Example driver.** `ternary_ferrite_growth_le.py` exposes it as
  `--newton-start extrapolated`.

Tests:
- **New:** the two start modes agree (to 1e-9 in `s`) for growing and receding
  interfaces without extra evaluations, and an unknown option is rejected.
- **Full suite:** 1,031 passed, 2 skipped.

**Why it is opt-in for now.** Turning it on everywhere passes every test except
the bitwise baseline. All four ternary baseline cases would move by at most
2e-10 relative, which is Newton-tolerance level; binary and three-phase are
unaffected. Making it the default is a one-line change plus a baseline
regeneration.

### Re-measured cost (Fe-Ni-C to 2 s, scheduled step 2% of t)

| Grid, t0 | Old-state start (after the branch fix) | Extrapolated start |
|---|---|---|
| 401/p3, 1e-7 | 60 s, 2,328 steps, 2,653 failed attempts | **3.0 s**, 842 steps, 0 failed |
| 801/p3.5, 1e-7 | 78 s, 2,855 steps, 3,471 failed | **3.7 s**, 842 steps, 0 failed |
| 801/p3.5, 1e-10 | 87 s, 3,271 steps, 3,617 failed | **4.9 s**, 1,189 steps, 2 failed (at t ≈ 1e-12 s) |
| 1601/p3.7, 1e-7 | 243 s (monkeypatch run) | **4.5 s**, 842 steps, 0 failed |
| 801/p3.5, 1e-4 | fails at start-up | still fails at start-up |

What changed:
- **Steps.** Every step is now accepted at the full scheduled size, so there is
  exactly one step per schedule point.
- **Evaluations.** About 4 candidate evaluations per step, which is the minimum:
  start, two Jacobian perturbations and one Newton update.
- **Wall time.** Compared with the original baseline (2,048 s for 801/p3.5),
  that is about 550× faster.

**Start-up from `t0 = 1e-4`.** This still fails, because the first step has no
motion history to extrapolate from. That is the remaining start-up problem;
Steps 2a and 2b are aimed at it.

### An important side effect: the earlier k values contained time-step error

With full-size steps, k at 2 s moved by about −0.2% to −0.3%. A time-step sweep
(`semi_log_dt` from 0.02 down to 0.00125, extrapolated start) shows this is
plain first-order backward-Euler time error. The error halves with each halving
of the step. In every run there were no retries, no forced acceptances, and
inventory drift was below 1e-13.

k at 2 s (µm/√s):

| semi_log_dt | 401/p3 | 801/p3.5 | 1601/p3.7 |
|---|---|---|---|
| 0.02 | 1.66876 | 1.66286 | 1.66041 |
| 0.01 | 1.67096 | 1.66504 | 1.66259 |
| 0.005 | 1.67207 | 1.66615 | 1.66369 |
| 0.0025 | 1.67262 | 1.66670 | 1.66424 |
| 0.00125 | 1.67290 | 1.66697 | 1.66452 |
| **dt → 0** (Richardson) | **1.67317** | **1.66725** | **1.66479** |
| error vs reference 1.66231 | +0.65% | +0.30% | +0.15% |

With time error removed, the spatial error halves with each grid refinement.
Extrapolating 801 → 1601 to zero grid spacing gives **k = 1.66233**. That
agrees with the semi-analytical NPLE reference to:
- **0.002%** for the iso-activity version (1.66231);
- **0.01%** for the cross-term version (1.66249).

This is a much stronger confirmation than the earlier grid sequence.

**Correction to the earlier sequence.** The earlier numbers (+0.6% / +0.25% /
+0.10%) mixed spatial error with time-step error. The retries had made the
effective steps about 3× smaller, which hid part of the time error, and the
time error has the opposite sign to the spatial error, so the two partly
cancelled.

For benchmark-quality numbers, use either:
- `semi_log_dt ≤ 0.005`, which gives a time error of about −0.07%; or
- two step sizes and Richardson extrapolation.

Both cost seconds now.

### Is Fix A (step-size memory) still needed?

No, not for this case. With the extrapolated start there are no retries after
the first ~1e-11 s. Fix A would only matter for a problem where a full
scheduled step genuinely fails, and none has been seen so far. Keep it on the
list as an optional safety net.

What remains:
- **Start-up from DICTRA's first step size (`t0 = 1e-4`).** This needs Step 2a
  (branch-constrained Newton) and/or Step 2b (treat an infeasible line-search
  trial as rejected instead of aborting). With a first step of 1e-7 s or
  smaller, start-up already works and costs essentially nothing.
- **Whether to make the extrapolated start the default.** Since done; see
  Update 3.

---

## Update 3: extrapolated start made the default

`interface_newton_start` now defaults to `'extrapolated'` in the ternary
two-phase model and in the example driver's `run_kawin` and `--newton-start`.
`'old_state'` keeps the previous behavior.

**Tests.** No test had to change except the bitwise baseline. It was
regenerated, and the reason is recorded in
`make_illingworth_substitutional_baseline.py`.
- The four ternary baseline cases moved by at most 2e-10 relative.
- The binary and three-phase cases are bitwise unchanged; neither model uses
  this option.
- Full suite: 1,031 passed, 2 skipped.

### Lee and Oh (1996) Fe-Cr-Ni cases

Source: `examples/ternaryExamples/FeCrNi/IllingworthTernaryTwoPhase_FeCrNi_Lee1996Cases.py`.

**Setup.** I used the script's own settings:
- PyCalphad surrogate;
- `composition_dependent_lagged` diffusivity;
- 100 + 1,000 linear-graded nodes for cases A–D (200 + 200 for fig9);
- semi-log steps;
- tolerance 1e-12;
- solved to 1,000 h.

The cases ran in pairs (both modes side by side) so each pair shared the same
machine load. The wall time is the solve only.

**Surrogate bundle.** The cached surrogate bundle was built before commit
2452580, which only added a docstring to `MovingBoundarySurrogates.py`. The
driver loaded the bundle by pointing that one implementation hash at a
byte-exact copy of the pre-docstring file, whose SHA-256 matches the hash saved
in the bundle. No bundle was rebuilt or modified.

**Run time and retries:**

| Case | Wall time, old / new | Speed-up | Steps, old / new | Failed attempts, old / new |
|---|---|---|---|---|
| fig9 | 55.4 / 54.4 s | 1.02× | 4,690 / 4,673 | 70 / 49 |
| A | 202.4 / 193.4 s | 1.05× | 18,505 / 18,505 | 0 / 0 |
| B | 230.2 / 227.4 s | 1.01× | 18,505 / 18,505 | 0 / 0 |
| C | 202.4 / 200.2 s | 1.01× | 18,505 / 18,505 | 0 / 0 |
| D | 187.0 / 183.8 s | 1.02× | 17,889 / 17,887 | 42 / 25 |

**Differences in results (new minus old):**

| Case | Max interface difference over the run | Max η difference | Final profile max difference | Inventory drift, old / new |
|---|---|---|---|---|
| fig9 | 5.7e-11 m (9.5e-6 of total motion) | 3.5e-7 | 1.8e-10 | 7e-15 / 1e-14 |
| A | 2.3e-12 m (6.5e-8) | 1.1e-6 | 4.9e-9 | 7e-13 / 5e-14 |
| B | 3.4e-12 m (1.1e-7) | 7.5e-7 | 1.8e-9 | 3e-13 / 2e-14 |
| C | 5.8e-12 m (1.7e-7) | 4.8e-8 | 4.3e-9 | 7e-13 / 7e-16 |
| D | 5.6e-11 m (1.6e-6) | 1.1e-7 | 3.7e-8 | 1e-12 / 7e-15 |

**Case D.** Its α layer dissolves, and both runs stop on the terminal
thin-phase rule. The stop times are 1,361,934 s (old) and 1,361,930 s (new),
which agree to 4 s out of 378 h. At 1, 10, 100 and 1,000 h every case's
interface position agrees to 0.01 nm or better.

### Interpretation

**Why there is almost no speed-up.** These cases are easy for the interface
Newton iteration:
- the fronts are smooth;
- there is no thin spike of a slow solute;
- the cell Courant numbers are small.

From the old state, Newton already converges in the minimum single update, at
4 candidate evaluations per step. The extrapolated start cannot go below that
floor. That leaves a 1–5% wall-time gain, and fewer retries where retries
happen: fig9 has the interface reversing direction, and D has α dissolving.

**Results.** The results are the same to within Newton tolerance accumulated
over about 18,500 steps. The extrapolated start also tends to converge further
below the tolerance, because Newton begins close to the answer. That is why
the inventory drift is 1–3 orders of magnitude smaller.

**Where the large gain comes from.** It is specific to sharp moving fronts,
such as the Fe-C-X cases. There, starting at zero displacement made Newton fail
on most steps.
