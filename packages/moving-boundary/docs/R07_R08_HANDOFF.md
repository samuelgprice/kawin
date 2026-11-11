# Handoff: start R07/R08 in the package (written 2026-11-11)

This handoff is for a fresh session. Each item is labelled **[verified]** (checked against the code
or the records when this was written), **[suggestion]**, or **[open]** (a decision for the user).
Verify claims against the code again before relying on them.

## 1. Do this first

1. Read this file, then [LEDGERS.md R07/R08](LEDGERS.md#release-review-follow-ups-r07-r12-open) and
   [LEDGERS.md "Authority switch"](LEDGERS.md#authority-switch-2026-11-11).
2. Run `git status`, and confirm that HEAD and the protected files match §2.
3. **Ask the user** for the commit cadence for R07/R08 (§7). Nothing is authorized for R07/R08 yet.
4. Reproduce R07 (§3.1) as a failing test **before** changing solver code.

The approval blocker in §6.4 affects only running `approve` in this worktree. Investigation,
implementation and testing can go ahead here; approval can happen later in a clean worktree.

## 2. Current checkpoint [verified]

- Worktree: `C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\moving-boundary-extraction`.
  The package is `packages/moving-boundary`.
- Branch `refactor/package-extraction`, HEAD `822ca69546b12d3fbbecf09a593963e257b4c383`,
  82 commits ahead of `main`. **Nothing is pushed.**
- **The authority switch is complete** (2026-11-11; every completion criterion is met; see the plan).
  - The package is the numerical authority. **Solver changes happen only in the package.**
  - Research main stays frozen at `7af24f4`. The clean, detached `...\code\mb-ref-7af24f4` is the
    last research oracle; it is never moved again, and only the explicit `historical` actions use it.
  - Normal `quick`, `full` and example acceptance use no research files and no Git.
- Working-tree changes, all **protected**: never discard, stash, commit or reformat them.
  - `packages/moving-boundary/examples/ternary_fecrni_lee1996.py`: the user's three Settings lines
    (`TDB = r"...kawin\examples\FeCrNi_Lee1993_..._withMobility.tdb"`, `CONFIG = PRODUCTION_B`,
    `REBUILD = True`).
  - `AGENTS.md` (untracked; it contains the environment rules, so follow it).
  - `branch_status/` (untracked: the branch ledger and earlier handoffs).
  - This handoff (untracked until the user says to commit it).

## 3. The two problems

LEDGERS records both as open and high priority. They were written on 2026-11-08, before the binary
R06 fix; the descriptions below were re-checked against the current code.

### 3.1 R07: the outer maximum timestep can invalidate an implicit step [verified]

- **Mechanism.**
  - Binary `solvers/illingworth/binary.py:_compute_dt` (around line 929) and two-phase ternary
    `solvers/illingworth/ternary.py:_compute_dt` (around line 1159) limit the trial dt by the
    schedule (`timeStep` or semi-log) and the remaining time only.
  - `getdXdt` then solves the complete implicit step at that dt, with retries
    (`_try_step_retries`), and returns `(new - old) / dt`.
  - The host loop `compatibility/solver.py:DESolver._getdXdt` (around line 56) then applies
    `dt = min(dt, self._dtmax)`, where `_dtmax = maxDtFrac × simTime`, limited to the remaining
    time. Explicit Euler commits `old + dXdt × dt_clamped`, a state for which the implicit
    equations were never solved.
  - The same host line also *raises* a dt below `_dtmin = minDtFrac × simTime` to `_dtmin`, unless
    it is the exact final remainder. A retry-reduced binary or ternary step below the outer
    minimum would be stretched in the same way. **[open]** Treat this as part of R07 (three-phase
    already does).
- **Already solved in three-phase** (`three_phase.py`):
  - `_outerMinTimeStep` and `_outerMaxTimeStep` are set from `minDtFrac`/`maxDtFrac` in `solve`
    (around line 1397).
  - `_compute_dt` includes `_outerMaxTimeStep` (around line 1433).
  - `_validate_pending_timestep_against_outer_minimum` rejects a subminimum retry unless it is the
    final remainder.
  - `_verify_pending_candidate` (around line 2793) refuses to commit unless the outer update
    reproduces the converged candidate's time and state.
  - Two-phase ternary already stages a `_pendingStepProjection` but does not verify the timestep.
- **Reproduction (LEDGERS R07, binary).**
  - Inputs from `examples/binary_constant.py` with D_ALPHA=1e-3, D_BETA=1e-1, `tolerance=1e-10`,
    `max_step_retries=15`, `solve(0.1, minDtFrac=1e-14)`.
  - `time_step=0.1, maxDtFrac=0.1`: interface 1.1029278493730248, inventory drift 2.511e-4.
  - `time_step=0.01, maxDtFrac=1`: interface 1.1246627054903873, drift 9.111e-12.
  - LEDGERS says the ternary family reproduces the mechanism too; no ternary numbers are recorded.
- **Effect on acceptance [verified for the inputs; outcome not yet run].**
  - No acceptance case or definition passes `maxDtFrac`, so every case runs at the default 1.
  - At the default, `_dtmax` equals the remaining time, which `_compute_dt` already applies.
  - **So an R07 fix that only adds the clamp and the verification should leave the exact gates
    bit-identical.** Confirm this with exact quick and full. If any case changes, stop and report.
- **Existing tests.** The only `maxDtFrac < 1` test is three-phase:
  `tests/test_illingworth_ternary_three_phase.py::test_three_phase_nonlinear_timestep_honors_outer_maximum_fraction`.
  Its siblings (`..._retry_below_outer_minimum_is_rejected_before_wrapper`,
  `..._final_remainder_below_outer_minimum_is_solved_and_accepted`,
  `..._tiny_final_remainder_commits_all_histories_and_allows_continuation`,
  `..._pending_candidate_rejects_outer_mismatch_without_commit`) are the templates. Binary retry tests:
  `test_illingworth_binary_variable_diffusivity.py::test_failed_step_is_retried_with_smaller_dt`,
  `::test_retry_exhaustion_reraises_last_error`. Ternary:
  `test_illingworth_ternary.py::test_ternary_getdxdt_reduces_timestep_after_failed_interface_solve`.

### 3.2 R08: binary convergence depends on length units [verified]

- **Code.** `binary.py:_take_implicit_step_planar` (around line 1256) iterates on the interface
  position and accepts when `abs(previous_future_s - future_s) <= self.tolerance and count >= 1`
  (around line 1296).
  - `tolerance` is an absolute length, default `1e-8` (constructor around line 330). That is 10 nm
    for metre coordinates.
  - There is no conservative interface-residual acceptance test.
  - Two-phase ternary and three-phase have a separate `residual_tolerance`, so R08 is binary only,
    as LEDGERS states.
- **Reproduction (LEDGERS R08).**
  - The R07 problem rescaled to R=5e-6, s0=1e-6, D=1e-15/1e-13, 501 nodes, 100 per phase;
    initial compositions 0.8/0.4, interface compositions 0.6/0; `time_step=0.1`,
    `max_step_retries=15`, solve to 0.5 with `minDtFrac=1e-14`.
  - `tolerance=1e-8`: s/R 0.2610915910, drift 1.75484e-3, 2 passes in the final step.
  - `tolerance=1e-14`: s/R 0.2581334961, drift 1.22648e-9, 11 passes, matching the unscaled problem.
  - R06's fine-grid iteration floor has a different cause (see LEDGERS R06).
- **Effect on acceptance [suggestion: expect changes].** Any change to the stopping rule can
  change every binary case's arrays: the binary rows of the LEDGERS capability table (C01-C04,
  C13-C15, C30-C31, LEGACY binary arrays, F04) and the binary contract models.
  - Find the actual set with exact quick and full; don't assume it.
  - Changed cases need an approval (§6).
- **Existing tests:**
  - `test_illingworth_binary_variable_diffusivity.py::test_implicit_step_satisfies_discrete_equations_at_converged_profile`;
  - `::test_variable_d_conserves_transformed_inventory`;
  - `test_binary_deviation_form.py::test_ni_p_fine_tolerance_completes_and_conserves`.

## 4. Suggested sequence and approaches [suggestion]

Do R07 first and R08 second, as separate changes, each with its own commits and gates.

**R07**
1. Write failing tests first: a binary run and a two-phase ternary run with `maxDtFrac < 1` must
   equal the same problem configured with that smaller `time_step` and `maxDtFrac=1` (bit-identical
   or within a stated round-off bound), with conservation at the unclamped level. Cover fixed and
   semi-log schedules, retry-reduced steps, and exact final remainders.
2. Port the three-phase pattern, sharing a helper in `shared.py` if that fits:
   - record the outer minimum and maximum in `solve`;
   - clamp in `_compute_dt`;
   - reject subminimum retries except the final remainder;
   - verify the committed time and state against the pending candidate in `postProcess`.
3. Keep `_nearFinalNoop`, the retry accounting (`_lastStepRetries`, `_currdt`) and floating-point
   order unchanged on the default path.
4. Gates: exact quick and full must stay bit-identical. If they do, no approval is needed.

**R08**
1. Write failing tests first: equivalent length-unit representations (the §3.2 pair) reach the
   same dimensionless result, with per-step and cumulative inventory error bounds; plus a test
   that a converged step satisfies the conservative interface residual.
2. Likely approach: accept only when both hold:
   - a position criterion with explicit semantics, either dimensionless (relative to the domain
     length or the cell width) or explicit absolute plus relative;
   - a conservative residual at the interface using the same bulk fluxes as the step.
   Document the units in the constructor docstring and README.
3. **[open] Decisions to discuss before coding:**
   - the meaning of `tolerance`: keep it as an absolute length and add `rtol` and
     `residual_tolerance`, or redefine it as dimensionless. Redefining changes the public
     constructor behaviour;
   - the default values;
   - whether to keep `count >= 1`.
4. Expect binary references to change. Collect the deviation report, then approve (§6) after the
   user agrees.

Numerical assumptions to preserve [verified from records]:
- floating-point operation order on unchanged paths;
- x/u and molar-volume conventions; R=8.3145 where used; temperature-array hours;
- the R06 deviation form (on by default; the floor projection is a policy, not an error bound);
- the explicit-Euler-only wrappers;
- the ternary thin-phase prompt default (changing it is a user decision).

## 5. Validation workflow

Follow `AGENTS.md`: every Python command runs through conda activation, never `python.exe`
directly, and with bounded timeouts. Tell the user before any run expected to exceed 30 minutes.

- **Activation:**
  - Research env (acceptance tools re-launch themselves in `mb-public-050`):
    `cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && <cmd>'`.
  - Package env: replace the env path with
    `C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\mb-public-050`.
- **`mb-public-050` must stay free of any `moving-boundary` install, editable or not**
  (the editable install was removed 2026-11-11; README "Local acceptance").
- **Targeted and full package tests**, run in `mb-public-050`, with a **new, empty target folder
  for every revision**. Never reinstall into an existing target: pip can succeed while keeping
  stale modules.

      mkdir <scratch>\mb-target-<rev> && python -m pip install <package> --no-deps --no-build-isolation --no-index --target <scratch>\mb-target-<rev>
      set "PYTHONPATH=<scratch>\mb-target-<rev>"
      set "MB_EXAMPLE_TDB=<package>\.verification\acceptance\inputs\FeCrNi.tdb"
      python -m pytest tests\test_illingworth_binary_variable_diffusivity.py -q -p no:warnings --basetemp <scratch>\pt-<rev>

  `PYTHONPATH=src` alone fails 5 FeCrNi example tests (no distribution metadata). Keep
  `--basetemp` outside the package. The full `tests/` run takes about 3.5 min.
- **Acceptance** (from the worktree root, research-env wrapper; each build installs a fresh wheel
  into a private target):
  - `python packages/moving-boundary/tools/acceptance.py quick --tolerance exact`: about 6.5 min,
    600 s budget;
  - `... full --tolerance exact`: about 8.5 min, 1200 s cap; includes O06 live TC, which skips with
    a reason if TC is unavailable;
  - `python packages/moving-boundary/tools/acceptance_examples.py run`: about 2.5 min, with tests;
  - `--record` only after a complete passing **public-tolerance** run, as the final step.
- **Latest results** (M10, worktree at `a58aa09`; commits after that are docs only):
  - exact quick: 30/30 runs + 5 contracts bit-identical, 33 repeat pairs, gate pytest 145, 385 s;
  - exact full: 40/40 + 5, 34 repeat pairs, 502 s;
  - public quick and full `--record`: all bit-identical;
  - examples: 10/10 + 56 tests;
  - full `tests/`: 1039 passed, 5 skipped (C25-C29 comparisons need `MB_ACCEPTANCE_STATE`),
    1 `slow` deselected;
  - the independence proof from Git-blocked exports also passed (plan M9).
- **Limitations of those results:**
  - one Windows machine only; Linux has not been checked;
  - O06 needs the TC licence server;
  - E11 has no fixture;
  - `approval_criteria` gate no ternary accuracy (conservation only);
  - 13 research tests are deferred, including the brass test (§8);
  - worktree runs include the user's Settings lines; they are outside the wheel, and the example
    cases use their own configs.

## 6. Numerical changes and approvals [verified unless marked]

1. **Never** weaken tolerances, `approval_criteria`, the binary profile mask, contract assertions
   or repeat checks, and never regenerate `reference/`, to get a pass. An intended change may
   differ from old references; that is what approval is for.
2. **What `approve` gates**, at public tolerance, stopping at the first failure; nothing is written
   on failure:
   - two bit-identical package runs for **every** case (O05/O08 seeded);
   - the fixed `approval_criteria` in `tools/acceptance_manifest.json` (user-approved 2026-11-10):
     - analytic error limits for C04/C11/C12/C31/F04/F11/F12 (2× the frozen error);
     - conservation limits for 39 cases (10×, floor 1e-13);
     - limits fixed once and never recomputed from a previous output;
   - the whole `tests/` suite on the freshly built targets, including the contract assertion
     groups for C25-C29, with no skips;
   - every **unlisted** case must pass its reference comparison;
   - listed cases' differences are reported, not failed.
3. **Usage:** `acceptance.py approve --cases <cases> --reason "LEDGERS Rxx" [--tier quick|full]`.
   - It always builds fresh targets from HEAD; `--reuse-targets` is refused.
   - It needs a **clean package tree** (`git status --porcelain --untracked-files=all` under the
     package) and every recorded approval to be an ancestor of HEAD.
   - It **writes** `tools/acceptance_manifest.json` and `docs/approvals/<id>.json` but commits
     nothing. Commit the solver change first; commit the approval separately afterwards (the
     report needs `git add -f`).
   - Budget: 900 s quick, 1500 s full. Rollback: `approve --rollback ID --reason ...`.
   - Then rerun exact quick and full: they must be bit-identical against the mixed reference set.
4. **Approval-only blocker:** the user's uncommitted Settings lines (and this handoff, while it
   is untracked) are inside the package tree, so `approve` refuses to run in this worktree. That
   affects approval only, not reproducing, implementing or testing R07/R08 here. Do not stash,
   discard or commit those lines. When an approval is due, use a clean worktree or clone of the
   committed solver change, and agree with the user how the written manifest, report and local
   `reference-approved/` files come back. M9 ran a synthetic approval in a scratch clone this way;
   that approval was synthetic and never came back.
5. **O04/O05 restriction:** `approve` refuses to list O04 or O05, because their example settings
   are hashed only in the local private record. R07/R08 are not expected to change these ternary
   cases. If they do, stop and ask.
6. **Small doc nit [suggestion]:** the `acceptance.approve` docstring still says "a committed
   report". Correct it to "writes" when you next touch that file.

## 7. Permissions and limits [verified from the records]

- The per-batch commit permission (2026-11-10: commit each completed batch after its checks pass;
  no push) was given for the **authority-switch plan**. Treat it as **not** extending to R07/R08
  until the user confirms.
- **Not authorized:**
  - pushing;
  - running `approve` for real (as opposed to synthetic approvals in a scratch clone);
  - changing tolerances, criteria or references;
  - editing research main;
  - moving the oracle;
  - capturing E11 or changing tiers;
  - starting R09-R12.
- Stop and report on:
  - any unexpected exact difference, including in R07's "should be bit-identical" gates;
  - any need for a tolerance, criteria or reference change;
  - scope changes;
  - substantial overruns, or `full` reaching its 1200 s limit.

## 8. Records

- [LEDGERS.md](LEDGERS.md): R07/R08 (and R09-R12), "Authority switch" with its migration log,
  and R06 for the deviation form.
- [NEXT_SESSIONS.md](NEXT_SESSIONS.md): definition of done and the current checkpoint.
- [AUTHORITY_SWITCH_PLAN.md](AUTHORITY_SWITCH_PLAN.md): §5 approval path, implementation log
  M0-M10, follow-ups, completion criteria.
- Acceptance reports: [acceptance.json](acceptance.json), [acceptance-deviations.json](acceptance-deviations.json),
  [approvals/](approvals/) (empty: no real approval exists).
- README "Local acceptance": environment rules and the independence guard
  (`MB_FORBID_RESEARCH`, `MB_FORBID_GIT`).
- [FUTURE_PORTS.md § Deferred tests](FUTURE_PORTS.md#deferred-tests). **Priority:**
  `test_illingworth_ternary.py::test_ternary_illingworth_quasi_binary_cu_zn_dummy_matches_binary_brass_case`
  checks binary/ternary agreement over a long run: accepted times, interface motion, profiles,
  conservation. It is directly relevant to both items: R07 changes binary and two-phase ternary
  stepping, and R08 changes only the binary side of that comparison. **[suggestion]** Port it (only its two builders, or a small fixture) before or
  alongside R07, so the parity check exists before either solver changes. Ask the user, since the
  deferral was their decision.
