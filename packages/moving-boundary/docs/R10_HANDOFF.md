# R10 implementation handoff (failure recovery: setup/reset rollback, step commits, atomic finalization)

Written 2026-11-13 at the end of the R09 session; committed with the R10c documentation commit.
**Status (2026-11-13): R10 implemented** under the user's authorization: R10a `c1b2aae`, R10b
`ded279d`, R10c docs (this file, LEDGERS R10, NEXT_SESSIONS). No reference changed; every gate was
bit-identical (see LEDGERS R10 "Resolution" for the test evidence and gates). Sections 1-11 below are
the pre-implementation handoff as reviewed; the one deviation is recovery guarantee (b), restated
by the user during R10a (section 5). Nothing was pushed or approved.

## 1. Checkpoint (read-only checks, 2026-11-13)

- Repository: `moving-boundary-extraction`, branch `refactor/package-extraction`, HEAD **`4bf4ba9`**
  (matches the expected checkpoint). Not pushed.
- Recent history:

  | Commit | Content |
  |---|---|
  | `3047510` | R09a: append-only sparse histories, flag-based finalize, live no-time getters |
  | `83b5887` | approval `20261113T174227Z-3047510-1` (C25, C29) |
  | `fc4b041` | R09b docs |
  | `d56f25e` | R09c-1: strict `r09-data-tail` layout tooling, O04/O05 settings carry-forward |
  | `24a2d39` | R09c-2: interval-1 duplicate `data` row removed |
  | `1993bb9` | R09c docs (pending approval) |
  | `6fd3be8` | approval `20261113T194742Z-1993bb9-1` (26 runs + C25, C26, C29) |
  | `4bf4ba9` | docs: R09 resolved |

- Protected user state, unchanged and never to be staged, reverted or reformatted:
  - modified `packages/moving-boundary/examples/ternary_fecrni_lee1996.py` (Settings cell:
    TDB path, `CONFIG=PRODUCTION_B`, `REBUILD=True`); the gates take E05/E06 settings from
    `tools/example_manifest.json`, not from this cell;
  - untracked `AGENTS.md`, `branch_status/` (three handoffs), `packages/moving-boundary/docs/R08_HANDOFF.md`;
  - six stashes from other branches (`stash@{0}`-`stash@{5}`);
  - this file, `docs/R10_HANDOFF.md`, untracked until the user decides to commit it.

## 2. Status of R09 and R10

- **R09 resolved** (LEDGERS R09):
  - R09a `3047510`: one row rule for all five history implementations, append-only rows; regular
    rows every `k` accepted steps counted from `reset` across solves; forced final rows appended
    without consuming a cadence slot; flag-based finalize (no time tolerance); live no-time getters;
    binary/two-phase `getTransformedState()` without p/q histories. C25 (model0, model4) and C29
    (model1) `profiles/compositions` changed (binary t = 0 models now report the reconstructed
    initial profile) and are approved as `20261113T174227Z-3047510-1`.
  - R09c `24a2d39`: the compatibility `MeshData.finalize` uses the same flag rule at interval 1, so
    binary/two-phase `data` no longer repeats its final row. Approved as
    `20261113T194742Z-1993bb9-1` with `approve --layout-correction r09-data-tail` for exactly C01-C08,
    C13-C19, C24, C30, C31, F04, LEGACY, O01, O03a, O04, O05, O06, O07, C25, C26, C29.
- **R10 open, not started.** R13 recorded, out of scope.

## 3. Local verification inputs another checkout needs

`.verification/` is git-ignored. A clean clone (needed for any future `approve`, which requires a
clean package tree) must have these copied in from the original worktree and verified byte for
byte (e.g. per-folder sha256 of every file, as done for the R09a/R09c clones):

- `.verification/public-environment.json` (pointer to the `mb-public-050` interpreter);
- `.verification/kawin-0.5.0-py3-none-any.whl` (stock wheel, hash-checked by `setup_targets`);
- `.verification/acceptance/inputs/` (adopted and private inputs, `private-inputs.json`, `FeCrNi.tdb`,
  the cached O-case inputs and `fecrni.surrogate/`);
- `.verification/acceptance/reference/` (frozen references, never rewritten);
- `.verification/acceptance/reference-approved/` with **all three** approval folders:
  `20261111T222849Z-5aedde7-1` (R08), `20261113T174227Z-3047510-1` (R09a),
  `20261113T194742Z-1993bb9-1` (R09c); immutable;
- `.verification/acceptance/example_settings/` (adopted settings; O04/O05 are validated only by
  their private digests; the R09c approval carries them forward unchanged);
- `.verification/acceptance/pilot/` (O05 pilot timing used by `normal_cases`).

Private measurements and private hashes are not reproduced here and must never be committed.
Committed approval reports contain public hashes and structural or public evidence only.

## 4. Environment and commands (AGENTS.md; every Python command through Conda activation)

- Research env wrapper (acceptance tools re-launch themselves in `mb-public-050`), from the worktree root:

      cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && <command>'

- Package env for direct tests: `C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\mb-public-050`.
  Never install into it. Use a **new, empty** target per revision:

      mkdir <scratch>\mb-target-<rev> && python -m pip install <package> --no-deps --no-build-isolation --no-index --target <scratch>\mb-target-<rev>
      set "PYTHONPATH=<scratch>\mb-target-<rev>"
      set "MB_EXAMPLE_TDB=<package>\.verification\acceptance\inputs\FeCrNi.tdb"
      cd <package>
      python -m pytest tests -q -p no:warnings --basetemp <scratch>\pt-<rev> -p no:cacheprovider

  Run pytest from the package directory (the extraction root has a frozen `kawin/` folder that would
  shadow stock kawin on `sys.path`).
- Gates (from the worktree root, research-env wrapper; each builds a fresh wheel and target):
  - `python packages/moving-boundary/tools/acceptance.py full --tolerance exact` (about 9-10 min,
    internal 1200 s cap; live O06 when TC is available);
  - `python packages/moving-boundary/tools/acceptance_examples.py run` (about 3 min);
  - full `tests/` on a fresh target (about 5 min).
- Bound every command (no command over 30 min without telling the user first; run long gates in the
  background and wait for completion). Do not edit `src/` while a gate that builds from the working
  tree is running.
- **Latest successful gates** (from `6fd3be8`, fresh targets, 2026-11-13, no layout flag): exact full
  40/40 runs and 5 contracts bit-identical against the selected references, live O06, 34 repeat pairs,
  557 s; examples E01-E10 pass with 68 tests (E11 a pending fixture); full `tests/` 1391 passed,
  5 skipped (C25-C29 comparisons outside an acceptance run), 1 deselected (`slow`). `4bf4ba9` is docs
  only. **R10 is expected to change no reference**: every R10 gate must be bit-identical against the
  current selected references; any difference is a stop.

## 5. Reviewed R10 stages (order and contracts as reviewed; R09c now precedes R10)

### R10a: setup and reset all-or-nothing (all three families)

- **Defects reproduced** (stock kawin 0.5.0):
  - *Setup failure* (e.g. the diffusivity provider raises): `isSetup=True` stays set by the host
    before Illingworth initialization completes; a second `setup()` returns immediately.
    Ternary/three-phase keep accepted state `None`, so the next solve crashes; binary is worse: the
    failure comes after `_p_curr` is set, so a later solve runs silently without the initial
    inventory record, `_initialInventory` or a reconstructed profile. A direct `setup()` with an
    invalid setting fails the same way (validation runs after the host set `isSetup`).
  - *Reset failure* (e.g. `therm.clearCache` raises): host `DiffusionModel.reset` sets
    `currentTime=0` and then raises before the histories are reset and before `isSetup=False`. After
    restoring the provider, the next solve runs from t = 0 with the t = 2dt state, and the history
    times go backwards (binary/ternary data `[0,1,2,...,1,1]`, three-phase `[0,1,2,1]` x dt).
- **Contract (D5, D-R1):** on any exception from `setup()` or `reset()`, every model-owned field is
  restored to its pre-call value and the exception is re-raised. This includes `currentTime`,
  `isSetup`, the accepted state, all histories, the pending candidate and latch, and diagnostics.
  Failed setup leaves `isSetup=False` with the exact pre-setup state (not a `reset()`). Covered
  failures: provider `clearCache`, interstitial u-fraction re-recording (`record_initial_u_fractions`),
  and the configuration validation at the end of binary/ternary `reset()`. Behavior change (accepted,
  D-R1): a validation failure at the end of `reset()` now keeps the previous simulation instead of
  leaving a cleared model; every existing `reset()` caller is on a success path.
- **Mechanism:** the setup/reset bodies stay exactly as they are (floating-point order unchanged). A
  small shared wrapper snapshots all model attribute bindings (`vars(model)`) plus deep copies of that
  family's existing histories (rare calls, so whole copies are fine), restores them on any exception,
  and re-raises. No general transaction framework. No new model-level attributes (if one becomes
  necessary: stop and propose a `tools/example_settings.py` BOOKKEEPING entry first; the example
  settings comparison reads plain model attributes).
- **Recovery guarantees:** after fixing the cause, (a) a continued solve equals an untouched model's
  continuation bit for bit; (b) a repeated `reset()` followed by `solve` equals a freshly constructed
  model bit for bit; (c) recovery after a failed setup (fix the cause, call `setup`/`solve`) equals an
  untouched model bit for bit.
  *(b) restated by the user on 2026-11-13:* a repeated reset + solve equals an untouched model's
  successful reset + solve, and also a fresh model for binary/two-phase ternary. Three-phase reset +
  solve already differed from a fresh model in the last bits before R10 (the host reset records the
  raw `mesh.y`, the constructor its u-fraction conversion); this is outside R10.
  *Later (2026-11-13, user-authorized follow-up):* three-phase reset now records the constructor's
  conversion, so (b) as originally written holds for all families (implemented; contract C27
  adoption pending; LEDGERS R10).

### R10b: binary/two-phase prepare-then-commit, plus atomic multi-history finalization

- **Defects reproduced** (binary and two-phase ternary `postProcess`; three-phase already correct):
  profile reconstruction raising advanced time 0 -> dt without moving the interface; one history's
  record raising advanced `data` but not the other histories; a coupled-model update raising after
  everything was committed and time advanced; the ternary composition check runs after time has
  advanced. R07's rejection latch held in every case. Ternary `_lastAcceptedDt` is set inside
  `getdXdt` (`_accept_step_result`), so after a rejected proposal it holds that rejected trial dt and
  sets the next step's `interface_newton_start="extrapolated"` Newton start.
- **Commit sequence** (binary and two-phase ternary, following three-phase):
  1. verify the update (R07 checks unchanged);
  2. consume the candidate and set the latch;
  3. prepare outputs without touching model state: clip and check the interface, apply the
     interface-node overrides, ternary composition check, profile reconstruction, inventory;
  4. commit as one unit: history records (restored on failure, including capacity), accepted state,
     `_s_old`/`_eta_old`, `currentTime`, projection totals, `_lastAcceptedDt = pending.dt`, then
     `updateCoupledModels`;
  5. clear the latch.
- **On any failure in steps 3-4:** time, accepted state, histories, cumulative projection totals and
  `_lastAcceptedDt` are unchanged; the candidate stays consumed and the latch stays set, so
  resubmitting it, or sending the originally correct update, is still refused ("unchanged" does not
  reopen R07). The only way forward is a new `getdXdt`, which must reproduce an untouched model's
  step bit for bit.
- **Coupled models (D6):** a failure in `updateCoupledModels` rolls back this model only, as
  three-phase does. Coupled models updated before the failing one keep their update; recovery
  equivalence then holds only if the caller restores or discards that external state before retrying
  (document in `postProcess`).
- **Atomic multi-history finalization (all families):** `postSolve` finalizes all of a family's
  histories as one unit. Per history, a targeted snapshot (no whole-history copy): the original
  `_y`/`_time` array references (finalize truncates by rebinding, so this restores capacity), a copy
  of the forced-target row if one exists, `N`, the cadence counter `currentIndex`, the row counter
  `_rowCount`, the flag `_lastRecordStored`, `currentY` and `currentTime`. On a failure, every
  history is restored and the exception re-raised. Calling `postSolve()` again is the retry and is
  idempotent. A continued solve without that retry is documented to record as if the first solve had
  never been finalized.
- **Three-phase:** its commit logic is unchanged. Its atomic history helpers
  (`_history_record_target`, `_ensure_history_record_capacity`, `_record_histories_atomically`,
  `_restore_history_snapshots` in `three_phase.py`) move to `shared.py` with the same behavior and the
  R09 row rule, and binary/ternary reuse them; `postSolve` uses the same restore-on-failure.
- **Unchanged:** the near-final no-op step; a direct `postProcess` with no proposal still commits
  unchecked; three-phase `solve(0)` rejection; binary/two-phase `solve(0)` setup-only no-op.
- **After a failed solve:** `postSolve` does not run, so histories hold only committed rows; no-time
  getters are live; the next successful solve adds the final row.

### R10c: documentation (LEDGERS R10 resolution, NEXT_SESSIONS checkpoint)

## 6. Rollback ownership (audit of in-place mutations in binary.py, ternary.py, three_phase.py)

| What changes | How | Restored by |
|---|---|---|
| History objects (`data`, `interfaceData`, `etaData`, `inventoryData`, `concData`, `pData`/`qData`, `profileData[i]`, `rightBoundaryData`, molar histories) | in place: `reset()`, `record()`, `setResponseAtN`, `data._y[0] = ...`, `currentY` assignment, `finalize` rebinding | setup/reset: deep copy of each existing history; step commit and `postSolve`: targeted per-row snapshot (section 7 fields + capacity) |
| Accepted state (`_p_curr`, `_q_curr`, `_s_*`, `_eta_*`, three-phase `_profiles_curr`, `_interfaces_*`, `_etas_*`, `_R`), `currentTime`, `isSetup`, `_lastAcceptedDt`, latch/candidate | always rebound (`getCurrentX` returns copies; commits build new arrays) | binding snapshot (all of `vars(model)` for setup/reset; the named fields for a commit) |
| Projection totals | arrays rebound (`a + b`), counters immutable ints | binding snapshot |
| Three-phase `_interfaceCompositionCache` | dict filled in place | not restored: a pure memo of eta -> compositions, leftovers harmless (document) |
| Three-phase `_laggedFaceDiffusivityCache` | rebuilt at the start of each step attempt | binding snapshot |

**External, never restored (document in the docstrings):** provider state and caches, including
`clearCache` effects (partially cleared caches after a failing provider); the ternary
`ThermodynamicTernaryInterfaceEquilibrium.thermodynamics`/`.phases` binding (assigned when unset);
`mesh.boundaryConditions` (created when absent); temperature parameters; coupled models; printed
diagnostics and warnings.

## 7. R09 history metadata that every transaction must restore

All five history implementations (binary `_ScalarHistory`/`_VectorHistory`, shared
`_ScalarHistory`/`_ArrayHistory`, compatibility `MeshData`) share the row helpers in
`src/moving_boundary/compatibility/recording.py`:

- `_history_row_target(history, force)`: enabled recording appends at `_rowCount` (regular calls only
  when `currentIndex % recordInterval == 0`, forced calls always); disabled recording overwrites row
  `N` (0);
- `_advance_history_rows(history, target, force)`: sets `N = target`, `_rowCount = target + 1`,
  `_lastRecordStored`; only regular calls increment `currentIndex`;
- `_finalize_history_rows(history)`: forced append only when `_lastRecordStored` is false (disabled:
  rewrite row 0), then truncate `_y`/`_time` to `N + 1` by rebinding.

Fields: `N` (last stored row; `toDict` index), `currentIndex` (cadence counter), `_rowCount`,
`_lastRecordStored`, `currentY`, `currentTime`, the `_y`/`_time` arrays (contents and capacity), and
`recordInterval`. `MeshData` adopts host-written rows through `_syncRowBookkeeping` in
`compatibility/diffusion.py`. Three-phase `_record_histories_atomically`/`_restore_history_snapshots`
already snapshot and restore `_rowCount` and `_lastRecordStored`; the moved shared helpers must keep
that.

## 8. Accepted state versus trial diagnostics (D7) and R07 requirements

- **Accepted** (changes only on a successful commit): accepted state, `currentTime`, histories,
  `_lastAcceptedDt` (moves from `getdXdt`/`_accept_step_result` to the commit, `= pending.dt`), and the
  cumulative projection totals (`_cumulativeProjectionInventoryChange`,
  `_cumulativeProjectionAbsoluteChange`, `_cumulativeProjectedNodes`, committed via
  `_commit_pending_step_projection`).
- **Last trial** (may reflect a rejected step; docstrings must say so): `_currdt`, `_lastStepRetries`,
  `_lastImplicit*`, interface D values (`_D_left`/`_D_right`), `_lastInterfaceCompositions`,
  `_lastStepProjection*`, and `_cumulativeProjectedSolves`, which already counts every linear phase
  solve including rejected ones.
- **R07 (must hold throughout):** `getdXdt` stages the converged candidate and clears the latch
  `_awaitingFreshCandidate` only on success; `postProcess` verifies time/state against the candidate
  (elapsed-step check, 4 ulp allowance, non-finite rejection) and consumes it. While the latch is set,
  every commit is refused, including resubmission of a rejected or interrupted update and of the
  originally correct update; a model that never proposed a step still commits directly. R10 failures
  leave the candidate consumed and the latch set.

## 9. Required tests (package-owned; written first)

New file `tests/test_illingworth_lifecycle_transactions.py`, reusing the small builders of
`tests/test_illingworth_history_contract.py` / `test_illingworth_numerical_controls.py`. Defect
tests must fail at the R10 starting commit **for the intended reason** (partial state, advanced time,
wrong rows), not because of import or setup errors; preservation tests must pass before and after.
Compare state bytewise (time, `isSetup`, accepted state, every history's rows, `N`, `currentIndex`,
`_rowCount`, `_lastRecordStored`, `currentY`/`currentTime`, capacity).

- **R10a defect (fail before R10):** reset `clearCache` failure after a two-step solve (raises; time,
  `isSetup`, state and histories byte-identical); its recovery tests (continued solve after fixing the
  provider == untouched continuation; repeated reset + solve == fresh model); validation failure at
  the end of `reset()` with the same rollback and recovery checks; setup failures: provider failure,
  invalid setting on a direct `setup()`, ternary closure failure (each leaves `isSetup=False` with
  byte-identical state; recovery == untouched). All reset-failure rollback and recovery tests are
  defect tests.
- **R10a preservation (pass before and after):** successful `reset()` and `setup()` behave and record
  exactly as before.
- **R10b defect (binary/two-phase):** reconstruction failure, ternary composition-check failure, a
  mid-sequence history-record failure (including capacity growth at an off-cadence row), and a
  coupled failure. After each: nothing changed; resubmission refused; fresh-candidate recovery ==
  untouched. Coupled test: a stateful recorder callback that succeeds followed by one that raises. The
  model rolls back while the recorder keeps its entry (documented limitation); if the test (as
  caller) removes that entry, recovery == an untouched model+recorder pair; if not, the recorder holds
  a duplicate, which the test asserts. Ternary `_lastAcceptedDt` unchanged by a rejected
  retry-reduced proposal, and the next extrapolated Newton start == an untouched model's.
- **Atomic finalization (all families):** inject a failure in the third history's finalization after
  earlier ones finalized, at interval 3 with 4 steps (forced row) and at interval 1 with 4 steps;
  rows, capacity, cadence and row counters, the flag and `currentY`/`currentTime` byte-identical to
  before `postSolve`; after removing the injection, `postSolve()` gives synchronized records with no
  omissions or duplicates in every history, `data` included (R09c is complete): `[0, 3, 4]`·dt at
  interval 3 and `[0, 1, 2, 3, 4]`·dt at interval 1; a second `postSolve()` adds nothing.
- **R10b preservation:** the three-phase injection set under the shared helpers; R07 latch tests
  (`tests/test_illingworth_outer_timestep_limits.py`); a direct `postProcess` with no proposal still
  commits; the R09 history-contract tests.
- **Compatibility already checked:** ported tests slice histories with `N + 1`; three-phase
  assertions on `N`/`currentIndex` stay valid; the R07 interrupted-commit test passes under rollback;
  no test reads ternary `_lastAcceptedDt`.

## 10. Scope, protected files, stop conditions, commit cadence, permissions

- **Out of scope:** R13; restart/`fromDict`; numerical defaults; solver algorithms; surrogate ports;
  consolidating the binary and shared history classes; a general recording or transaction framework;
  other cleanup. Do not modify research main, installed host code, `mb-public-050`, hash-pinned
  ported tests or generated definitions (`tools/definitions`, `docs/*provenance*.json`).
- **Protected:** section 1 items, all `.verification/` references and approval folders (immutable),
  `example_settings/`, private inputs.
- **Stop and report if:** any exact case or contract is not bit-identical; a pinned or generated file
  would need to change; a defect test does not fail for the intended reason or a preservation test
  fails before the fix; the example-settings comparison fails; O06 fails at Thermo-Calc licence
  startup; a command would exceed 30 minutes; a new model-level attribute seems necessary; anything
  would touch the host, environments or protected files.
- **Commit cadence and gates (local only, never pushed; decided 2026-11-13):** tests at the starting
  commit (no commit, pre-fix evidence) -> R10a implementation, gated by the focused tests and the full
  package suite on a fresh target before its commit `R10a fix: make Illingworth setup and reset
  all-or-nothing (R10)` -> R10b implementation, then focused tests, full `tests/`, exact full
  acceptance and E01-E10 examples before `R10b fix: prepare binary/two-phase outputs before
  committing accepted steps and finalize histories atomically (R10)` -> `R10c docs(ledgers): record
  R10` -> stop for review. Add an intermediate acceptance run after R10a only if findings justify it.
  No approval is expected or permitted.
- **This handoff file:** committed in the R10c documentation commit with its status updated (done).
- **Already decided by the user (reviewed 2026-11-12/13):** D5 (restore exact pre-setup state on setup
  failure; reset failure is a defect fix, all-or-nothing), D-R1 (a validation failure at the end of
  reset also restores the previous simulation), D6 (model-only rollback on coupled failure, external
  coupled updates not undone, conditional recovery, the stateful-recorder test), D7 (`_lastAcceptedDt`
  and projection totals become accepted state; the rest stay documented last-trial diagnostics), R07
  candidate consumption and latch preserved, targeted per-row snapshots for commits (no whole-history
  copies), atomic multi-history `postSolve` with failure-injection tests, retry = repeated `postSolve`.
  Order changed by the user: R09c before R10 (done).
  Practical decisions (2026-11-13): R10a gated by focused tests and the full package suite before its
  commit; exact full acceptance and examples after R10b, with an intermediate acceptance run only if
  findings justify it; this handoff stays uncommitted and goes into the R10c docs commit.
- **Permission:** granted 2026-11-13 for R10a, R10b and R10c as three local commits (no push, no
  approval); used as granted.

## 11. Genuinely unresolved decisions

None for R10. The three-phase reset/fresh-construction last-bit difference recorded here is being
corrected by a user-authorized follow-up (implemented; contract C27 adoption pending; LEDGERS R10).
