# Definition of done

The end state: my Illingworth solvers (binary, ternary two-phase, ternary three-phase,
with interstitial and molar-volume support) live in this package. The package depends
on UNMODIFIED public kawin (a chosen release, currently 0.5.0) only for what it
really needs: the model/solver loop, mesh and recording, TemperatureParameters, and
the pycalphad/thermo side. Fork-only host behavior lives in the package's own
compatibility layer. A single command shows that the runs I actually use still give
the same results as the frozen research oracle (`mb-ref-7af24f4`), except cases with logged
approved references (`acceptance.py approve`; LEDGERS.md "Authority switch"). That command
needs no research checkout and no Git.

> **Deferred ports backlog:** [FUTURE_PORTS.md](FUTURE_PORTS.md) lists research functionality to
> port *after* the authority switch (surrogate diagnostics, comparison and benchmarking,
> visualization and convergence tools, training-data utilities, material-specific workflows),
> plus the research tests deferred with them. Sequence: authority switch → R07/R08 →
> diagnostics/comparison → the rest. **Revisit it after R07/R08** (both done 2026-11-11). None of it is needed by
> current acceptance.

## Current checkpoint

### Current (2026-11-15)

Paths below are relative to `packages/moving-boundary`. The last pushed revision is `522c695` on
`origin/refactor/package-extraction`, pushed 2026-11-15. It is the pre-transfer checkpoint. Four local
commits follow it, besides the docs commits that record them (`f9855d5` and the entries 1-2 docs
commit); as of writing none is pushed:

- `9e9464b` adds the hash-checked transfer of local acceptance artifacts (`tools/acceptance_transfer.py`;
  [README "Transferring local artifacts to another checkout"](../README.md#transferring-local-artifacts-to-another-checkout)).
  Its validation:
  - full `tests/` on a fresh installed target: 1847 passed, 5 skipped (the C25-C29 contracts outside an
    acceptance run), 1 deselected (`slow`);
  - a real-artifact smoke check: export (262 files: 246 required, 15 private, 1 optional), `check`, a dry
    run in the original worktree (all 262 present and matching, nothing written), a restore into a fresh
    short-path clean clone of `9e9464b` with LF line endings and passing post-install verification, and a
    repeat restore that changed nothing;
  - exact quick in that restored clone: 31 runs and 5 contracts bit-identical, 34 repeat pairs, 145 gate
    tests, 398 s.
  The environment pointer was supplied separately, as documented.
- `b2528a6` keeps digests and private contents out of transfer refusals: bundle entries are named by
  index with a fixed description of the invalid field, and refused values are never echoed. It also
  tightens one check: a boolean or negative `size` in a bundle entry is now refused as malformed (a
  boolean previously passed the integer check). Validation: 196 focused tests passed, including all 90
  transfer tests. The full suite and the numerical gates were not repeated after this commit.
- `fa22e73` ports surrogate diagnostics ([FUTURE_PORTS](FUTURE_PORTS.md) entry 1) as
  `moving_boundary.surrogate_diagnostics`, with the optional `[diagnostics]` extra (Plotly), generated
  test-port batch 5 and the 3 construction-diagnostic tests of `test_illingworth_ternary.py`
  un-deferred. Validation: oracle-versus-package parity, 45 cases bit-identical; fresh-target full
  `tests/` 1903 passed, 5 skipped, 1 deselected; exact quick 31 runs and 5 contracts bit-identical
  (431 s).
- `f78a13a` ports saved-surrogate comparison (entry 2) as `moving_boundary.surrogate_comparison`
  and strengthens the import-guard test (kawin origin asserted; the child installs the independence
  guard itself). Validation: comparison parity identical on shared and own-published bundles;
  fresh-target full `tests/` 1918 passed, 5 skipped, 1 deselected; exact quick (424 s) and exact
  full (42 runs and 5 contracts, 558 s) bit-identical. Usage:
  [README "Diagnosing and comparing surrogates"](../README.md#diagnosing-and-comparing-surrogates).

State at `522c695`, unchanged by these local commits. They add the transfer tool, the new
surrogate diagnostics/comparison modules, tests, test-port tooling and docs, and change no solver,
surrogate or provider code, numerical behavior, references or approval criteria. Acceptance tooling
did change in `9e9464b`, for the transfer only: `tools/acceptance.py` gained the
`STOCK_WHEEL_SHA256` constant (the value it already asserted) and an optional input-folder argument
to `private_input_problems`, and `tools/acceptance_baseline.py` documents its reuse by the transfer.
The state:

- C33 (quick) and F33 (full_drift), the package-owned two-phase ternary analytic cases, are active since
  `65ce32c` (capture `20261115T224052Z-a26d099-1`); their docs and the R13 retrospective are in `522c695`
  ([LEDGERS "Two-phase ternary analytic acceptance cases (C33/F33)"](LEDGERS.md#two-phase-ternary-analytic-acceptance-cases-c33f33)).
- Latest recorded gates (`--record`, at `65ce32c`): quick 31 runs and 5 contracts, 145 tests, 413.63 s;
  full 42 runs and 5 contracts, 521.06 s; every run bit-identical, live O06 included, no case skipped.
- Validation of `65ce32c` (fresh clean LF clones):
  - full `tests/` 1772 passed, 5 skipped, 1 deselected;
  - exact quick 31 runs and 5 contracts bit-identical, 34 repeat pairs;
  - exact full 42 runs and 5 contracts bit-identical with live O06, 35 repeat pairs. Exact full needs a
    sufficiently short clone path for O06 (Windows 260-character limit).
- Five approvals are selected: R08 `20261111T222849Z-5aedde7-1`, R09a `20261113T174227Z-3047510-1`, R09c
  `20261113T194742Z-1993bb9-1`, R10 reset parity `20261113T232720Z-9595574-1` and R13
  `20261115T181200Z-6caa07b-1`. Their artifacts, the C33/F33 baselines and the other local inputs are
  git-ignored under `.verification/`. Another checkout gets them with `tools/acceptance_transfer.py`
  (`export` in the source, `check`, then `restore --dry-run` and `restore` in the destination).
  - Bundles contain private and TC-derived files and private digests. Keep them local, outside any
    repository, and never commit them.
  - Each destination supplies its own `.verification/public-environment.json`, either copied from a
    checkout on the same machine that uses the same `mb-public-050` or written by hand.
  - A restore never creates, changes or installs into an environment.

### Historical checkpoints

The entries below keep their original results and wording as of their dates.

Authority switch complete (2026-11-11): the package is the numerical authority; normal acceptance
needs no research checkout and no Git (LEDGERS.md "Authority switch"). Final recorded gates: exact
quick 30/30 and full 40/40 runs plus 5 contracts bit-identical; Quick: 30 run comparisons, 5 contracts and 145 pytest tests passed in 381.09 s. Full: 40 run comparisons, 5 contracts and 145 pytest tests passed in 491.61 s (public tolerance, `--record`, 2026-11-11, after the authority switch; every run bit-identical).
Examples 10/10 with 56 tests; full `tests/` 1039 passed, 5 skipped (contract comparisons outside an
acceptance run), 1 deselected (`slow`).

R07 and R08 are fixed since then. R08 changed binary results by design; its references are approved
as `20261111T222849Z-5aedde7-1` (C02, C03, C04, C13, C14, C15, C30, C31, F04, O07, C29; commit
`34c02d2`). Gates at that commit (fresh targets, 2026-11-11, not `--record`ed): exact full 40/40 runs
and 5 contracts bit-identical against the selected references, live O06, 34 repeat pairs, 515 s;
examples E01-E10 pass with 68 tests (E11 pending); full `tests/` 1158 passed, 5 skipped, 1 deselected.
The approved artifacts are local (`.verification/acceptance/reference-approved/`): another
checkout needs them copied in before quick/full can pass.

Earlier checkpoint: Illingworth solvers, shared/interstitial helpers and numerics moved under `solvers/illingworth/`, leaving room for sibling solver families.
Commit: `refactor: group Illingworth modules under solvers/illingworth`; permitted docstring changes: `solvers/__init__.py`, `solvers/illingworth/__init__.py` and the approved `solvers/illingworth/numerics/__init__.py`.
Fresh installed-wheel full `--record`: 40 runs and 5 contracts bit-identical, 88 tests, 34 repeat pairs, 578.02 s; solver origins `moving_boundary.solvers.illingworth.*`.

Whole-module relocation is complete: binary 8233ad2, ternary e08eed0, three-phase
693595d. Each separate installed-wheel quick gate passed 29 exact run comparisons
and 22 package tests. The method-by-method subclass strategy is replaced.

Stage 1B is complete against unmodified public kawin 0.5.0 in its separate local
environment, with bounded package policy adapters and no fork-only runtime imports.
Quick: 30 run comparisons and 88 pytest tests passed in 427.34 s. Full: 40 run comparisons and 88 pytest tests passed in 578.02 s.
See PUBLIC_KAWIN_MILESTONE.md, acceptance.json and README for scope and command.

The final quick gate used 427.34 s of its unchanged 600 s
budget, below the 540 s review threshold. It changed zero tracked files. Full
used `--record` and completed in 578.02 s of its unchanged
1200 s cap. All 33 quick/contract reference pairs remain; full adds O06's
unseeded pair for 34. No case was skipped on this configured machine. Only
the new ignored O06 fixture pair was captured; all 89 older files are unchanged.

## Real-usage examples (implemented; tiers pending approval)

`examples/` has six installed-API teaching examples plus the minimal `binary_constant.py`, and
`moving_boundary.analytic` provides exact solutions. `tools/acceptance_examples.py` compares their
quick-check settings with existing frozen fixtures; it isn't part of `acceptance.py`, whose inputs
and fixtures are unchanged. After adding `analytic`, an unchanged full gate gave 40 runs and 5
contracts bit-identical, 34 repeat pairs, 570 s (not recorded). All ten example cases (E01–E10)
are bit-identical with matching settings; last run 2026-11-11 on `34c02d2` (160.7 s), with E01/E02
compared against the R08-approved C04/F04. The settings comparison skips the named run-bookkeeping
attributes in `tools/example_settings.py` (`BOOKKEEPING`).

Changes on 2026-11-07/08 (commits 22114b7 to 22c8645):

- **Fe-Cr-Ni.** `Config` gained `terminal_thin_phase_policy`, `compare_frozen` and `min_dt_frac`
  (defaults leave solver inputs unchanged; E05/E06 stayed bit-identical). `PRODUCTION_B` has
  the research case B settings, untested end to end (about 5 min sampling plus 4 min solve,
  extrapolated). The build spec records the `moving-boundary` version. `simplex_bulk_points()`
  now projects symmetrically onto the buffered simplex and reproduces the research
  `allValid_3Element_compositions_*_projTo*.pkl` grids exactly, values and order.
  `check_limits()` evaluates every workflow check the tests assert, from `LAGGED_THRESHOLDS`,
  for `LAGGED_FAST` and `LAGGED_DEMO` only. These change the `LAGGED_*` surrogate inputs, so
  existing lagged bundles need `REBUILD`.
- **Bundles.** Both bundle-building examples refuse a non-empty folder without `manifest.json`
  and give a rebuild hint for damaged bundles. `surrogate_lifecycle.run_differences()` compares
  times, interface, eta, inventory and both phases' profiles.
- **New `binary_tlp_bonding.py`.** The Ni-P bonding case of Zhou and North as reproduced in
  Illingworth and Golosnoy Fig. 3, with an early-time exact solution, the mass-balance width
  bound and conservation as checks. DEMO (81/321 nodes, `semiLog_dt` 0.005, to 7e4 s) takes about
  2 s. A space and time convergence sweep gave a converged peak of 23.1096 um; DEMO's is 0.06% low.
  It has tests but no E case or fixture.
- **Citations.** Every example cites its literature sources in-line with a References cell;
  `examples/README.md` ("Citing sources") makes this the practice for new and changed examples.

Tests: `tests/test_examples.py` 46 passed and the slow test deselected, about 90–120 s, with
`MB_EXAMPLE_TDB` set (2026-11-08). The reviewer simulation above (117 passed, 8 expected skips,
167 s, on a fresh venv with the wheel and PyPI kawin 0.5.0) predates these changes and hasn't
been rerun.

Awaiting decisions: E-case tiers and proposed limits (ACCEPTANCE_CANDIDATES.md, now including
the Ni-P example), whether the Ni-P example gets an E case, go-ahead for the E11 lagged fixture
(its inputs changed in 22114b7, so freeze the config first), Lee TDB redistribution, and merging
the R05 branch. E11 isn't captured, so no gate is yet the completed expanded suite. Linux and
golden tolerance work hasn't been done. Thermo-Calc (NiTiNb) remains a later optional extension.
R06 in LEDGERS.md (binary interface iteration stalls on fine grids) is fixed by the
deviation-form solve from research 25bf6cd, ported to `binary.py` 2026-11-09; the two-phase
ternary counterpart (research 2741135, merged as 720d122) is ported to `ternary.py`/`shared.py`
the same day, and the three-phase counterpart (research fae6149, merged as 7af24f4) is ported
to `three_phase.py`/`shared.py` on 2026-11-10. Open solver findings:
[release review follow-ups R07-R12](LEDGERS.md#release-review-follow-ups-r07-r12-open)
(2026-11-08): outer timestep clamping, binary tolerance scaling, sparse histories,
failure recovery, incomplete restart loading and numerical-control validation.
R07/R08 are the highest-priority numerical investigations. R07 (outer timestep clamping) is
fixed in the package (2026-11-11), including a follow-up to its commit guard (elapsed-step
timestamp check; commits refused after a rejected proposal until a new candidate is solved).
R08 (binary tolerance units) is implemented in the package (2026-11-11): dimensionless
position tolerance plus a conservative, inventory-normalized residual check, with the
call sites migrated (generated files through the recorded `api_migration` transform).
Its binary reference changes are approved as `20261111T222849Z-5aedde7-1` (commit `34c02d2`;
LEDGERS R08). R11 (restart loading) is fixed by rejection (2026-11-12, commit `6296bd9`):
binary and two-phase ternary `fromDict`/`load` raise before changing state, output unchanged;
quick exact 30/30 runs and 5 contracts bit-identical, full `tests/` 1176 passed. R12
(numerical controls) is fixed (2026-11-12, commit `cdb6c68`): ternary tolerances and active
semi-log controls are validated before the host constructor and at the start of `solve`, and
binary/two-phase reject negative, NaN and infinite durations before setup; exact full 40/40 runs
and 5 contracts bit-identical (live O06), examples E01-E10 pass, full `tests/` 1284 passed.
R09 is partially resolved (2026-11-13, R09a commit `3047510`): sparse histories are append-only
with initial, regular and forced final rows, dedup is flag-based, and no-time getters are live.
Contracts C25/C29 (binary time-zero `profiles/compositions`) are approved as
`20261113T174227Z-3047510-1` (commit `83b5887`). Gates from `83b5887`: exact full 40/40 runs and
5 contracts bit-identical (live O06, 34 repeat pairs), examples E01-E10 pass, full `tests/` 1332
passed. R09 is resolved (2026-11-13): R09c (tooling `d56f25e`, fix `24a2d39`) removed the
interval-1 binary/two-phase `data` duplicate row, and its 26 runs and contracts C25, C26, C29 are
approved as `20261113T194742Z-1993bb9-1` (commit `6fd3be8`; `approve --layout-correction
r09-data-tail`, strict layout proof, O04/O05 private settings carried forward). Gates from `6fd3be8`
(fresh targets, no layout flag): exact full 40/40 runs and 5 contracts bit-identical (live O06, 34
repeat pairs), examples E01-E10 pass, full `tests/` 1391 passed, 5 skipped, 1 deselected. The
approved artifacts are local (`.verification/acceptance/reference-approved/`, three approvals); another
checkout needs them copied in. R10 is resolved (2026-11-13, no reference changed): setup/reset are
all-or-nothing in every family (R10a `c1b2aae`), and binary/two-phase steps prepare outputs before
an atomic commit, ternary `_lastAcceptedDt` is accepted state and `postSolve` finalizes all
histories atomically (R10b `ded279d`). Gates at `ded279d` (fresh targets): exact full 40/40 runs and
5 contracts bit-identical (live O06, 34 repeat pairs), examples E01-E10 pass, full `tests/` 1448
passed, 5 skipped, 1 deselected. Follow-ups: interrupted three-phase commits roll back (`b04b818`);
three-phase reset reproduces fresh construction bit for bit (fix `9595574`; contract C27 approved as
`20261113T232720Z-9595574-1`, commit `ff01469`). Gates from `ff01469` (fresh targets): exact full
40/40 runs and 5 contracts bit-identical (live O06, 34 repeat pairs), examples E01-E10 pass.
Two-phase ternary reset also reproduces fresh construction (initial profile through the
constructor's conversion, and `initialEta` restored to NaN); binary needed no change (identical
bits); no reference changed, exact full and E01-E10 bit-identical, full `tests/` 1506 passed. The
approved artifacts are local (four approvals). R13 (ternary/three-phase residual normalization) is
resolved (2026-11-15): 19 runs (C09-C12, C16-C21, C32, F11, F12, O01, O02, O03b, O04, O05, O06) and
contract C27 are approved as `20261115T181200Z-6caa07b-1` (commit `e1bb66d`; `approve --tier full
--verify-private-settings` from a clean clone of `6caa07b`; O04/O05 private settings verified equal
and carried forward; candidates bit-identical to the reviewed `4ad05c4` run). C32 takes 15 timestep
halvings instead of 13, accepted by the user as intended. Gates from `e1bb66d` (fresh targets, no
special flags): exact full 40/40 runs and 5 contracts bit-identical against the selected references
(live O06, 34 repeat pairs), examples E01-E10 pass. The approved artifacts are local and ignored
(`.verification/acceptance/reference-approved/`, five approvals, relative to `packages/moving-boundary`);
another checkout gets them, hash-checked, through the transfer workflow
([README](../README.md#transferring-local-artifacts-to-another-checkout); LEDGERS R13 *Adoption*).

## Installed solving and research-owned generation

The installed package with stock kawin alone supports solving with user-supplied
analytic providers, saved surrogates, or cached tables supplied through the solver's
provider interface. Those solves do not require the research checkout once their
provider inputs are assembled using installed APIs.

Generating data with pycalphad now works from the installed package through
`moving_boundary.providers.pycalphad.create_pycalphad_thermodynamics_source`.
The engine is imported lazily; stock kawin 0.5.0 already requires pycalphad>=0.11,
so no separate extra is needed. Stage 2B adapter relocation is complete for the
accepted O05 restricted-phase generation and O08 default-phase endpoint/equilibrium
sampling. Both use the installed adapter under the examples-import/source-read
guard; reference workers use the original adapter from clean detached 06dc7b4.
O08 has two valid BCC/FCC probes and one invalid FCC-only probe at 1373 K, using
the same FeCrNi database. It compares all arrays bitwise and phase labels in order.
Only O05/O08 keep PYTHONHASHSEED=0; R01 remains open on the research branch.

Building Thermo-Calc or pycalphad surrogates works from the installed package:
pass the installed provider to from_database, optionally use
prepare_surrogate_artifact for a verified, reloadable bundle, and use
merge_phase_diffusivity_surrogates to combine one phase's diffusivity samples.
See [README's installed pattern](../README.md#building-surrogates). The batch-safe
TC adapter is installed in providers/thermocalc.py; TC-Python and its licence are
separately required. Its real O06 generation acceptance is described below.

Anything importing or reading examples/ still needs the research checkout.
What stays there is script orchestration: the FeCrNi demo wrapper,
training_data.py dataset sampling/export/resume, case assembly, plotting, and
research scripts' probe-point search settings, failure/diagnostic reports and
bundle build specs. Real research workflows call the installed from_database
directly, usually inside prepare_surrogate_artifact; NiTiNb also uses the installed
merge_phase_diffusivity_surrogates.

Decision: do not move training_data.py:build_moving_boundary_surrogate into the
package. It is a thin FeCrNi-demo wrapper with demo-specific defaults; only
FeCrNi_TC_Python.py and kawin/tests/test_thermocalc_examples.py use it, with no real
research workflow depending on it. A general helper limited to session handling
and provenance metadata, with no system defaults, remains possible but is not
planned.

surrogates.py, surrogate_artifacts.py, _diffusivity_domain.py and equilibrium.py
were copied whole and unchanged in implementation, apart from required import
relocations and documented whitespace-only edits. Their presence in the package
does not complete the broader Stage 2 provider/surrogate separation. Stage 2B
adapter relocation is complete within the O05/O08 scope; R01 remains a separate
research-branch scientific issue.

## Completed implementation order

Approved list/tier changes were recorded in ACCEPTANCE_CANDIDATES.md, followed by
item 5 batch guards, item 6 documentation/test-stub cleanup, frozen inputs and
quantity tolerances, detached 06dc7b4 reference capture, then the three separate
item 2 relocations and item 3 public-host compatibility. No approval is outstanding
for that completed work. All TC-derived fixtures remain ignored; O06 is enabled in full when available.

## Research reproducibility issue R01 (open)

O05 restricted and O08 default-phase generation are sensitive to Python hash order
in third-party pycalphad
0.11.1: model.py:893/895/903 builds the ideal-mixing expression from an unordered
Species set (variables.py:215), then sympydiff_utils.py:137 compiles a hash-order-
sensitive Hessian with SymEngine CSE. A diagnostic ordered-iteration control alone
eliminated all O05 hash-0/hash-1 differences. NumPy seeding was unnecessary.
This affects rebuilt pycalphad-backed surrogate data and subsequent histories;
the 42 other cases repeat bitwise unseeded. O08 hash 0/1 reference workers differ
in five of seven arrays (endpoints, phase amounts/compositions and chemical
potentials), while a fresh hash-0 repeat is bit-identical. See LEDGERS.md R01 for
controls and main-06dc7b4 call sites. Only O05/O08 retain PYTHONHASHSEED=0; no
NumPy seed is imposed. Every unaffected quick case/contract is checked twice unseeded.

Propose a research-branch provider regression across hash seeds, followed by
validated deterministic symbolic ordering or a dependency/compiler fix. Do not
implement that scientific fix independently in the package, on main, or by
refreshing references. Numerical code, references and tolerances remain unchanged.

## Stage 2B: adapter relocation complete; acceptance bounded to O05/O08

The source-mapped provider facade and phase-universe/g_offset/endpoint-ordering
helpers now live in providers/pycalphad.py. Their arithmetic, iteration order,
component conventions and R01 hash sensitivity are unchanged. The only closure
outside the source adapter was an unused conditional import of examples.debugInPlace;
its call was commented out. That import is omitted; no debugger helper or larger
examples tree was copied. See PUBLIC_KAWIN_MILESTONE.md for the full source map.

O05 samples restricted-phase surrogate data and O08 samples default-phase
endpoints/equilibrium through the installed adapter while rejecting examples/debugger
imports and direct examples source reads. The original reference adapter remains
at detached 06dc7b4. O08 exercises its invalid-endpoint branch, including the unused
research debugger import (LEDGERS.md R02); the package copy omits that import.
O08's newly captured fixture is frozen by JSON/NPZ hashes in the manifest; every
older reference, input hash and preregistered tolerance remains unchanged. O08 is
bitwise-only even in public mode. No cached/fake provider replaces real generation.
This acceptance does not cover arbitrary databases/phase universes, default-phase
surrogate training, or every delegated kinetics/diagnostics method.
Surrogate building uses the installed APIs (README); the research script
orchestration listed above stays in the checkout. Databases, trained artifacts
and TC outputs stay local.

The broader Stage 2 separation of generation, surrogate evaluation, persistence/
domain validation and equilibrium interfaces is still open. The four whole copied
modules above have not been refactored into that split. R01 is still open on the
research branch; Stage 2B makes no species/term sorting or CSE change.

See [LEDGERS.md R04](LEDGERS.md#research-phase-name-issue-r04-open-research-cleanup-o06-convention-resolved) for the passing suffixed-name probe and remaining research-branch work.

## Stage 2C complete within O06

providers/thermocalc.py is a source-mapped, batch-safe copy of the frozen adapter.
Installed generation uses it and installed surrogate APIs under the examples
import/source-read guard; the reference uses the original 06dc7b4 adapter and
training_data.py. Every availability/generation/repeat worker gets its own fresh,
empty ignored TC cache. TC-Python is lazy, separately installed and licensed;
it is not a pip dependency. TC inputs, caches, outputs and numerical metrics
remain ignored; committed evidence contains only sanitized counts/hashes.

O06 accepts only the approved suffixed FeCrNi two-phase input: three real tie
lines, two bulk points and the same six-step solve. Its two unseeded pilot runs
and hash-1 control were bit-identical. It runs unseeded in full only, with a fresh
unseeded reference repeat pair inside full's unchanged 1200 s cap. Every numerical
array requires identical shape, dtype and bytes. All structural arrays are also
compared; only metadata_json.thermocalc_config.cache_dir is replaced with null,
retaining every other JSON character. This exception is recorded in the manifest.
The new detached-06dc7b4 JSON/NPZ fixture refuses overwrite and is hash-checked
before comparison. All 89 older reference files and existing inputs/tolerances
are unchanged. A missing runtime/licence/local input produces an explicit skip;
a present input hash mismatch or unrelated generation failure fails the gate.

Surrogate building is installed; the script orchestration listed above remains
in the research checkout, including the deliberately retained FeCrNi demo wrapper.
O06 follows that demo; a future TC case using the Lee 1996 script's probe-point search plus a prepare_surrogate_artifact bundle is an option for the user to decide, not an approved addition.
NiTiNb/other TC databases or phase
universes, arbitrary three-phase generation, default-phase invalid probes with TC,
broader diagnostics/retry settings and physical convergence are not accepted by
O06. The broader Stage 2 generation/evaluation/persistence/domain/equilibrium split
is still open. Scientific fixes remain research-branch work.

See [LEDGERS.md R03](LEDGERS.md#research-dependency-issue-r03-tc-debugger-importcall-open) for main's remaining TC debugger dependency.

## Next scoped work

- Stage 2C is complete within O06. Expand TC acceptance only with a separately
  approved scope and inputs; retain the frozen adapter behavior and exact gate.
- Broaden fallback/Numba, plotting/result workflows, restart/coupling and scientific
  convergence coverage before claiming those capabilities. Parity is not proof of
  mathematical convergence or support for untested combinations.
- Preserve one numerical authority: since the authority switch (2026-11-11) it is this
  package. Solver changes (R07, R08, ...) are made here only. A deliberate numerical change
  is adopted with `acceptance.py approve --cases ... --reason "LEDGERS Rxx"` after its
  correctness gates; the frozen `reference/` files are never rewritten.
- R13 follow-ups, open and not part of R13
  ([LEDGERS R13 *Follow-up*](LEDGERS.md#r13-ternary-and-three-phase-residual-normalization-resolved)):
  - a tighter default ternary/three-phase tolerance, which was promising on the survey fixtures but is not
    established for TC-derived cases or very fine grids;
  - `bulk_picard_atol=1e-12` is an absolute composition, so the implicit-mode inner tolerance is also
    dilution-sensitive (not examined);
  - the two-phase startup eta estimate fails when one component is diluted by 1e-4 or more unless
    `initial_eta_componentwise_scaling` is set.

  Defaults are unchanged; any change needs its own plan, correctness gates and approval.
- FUTURE_PORTS entries 1-2 are ported. The planned follow-on is an open-data Fe-Cr-Ni example that
  builds pycalphad (and optionally Thermo-Calc) bundles, diagnoses and compares them and runs a small
  simulation from each. It starts with a database provenance and licence check of the
  MatCalc-derived `examples/FeCrNi.tdb` at the repository root (it is vendored only if that check
  passes) and a bounded feasibility study, each followed by a user checkpoint. Entries 3-6 are
  prioritized at the backlog review ([FUTURE_PORTS.md](FUTURE_PORTS.md)).
- Example acceptance decisions are pending ([ACCEPTANCE_CANDIDATES.md](ACCEPTANCE_CANDIDATES.md)): E-case
  tiers and limits, the E11 lagged fixture (freeze its configuration first), and whether the Ni-P bonding
  example gets an E case.

## Open decisions reserved for the user

- Final package name and intended audience/distribution/maintenance ownership.
- Pushing branches, publication, MAP licensing outreach and upstream contribution.
- Changing either ternary solver's thin-phase prompt default.

## Baseline and research policy

The extraction's kawin/, examples/ and root packaging remain frozen at ae9fb29.
The approved numerical oracle is the clean detached 90700be worktree (`mb-ref-90700be`):
research main with the R05 and R06 merges, moved from 06dc7b4 on 2026-11-09 (LEDGERS.md
R06); the frozen O06/O08 fixtures stay from 06dc7b4. Since the same day the oracle is
`mb-ref-720d122` (adds the two-phase ternary R06 merge); O06 was recaptured there, O08 stays
from 06dc7b4. Since 2026-11-10 the oracle is `mb-ref-7af24f4` (adds the three-phase R06 merge;
O06 and O08 fixtures unchanged). Since the authority switch
(2026-11-11) the oracle is frozen and is never moved again; it is needed only for the
explicit historical actions (`historical capture|repeat --research`). Do not merge research
HEAD into the extraction or develop scientific repairs independently in both locations.
Preserve floating-point order, x/u and volume conventions, R=8.3145 where used,
and temperature-array hours. Never regenerate references to hide a difference.
The shared Conda environment is unchanged. No push, research-branch merge,
upstream contact, MAP-author contact or blanket redistribution permission is given.
