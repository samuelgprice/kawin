# Capability and dependency ledgers

## Current capability inventory

Quick: 31 run comparisons, 5 contracts and 145 pytest tests passed in 413.63 s. Full: 42 run comparisons, 5 contracts and 145 pytest tests passed in 521.06 s (public tolerance, `--record`, 2026-11-15, at `65ce32c` after the C33/F33 activation; every run bit-identical against the selected references, live O06 included, no case skipped).
These are regression claims for the approved frozen inputs, not a full release or
physical convergence audit. See ACCEPTANCE_CANDIDATES.md for exact input variants.

| Capability | Evidence and supported scope |
|---|---|
| Binary constant, lagged and implicit diffusivity | C01-C04, C13-C15, C30-C31 and legacy 33 histories/12 bulk probes |
| Ternary two-phase and motion directions | C05-C08, interstitial x/u C16-C19, saved surrogate C24; analytic accuracy C33/F33 |
| Three-phase, moving geometry and molar volume | C09-C12, C20-C23; equal/unequal explicit volumes; binary/two-phase unequal volumes unsupported |
| Exact/similarity diagnostics and drift | C04/C11/C12/C33 and full F04/F11/F12/F33; analytic errors recorded independently of reference; C33/F33 compare the two-phase ternary solver with its similarity solution against package-captured baselines (see "Two-phase ternary analytic acceptance cases (C33/F33)") |
| Lifecycle, rollback, failures and terminal stop | C25-C29 package pytest main/package contracts; C32 configured continue stop using real retries; no topology transition |
| Cached material/provider workflows | O01/O02/O03a/O03b/O04/O07 run whenever present; private TC-derived inputs/references remain ignored |
| pycalphad generation (Stage 2B relocation complete within O05/O08) | Installed providers/pycalphad.py; real restricted O05 and default-phase O08 sampling, seed 0, package examples import/read guard; O08 has two valid tie lines and one invalid probe, bitwise fixture/metadata gate |
| TC-Python live generation | Stage 2C complete within full-only O06: installed batch-safe adapter, real guarded sampling, approved suffixed local input, six steps and unseeded repeat pair; exact arrays, only metadata cache path normalized |
| Surrogate diagnostics and saved-bundle comparison (FUTURE_PORTS entries 1-2, 2026-11-15) | `surrogate_diagnostics.py`, `surrogate_comparison.py`: package tests only (generated ports of the research tests plus import/optional-Plotly guards), no acceptance case; oracle-versus-package parity bit-identical on synthetic surrogates and bundles. Plotly figures are checked structurally; the optional `[diagnostics]` extra is needed only for plotting. 2026-11-16: the comparator's optional `phase_mapping` (package extension, hand-written tests; unchanged without a mapping) and the open Fe-Cr-Ni workflow example (example tests only, no E case) |
| Unaccepted workflows | Arbitrary plotting/results/restart/coupling, Numba fallback, other kawin versions, alternative moving-boundary solvers and curvature/precipitation APIs are outside this gate |

## Ownership and provenance

Original whole-module solver/numerics/mesh/equilibrium/provider/domain copies map
to 06dc7b4 in relocation-map.json and BINARY_SLICE.md; the diagnostics and comparison modules
(FUTURE_PORTS entries 1-2) record 7af24f4, where their sources are identical to 06dc7b4, with the
source sha256 and their two message-only adaptations; the comparator's `phase_mapping` is a
package extension with no research counterpart (relocation-map.json `package_extensions`). Stage 2B generation maps
function-by-function in PUBLIC_KAWIN_MILESTONE.md and the provider docstring.
Building Thermo-Calc or pycalphad surrogates uses installed providers and
from_database, optionally prepare_surrogate_artifact bundles and
merge_phase_diffusivity_surrogates; see [README's pattern](../README.md#building-surrogates).
Research-checkout dependencies are script orchestration: the FeCrNi demo wrapper,
training_data.py dataset sampling/export/resume, case assembly, plotting, and
scripts' probe-point search settings, failure/diagnostic reports and bundle build
specs. The demo wrapper will not move: its defaults are demo-specific and no real
research workflow uses it. A general session/provenance helper without system
defaults is possible but not planned; see NEXT_SESSIONS.md's installed-building
scope. The TC adapter is package-owned and accepted within O06, which follows
the FeCrNi demo; a future real-usage TC case (Lee 1996 probe-point search plus a
prepare_surrogate_artifact bundle) remains a user decision, not an approved case.
The broader Stage 2
generation/evaluation/persistence/domain/equilibrium split is open; the four whole
copied modules remain unchanged. R01 is open on the research branch. Shared private
helpers live in solvers/illingworth/shared.py. No fork solver inheritance remains. Public unmodified
kawin==0.5.0 supplies model/solver iteration, mesh bases, recording, temperature and
thermo/pycalphad. PUBLIC_KAWIN_MILESTONE.md maps every bounded policy method:
minimum-step/final-remainder, initial adjustment, local floor, recording additions,
endpoint metadata and explicit-R tracer wrapper. No kawin globals are patched.

Stock wheel SHA256:
9f7a1497c5dff4314536860ac6fd0b96a68243dd681db8eb106dbe33fe25cbaf.
All 73 installed stock files match. The separate offline clone is mb-public-050;
the shared kawin environment is unchanged. The wheel contains only moving_boundary
and distribution metadata, no kawin sources or reference/provider datasets.

Numerical source for new references is clean detached 06dc7b4. Every reference
worker asserts its import origin and source SHA; installed workers assert stock
host and package origins. The extraction's kawin/examples/root packaging stay
at ae9fb29. Existing inputs and tolerances were frozen before comparisons. O05/O08
need the narrow hash exception described in R01 below. O08 adds one newly frozen
detached-06dc7b4 JSON/NPZ fixture with manifest hashes; original fixtures remain
unchanged. No worker imposes a NumPy seed.

## Authority switch (2026-11-11)

The package is now the numerical authority (plan: docs/AUTHORITY_SWITCH_PLAN.md, items 7 and
14-18). Research main stays frozen at `7af24f4`, and the clean detached `mb-ref-7af24f4` is the
last research oracle; it is never moved again.

- **Ownership.** Case definitions are package-owned in `tools/definitions/` (outside the
  wheel). They are materialized from the research builders by `tools/materialize_definitions.py`
  and hash-checked against `docs/definitions-provenance.json` on every run. The adopted local
  inputs (`open-surrogate.npz`, `pycalphad-reference.npz`, the example settings) keep their
  original hashes, and package-owned generation paths were shown to recreate them
  (`docs/regeneration-verification.json`, 12/12 verified). LEGACY's baseline is
  `reference/LEGACY.npz`, after a one-time bitwise proof against the research
  `illingworth_substitutional_baseline.npz`. 301 of 314 research tests are ported
  (`docs/test-port-provenance.json`); the 13 deferred ones are listed in FUTURE_PORTS.md.
- **Normal and historical actions.** `quick`, `full` and example acceptance read no research
  file and run no Git. A `git archive` export passed them with research reads and Git blocked,
  with the approved set both empty and active (plan M9). `historical capture|repeat --research
  PATH` is the only research use. `tooling_sha256` was removed.
- **Approved references.** A deliberate numerical change is adopted with `acceptance.py approve
  --cases ... --reason "LEDGERS Rxx"`. It runs the whole test suite, a repeat pair for every
  case, the registered `approval_criteria`, and a public-tolerance check of every unlisted case.
  It then promotes immutable, hash-recorded artifacts under
  `.verification/acceptance/reference-approved/` (local, never committed) and writes the
  manifest entry (`tools/acceptance_manifest.json`) and the report (`docs/approvals/<id>.json`).
  `approve` does not commit anything. Committing those two files is a separate manual step,
  done separately from the solver change; the report must be force-added (`git add -f`). The
  approval takes effect locally as soon as it is written. `approve --rollback ID` points those
  cases back, and its manifest change is committed the same way.
  `reference/` is never rewritten. The first real approval is `20261111T222849Z-5aedde7-1`
  (LEDGERS R08, 11 binary cases, commit `34c02d2`). A pure storage-layout change is adopted with
  `approve --layout-correction RULE`, which adds a strict structural proof on top of every gate
  (`tools/acceptance_layout.py`; LEDGERS R09, stage R09c).
- **Item 7.** Contract cases C25-C29 compare with per-model molar-volume scales
  (`contract_model_scales`, user-approved), as run cases do.
- **Item 14 supersession.** Supersession is recorded only in the package docs (this entry,
  NEXT_SESSIONS.md's definition of done, the branch ledger). Research main is not edited.

Migration log (branch `refactor/package-extraction`, not pushed):

| Batch | Commits |
|---|---|
| M0 classification, materializer, artifact hashes | `7f7755f` |
| M1 definitions C01-C23, C30-C32, LEGACY | `e3b9bfd`, `8f74746` |
| M2 contracts and per-model scales (item 7) | `045f0b9`, `3b4926c` |
| M3 C24 and prepare | `deebac9` |
| M4 F04/F11/F12, O01-O08 | `3321479` |
| M5 example worker, hashed settings | `1cee752`, `9c37a44` |
| M5b regeneration verification | `2a239a8` |
| M6 normal/historical split | `3c2c4a6`, `a4d38a6` |
| M7 approval path | `3ca6911`, `e86ff94`, `429b3fe`, `e857424` |
| M8 test port | `52601f2`, `833e620`, `b3606de`, `42f32b1` |
| M9 independence proof | `0f71c19`, `4d64aa0`, `d879153` |
| M10 docs and final recorded gates | this entry; results in acceptance.json and the plan log |

## Research reproducibility issue R01 (open)

O05 restricted generation and O08 default-phase sampling are hash-sensitive.
The 42 unaffected cases (33 quick runs/contracts plus nine full additions) each
matched in two fresh processes with neither seed override.
The cause is third-party pycalphad 0.11.1 symbolic construction, not an unseeded
sampler or moving-boundary solver. pycalphad/variables.py:215 hashes Species by
its name; pycalphad/model.py:893 constructs an unordered active-species set,
:895 iterates it and :903 accumulates ideal_mixing_term. Hash-dependent insertion
order changes the ordered SymEngine expression tree even when its printed algebra
is identical. pycalphad/codegen/sympydiff_utils.py:137 compiles Hessians through
SymEngine 0.13.0 lambdify with CSE enabled; compiled Hessian roundoff and the first
interface endpoints already differ before the moving-boundary solve.

Controlled hash 0/1 reference runs differed in 25 arrays. Sorting only the
ideal_mixing_energy active-species iteration in an isolated diagnostic process
made every O05 array identical across those hashes; sorting normalization/moles
alone did not. Compile-only no-CSE Hessian evaluation also removed the observed
test-point discrepancy. These were diagnostic controls, not source fixes.
NumPy's global RNG state was unchanged and no RNG calls were observed; a NumPy
seed alone failed to restore repeatability. Only O05/O08 retain PYTHONHASHSEED=0;
no worker calls np.random.seed. Other cases explicitly require no hash override.

Affected research workflow at main 06dc7b4: surrogate generation in
kawin/diffusion/MovingBoundarySurrogates.py:1810/:887 and the restricted provider
factory examples/ternaryExamples/pycalphad_default_phase_adapter.py:86 delegates
to these third-party models. Rebuilding pycalphad-backed training data can propagate
roundoff into sampled endpoints/matrices and then solver histories. Analytic,
saved-surrogate and cached-table solves do not invoke that model generation.
O08 exercises the default-phase facade at
examples/ternaryExamples/pycalphad_default_phase_adapter.py:169. Fresh hash-0/hash-1
reference workers differ in five of seven O08 arrays: alpha/beta endpoints, phase
amounts, full compositions and chemical potentials. A hash-0 repeat is bit-identical.
This real full-phase equilibrium path shares the diagnosed pycalphad symbolic/CSE
dependency. Two preliminary unseeded pilots happened to match hash 0; the explicit
hash-1 control establishes sensitivity, so O08 has the same narrow seed exception.
The installed Stage 2B copy retains R01: isolated real hash-1 sampling differs in
13 of 21 numeric provider arrays from the accepted hash-0 generation. That check
uses ignored temporary artifacts and leaves accepted outputs unchanged.

Proposed research-branch fix: add cross-hash subprocess regression coverage at
the provider boundary, then evaluate deterministic symbolic species/term ordering
(or a validated dependency/compiler change), including raw derivatives, endpoint
ordering, sampled matrices and downstream solves. This is a scientific change:
author/review it on the designated research branch and port deliberately. No fix
was applied to main, extracted numerics or installed dependencies. No reference
or preregistered tolerance was changed. Diagnostic traces remain ignored locally.

The mandatory quick/full gate now repeats every unaffected quick case and C25-C29
contract twice in fresh unseeded reference workers, comparing shape, dtype and raw
bytes, including signed zero/NaN payloads. Their diagnostic outputs never overwrite
fixtures. The measured 33 reference pairs took 290.98 s; together with the former
146.36 s quick gate the estimate is 7-8 minutes, below its unchanged 600 s cap.

## Research dependency issue R02: invalid-endpoint debugger import (open)

Main still imports examples.debugInPlace inside the invalid-endpoint branch of
examples/ternaryExamples/pycalphad_default_phase_adapter.py:190 (also present at
06dc7b4). The actual debugInPlace() call is commented out, but the import creates
a research-checkout dependency whenever a probe is outside the requested two-phase
field. O08's real FCC-only probe executes that branch in the unchanged reference
adapter and confirms the helper was imported. The package's source-mapped copy
omits only this unused import and passes its generation import/read guard.
Main and the frozen oracle are unchanged. A research-branch cleanup of the unused
import is proposed separately; no repair is made on main in this extraction.

## Research dependency issue R03: TC debugger import/call (open)

Main 5f4ad57 still imports examples.debugInPlace at
examples/ThermoCalc/tc_python_adapter.py:21 and calls debugInPlace() in the
generic calculation-wrapper exception branch at :1289 (also frozen at 06dc7b4).
Inside Jupyter the helper can wait for a debugger client; its module import also
creates a research-checkout dependency. The package copy omits only the import
and live call, retaining diagnostics, error type/message/cause and retry logic.
Licence-free failure tests guard input/debugger imports and preserve __cause__.
Main and the frozen reference are unchanged. Any source cleanup belongs on the
research branch; the installed batch-safe copy does not resolve this main issue.

## Research phase-name issue R04 (open research cleanup; O06 convention resolved)

In examples/ThermoCalc/tc_python_adapter.py, calculate_equilibrium:452 stopped
applying base_phase_name in cd87f88 (2026-08-22), preserving composition-set
names such as FCC_A1#1. With the original O06 FeCrNi base-phase configuration,
the exact phase-key match at :1754 fails, every tie line is invalid, and
kawin/diffusion/MovingBoundarySurrogates.py:873 rejects the endpoints. The
examples/ThermoCalc tree is identical at frozen 06dc7b4 and main 5f4ad57, so
main's unsuffixed FeCrNi TC workflow is affected too.

The diagnostic probe using the frozen 06dc7b4 adapter and the same O06 sampling
inputs with phases=("BCC_A2#1", "FCC_A1#1") passed configuration validation,
all three tie-line checks, ten kinetics queries and the surrogate build; the
session closed normally. No adapter change was made. This suffixed-name
convention, already used by the NiTiNb TC script
examples/ternaryExamples/IllingworthTernaryThreePhaseNiTiNb_TC.py:166, resolves
O06's phase-name blocker. The user approved that sole O06 input revision;
the reference adapter remains frozen at 06dc7b4. The revised two unseeded
runs and hash-1 control were bit-identical. O06 is now accepted in full,
unseeded with its own repeat pair and no numerical tolerance. Only the
per-worker metadata cache-directory field is normalized in comparison.

The duplicate-base-phase check at :453 cannot detect separately numbered
composition sets, and the docstring at :441-446 still describes base-name
merging. Composition-set identifiers can distinguish physically distinct
phases in NiTiNb; do not strip them unconditionally in the package or on main.
Remaining research-branch work, validated on both FeCrNi and NiTiNb:

- Update FeCrNi_TC_Python.py to configure suffixed names.
- Raise a clear error when a stable phase matches a configured phase by base
  name but not by exact name.
- Correct the stale docstring at :441-446 and the dead duplicate check at :453.

## Research comparison issue R05: x compared with u (fixed on a research branch, not merged)

`compare_model_to_similarity` and its three-phase counterpart subtracted the exact solution,
which is in the solver state basis, from `getPhysicalPhaseProfiles`, which returns mole
fractions. With interstitials that compares x with u, leaving u²/(1+u) for carbon: up to
about 8.7e-4 at u_C = 0.03, against the fork test's 1e-3 bound. Substitutional results are
unaffected. No solver numerics were involved, and eta, k, interface and inventory metrics
were never affected.

Fixed in `bc62481` on `fix/r05-similarity-state-basis` (research worktree, from main 5f4ad57;
not pushed or merged), with a focused no-solve test. All 30 tests in the four affected fork
files pass with no bound changed. At the source configuration the corrected carbon profile
error is 4.75e-4. `analytic/` is ported from that commit. The 06dc7b4 acceptance oracle is
unchanged: `collect` doesn't use this helper, and existing `exact/*` arrays come from their
own branches.

## Research numerical issue R06: binary interface iteration stalls above its tolerance (binary fixed: research `25bf6cd`, merged to research main `90700be`, ported, oracle moved 2026-11-09; two-phase ternary fixed: research `2741135`, merged `720d122`, ported, oracle moved 2026-11-09; three-phase fixed: research `fae6149`, merged `7af24f4`, ported, oracle moved 2026-11-10)

Found with `examples/binary_tlp_bonding.py` (Ni-P bonding, um and s units), 2026-11-08. Late in
the solidification stage, the binary Illingworth model's interface fixed point stops reducing
its error just above `tolerance` and raises after `max_iterations`:

- Tolerance 1e-9 um, DEMO settings: stalls at exactly 2.027e-9, with 200 or 2000 iterations.
- Tolerance 1e-8 um: power-law grids with exponent 2 in the liquid fail between 2,820 s and
  8,951 s (floors 1.0–2.0e-8), and exponent 3 fails even at 5e-8; 161/641 linear-graded nodes
  at `semiLog_dt` 0.01 fail at 63,863 s (floor 2.1e-7). Coarser grids and smaller steps complete.
- The floor rises as the smallest cell shrinks, and for a fixed grid, larger time steps fail
  earlier. The relative inventory drift also grows with refinement (about 1e-8 to 1e-7 at the
  finest grids, against 5e-9 at DEMO), consistent with the drift being set by the iteration
  tolerance rather than the scheme.

The example works around it: DEMO uses 81/321 linear-graded nodes, `semiLog_dt` 0.005 and
tolerance 1e-8 um, and its notes warn that finer grids run to the full duration can fail late.
No solver code was changed. Any fix starts on the research branch, per the policy above.

**Diagnosis (2026-11-09; scratch scripts and a monkeypatch prototype, no repo code changed).**
The stall is round-off in the fixed-point map, not slow or cyclic iteration dynamics:

- At the first stalled step of DEMO at tolerance 1e-9 um (t = 30,166.6 s, dt = 151.2 s,
  s = 9.641 um, interface cells 0.024 um liquid and 0.094 um solid) the map
  `s_new = G(s_trial)` is a strong contraction (`G'` about 0.002-0.008). It converges to about
  1e-9 in three passes, then cycles among values up to 7e-9 apart.
- Evaluated at 11 trial positions 1e-12 um apart, the float `G` spreads over 3.5e-9 um; the
  same `G` with assembly, Thomas solve and interface formula in 50-digit `decimal` arithmetic
  spreads over 7e-15 um (smooth). Exact solves of the float-assembled systems, and an exact
  interface formula on the float profiles, both stay noisy (3.4e-9, 3.5e-9 um).
- The noise is in the liquid node next to the interface. The exact `c_L - p[-2]` is 4e-21
  (the liquid is uniform), but float `p[-2]` is 1-5 ulps of `c_L` (1e-17 to 7e-17) away from
  it. The flux `(c_L - p[-2]) / (1 - u[-2]) * D_L * dt / s` multiplies that by about 3e7 here,
  giving spurious liquid flux of about 1e-10 against a solid flux of 3.4e-3 (composition um)
  and nanometre noise in `s`. The amplification scales with `dt / dz`, which matches the pattern
  above (smaller interface cells or larger steps fail earlier). The drift growing with
  refinement has the same cause.

**Correction (2026-11-09, recorded with the ternary fix; the bullet above is left as first
written).** Two figures in the third bullet are wrong. (1) "nanometre noise in `s`": Ni-P runs
in um, so the 3.5e-9 um spread is 3.5 femtometres. (2) "multiplies that by about 3e7": the
factor `D * dt / dz` from the entry's own numbers (D = 500 um^2/s, dt = 151.2 s, interface cell
0.024 um) is about 3.2e6 um, which also reproduces the stated spurious flux of about 1e-10 from
the 1e-17 to 7e-17 error in `p[-2]`. The commit message of research `25bf6cd` has the same
"nanometre" wording; the history is left unchanged. `R06_deviation_form_explainer.md` on research
main uses the corrected figures.

**Candidate fix, prototyped only.** Solve each phase for the deviation from its interface
composition, `w = p - c_L` (left) and `w = q - c_S` (right). Every row of the bulk systems sums
to its old-volume coefficient, so the deviation system has the same matrix and the same
right-hand side with `p_old - c` in place of `p_old` (an exact identity). The interface update
then uses `w[-2]` and `w[1]` directly. As a monkeypatch on the Ni-P example this finished every
case above (baseline / deviation form):

| Case | Baseline | Deviation form |
|---|---|---|
| DEMO, tol 1e-8 | ok, drift 7.5e-9 | ok, drift 2.1e-9 |
| DEMO, tol 1e-9 | fails at 30,167 s | ok, drift 8.2e-11 |
| DEMO, tol 1e-10 | fails at 4,767 s | ok, drift 1.4e-11 |
| 161/641 linear, `semiLog_dt` 0.01, tol 1e-8 | fails at 63,863 s | ok, drift 1.0e-9 |
| 161/641 power 2/2 to 1e4 s, tol 1e-8 | fails at 8,951 s | ok, drift 1.3e-9 |
| 81/321 power 3/2 to 1e4 s, tol 5e-8 | fails at 5,190 s | ok, drift 3.6e-9 |

On common steps the widths differ by at most 1.3e-6 um; the DEMO final width converges to
2.53782703 um (baseline at 1e-8: 2.53782721). The changes originate in round-off, but the
flux amplification makes them far larger than a few ulps of the outputs (up to 1.3e-6 um in
width here), so they need deliberate validation and baseline review: the binary entries of
`kawin/tests/data/illingworth_substitutional_baseline.npz` (bitwise) would change. The dimensionless residual test that R08 proposes would not remove this floor:
the two-phase ternary documents the same limit (`ternary.py` class docstring, "Resolution
limit", noise `eps * sum_j |D_kj| * |u_j| * dt / dz`) and retries instead. The `|u_j|` factor
suggests the deviation form would help there too (not tested).

**Status and decisions (user, 2026-11-09):** binary first, on by default; ternary afterwards as
a separate prototype (its interface compositions vary with eta, and the three-phase middle phase
has two interface compositions); R08 as a separate change after R06.

**Research fix (2026-11-09, commit `25bf6cd`, not merged):** worktree `...\code\kawin-r06`,
branch `fix/r06-binary-deviation-solve` from research `main` `5f4ad57`. In
`kawin/diffusion/MovingBoundaryIllingworthFDM.py`, both phase solves assemble the deviation
right-hand side directly and return the deviation; `_BinaryBulkSolveResult` carries the
interface-adjacent deviation; `_new_interface_planar` takes the deviations; Picard still queries
D and measures its update with absolute compositions. Verification on that branch:

- New `kawin/tests/test_illingworth_binary_deviation_form.py`: the row-sum identity on every
  row for both motion directions and scalar or face D; exact preservation of a state at the
  interface compositions in all three diffusivity modes; Ni-P at tolerance 1e-10 um to 6,000 s
  (drift < 1e-10); smoothness of the interface map at 30,000 s. The preservation and Ni-P tests
  fail on `main` (the Ni-P run stalls). A further 12-case test compares one step with 50-digit
  arithmetic where profiles are far from or cross their interface compositions (including
  `c = 0` and `c` near 1 beside bulk near 0), both directions, scalar and face D: profiles and
  deviations within 1.6 ulps of the composition scale (bound 8), interface update correctly
  rounded (bound 4 ulps of `s`), so the shift costs nothing where it gains nothing.
- 50-digit `decimal` evaluation of the interface map at four Ni-P steps (two growing, two
  shrinking): float minus exact at most 1.8e-15 um (about one ulp of `s`), against 3.5e-9 before.
- Bitwise substitutional baseline: only binary arrays changed, by at most 1e-15 relative; every
  binary interface history, right-phase profile, ternary and three-phase array is unchanged.
  Regenerated deliberately and logged in `make_illingworth_substitutional_baseline.py`.
- Python versus the authors' MAP C++ (`compare_illingworth2005_planar.main()`, author default
  case): identical to `main`, max interface difference 4.74e-6 relative on both.
- The Fig. 3 "present work" run of that harness (paper Ni-P, about 1e5 steps) stalls on `main`
  ("10000 iterations; final interface error was 1.264e-10") and completes on the fix branch.
- All 517 `kawin/tests` Illingworth tests pass after one deliberate change, approved by the user:
  `test_ternary_illingworth_quasi_binary_cu_zn_dummy_matches_binary_brass_case` compared the
  ternary with the binary interface to 1e-10 um and the gap became 1.7e-10. A 50-digit march of
  that case (same time points) gives errors against the exact discrete solution of 2.0e-10 um
  for the old binary, 3.1e-11 for the fixed binary and 2.0e-10 for the ternary: the old
  agreement came from shared absolute-form round-off. The bound is now 3e-10, with a comment to
  restore 1e-10 once the ternary also solves in deviation form. This is also direct evidence
  that the ternary carries the same round-off error.

**Port (2026-11-09).** Source `25bf6cd60732ed34ed6f70b94716a515c76b714c`; affected functions
`_BinaryBulkSolveResult`, `_new_interface_planar`, `_new_concentration_left_planar`,
`_new_concentration_right_planar`, `_solve_bulk_picard`, `_solve_bulk_phases`,
`_take_implicit_step_planar` and the class docstring of `solvers/illingworth/binary.py`. The
research diff applied unchanged (offset only); the module now equals the source apart from its
header and imports. `tests/test_binary_deviation_form.py` ports the 21 research tests with the
research helpers reproduced locally. Rationale and expected changes as above: round-off-
originated binary differences, no ternary or three-phase change.

- Package tests: the 21 ported tests plus bridge/tridiagonal/compatibility (34 passed);
  `tests/test_examples.py` 46 passed, 1 slow deselected.
- `acceptance.py quick` (public tolerance, 06dc7b4 oracle, 487.7 s): 29 runs pass, 20 bit-identical
  and 9 within tolerance (C01-C04, C13-C15, C30, C31; worst C04 analytic error 5.7e-15 against
  1e-12, none within 10x); contracts 3 bit-identical, 2 within tolerance (C25, C29; worst
  4.4e-16); 88 tests; 32 repeat pairs identical. `LEGACY` fails by design: it compares binary
  output bitwise with the 06dc7b4 substitutional baseline (`binary_phase_uniform__data_y`
  first; 460/3507 elements differ).
- `acceptance_examples.py run --skip-tests`: E03-E10 pass; E01 (vs C04) and E02 (vs F04) fail
  their bitwise fixture comparison, with differences at most 2.7e-14 absolute (interface 3e-13
  relative).
- Ni-P DEMO: unchanged except inventory drift 7.5e-9 -> 2.1e-9; the R06 grids (161/641
  `semiLog_dt` 0.01 to 7e4 s; 161/641 power 2/2 and 81/321 power 3/2 to 1e4 s; DEMO at 1e-10)
  complete with the ported solver. The example's notes and drift comments are updated; limits
  and settings are unchanged.

**Research merges and oracle move (2026-11-09, user-approved).** Research main gained
`10615ad` (merge of R05 `bc62481`) and `90700be` (merge of R06 `25bf6cd`), both `--no-ff` from
`5f4ad57`; 531 Illingworth/similarity tests pass on main. The acceptance oracle moved from
detached `06dc7b4` to a new clean detached worktree `...\code\mb-ref-90700be`:

- `acceptance_manifest.json`: top-level `reference_commit` and the 6 changed of 82
  `source_sha256` entries (the two similarity examples, `make_illingworth_substitutional_baseline.py`,
  `test_illingworth_binary_variable_diffusivity.py` and `test_illingworth_ternary.py` from R05/R06;
  `test_illingworth_ternary_three_phase.py` from `5f4ad57`). `acceptance.py` `REFERENCE` points at
  the new worktree. O06/O08 keep their frozen 06dc7b4 fixtures and hashes (untouched).
- Capture of 43 cases (all but O06/O08) passed in 245.6 s; old references are kept locally in
  `.verification/acceptance/reference-06dc7b4-backup`. Bytes changed only for binary-solver
  cases: C01-C04, C13-C15, C25, C29, C30, C31, F04, LEGACY and O07 (Fe-C `fe_c_exb1a`, at most
  9e-14 relative); every ternary, three-phase and sampling reference is byte-identical.
- `exact` quick: 30 runs and 5 contracts bit-identical, 88 tests, 33 repeat pairs, 520.8 s.
  `exact` full: 40 runs (O06 included, nothing skipped) and 5 contracts bit-identical, 88 tests,
  34 repeat pairs, 686.1 s. `acceptance_examples.py run --skip-tests`: E01-E10 pass, 99.1 s.
- Public quick and full with `--record` (same counts, 493.7 s and 625.2 s) updated
  `docs/acceptance.json` and `docs/acceptance-deviations.json`: the reference commit, run times,
  and the recorded inventory allowance of C09, C10 and C20-C23. The old reference JSONs lacked
  `minimum_molar_volume` (captured by an earlier worker; the three-phase solver is unchanged),
  so `compare()` used its default of 1; recaptured references carry it. `compare()` then divided
  every inventory-group array by it, although the manifest's rule is "initial domain length;
  divided by minimum explicit Vm for mole inventories". That loosened C23's composition-based
  inventories (`inventoryData`, `state/_initialInventory`, about 0.2) from an absolute floor of
  1e-12 to 1.4e-7. Fixed the same day: `acceptance.py` `inventory_scale()` divides only the mole
  arrays (`molarInventoryData`, `totalMolesData`, `dependentMoleClosureErrorData`, all in moles)
  by the minimum molar volume, matching the manifest, with a unit test; the records were
  re-recorded. No result changed; every run is bit-identical.
- Re-record with `inventory_scale()` (public quick 528.4 s and full 647.0 s; 30/40 runs and 5
  contracts bit-identical, 89 tests, 33/34 repeat pairs identical): C09, C10, C20-C23 and C32 now
  record two inventory allowances, 1e-12 x length for composition-based arrays and that divided
  by the minimum molar volume for mole arrays (C23: 1e-12 and 1.4e-7); cases without explicit
  volumes are unchanged.

**Ternary research fix (2026-11-09, commit `2741135`, merged to research main as `720d122`,
user-approved).** Branch `fix/r06-ternary-deviation-solve` (worktree `...\code\kawin-r06t`) from
research main `90700be`; the `--no-ff` merge `720d122` has the same tree as `2741135`. Six files:
`kawin/diffusion/MovingBoundaryIllingworthTernaryFDM.py`, `kawin/tests/test_illingworth_ternary.py`,
new `kawin/tests/test_illingworth_ternary_deviation_form.py` (57 tests),
`kawin/tests/make_illingworth_substitutional_baseline.py` (log entry), the regenerated
`kawin/tests/data/illingworth_substitutional_baseline.npz`, and new
`R06_deviation_form_explainer.md` (binary and ternary, before and after). Two-phase ternary only;
the three-phase solver is untouched.

- Each phase solve is shifted by the trial candidate's `c(eta)`: the right-hand side uses
  `p_old - c`, the Dirichlet row is 0, and the compiled fill receives `p - c`. The flux and the
  residual's swept-inventory terms use the solved interface-adjacent deviation, carried by the
  required `_BulkPhaseSolveResult.interface_deviation` and `.projection` fields. Diffusivity
  queries, Picard norms and stored profiles stay absolute.
- `w + c` resolves each component only to about `eps * (|c| + |w|)`, so a dilute value just above
  `minComposition` can round below it. `_project_shift_roundoff_to_floor` moves such violations
  onto the floor within a fixed policy threshold, `16 * eps * (|c| + |w|)` (a policy, not an
  error bound), behind the fast exact screen `_profile_within_floor_bounds`. Its signed inventory
  change is added to `_interface_residual`; projection totals are staged in
  `_accept_step_result` and committed in `postProcess`. Cost: about 15 us per phase solve.

Verification (fix branch unless stated; 588 Illingworth/similarity tests pass there, and again
on main after the merge, 140.7 s, including the bitwise baseline test):

- Residual noise (scaled carbon residual, float minus 50-digit evaluation of the same discrete
  equations, one motion branch and one closure segment): Fe-Mn-C 3201/p3.7 at t = 1 s,
  2.5e-11 -> 7.3e-16 at dln t 0.005 and 6.7e-11 -> 1.8e-15 at 0.01; at t = 1e-4 s,
  4.6e-15 -> 1.3e-18. Synthetic Fe-C-Mn-like case at t = 30 s: 2.0e-10 -> 1.7e-17.
- Brass quasi-binary: ternary-vs-binary interface gap 1.70e-10 -> 4.9e-11 um. Error against a
  50-digit march: ternary 2.01e-10 -> 4.1e-11 um, binary 3.5e-11 um. The bound of
  `test_ternary_illingworth_quasi_binary_cu_zn_dummy_matches_binary_brass_case` is restored
  from 3e-10 to 1e-10, as the binary record above anticipated.
- Fe-Mn-C to 2 s, main -> fix: 1601/p3.7 at dln t 0.01, retries 2144 -> 554 and drift
  1.9e-5 -> 9.3e-6; 3201/p3.7 at 0.01, retries 1470 -> 1619; at dln t 0.005, 0 retries on both
  trees. Every late-time failure disappears (1601: all failures in [1e-4, 1e-3) s; 3201: the 8
  at t >= 0.1 s). The remaining failures are all at t < 1e-3 s ("line search failed", best
  residual 1-3e-10), where the residual is accurate to about 1e-18, so they are not round-off.
  **Open, not R06** (possibly related to R07/R08); recorded only.
- Synthetic 401/p3.5 to 100 s: retries 128 -> 0, carbon drift -3.8e-9 -> 9.0e-11. On 801/p3.7:
  16,052 retries in 185 s -> 0 in 4.4 s.
- Bitwise substitutional baseline: 28 of 151 keys changed, all in the four ternary cases, at
  most 3.0e-15 relative. Interface and eta histories and all binary and three-phase arrays are
  bitwise unchanged. Regenerated deliberately and logged in
  `make_illingworth_substitutional_baseline.py`.
- Existing tests changed: private-signature callers, two solver mocks, the branch-spy index, and
  two outer-failure tests, which relied on a round-off residual of about 1e-23 and now start off
  equilibrium.
- **Open question:** k(2) shifts by about 1e-4 relative at dln t 0.005, where neither tree
  retries, which suggests sensitivity to the 1e-10 acceptance tolerance. Unexplained.

**Ternary port (2026-11-09).** Source `2741135c9a4ba2644a1facf6ad604112dba35309` (merged to research
main as `720d122`). The package keeps the research module's class in `solvers/illingworth/ternary.py`
and its pre-class helpers in `solvers/illingworth/shared.py`, both otherwise verbatim, so the
research diff was applied by hand-placed hunks: the 32 class-body hunks applied with exact
context at a uniform offset (-849 lines; a fuzzy apply misplaced six of them and was discarded),
and the three module-level hunks went to `shared.py`. Both files now equal research `2741135`
apart from their header comments and imports (`ternary.py` additionally imports
`_project_shift_roundoff_to_floor` from `shared.py`). Affected functions: in `shared.py`,
`_InterfaceCandidate` (projection fields), `_BulkPhaseSolveResult` (`interface_deviation`,
`projection`), the new `_FloorProjection`, `_read_only_zeros`, `_no_floor_projection`,
`_profile_within_floor_bounds`, `_project_shift_roundoff_to_floor`, `_FLOOR_PROJECTION_ULPS`,
`_FLOOR_PROJECTION_MAX_NEXTAFTER_STEPS` and the `_INTERFACE_ETA_FD_STEP` comment (kept verbatim,
although `three_phase.py` also imports the constant and is not in deviation form); in
`ternary.py`, the class docstring, `__init__`/`reset` projection totals,
`_reset_implicit_diagnostics`, `_record_implicit_success`/`_failure`, both interface-flux helpers,
the new `_project_floor_roundoff`, both planar solves, both Picard solves, `_interface_residual`,
`_evaluate_interface_candidate`, the new `_commit_pending_step_projection`, `_accept_step_result`
and `postProcess`. The compiled fills (`numerics/ternary.py`) and the three-phase solver are
unchanged. `tests/test_ternary_deviation_form.py` ports the 57 research tests; bodies are
unchanged except that two helpers are read from `shared.py`, and the fine-grid model's research
helpers (from `test_illingworth_interstitials_ternary.py`) are reproduced locally.
`docs/relocation-map.json` still names 06dc7b4 and the old `shared.py` symbol list, as after the
binary port.

- Package tests: the 57 ported tests pass (47.7 s); full `tests/` 215 passed, 8 skipped,
  1 deselected; `tests/test_examples.py` 46 passed, 1 slow deselected (with `MB_EXAMPLE_TDB` set
  to the acceptance copy; 43 passed, 3 skipped without it).
- `acceptance.py quick` (public tolerance, 90700be oracle, 516.8 s): 30 runs pass, 20 bit-identical
  and 10 within tolerance (C05-C08, C16-C19, C24, O05; worst C18 profiles 7.2e-14 against
  2.9e-12, none within 10x); contracts 4 bit-identical, 1 within tolerance (C26, 2.2e-16); 89
  tests; 33 repeat pairs identical. Every binary, three-phase (C09-C12, C20-C23, C32) and LEGACY
  run is bit-identical.

**Oracle move to 720d122 (2026-11-09, user-approved).** The acceptance oracle moved from detached
`90700be` to a new clean detached worktree `...\code\mb-ref-720d122` (research main with the
two-phase ternary merge):

- `acceptance_manifest.json`: top-level `reference_commit` and the 2 changed of 82 `source_sha256`
  entries (`make_illingworth_substitutional_baseline.py`, `test_illingworth_ternary.py`; the
  solver module itself is not hashed). `acceptance.py` `REFERENCE` points at the new worktree.
- Capture of 43 cases (all but O06/O08) passed in 220.6 s; old references are kept locally in
  `.verification/acceptance/reference-90700be-backup`. Array bytes (`.npz`) changed only for the
  worker's two-phase ternary family: C05-C08, C16-C19, C24, C26, O01, O03a, O04 and O05. Every
  binary, three-phase and LEGACY reference is byte-identical; the JSON files differ only in
  `kawin_origin` and `elapsed_seconds`.
- `exact` quick: 30 runs and 5 contracts bit-identical, 89 tests, 33 repeat pairs, 512.0 s.
- **O06 fixture refresh (user decision).** The first `exact` full failed only O06 (Thermo-Calc
  sampling, two-phase ternary), whose frozen 06dc7b4 fixture requires bytes in every mode: its
  46 sampled provider arrays, accepted times and grids were byte-identical, and only the 11
  solve-output arrays differed (profiles 3.4e-14 absolute / 4.5e-13 relative, eta 1.3e-15,
  interface 8.7e-19, inventory 1.1e-19), as in the other two-phase ternary cases. The frozen
  fixture was moved to `.verification/acceptance/reference-O06-06dc7b4-frozen` (hashes as
  before) and O06 recaptured from `mb-ref-720d122` with live Thermo-Calc (24.8 s): the same 11
  arrays changed by the same amounts, `provider/metadata_json` differs only in the normalized
  `thermocalc_config.cache_dir`, and every other array is byte-identical. The manifest's
  `thermocalc_sampling.reference_sha256` and `reference_commit` now name the new fixture and
  720d122; the byte-only gate itself is unchanged. The TC sampling sources
  (`examples/ThermoCalc/training_data.py`, the default-phase adapter) are unchanged between
  06dc7b4 and 720d122. O08 keeps its frozen 06dc7b4 fixture. `docs/acceptance.json`
  `stage_2c` still records the original O06 fixture hashes as the Stage 2C history.
- `exact` full after the refresh: 40 runs (O06 included) and 5 contracts bit-identical, 89 tests,
  34 repeat pairs, 620.8 s. Public quick and full with `--record` (same counts, 521.2 s and
  620.3 s) updated `docs/acceptance.json` run times and `docs/acceptance-deviations.json`
  `reference_commit`.
- `acceptance_examples.py run --skip-tests` first failed E03-E06 on "setting differs" only: the
  derived local reference settings (`.verification/acceptance/example_settings`, recorded once
  from 90700be) lacked the ten new projection-diagnostic attributes. They were moved to
  `example_settings-90700be-backup` and recaptured from 720d122: C04, C11, C12, F04, F11 and F12
  are byte-identical, and C16, C24, O04 and O05 differ only by those ten added attributes (no
  removed or changed entries). The rerun passes E01-E10 in 85.4 s (E11 pending).

**Three-phase research fix (2026-11-10, commit `fae6149`, merged to research main as `7af24f4`,
user-approved).** Branch `fix/r06-three-phase-deviation-solve` (worktree `...\code\kawin-r06tp`)
from research main `720d122`; the `--no-ff` merge `7af24f4` has the same tree as `fae6149`. Six
files: `kawin/diffusion/MovingBoundaryIllingworthTernaryThreePhaseFDM.py`, the floor-projection
helper in `kawin/diffusion/MovingBoundaryIllingworthTernaryFDM.py`, new
`kawin/tests/test_illingworth_ternary_three_phase_deviation_form.py` (93 tests),
`kawin/tests/make_illingworth_substitutional_baseline.py` (log entry), the regenerated
`kawin/tests/data/illingworth_substitutional_baseline.npz`, and a three-phase section in
`R06_deviation_form_explainer.md`.

- Mechanism: the interface-adjacent transfers were `L x0 + R x1` from absolute compositions, so
  each carried `eps * |A| * |c|` with `A = dt D / (L dxi)`. Each interval is now solved for
  `w = C - r`, with `r` the interface composition in the outer phases and, in the middle phase,
  `c_L + xi (c_R - c_L)` with exact end values. Every free row sums to `old_w I`, so the matrix is
  unchanged; free rows get `old_w (p_old - r) - lower dr_{i-1} - upper dr_{i+1}` (`dr` from grid
  differences), Dirichlet rows 0. The compiled fill receives `p - r` and zero fixed values; the
  middle-phase slope terms are added after either fill, which stay bitwise equal. End transfers
  use `(L + R)_f = d_f I` exactly: `H_0 = d_0 c_L + R_0 (dr_1 + w_1)`,
  `H_f = d_f c_R + L_f (w_{n-2} - dr_{n-1})`.
- Floor projection as in the two-phase fix. The shared helper accepts a per-node shift (threshold
  `16 eps (|r_i| + |w_i|)`, unchanged policy), several fixed nodes and optional node weights;
  two-phase results are bitwise unchanged. Outer phases add their signed change to their own
  interface residual; the middle phase splits its change node by node `(1 - xi, xi)` between A|B
  and B|C (user decision). `projection`/`projection_split` are required `_ThreePhaseBulkResult`
  fields; totals are staged with the pending candidate and committed in `postProcess`.

Verification (fix branch; 680 Illingworth/similarity tests and the 3 three-phase example tests
pass there, and 681 + 3 on main after the merge, 163 s):

- Fine-grid probe (similarity "opposite", 201 nodes per phase, end spacing ratio 1e-8, tolerance
  1e-11): scaled residual float minus 50-digit evaluation 5e-11..3e-10 -> at most 6.8e-19; retries
  401 in 645 steps -> 0 in 396; march 23.6 s -> 4.5 s. Default grids: 1.8e-18 -> 1.5e-19, no
  retries either way.
- Single intervals against 50-digit arithmetic (15 cases, all three phases): transfer errors 14 to
  1e7 times smaller, profile errors at most 9e-15 (were up to 4e-12). On graded grids the remaining
  transfer error is about `eps (|H| + |A| |w|)` plus input-geometry rounding, not `eps |H|`.
- Full runs with the middle phase projecting (B interior held at the composition floor by a 1e-8
  Y diffusivity beside interface Y values of 0.15/0.20; static and moving interfaces; all three
  diffusivity modes): 100 steps, no retries, the A|B/B|C split sums to the total, no diffusivity
  query below the floor. The moving case fails at its first step on `720d122` and with the
  projection disabled.
- Similarity convergence metrics unchanged. Substitutional baseline: only the 36 three-phase arrays
  moved (eta histories at most 1.2e-10 relative / 5.7e-11 absolute, interface 4.4e-12, profiles
  2.5e-11), regenerated deliberately and logged. Cost: default-grid march +12% wall time.

**Three-phase port (2026-11-10).** Source `fae6149ccc6ebfda2100c4ed5a9b99b962e77620` (merged to
research main as `7af24f4`). The research diff applied with exact context at uniform offsets: 18
hunks to `solvers/illingworth/three_phase.py` (+2 lines) and 4 hunks to the helper in
`solvers/illingworth/shared.py` (+1). `three_phase.py` now equals research `fae6149` apart from its
header and import paths; the `shared.py` helper section is identical to `fae6149` (only the helper
changed there since `2741135`, so `ternary.py` still equals the source class). Affected functions:
in `three_phase.py`, the class docstring, `_ThreePhaseBulkResult` (`projection`,
`projection_split`), `_ThreePhasePendingState` (`projection`), `__init__`/`reset` and the new
`_reset_projection_diagnostics`, `_solve_interval_planar`, the new `_interval_reference`,
`_apply_reference_slope_correction`, `_interval_cell_widths` and `_project_floor_roundoff`,
`_finish_interval_solve`, `_solve_interval_picard`, `_interface_residuals`, `getdXdt`,
`_store_pending_candidate`, the new `_record_step_projection`/`_commit_step_projection`,
`_accept_step_candidate` and `postProcess`; in `shared.py`, `_project_shift_roundoff_to_floor`.
The compiled fills are unchanged. `tests/test_three_phase_deviation_form.py` ports the 93 research
tests: bodies unchanged apart from package imports and calling the interstitial builder directly;
the helpers the research file took from three other research test modules are reproduced
verbatim and exposed as `tpt`/`sim` (interstitial helpers prefixed, the similarity `CASES` renamed
`_SIMILARITY_CASES`, to avoid clashes in one module).

- Package tests (`mb-public-050`, `src` on `PYTHONPATH`, `MB_EXAMPLE_TDB` set to the acceptance
  copy): the 93 ported tests pass (19.7 s); full `tests/` 311 passed, 5 skipped, 1 deselected
  (112.4 s).
- `acceptance.py quick` against the `720d122` oracle (public tolerance, 475.8 s) **fails for
  three-phase cases, as the solver change predicts**: 21 runs bit-identical, 7 within tolerance,
  2 failed; 87 of 89 pytest tests pass. Every binary, two-phase ternary, LEGACY, O05 and O08 run
  passes. C09 and C10 fail on eta: 5.72e-11 against an allowance of 4.69e-11 (C09 interface 3.1e-12,
  profiles 5.7e-12 absolute / 2.5e-11 relative, all within tolerance), digit for digit the change
  logged for the research baseline's three-phase cases. Contract C27 (three-phase lifecycle) fails
  on eta in 3 of its 5 models. Contract C28 (stationary unequal-volume three-phase) fails on
  `dependentMoleClosureErrorData`: 1.46e-11 = 2^-36, one ulp of a total of about 1e5 mol/m^2,
  against an absolute allowance of 1e-12 that is not scaled by the minimum molar volume in this
  contract (compare C23's recorded mole allowance, 1.4e-7). The eta shift is at the interface
  Newton tolerance: the accurate residual stops Newton at a slightly different root than the
  noisy one did. Unlike the two earlier R06 moves, this change is larger than the public
  tolerance, so the oracle move below adopts it deliberately (user decision, 2026-11-10); no
  tolerance was changed. Open question: whether contracts should apply the molar-volume scale to
  mole arrays, as `inventory_scale()` does for run cases.

**Oracle move to 7af24f4 (2026-11-10, user-approved).** The acceptance oracle moved from detached
`720d122` to a new clean detached worktree `...\code\mb-ref-7af24f4` (research main with the
three-phase merge):

- `acceptance_manifest.json`: top-level `reference_commit` and the 1 changed of 82 `source_sha256`
  entries (`make_illingworth_substitutional_baseline.py`; the solver modules are not hashed).
  `acceptance.py` `REFERENCE` points at the new worktree. `thermocalc_sampling.reference_commit`
  stays `720d122`: the O06 fixture is unchanged (two-phase results are bitwise unchanged).
- Capture at `--tier full` of 43 cases passed in 214.0 s (O08 refused overwrite, as designed; O06
  not recaptured); old references are kept locally in
  `.verification/acceptance/reference-720d122-backup`. Array bytes (`.npz`) changed only for the
  15 three-phase cases: C09-C12, C20-C23, C27, C28, C32, F11, F12, O02 (exb5 gamma/alpha/gamma)
  and O03b (symmetric three-phase check). Every binary, two-phase ternary and LEGACY reference is
  byte-identical; the JSON files differ only in `kawin_origin` and `elapsed_seconds`.
- `exact` quick: 30 runs and 5 contracts bit-identical, 89 tests, 33 repeat pairs, 494.8 s.
  `exact` full: 40 runs (O06 included with live Thermo-Calc, against its unchanged frozen fixture)
  and 5 contracts bit-identical, 89 tests, 34 repeat pairs, 631.1 s. Public quick and full with
  `--record` (same counts, all bit-identical, 471.9 s and 607.6 s) updated `docs/acceptance.json`
  run times and `docs/acceptance-deviations.json` `reference_commit`.
- `acceptance_examples.py run --skip-tests` first failed E07-E10 on "setting differs" only: the
  derived local settings lacked the nine new three-phase projection-diagnostic attributes. They
  were moved to `example_settings-720d122-backup` and recaptured from 7af24f4: C04, C16, C24, F04,
  O04 and O05 are byte-identical, and C11, C12, F11 and F12 differ only by those nine added
  attributes (none removed or changed). The rerun passes E01-E10 in 85.4 s (E11 pending).

Ternary R06 handed off 2026-11-09 (prompt: ternary-r06-deviation-form).
Handed off 2026-11-09 (prompt: r06-ternary-merge-port-threephase).
Three-phase R06 option (a) handed off 2026-11-09 (prompt: three-phase-r06-option-a).

## Release review follow-ups R07-R12 (open)

Recorded 2026-11-08 from the core Illingworth review; extraction branch tip
`f9bfd6b2fc3cc648bfda3693c2730d406297ab9c`. These are investigation items, not
implemented fixes or changes to the frozen acceptance contract. R07/R08 are the
highest-priority numerical findings; R09-R12 concern output, lifecycle and input
contracts. Diagnostic cases below are synthetic checks, not production benchmarks.

Evidence: 23 focused package tests passed using the activated kawin interpreter
with extracted package sources and stock kawin 0.5.0 host modules on the import
path. Additional diagnostic runs reproduced the observations below. Small binary,
interstitial two-phase and three-phase runs also completed with Numba imports
blocked. The full installed-wheel acceptance suite was not rerun for this review.
Scientific fixes must follow the existing authoritative-source/port policy; do not
refresh reference fixtures or widen tolerances to conceal changed behavior.

### R07: outer maximum timestep can invalidate an implicit step

**Status:** fixed 2026-11-11 (see Resolution below). **Scope:** binary and two-phase
ternary. Three-phase already limits the trial timestep before solving and verifies
its pending candidate during acceptance.

`binary.py:_compute_dt` and `ternary.py:_compute_dt` do not include the outer
maximum timestep. After the complete implicit solve returns an increment divided
by its trial dt, `compatibility/solver.py:DESolver._getdXdt` can shorten dt to
`_dtmax`. The driver then scales that increment, committing a state for which the
implicit equations were not solved. Both solver families reproduced the mechanism.

Binary reproduction: use `examples/binary_constant.py` inputs, with constant phase
D values ALPHA=1e-3 and BETA=1e-1, `tolerance=1e-10`, `max_step_retries=15`, and
`solve(0.1, minDtFrac=1e-14)`. With `time_step=0.1, maxDtFrac=0.1`, the final
interface was 1.1029278493730248 and relative inventory drift was 2.511e-4.
Actually solving at `time_step=0.01, maxDtFrac=1` gave 1.1246627054903873 and
drift 9.111e-12. The clamped run's interface advance was about 17% smaller.

**Investigate:** honor the outer maximum before each nonlinear solve and verify
that the committed timestep/state matches its converged candidate, following the
three-phase approach. Add regression coverage for fixed and semi-log schedules,
retry-reduced steps, and exact final remainders. A clamped run should agree with a
run explicitly configured to take the same smaller steps.

**Resolution (2026-11-11, package only; three-phase unchanged).**
- `_compute_dt` also takes the outer maximum `maxDtFrac * (finalTime - initialTime)`
  (the host wrapper's expression), so each implicit step is solved at the dt the
  wrapper commits. At the default `maxDtFrac=1` this equals the remaining time,
  which the schedule already applied.
- Outer minimum: the package wrapper never stretched a short step; it raised
  `ValueError`, but only after `getdXdt` had staged the step. `getdXdt` now rejects
  a converged step below `minDtFrac * simTime` itself ("reduce minDtFrac"), with the
  wrapper's own exact-final-remainder exemption, and clears the staged candidate
  (ternary: also the staged projection totals).
- `getdXdt` stages the converged candidate; `postProcess` raises `RuntimeError`
  and commits nothing if the wrapper's time or state does not reproduce it.
  Unlike three-phase, `postProcess` with no staged candidate is not an error,
  because an existing ternary test commits directly.
- Helpers are in `shared.py`. The trial-only diagnostics of a rejected step
  (`_currdt`, `_lastStepRetries`, interface D, ternary `_lastAcceptedDt`) are R10's.
- After the fix, the binary reproduction gives 1.1246627054903873 and drift
  9.110e-12 for both runs. The two-phase ternary analogue (`time_step=1e-2,
  maxDtFrac=0.1` against `time_step=1e-3`, test builder) gives 0.4505991417849936
  and drift 2.80e-11 for both; before the fix the clamped run gave
  0.45059520500044936 and drift 2.6e-7.
- Tests: `tests/test_illingworth_outer_timestep_limits.py` (fixed and semi-log
  schedules, retry-reduced steps, subminimum rejection, exact final remainder,
  and commit refusal, for both families). All 12 failed before the fix, except the
  two final-remainder guards.
- Gates (unchanged, no approval needed): full `tests/` 1051 passed, 5 skipped,
  1 deselected (fresh target); exact quick 30/30 + 5 contracts bit-identical,
  33 repeat pairs, gate pytest 145, 454 s; exact full 40/40 + 5 bit-identical,
  34 repeat pairs, 552 s, O06 run against live TC.

**Follow-up (2026-11-11): two commit-guard gaps in the resolution above.**
Both reproduced against `ce1c990` in both families.
- *Large time origins.* With `currentTime=1e9` and a 1e-4 step, `postProcess` at
  `start + 5e-5` with the full candidate state was accepted, because the relative
  tolerance on the absolute timestamp (2e-13 * 1e9 = 2e-4) hid the half step. The
  check now compares the elapsed step, `(time - start) - dt`, with 4 ulp of the
  largest time involved. That is the resolution of the wrapper's `start + dt`:
  about 4.77e-7 at t = 1e9, 3.55e-15 at t = 4, and 4 ulp of dt from t = 0.
  (Corrected; the first version of this record gave 1 ulp at t = 1e9, 1.2e-7.)
- *Rejection lifecycle.* A rejected update cleared the candidate, so resubmitting
  it committed through the "no candidate" exemption. `_awaitingFreshCandidate`
  is now set:
  - from the start of `getdXdt` until it succeeds;
  - on a mismatch, or an exception while checking;
  - while a commit is in progress.

  While it is set, every commit is refused, and everything staged (ternary: also
  the projection totals and the terminal thin-phase stop) is discarded, until
  `getdXdt` solves a new candidate. A model that never proposed a step still
  commits directly, as before.
- Not changed: an interrupted commit can still have advanced time or state before
  it raised (R10). Only its resubmission is refused.
- Tests: 20 more in `test_illingworth_outer_timestep_limits.py`. They cover valid and
  wrong timestamps at t = 1e9 (including a complete solve), repeated rejection,
  the originally correct update after a rejection, malformed updates, failed and
  subminimum proposals, interrupted commits, recovery through a new candidate,
  and direct-call compatibility. The 14 rejection tests failed against `ce1c990`;
  the 6 valid-update tests passed there too.
- Gates (unchanged, no approval needed): full `tests/` 1071 passed, 5 skipped,
  1 deselected (fresh target); exact quick 30/30 + 5 contracts bit-identical,
  33 repeat pairs, 454 s; exact full 40/40 + 5 bit-identical, 34 repeat pairs,
  590 s, O06 run against live TC.
- *Non-finite times (second follow-up, 2026-11-11).* `ulp(inf)` is infinite, so
  `postProcess(inf, valid candidate state)` was accepted in both families and
  recorded an infinite time. NaN was rejected, but only because NaN comparisons
  are false. A non-finite start, accepted or committed time or candidate dt is now
  reported as "non-finite time" before the allowance is formed. Rejection leaves
  the latch set and recovery works through a new candidate. The finite allowance
  is unchanged. Tests: 15 more (both families with inf, -inf and NaN; each time
  field; the finite four-ulp allowance at t = 1e9). This is input rejection only,
  so the exact acceptance gates were not rerun.

### R08: binary convergence tolerance depends on length units

**Status:** fixed in the package (2026-11-11, resolution below); binary references
approved as `20261111T222849Z-5aedde7-1` (commit `34c02d2`). **Scope:** binary fixed-point
interface iteration in `binary.py:_take_implicit_step_planar`.

The convergence test is `abs(previous_future_s - future_s) <= tolerance`, without
a conservative residual acceptance test. The default 1e-8 is an absolute length:
it means 10 nm for metre coordinates. Equivalent unit representations can stop
at different iteration counts and give materially different conservation/advance.

Reproduction: rescale the binary R07 problem to R=5e-6, s0=1e-6, D values 1e-15
and 1e-13; retain 501 physical mesh nodes, 100 nodes per phase, initial left/right
compositions 0.8/0.4, interface compositions 0.6/0, `time_step=0.1`,
`max_step_retries=15`, and solve to 0.5 with `minDtFrac=1e-14`.
With `tolerance=1e-8`, final s/R was 0.2610915910 and relative inventory drift
1.75484e-3 (about 0.18%; final step took two fixed-point passes). Scaling the
tolerance to 1e-14 gave s/R=0.2581334961 and drift 1.22648e-9 (11 passes),
matching the corresponding unscaled problem. Interface advance differed by about 5%.

**Investigate:** combine a dimensionless or explicit absolute/relative position
criterion with a conservative interface-residual criterion using the same bulk
fluxes. Document tolerance units and semantics. Test equivalent length-unit
representations and per-step as well as cumulative inventory errors. R06's
fine-grid iteration floor has a different cause (round-off in the interface flux; see R06).

**Resolution (package, 2026-11-11; binary references approved 2026-11-11).**
Decisions are the user's (R08_HANDOFF §2-§4, all recommended options approved).
- *Stopping rule.* A trial position `s*` is accepted (never on pass zero) only when
  `|s* - s_prev| <= tolerance * R` and `|r| <= residual_tolerance * scale`.
  `tolerance` is dimensionless (default 1e-12; it was an absolute length, default
  1e-8, so an old value `t` is `t / R`). `residual_tolerance` is new (default 1e-13).
  `r = lhs * (s* - s) - rhs` is the planar conservative balance formed from the bulk
  solves at `s*` (deviations and interface-face D), on the motion branch of `s* - s`,
  through the same `_interface_balance_terms` as the update (which is bit-identical).
  It equals the step's transformed-inventory change to within a few eps * R * max|c|.
- *Normalization.* `scale = max(|I_old|, R * c_floor)`, fixed per step from the old
  state; `c_floor` is the new `interface_residual_composition_scale` or else
  `constraints.minComposition`. Per step `|dI| <= residual_tolerance * scale` up to
  round-off: relative while `|I_old| >= R * c_floor`, otherwise an absolute
  mean-composition bound `residual_tolerance * c_floor`. The floor can bind for valid
  data (interface compositions are not clamped). Only when `I_old` and `c_floor` are
  both exactly 0 is the scale `64 eps R` (an absolute round-off bound); every other
  non-finite or non-positive input or scale raises `ValueError`.
- *Picard.* `bulk_picard_rtol=None` is now 1e-10 in implicit mode, independent of
  `tolerance`. C03 and LEGACY implicit keep 1e-8 and C15 keeps 1e-10, explicitly.
- *Validation and diagnostics.* `tolerance`, `residual_tolerance` and an explicit
  composition scale must be positive and finite (checked before the host
  constructor). New `_lastImplicitResidual`, `_lastImplicitResidualScale`;
  `_lastImplicitError` stays `|delta s|`.
- *Call sites.* Test and acceptance builders keep their intent as `old / R` (brass
  1e-10 um -> 1e-10/565, required for its parity test); the three binary examples use
  the new defaults. Exception by user decision (2026-11-11, superseding "keep 1e-8"):
  C04, C31 and F04 (R = 1) run at the new default 1e-12 (`VALIDATION_PARAMS["tolerance"]`,
  a module-level `api_migration`), so examples E01/E02, which use the defaults,
  reproduce C04/F04 bit for bit with equal settings. Generated definitions and ported tests are migrated by a recorded
  `api_migration` transform (`tools/api_migrations.py`) that both generators apply
  and record; `--check` reproduces every file from oracle `7af24f4` plus those edits,
  and the materializer's AST proof applies the same edits to the recorded AST.
- *Tests.* `tests/test_illingworth_binary_tolerance.py` (51): length-unit and dilution
  invariance, residual identity and both checks in all three modes, floor and
  safeguard cases, Picard default, validation, pass zero, fixed-point
  nonconvergence. 45 failed before the change; 6 controls passed.
- *Not changed:* ternary and three-phase numerics (their R-scaled residual default
  is R13), R07's outer-timestep contract, `_nearFinalNoop`, the retry accounting.
- *Example settings.* The example settings comparison now skips the named run-bookkeeping
  attributes in `tools/example_settings.py` `BOOKKEEPING` (R07 commit-guard state, copies of
  the `solve()` arguments and the limits derived from them, last-step diagnostics); adopted
  settings files and hashes are unchanged and every other attribute is still compared.
- *Acceptance.* Binary arrays change by design. Exact acceptance before approval: every
  ternary, three-phase and unchanged binary run bit-identical (C01 and LEGACY reach an
  exact floating-point fixed point); C02-C04, C13-C15, C30, C31, F04, O07 and contract C29
  differed. O06 failed twice at Thermo-Calc licence startup (`take_l` Error[28]); the cause
  was not established, and it then passed repeatedly against live TC.
- *Approval* `20261111T222849Z-5aedde7-1` (user-authorized; `approve --tier full` in a clean
  clone of `5aedde7`, fresh targets; commit `34c02d2`): C02, C03, C04, C13, C14, C15, C30,
  C31, F04, O07 and C29. Gates: 40 run cases (live O06) each repeated bit-identically,
  `approval_criteria` for every case, whole suite on the installed wheel 1163 passed, 34
  unlisted cases and contracts within public tolerance, nothing skipped; settings captured
  for C04/F04 (E01/E02). `reference/` is unchanged. After the approval, from the committed
  revision: exact full 40/40 runs and 5 contracts bit-identical against the selected
  references (live O06, 34 repeat pairs, 515 s); examples E01-E10 pass (E01/E02 against the
  approved C04/F04; E11 still a pending fixture); full `tests/` 1158 passed, 5 skipped,
  1 deselected.
  Analytic errors (exact acceptance, 2026-11-11; frozen reference -> approved R08 values at
  tolerance 1e-12, unchanged limits): interface C04 0.001701657985 -> 0.001701657998
  (relative change 7.6e-9, 10th significant digit; limit 0.0035), C31 0.0004145331289
  -> 0.0004145330724 (1.4e-7, 7th digit; limit 0.00083), F04 0.007035469669 ->
  0.007035471002 (1.9e-7, 7th digit; limit 0.015); profile C04 0.03803348829 ->
  0.03803348827 (6e-10; limit 0.077), C31 0.01521744308 -> 0.01521744306 (1.5e-9; limit
  0.031), F04 0.002771259736 -> 0.002771260644 (3.3e-7, 7th digit; limit 0.0056).
  Inventory drift: C04 4.5e-12 -> 3.7e-16 (limit 4.6e-11), C31 2.4e-11 -> 1.2e-14
  (2.5e-10), F04 5.1e-10 -> 1.3e-13 (5.2e-9). With 1e-8 kept these differed from the
  1e-12 values by at most 3e-12 (F04 interface error).

### R09: sparse recording mixes live state with recorded snapshots

**Status:** resolved 2026-11-13. R09a (commit `3047510`) with C25/C29 approved as
`20261113T174227Z-3047510-1` (commit `83b5887`); R09c (tooling `d56f25e`, fix `24a2d39`) with its
26 runs and 3 contracts approved as `20261113T194742Z-1993bb9-1` (commit `6fd3be8`). Resolutions
below. **Scope:** binary history classes, shared
`_ScalarHistory`/`_ArrayHistory`, and solver getters that delegate to them.

With recording every third step, the binary `getInterfacePosition()` returned
1.0 during accepted steps whose live positions were 1.0000498908 and
1.0000996877. The no-time history lookup returns the last stored row rather than
`currentY`. Related profile/eta getters need the same current-state audit; the
three-phase transformed-state getter already accesses live profiles directly.

Finalization has a separate confirmed defect: a one-step binary run with recording
interval 3 left interface and transformed-profile time arrays equal to `[0.1]`,
overwriting the initial record. Forced recording computes its destination from
`int(currentIndex / recordInterval)` instead of independently appending a snapshot.

**Investigate:** distinguish live state from explicitly requested historical
snapshots, preserve the initial record, and append forced final records without
corrupting the sampling cadence. Cover intervals greater than one, runs shorter
than the interval, continued solves, disabled recording and disabled profile
history. Check synchronization of interface, eta, profile and inventory histories.

**Resolution, stage R09a (2026-11-13, commit `3047510`; decisions D2-D4 and the flag-based
dedup are the user's).**
- *Row rule.* One helper set in `compatibility/recording.py` (`_history_row_target`,
  `_advance_history_rows`, `_finalize_history_rows`) is used by the binary
  `_ScalarHistory`/`_VectorHistory`, the shared `_ScalarHistory`/`_ArrayHistory`, the compatibility
  `MeshData` and three-phase `_history_record_target`. Rows are append-only: row 0 is never
  overwritten; a regular row is stored every `k` accepted steps, counted from `reset` across
  continued solves (`currentIndex` is the cadence counter); a forced final row is appended and uses
  no cadence slot. `N` still means "last stored row", so `toDict` index fields keep their meaning.
  The new bookkeeping (`_rowCount`, `_lastRecordStored`) lives on each history, not on the model.
- *Final dedup.* `finalize` appends the current state only if the most recent `record` call stored
  no row. No time tolerance is used, so a distinct snapshot 5e-15 after the last stored row is kept
  (the 1e-14 rule dropped it). Repeated `postSolve` calls add nothing, except the legacy duplicate
  below. Three-phase `postSolve` uses the same rule for every history, `data` included, and its
  atomic commit snapshot/restore also covers the two new fields.
- *Getters.* With no time, every getter returns the current accepted state at any interval,
  including disabled recording (`y(None)` returns `currentY`). A timed request keeps the historical
  rules: scalars interpolate and clamp, arrays need an exactly recorded time (atol 1e-14) or raise,
  `MeshData` interpolates unless interpolation is disabled. Asking for the current, unstored time
  returns the last stored row (D3). Binary/two-phase `getTransformedState()` with no time returns
  the accepted `p`/`q` without p/q histories (D4); a timed call still raises.
- *Examples at interval 3.* 1 step: `[0, 1]`; 4 steps: `[0, 3, 4]`; 2 + 2 steps: `[0, 2, 3, 4]`
  (all times x dt, identical in every history); `solve(0)` adds nothing.
- *Kept for R09c (deliberately).* At interval 1 the compatibility `MeshData.finalize` still always
  appends a row, so binary/two-phase `data` ends `[0, 1, 2, 2]` while the other histories end
  `[0, 1, 2]`, and each continued solve adds another duplicate. Binary `data` row 0 still holds the
  profile before the setup reconstruction (documented exception). Recording-disabled histories keep
  one row.
- *Reference effect.* Every run case and the interval-1 layout are bit-identical. Contracts C25
  (model0, model4) and C29 (model1) changed in `profiles/compositions` only: binary models at t = 0
  with no committed step, captured with `getCompositions()`. That now returns the reconstructed
  initial profile the solver starts from (`data.currentY`) instead of stored row 0 (max change
  0.30 and 0.34 in mole fraction); `getCompositions(0.0)` still returns row 0. The user chose to
  adopt this (no setup-only exception).
- *Tests.* `tests/test_illingworth_history_contract.py` (48). 26 defect tests failed before the
  change for the intended reason: overwritten or missing rows (1, 4 and 2 + 2 steps at interval 3,
  all families), stale no-time getters (all families), an extra `data` row from `solve(0)` or a
  repeated `postSolve` at interval 3, `ValueError` without p/q histories, the dropped close-time
  snapshot (four package history classes and binary/ternary direct commits), and the binary t = 0
  live profile. 22 passed before and after: interval-1 layout including the duplicate, repeated
  `postSolve` on package histories, disabled recording, on-cadence final row, final state equal
  for `record=1, 3, False`, explicit-time rules, D3, three-phase `solve(0)` rejection, three-phase
  failed-commit restoration of every history field, and the `MeshData` close-time case (its host
  finalize never deduplicated).
- *Approval* `20261113T174227Z-3047510-1` (user-authorized; `approve --tier full` in a clean clone
  of `3047510` with fresh targets and the local inputs copied hash-verified; commit `83b5887`):
  exactly C25 and C29. Gates: 40 run cases (live O06) each repeated bit-identically,
  `approval_criteria` for every case, whole suite on the installed wheel 1337 passed with nothing
  skipped, every unlisted case and contract within public tolerance, 1006 s. Beyond that, an exact
  full run of `3047510` in the same clone gave 40/40 runs and C26-C28 bit-identical, with C25/C29
  differing in exactly the three arrays above. `reference/` and the R08 approval artifacts are
  unchanged; the new artifacts were copied back and match the manifest hashes.
- *Gates after the approval* (from `83b5887`, fresh targets, 2026-11-13, not `--record`ed): exact
  full 40/40 runs and 5 contracts bit-identical against the selected references, live O06, 34
  repeat pairs with no differences, 542 s; examples E01-E10 pass with 68 tests (E11 a pending
  fixture), 157 s; full `tests/` 1332 passed, 5 skipped (C25-C29 comparisons outside an acceptance
  run), 1 deselected (`slow`), 298 s.

**Resolution, stage R09c (2026-11-13; implemented, reference adoption pending; order changed by the
user to R09c before R10).**
- *Tooling (R09c-1, commit `d56f25e`).* `tools/acceptance_layout.py` classifies a candidate against
  the case's previously selected reference as `identical` (key set, dtype, shape, bytes),
  `tail-removal` or `other`. `tail-removal` requires the same key set, every array outside the
  declared pairs identical, and each changed pair (`data/time`+`data/values`,
  `model<i>/data/time`+`/values`, LEGACY `<mode>__data_time`+`__data_y`) to have lost exactly its
  last row, a bitwise duplicate of the row before it (`candidate == reference[:-1]`). Only O06's
  existing metadata-path normalization applies; a change within public tolerance is `other`.
  - `quick|full --layout-correction r09-data-tail` and `acceptance_examples.py run
    --layout-correction r09-data-tail` are non-promoting checks: every normal gate runs, normal
    comparisons are recorded and reported, any `other` fails, and corrected mapped cases need
    case-versus-case settings equality.
  - `approve --layout-correction r09-data-tail --cases ...` keeps every existing gate and adds:
    listed cases and contracts must be `tail-removal`, unlisted ones `identical` (so the listed set
    equals the differing set). For mapped cases the selected settings are hash-validated first,
    fresh `--settings-only` settings are captured, and
    `example_settings.differences(selected, captured, allowed=())` must be empty (BOOKKEEPING only;
    no example-versus-case exceptions; the 6 R07 bookkeeping fields missing from older records are
    thereby tolerated). Public cases store the full capture with its committed hash as before.
    O04/O05 only, and only in this mode, get the `private-carried-forward` marker: the adopted file
    and its private hash validation stay, no settings file or hash is published, and temporary
    captures are deleted before promotion. The committed report carries structural verdicts
    (key names) instead of numerical deviations.
  - Tests: `tests/test_acceptance_layout.py` (58): every structural rejection, O06 normalization,
    the listed/differing-set rule, failed criteria, repeat mismatch and package-test/contract
    failures blocking promotion even when the layout passes, settings (changed genuine settings,
    unrecorded attribute, older records, hash and private-hash failures, no example exceptions),
    marker restrictions and validation, crash-safe promotion and rollback with a carried-forward
    case, example checks and CLI refusals. Normal gates unchanged at `d56f25e` (exact full 40/40
    runs and 5 contracts bit-identical, live O06, 34 repeat pairs, 553 s; examples E01-E10 pass;
    full `tests/` 1390 passed, 5 skipped, 1 deselected).
- *Fix (R09c-2, commit `24a2d39`).* The compatibility `MeshData.finalize` uses the shared flag rule
  at every interval, so binary/two-phase `data` holds each accepted time once, is synchronized with
  the other histories, and repeated `postSolve` adds nothing. Tests: the R09a test that preserved
  the duplicate is replaced; 5 tests (interval-1 synchronization and repeated `postSolve` for
  binary/ternary, `MeshData` interval 1) failed against `d56f25e` for the duplicate row, and all 49
  history-contract tests pass after.
- *Affected set, confirmed against the selected references* (frozen, R08 approval, and the R09a
  approval for C25/C29): every selected reference with a duplicate data tail has exactly one, with
  no interior repeats. Runs (26): C01-C08, C13-C19, C24, C30, C31, F04, LEGACY (all three modes),
  O01, O03a, O04, O05, O06, O07. Contracts (3): C25 (model1-3), C26 (model1-3), C29 (model0).
  Examples: E01-E06 through C04, F04, C16, C24, O05, O04.
- *Gates at `24a2d39`* (fresh targets, 2026-11-13, nothing `--record`ed, no approval):
  - full `tests/` 1391 passed, 5 skipped, 1 deselected;
  - `full --tolerance exact --layout-correction r09-data-tail`: 29 `tail-removal` (exactly the set
    above), 16 `identical` (all three-phase cases and contracts, O02, O03b, O08), 0 `other`; live
    O06; 34 repeat pairs with no differences; contract assertion groups and the gate's package tests
    passed; settings equal case versus case for C04, C16, C24, F04, O04, O05; 628 s;
  - example layout check: E01-E06 `tail-removal`, E07-E10 `identical`, 68 tests passed;
  - normal gates fail by design until the approval: exact full has 14 runs bit-identical and fails
    exactly the 26 runs (only their `accepted_times`/`profiles` groups, i.e. `data/time` and
    `data/values`) and contracts C25, C26, C29; normal examples fail E01-E06 on `data/time` and
    `data/values` only, E07-E10 pass.
- *Approval* `20261113T194742Z-1993bb9-1` (user-authorized; commit `6fd3be8`): `approve --tier
  full --layout-correction r09-data-tail` in a clean clone of `1993bb9` with fresh targets and the
  local inputs copied hash-verified, for exactly the 29 cases above (reason "LEDGERS R09c: remove
  the interval-1 duplicate final data row (exact r09-data-tail layout correction)"), 952 s.
  - Gates: 40 run cases (live O06) each repeated bit-identically; `approval_criteria` for every
    case; whole suite on the installed wheel 1396 passed, nothing skipped; all 29 listed cases
    `tail-removal` against their previously selected references; all 16 unlisted cases and
    contracts bit-identical (and within public tolerance).
  - Settings: equal case versus case. C04, F04, C16 and C24 captures are stored with the approval
    under their committed hashes. O04/O05 carry `private-carried-forward`: no settings file in the
    approval folder, and none of the local private hashes appears in the manifest or report. Example
    acceptance still checks their unchanged adopted `example_settings/O0x.settings.json` against
    the private digests.
  - Artifacts: the 62 approved files were copied back and match the manifest hashes, as does the
    committed report. `reference/`, the R08 and R09a approval folders, `example_settings/` and
    `private-inputs.json` are byte-identical to before the approval (218 files).
- *Gates after the approval* (from `6fd3be8`, fresh targets, 2026-11-13, no layout flag, not
  `--record`ed): exact full 40/40 runs and 5 contracts bit-identical against the selected
  references, live O06, 34 repeat pairs with no differences, 557 s; examples E01-E10 pass with 68
  tests (E11 a pending fixture), 199 s; full `tests/` 1391 passed, 5 skipped (C25-C29 comparisons
  outside an acceptance run), 1 deselected (`slow`), 288 s.

### R10: failed initialization or output preparation can leave partial state

**Status:** fixed 2026-11-13 (R10a `c1b2aae`, R10b `ded279d`; resolution below; no reference
changed). **Scope:** all three setup methods; binary and
two-phase ternary accepted-state preparation/commit paths.

Each solver calls host `setup()` before completing initialization; the host sets
`isSetup=True`. A diagnostic ternary diffusivity-provider failure left
`isSetup=True` and `_p_curr=None`. After restoring the provider, a second setup
returned immediately without completing initialization.

Binary/two-phase `postProcess` also calls `GenericModel.postProcess` before all
output preparation succeeds. Injecting a binary reconstruction failure advanced
`currentTime` from 0 to 0.1 while its accepted interface remained at the old value.
Three-phase already separates preparation and atomic commit.

**Investigate:** mark setup complete only after successful initialization, or
restore a coherent uninitialized state on failure. Prepare/validate outputs before
advancing accepted time/state and histories; audit trial-only diagnostics such as
`_lastAcceptedDt` too. Test recovery after provider, validation, recording and
coupled-model failures without silently advancing accepted state.

**Resolution (2026-11-13; user-reviewed decisions D5, D-R1, D6, D7; handoff
`docs/R10_HANDOFF.md`).** Tests: `tests/test_illingworth_lifecycle_transactions.py`.
- *R10a (commit `c1b2aae`): setup and reset are all-or-nothing in every family.* A shared wrapper
  (`shared._all_or_nothing`) snapshots every attribute binding of the model plus the contents of each
  reachable history, and on any exception from `setup()`/`reset()` restores them and re-raises: time,
  `isSetup`, accepted state, every history (rows, capacity, cadence and row counters, stored-row
  flag), candidate, latch and diagnostics. The method bodies are unchanged. A failed setup leaves
  `isSetup=False` with the exact pre-setup state (D5); a reset failing in the host part or provider
  `clearCache`, the interstitial u-fraction re-recording, or the configuration validation at its end
  keeps the previous simulation (D-R1). An already set-up model's `setup` takes no snapshot.
  - Defect tests (31) failed at `4bf4ba9` for the intended reasons: a failed reset rewound
    `currentTime` to 0 (clearCache) or cleared state, histories and time (validation); a failed
    setup left `isSetup=True` (provider, invalid setting on a direct `setup()`, tie-line closure).
    Their recovery parts check: a continued solve equals an untouched continuation; after a failed
    setup, setup/solve equal an untouched model.
  - Recovery guarantee (b) restated (user decision 2026-11-13): a repeated reset + solve equals an
    untouched model's successful reset + solve; for binary and two-phase ternary it also equals a
    freshly constructed model. Three-phase reset + solve differs from a fresh model in the last bits,
    before and after R10: the host `reset` records the raw `mesh.y` as row 0, while the constructor
    stores its u-fraction conversion (11 of 31 nodes differ by up to 2.8e-17 in the test fixture).
    This pre-existing limitation is outside R10 and pinned by a preservation test. (Superseded:
    resolved by the reset-parity follow-up below, user-authorized 2026-11-13; (b) now holds against a
    fresh model for every family.)
  - Gates before the commit (fresh target): focused tests 275 passed; full `tests/` 1428 passed,
    5 skipped, 1 deselected.
- *R10b (commit `ded279d`): binary/two-phase prepare-then-commit and atomic finalization.*
  - `postProcess` verifies the update (R07 unchanged), consumes the candidate and sets the latch,
    prepares every output without touching model state (interface clip and check, interface-node
    overrides, ternary composition check, profile reconstruction, inventory), then commits history
    records, accepted state, `currentTime`, cumulative projection totals, ternary `_lastAcceptedDt`
    and `updateCoupledModels` as one unit, and clears the latch. On a failure all of these are
    unchanged; the candidate (with its staged projection totals and terminal thin-phase stop) stays
    consumed and the latch set, so a resubmission, or the originally correct update, is refused and
    a new `getdXdt` reproduces an untouched model's step bit for bit.
  - Coupled models (D6): a failure in `updateCoupledModels` rolls back this model only. Coupled
    models updated before the failing one keep their update; recovery equals an untouched run only if
    the caller restores or discards that state (tested with a stateful recorder: with the entry
    discarded the pair equals an untouched pair, otherwise the recorder holds a duplicate).
  - D7: ternary `_lastAcceptedDt` is accepted state, set to the committed candidate's dt in
    `postProcess` (no longer by `getdXdt`), so a rejected retry-reduced proposal changes neither it
    nor the next extrapolated Newton start. `_currdt`, `_lastStepRetries`, `_lastImplicit*`,
    interface D values, `_lastInterfaceCompositions`, `_lastStepProjection*` and
    `_cumulativeProjectedSolves` remain documented last-trial diagnostics.
  - `postSolve` (all families) finalizes every history as one unit with targeted per-history
    snapshots (original `_y`/`_time` arrays, the forced-target row, `N`, counters, flag,
    `currentY`/`currentTime`); a failure restores every history and repeating `postSolve` is the
    retry. A continued solve without that retry records as if the first solve was never finalized.
  - The three-phase atomic history helpers moved to `shared.py` with unchanged behavior (value shape
    check extended to the binary p/q histories) and are used by all families; three-phase commit
    logic is unchanged.
  - Defect tests (20) failed at `c1b2aae` for the intended reasons (time advanced by a failed
    reconstruction or composition check; histories recorded before a failing record or coupled
    model; ternary `_lastAcceptedDt` left at the rejected 0.005 instead of the committed 0.01; a
    failed `postSolve` changed the histories finalized before it, at intervals 3 and 1).
  - Gates before the commit (fresh targets, 2026-11-13, nothing `--record`ed, no approval):
    focused tests 399 passed (with the R07, R09 and three-phase suites); full `tests/` 1448 passed,
    5 skipped, 1 deselected; exact full 40/40 runs and 5 contracts bit-identical against the
    selected references, live O06, 34 repeat pairs with no differences, 579 s; examples E01-E10
    pass with 68 tests (E11 a pending fixture), 166 s.
- *Not restored (documented):* provider state and caches (a partially cleared cache), the ternary
  `ThermodynamicTernaryInterfaceEquilibrium.thermodynamics`/`.phases` binding, a created
  `mesh.boundaryConditions`, temperature parameters, coupled models, printed diagnostics. Unchanged:
  the near-final no-op step, a direct `postProcess` without a proposal (commits unchecked),
  three-phase `solve(0)` rejection and the binary/two-phase `solve(0)` setup-only no-op.
- *Follow-up (2026-11-13, from review): three-phase interruptions.* Three-phase
  `_commit_accepted_state` and `postProcess` caught only `Exception`, so a `KeyboardInterrupt` from
  a coupled-model update advanced accepted state, time and every history and kept a staged terminal
  thin-phase stop (reproduced: 3 new three-phase tests failed at `7e81f53`; binary/two-phase already
  rolled back). Both now handle `BaseException` like binary/two-phase: this model's state and
  histories are restored, the candidate is consumed, the terminal stop cleared and the interruption
  re-raised unchanged; external coupled updates are still not undone (D6). Tests (11, all families):
  interruption rollback with resubmission refused, recovery through a fresh candidate equal to an
  untouched model, terminal-stop cleanup, and preservation of successful commits with a coupled
  model. Gates (fresh targets, no approval): focused lifecycle/R07/R09/three-phase suites 519
  passed; full `tests/` 1459 passed, 5 skipped, 1 deselected; exact full 40/40 runs and 5 contracts
  bit-identical, live O06, 34 repeat pairs, 630 s; examples E01-E10 pass with 68 tests.
- *Follow-up (2026-11-13, user-authorized): three-phase reset parity (resolved: fix `9595574`, C27
  approved as `20261113T232720Z-9595574-1`, commit `ff01469`).* A three-phase model that has run,
  then successfully resets and solves, now equals a freshly constructed model bit for bit.
  - Cause: the constructor runs `reset()` and then `_validateMeshComposition()` (clamps near-bound
    compositions in `mesh.y`, whose flattened form is a view, then converts x to u). Three-phase
    `reset` re-recorded the raw `mesh.y` (substitutional) or the interstitial-only
    `record_initial_u_fractions` conversion, which differ from the constructor's in the last bits
    (reproduced: 11/31 nodes up to 2.8e-17 substitutional, 21/31 up to 5.6e-17 interstitial; every
    accepted field and history then differed after a solve).
  - Fix: the conversion and recording step of the compatibility `_validateMeshComposition` is
    extracted unchanged as `_recordInitialMeshComposition`; three-phase `reset` applies it to
    `mesh.y` (already clamped, not modified, converted once). The constructor, binary and two-phase
    ternary are unchanged (they still use `record_initial_u_fractions` after reset).
  - Tests: the preservation test that pinned the discrepancy is replaced by parity regression tests
    (both fixtures: initial profile, histories, time and `isSetup` after one and two resets, setup and
    solve parity, a failed re-recording that keeps the run and recovers); 8 failed before the fix.
    Focused suites 449 passed; full `tests/` 1465 passed, 5 skipped, 1 deselected.
  - Acceptance: 40/40 runs, contracts C25, C26, C28, C29, 34 repeat pairs, live O06 and E01-E10
    bit-identical. Contract C27 differs only in `model4` (the contract's run, `reset()`, `setup()`
    three-phase model), in 9 arrays at round-off level (`data`, `etaData`, `inventoryData`,
    `profileData0`/`1` values, `profiles/compositions`, `profiles/phase0`/`1/x`,
    `state/_initialInventory`; max abs 1.9e-16, max rel 1.3e-15): the frozen reference recorded the
    old reset. The candidate `model4` equals the contract's fresh `model0` in all 34 arrays.
  - Approval `20261113T232720Z-9595574-1` (user-authorized; commit `ff01469`): `approve --tier full
    --cases C27` in a clean clone of `9595574` with the local inputs copied hash-verified (316
    files), 870 s. Before it, a non-promoting exact full run in the clone confirmed exactly the 9
    `model4` arrays above, `model4` equal to `model0` in all 34 arrays, C25, C26, C28, C29 and all 40
    runs bit-identical. Gates: 40 repeat pairs bit-identical, `approval_criteria`, whole suite 1470
    passed, 44 unlisted cases and contracts passed. The promoted `C27.npz` is byte-identical to the
    checked candidate; only C27 changed selection (from `reference`); no private hash is in the
    manifest or report. The approved artifacts were copied back and match the manifest; the 314
    pre-existing reference, approval, input, settings and pilot files are unchanged.
  - Gates after the approval (from `ff01469`, fresh targets, no special flags, not `--record`ed):
    exact full 40/40 runs and 5 contracts bit-identical against the selected references, live O06,
    34 repeat pairs with no differences, 550 s; examples E01-E10 pass with 68 tests (E11 a pending
    fixture), 157 s.
- *Follow-up (2026-11-13, user-authorized): binary and two-phase ternary reset parity.* Goal: with
  identical configuration and deterministic inputs, a successful reset followed by setup/solve equals
  a freshly constructed model bit for bit, including the initial profile and every history. The two
  families were checked separately with fixtures whose constructor conversion is not exact (scratch
  diagnostics, not committed).
  - Binary: no defect reproduced; unchanged. With one independent component the constructor's
    conversion and the reset record give identical bits: without interstitials the divisor
    `fl(fl(1 - x) + x)` rounds to exactly 1 (the rounding error of `1 - x` is at most 2^-54), and with
    an interstitial both divide `x` by the same `1 - x`. Evidence: 0 of 400000 random profiles differ
    in each case; the shared fixture, a 0.1/0.3 profile and two Fe-C interstitial profiles match a
    fresh model after reset, setup and solve. Binary reset still uses `record_initial_u_fractions`.
  - Two-phase ternary, defect 1 (initial profile): reset recorded the raw `mesh.y` (no interstitials)
    or `record_initial_u_fractions` (interstitials, divisor `1 - x_C`), while the constructor records
    `x_to_u_frac(expand_x_frac(x))`, whose divisor `((1 - (x1 + x2)) + x1) + x2` (or `... + x_Mn`) can
    round away from 1 (or from `1 - x_C`). Reproduced: 4% of random substitutional and 26% of random
    Fe-C-Mn profiles differ; a substitutional fixture (left 0.176/0.076) differs in 18 of 42 values
    (up to 2.8e-17) and an Fe-C-Mn fixture in 400 of 800 values (up to 1.7e-18); setup then starts
    from a different `c0`, and every accepted field and history differs after a solve. The existing
    shared and interstitial fixtures convert exactly, which is why the earlier preservation test
    passed. Fix: reset applies the constructor's `_recordInitialMeshComposition` to `mesh.y`, as
    three-phase does (`mesh.y` is already clamped, is not modified and is converted once; the
    constructor and the solver equations are unchanged).
  - Two-phase ternary, defect 2 (stale eta): `setup` stores its eta estimate in `initialEta`, and
    reset recorded that estimate as `etaData` row 0, while a fresh model has the unestimated NaN; so
    between reset and setup `initialEta`, `etaData` and `getInterfaceEta()` returned the previous
    run's eta (all four ternary fixtures). The next `setup` resets and re-records `etaData`, so
    setup/solve results were not affected. Fix: reset restores `initialEta = NaN` before recording
    (user decision 2026-11-13); configured initialization inputs (method, bracket, guesses, root
    controls) are unchanged.
  - R10a guarantees hold for both steps (reset stays wrapped): a failure while re-recording the
    profile, or later in reset after `initialEta` was restored, keeps the previous run (including
    `initialEta` and `etaData`); continuation equals an untouched continuation and a successful retry
    equals the untouched reset and a fresh model.
  - Tests: the three-phase parity tests are generalized to all families (8 fixtures: binary
    substitutional and Fe-C, ternary shared, inexact substitutional, Fe-C-Mn and inexact Fe-C-Mn,
    three-phase substitutional and interstitial): fixture exactness guard, initial profile and `data`
    after one and two resets with `mesh.y` unchanged, the full state after reset, setup and solve
    parity after one and two resets, recovery from a failed re-recording; plus ternary eta tests
    (NaN state after one and two resets, unchanged inputs, setup/solve parity, rollback of a later
    reset failure). 17 failed at `af52be7`, each for its stated cause (initial-profile bytes, stale
    eta, or "did not raise" because substitutional reset did not re-record); binary and three-phase
    cases passed (preservation). Focused suites (lifecycle, history, outer time-step limits, binary, ternary and three-phase interstitials, setup and diagnostics, ternary, ternary physical scale) 532 passed; full `tests/` 1506 passed, 5 skipped, 1 deselected.
  - Acceptance: no reference changed. Exact full 40/40 runs and 5 contracts (C25, C26 included) bit-identical, live O06, 34 repeat pairs with no differences, 528 s; examples E01-E10 pass with 68 tests (E11 a pending fixture), 159 s. No approval was made.

### R11: inherited deserialization does not restore the solver state

**Status:** fixed 2026-11-12 by rejecting restart (resolution below); full restart and a
historical-output loader are not implemented. **Scope:** binary and two-phase ternary
`fromDict` inherited from the host; ternary `toDict` documentation.

Host deserialization restores physical snapshots and time without restoring the
transformed profiles, interface, eta, previous-step state or reference inventory.
A binary dictionary saved at t=0.1, s=1.0000498908 loaded without error into a
freshly set-up model at t=0.1, but its internal/reported interface stayed at 1.0
and its transformed profile remained initial. Two-phase ternary has the same
inherited loader; its `toDict` docstring incorrectly calls the output a restart
dictionary. Three-phase already rejects incomplete restart loading explicitly.

**Investigate:** reject unsupported restart clearly in binary/two-phase, and
correct the output-dictionary documentation. Full restart can be separate work;
if implemented, verify complete state restoration and continued-solve equivalence.
Historical output loading should have an explicit contract distinct from restart.

**Resolution (2026-11-12, commit `6296bd9`).**

- *Contract.* Binary and two-phase ternary `fromDict` are class methods that raise
  `NotImplementedError` ("... cannot be restarted ...") before changing any state, as
  three-phase already did. The inherited `load` calls `fromDict`, so it raises too; its API
  is unchanged. `toDict`/`save` output is unchanged (no marker key added). Saved output is
  read with `numpy.load`. The ternary `toDict` docstring now calls the output an output
  dictionary, and both class docstrings list the restriction.
- *Compatibility.* `model.load(file)`/`fromDict(data)` on these two families now raise instead
  of partially loading. No package, example, tool or ported-test caller existed; the research
  `save`/`load` tests exercise other `MovingBoundaryFDM` models.
- *Tests.* `tests/test_illingworth_restart_contract.py`. 14 defect tests failed before the fix:
  12 with `DID NOT RAISE NotImplementedError` (instance `fromDict` into a fresh, set-up and the
  source model; instance `load` into fresh and set-up models; continued solve after a refused
  load), and the 2 class-level `fromDict` calls with `TypeError` (the host loader is an
  instance method). 4 preservation tests (`toDict` keys and history identity, `save`
  contents) passed before and after. The no-mutation checks compare time, setup flag,
  accepted state and every history bytewise.
- *Gates* (fresh installed targets, 2026-11-12): focused 18 new + the 2 three-phase
  rejection tests passed; full `tests/` 1176 passed, 5 skipped (C25-C29 comparisons outside
  an acceptance run), 1 deselected (`slow`), 234 s; `acceptance.py quick --tolerance exact`
  30/30 runs and 5 contracts bit-identical against the selected references, 33 cases
  repeated with no differences, no skips, 414 s (not `--record`ed). No reference approval.

### R12: ternary numerical controls lack consistent validation

**Status:** fixed 2026-11-12 (resolution below). **Scope:** both ternary configuration
validators; compare binary's existing checks and three-phase solve-argument checks.

A diagnostic two-phase constructor accepted `residual_tolerance=inf`; its
configuration validator also accepted `tolerance=inf`. Static inspection found no
positive-finite checks for the ternary semi-log spacing/initial-target parameters,
beyond checking their presence in semi-log mode. A negative log spacing can leave
no interior scheduled targets before the final target is appended.

**Investigate:** require positive finite convergence tolerances and active semi-log
parameters. Validate solve duration and min/max timestep fractions consistently
across families. Add active-mode checks for zero, negative, NaN and infinity, and
ensure invalid inputs fail before setup/state mutation or schedule allocation.
The active semi-log failure variants still need dedicated reproduction tests.

**Resolution (2026-11-12, commit `cdb6c68`).** A reproduction on the installed `f0c3201` wheel
found the two-phase constructor accepting `tolerance` and `residual_tolerance` of 0, -1e-12,
NaN and inf, and semi-log `semiLog_dt`/`semiLogT0` of 0, negative, NaN and inf (a negative
spacing left only the final target; `semiLogT0<=0` was silently clamped to 1e-15). Binary and
two-phase `solve` accepted negative and NaN durations after running setup; an infinite one fails
after setup (binary) or never terminates in fixed mode (two-phase ternary, by inspection).

| Control | Active when | Rule now | Where checked |
|---|---|---|---|
| `time_step` | always | > 0, finite (unchanged) | ternary families: shared controls check |
| `tolerance` | residual fallback; implicit Picard rtol fallback | > 0, finite, **always** (invariant of the base tolerance) | shared controls check |
| effective `residualTolerance` | always | > 0, finite | shared controls check |
| `semiLog_dt`, `semiLogT0` | `dt_mode='semi_log'` | present, > 0, finite; ignored in fixed mode | shared controls check |
| `simTime` | every `solve` | binary/two-phase: finite, >= 0 (0 = setup-only no-op); three-phase: finite, > 0 (unchanged) | top of `solve`, before setup |
| `minDtFrac`/`maxDtFrac` | every `solve` | unchanged (already rejected before setup in all families) | top of `solve` |

- *Timing.* `shared._validate_ternary_numerical_controls` only reads attributes. Both ternary
  constructors call it before the host constructor; the configuration validators call it (it
  replaces their `time_step`, `dt_mode` and semi-log presence checks, same messages); and
  two-phase/three-phase `solve` call it after the duration and fraction checks, before
  `_solveMinDtFrac`/`_solveMaxDtFrac`, `setup`, `setTimeInfo` or any schedule. A control
  changed on an existing model is therefore refused by `solve` without changing it. General
  failure recovery during setup remains R10.
- *Unchanged.* Defaults, valid-input behavior, timestep semantics, final-remainder handling,
  residual normalization (R13), the Picard zero-tolerance conventions, binary's validation
  (including its fixed-mode semi-log checks) and the 1e-15 clamp (now reachable only for
  positive `semiLogT0` below 1e-15).
- *Tests.* `tests/test_illingworth_numerical_controls.py`. 84 defect tests failed before the
  fix for the intended reason: 32 constructor cases (both ternary families, each control at 0,
  negative, NaN, inf) reached the host-constructor sentinel; 40 changed-control cases (stored
  `residualTolerance`, `tolerance`, `dtMode`+semi-log values; fresh and set-up models) and 12
  invalid-duration cases (binary/two-phase, fresh and set-up) reached the `setup` sentinel.
  After the fix each raises a `ValueError` naming the control, with time, bookkeeping,
  schedule, accepted state and every history bytewise unchanged; after restoring every changed
  setting (including `dtMode`) a continued solve matches an untouched model bit for bit.
  `setup`/`setTimeInfo` sentinels keep unfixed infinite durations out of the solver loop.
  24 preservation tests passed before and after: valid controls reach the host constructor,
  unused semi-log values accepted in fixed mode, binary/two-phase `solve(0)` takes no step,
  three-phase still rejects zero, invalid dt fractions still rejected before setup, and a
  final partial step still ends at the requested time.
- *Gates* (fresh installed targets, 2026-11-12): focused 224 passed (R12, R11, R07 and binary
  tolerance files); full `tests/` 1284 passed, 5 skipped (C25-C29 comparisons outside an
  acceptance run), 1 deselected (`slow`), 243 s; `acceptance.py full --tolerance exact`
  40/40 runs and 5 contracts bit-identical against the selected references, live O06, 34
  repeat pairs with no differences, 594 s (not `--record`ed); examples E01-E10 pass with 68
  tests (E11 a pending fixture), 160 s. No configuration was newly rejected and no reference
  approval was made.

### R13: ternary and three-phase residual normalization (resolved)

**Status:** resolved 2026-11-15. Implemented 2026-11-14, review follow-ups 2026-11-15; references
adopted as `20261115T181200Z-6caa07b-1` (approval commit `e1bb66d`; see *Adoption* at the end of
this entry). Recorded 2026-11-11 from the R08 survey. The text up to *Adoption* is the
pre-approval record: there, "pending" and "until a separately authorized approval" describe the
state before `e1bb66d`.
**Scope:** `ternary.py:_interface_residual_scale`, `three_phase.py:_residual_scale`, their
validation, and the docstring and README descriptions of `interface_residual_composition_scale`.
Binary numerics, the attempted Newton update and its fallback, clamping and floor projection,
Picard tolerances, convergence diagnostics, history and lifecycle code are unchanged.

**Finding (verified against the code).** Per step and up to floating-point round-off (at most
6e-16 of the inventory in the survey fixtures), the residual is the inventory defect:
two-phase `dI_k = r_k`; three-phase `dI~_k = r_ab,k + r_bc,k`, with
`I~_k = sum_alpha (rho_alpha / rho_A) w_alpha <c_k>_alpha` (physical moles `rho_A * I~_k`), for
equal and unequal molar volumes, a moving right boundary and u-fractions.

- *Earlier default.* Two-phase: `scale_k = max(|I_k|, R, 1e-300)`. This reduces to `R` only where
  the inventory does not exceed the length term and the `1e-300` floor is inactive. That holds
  for mole fractions. It is not universal: an interstitial u-fraction can exceed one, and its
  inventory is then the scale. Three-phase: `L` for all four rows (`L = R0`, or `N_tot / rho_A`
  with unequal volumes), without looking at the inventory.
- *What it bounded.* Where the scale was the length term, the tolerance bounded a change of mean
  composition per step (`tol`; three-phase total `2 * tol`), not a relative solute loss. The
  permitted defect relative to the inventory of a component was `tol / cbar_k` (three-phase
  `2 * tol / cbar_k`), with no upper bound as the mean composition `cbar_k` approaches zero. The
  docstring wording "relative solute loss per step" was wrong; it was corrected first (`cdde760`).
- *Three-phase rows.* A small sum does not bound the rows: opposite-signed `r_ab`, `r_bc`
  conserve the total but misplace solute between the interfaces, so both rows stay in the test.
  The dependent component has no row; its error is minus the sum of the substitutional
  independent ones.

**Change.** Without an explicit `interface_residual_composition_scale`, per component:
`scale_k = max(|I_k|, L * constraints.minComposition)` (`shared._componentwise_residual_scale`,
the binary R08 rule applied per component; binary code unchanged). Three-phase uses `I~_k` and the
same `scale_k` for both interfaces.

- *Enforced per step, up to round-off.* Two-phase `|dI_k| <= tol * scale_k`. Three-phase: each
  row `|r| <= tol * scale_k`, hence `|dI~_k| <= 2 * tol * scale_k`. These are relative to the
  inventory of the component itself while `|I_k| >= L * minComposition`. Where the floor binds
  the bound is absolute: a change of mean composition of at most `tol * minComposition`
  (three-phase total `2 * tol * minComposition`). This is what the test enforces per step; it is
  not a statement about accumulated drift or solution accuracy.
- *Zero and invalid values (default path only).* Inventory and `minComposition` both exactly zero
  give `64 eps L`; a non-finite inventory, a non-positive or non-finite `L`, a negative or
  non-finite `minComposition`, and a scale that overflows or underflows raise `ValueError`. No
  `1e-300` clamp on this path.
- *Explicit scales.* Unchanged bit for bit, including the `1e-300` lower bound:
  `max(max(|I_k|, L * s_k), 1e-300)`. `1.0` requests the earlier default (three-phase: while
  `|I~_k| <= L`).
- *Tolerance defaults.* Unchanged (`tolerance=1e-8`, `residual_tolerance=None`, meaning
  `tolerance`).
- *Validation.* The accepted values are unchanged; only the time of rejection is. An explicit
  scale (a scalar or two values, positive and finite) is checked by
  `shared._validate_ternary_numerical_controls`: before the host constructor, in the
  configuration validator and at the start of every `solve`. `constraints.minComposition` (finite,
  non-negative) is checked only on the default path, in the configuration validator and at the
  start of `solve` (`shared._validate_default_residual_floor`), because `constraints` does not
  exist before the host constructor. In `ternary.py:_solve_interface_planar` the scale is
  computed before `_bulkDiffusivityCountingActive` is set.

**Callers.** Definitions, examples and benchmark cases that relied on the default run the new
default with no source change (user decision 2026-11-14). The only definition with an explicit
scale is `tools/definitions/ferrite_growth_le.py` (O03a), which is unchanged.

**Tests.** `tests/test_illingworth_ternary_residual_normalization.py` (113 items) was written first
and run on the unchanged solver (`38a6595`): 61 failed, 52 passed.

- *Defect tests* (61 items). 59 failed before the change for their stated reasons: the scale
  helper did not exist; the default scale was the length term (default scale values, floor
  binding, zero and invalid values); iteration counts, interface positions and eta depended on
  dilution; per-step rows and inventory changes exceeded `tol * |I_k|` (also undiluted, in all
  five fixtures); an invalid explicit scale reached the host constructor or `setup`; a
  wrong-shaped scale and an invalid `minComposition` were not refused; the accumulated drift of
  a 201-node interstitial run exceeded `N * tol * |I_k|`.
- *Two dilution-invariance items passed before the change:* the two-phase and the three-phase
  substitutional couples with only component Y diluted (by 1e-6 and 1e-4). In those runs the
  row of the undiluted component decided every step, so loosening the Y row changed neither the
  iteration counts nor the results to 1e-10. They are kept as written; they show no defect for
  that pattern and guard it now.
- *Characterization* (1 item, differs before the change by construction): the interface-solve
  diagnostics of the superseded research test on the new default. The counts are the same; the
  scaled residual is the same physical defect over the component inventory.
- *Preservation and robustness* (51 items) passed before and after: the residual and inventory
  identities; explicit scalar and per-component scales equal to the earlier expression bit for
  bit, including `1e-200` and the subnormal `1e-310` with ordinary and zero inventories; `1.0`
  equal to the earlier default scale and reproducing recorded pre-R13 default runs (iterations,
  positions and eta to 1e-13); length-unit invariance; the attempted Newton update and its
  fallback in both families; a 201-node interstitial run at 1e-13 without retries.
- *Corrections to the new tests* (recorded so the counts can be reproduced). On their very first
  run on the unchanged solver the result was 63 failed, 50 passed, because of two faults in the
  tests themselves: the length-unit test compared against a run of a different duration (it was
  rewritten to compare metres with micrometres), and the fine-grid test mixed a no-retry check
  (preservation) with a drift bound (defect) and was split in two. The counts above are from the
  corrected file on the same unchanged solver. After the change, one expected error message was
  widened: the two-phase inventory integral already refuses a non-finite profile with its own
  `ValueError`.
- *Research tests that pinned the earlier default* (five; under the candidate default the
  existing suite gave 1051 passed, 5 failed):
  - `test_illingworth_ternary_three_phase.py::test_three_phase_molar_volume_state_is_immutable_and_equal_volume_kinematics_are_exact`
    and `test_illingworth_ternary_three_phase_similarity.py::test_three_phase_interface_residual_composition_scale_rows`
    assert values of the earlier default scale. Their "default" model now requests it with
    `interface_residual_composition_scale=1.0` (recorded `api_migration` edits scoped to the test
    function; reason in `docs/test-port-provenance.json`).
  - `test_ternary_deviation_form.py::test_interface_residual_is_smooth_near_a_fine_grid_step`
    (hand-ported file): a hand edit sets the scale to 1.0 on its model, because its `1e-15` noise
    threshold is in units of the earlier scale. The no-retry run in that file stays on the new
    default and passes.
  - `test_illingworth_ternary.py::test_ternary_interface_success_diagnostics_count_candidate_and_jacobian_work`
    and `test_illingworth_ternary_three_phase_similarity.py::test_three_phase_unit_composition_scale_matches_default_bitwise`
    are superseded (`tools/port_tests.py:SUPERSEDED`, recorded with reason and replacement). The
    builder of the first takes no keyword arguments and serves many tests that should stay on
    the new default; the premise of the second, "1.0 equals the default", is no longer true.
    Package tests repeat the assertions of the first verbatim with the scale set to 1.0 and cover
    the second with the explicit-1.0 runs.
- Full `tests/` on a freshly installed target: 1653 passed, 5 skipped (the same five as before the
  change), 1 deselected. Both generators reproduce every file (`--check`).

**Tooling added for R13.**

- `tools/api_migrations.py`: an edit may carry its own reason; `append_keyword` adds a keyword to a
  keyword-free call. `tools/port_tests.py`: `SUPERSEDED` tests are removed like deferred ones and
  recorded with their reason and the package test that replaces them (`9157592`).
- `tools/acceptance.py approve --verify-private-settings` (`38a6595`): an ordinary numerical
  approval may list O04/O05. Every existing gate runs unchanged. Before promotion the adopted,
  privately hashed settings must equal a fresh package capture; a missing, unhashed, changed or
  different setting refuses the approval. The cases keep their adopted settings
  (`private-carried-forward`) and the record states `private_settings: verified-equal`. No
  settings content, settings hash or private digest is committed, and the per-array deviations of
  O04/O05 are committed as verdicts only. Tested on synthetic temporary projects only; no
  approval was made with it.

**Survey (planning phase, open-input fixtures only; fixture observations, not guarantees).** Five
fixtures (two-phase substitutional and interstitial; three-phase with equal volumes, unequal
volumes and interstitial), each diluted by factors down to 1e-6 with the exact `(s, eta)` path
unchanged; 488 runs, no step retry, no step failure, no forced-update fallback.

- The earlier default behaved exactly as a tolerance of `tol / g` under a dilution `g`; the new
  default gave the same iteration counts and results at every dilution, for all or one
  component, substitutional and interstitial, equal and unequal volumes.
- Once the earlier tolerance was loose enough, every step took exactly one Newton update and the
  measured errors stopped growing with further dilution. This was observed in these
  well-conditioned fixtures. It is not a bound: the earlier rule permitted the relative defect
  above, and the fallback can accept a start with no update.
- *Newton iteration counts, same tolerance, earlier to new default, undiluted:* unchanged for
  the two substitutional fixtures at 1e-8 (556 and 925); two-phase interstitial at the default
  1e-8: 556 to 651 (about +17%); two-phase substitutional at 1e-11: 606 to 687 (+13%);
  two-phase interstitial at 1e-10: 632 to 717 (+13%); three-phase at 1e-11: 1095 to 1120 (+2%).
- *Measured runtime* (single runs, not a benchmark, reported separately from iteration counts):
  the new default at 1e-13 against 1e-8 took about 25% to 45% longer for 19% to 30% more
  iterations. Runtime at equal tolerance was not measured in the survey.
- The two-phase interstitial fixture at the default 1e-8, undiluted: relative inventory drift
  4e-6 with the earlier default and 2e-7 with the new one.

**Follow-up (open, not part of R13).**

- *Tighter default tolerance.* On the fixtures, 1e-11 gave drift of at most 2e-10 for about 20%
  more iterations than 1e-8, and the new default converged at 1e-15 on 201-node grids and at
  1e-13 with composition-dependent diffusivity (lagged and implicit). This is not established for
  TC-derived cases or very fine grids (the documented Fe-Mn-C 3201-node floor is about 1e-13
  relative to carbon). The defaults are unchanged.
- *Picard absolute tolerance.* `bulk_picard_atol=1e-12` is an absolute composition, so the
  implicit-mode inner tolerance is also dilution-sensitive. Not examined.
- *Startup eta estimate.* With only one component diluted by 1e-4 or more, the two-phase startup
  eta estimate fails without `initial_eta_componentwise_scaling`, under both normalizations (16
  survey runs stopped at setup). Not an R13 effect.

**Affected references (measured; approved under *Adoption*).** The planning estimate was every run that
relied on the default (C05-C12, C16-C24, C32, F11, F12, O01, O02, O03b, O04, O05, O06; contracts
C26-C29; examples E03-E10). The measured set is smaller. Exact full acceptance and example
acceptance on clean clones of `b2b3021` and of `4ad05c4` (the final code; identical outcome)
gave:

- *Differ from their selected references:* runs C09, C10, C11, C12, C16, C17, C18, C19, C20, C21,
  C32, F11, F12, O01, O02, O03b, O04, O05, O06 (19 of 40); contract C27 (one model, differences
  at the 1e-16 level); examples E03, E05, E06, E07, E08, E09, E10 (through their mapped cases).
- *Bit-identical although they run the new default:* C05-C08, C22-C24, contracts C26, C28, C29,
  example E04. The Newton step does not depend on a row scaling of the residual beyond round-off
  that the update absorbs, so results change only where a convergence decision changes.
- *Unchanged, as expected:* all binary runs, C25, O03a, O07, O08, LEGACY, E01, E02.
- These differences are reference-comparison failures until a separately authorized approval.
  Acceptance limits, `approval_criteria` and contract assertions are not edited, regenerated or
  relaxed. No difference appeared outside the planning estimate.

**Candidate results on `4ad05c4`** (clean clone with LF line endings, fresh install; nothing
promoted; the same checks on `b2b3021` gave the same results).

- *Correctness (passing).* Every case covered by the unchanged `approval_criteria` passes on its
  saved candidate output (39 cases, LEGACY and live O06 included, evaluated with the unchanged
  `acceptance.criteria_results`). All five contracts pass their assertion groups
  (`acceptance.contract_assertions`). The private-settings proof for O04 and O05, run without
  promotion, reports both verified equal with no capture left. Unseeded repeatability: 34 cases
  checked twice, no difference. Full `tests/`: 1720 passed, 5 skipped, 1 deselected.
- *Size of the differences, public cases.* Largest absolute differences: C09/C10 eta 3.7e-8;
  C11/F11 eta 6.0e-8; C12/F12 eta 7.0e-8; C16-C19 profiles 6.8e-8; C20/C21 eta 1.1e-13 (within
  the public tolerance; all others are outside it); C32 profiles 1.9e-9 and its final time
  (below). Conservation criteria, reference to candidate, all within their unchanged limits:
  C09/C10 component moles 1.2e-9 to 2.0e-13; C11/F11 2.5e-10 to 4.4e-11; C12/F12 1.9e-10 to
  1.4e-11; C16-C19 6.7e-9 to 2.0e-11; C32 1.3e-10 to 3.2e-11; C20/C21 unchanged. The analytic
  errors of C11, C12, F11 and F12 are unchanged to four digits.
- *Private cases (verdicts only).* O01, O02, O03b, O04, O05 and O06 each differ in interface,
  profiles, eta and inventory, are outside the public tolerance and pass their conservation
  criterion. Their numbers stay in ignored local output.
- *Convergence.* Accepted times are bit-identical in every changed case except C32, so no other
  case changed its retries. No case failed to converge.

**C32 (accepted behavior change; user decision 2026-11-15).** C32 runs with `maxIterations = 1`,
one normal retry and 25 terminal thin-phase retries. With one iteration a step can be accepted
only when its unmodified start already meets `tol * scale` (through the existing forced-update
fallback), so the timestep is halved until it does.

- *Evidence.* The physical start residual is linear in the timestep and is the same under both
  normalizations; only its divisor changed, from the length term 0.983 to the component
  inventories 0.258 and 0.138. The scaled residual at each trial timestep is therefore larger by
  that ratio, and the first timestep that passes `1e-10` is `2e-6 / 2^15 = 6.1e-11` where it
  was `2e-6 / 2^13 = 2.44e-10`: 15 halvings where there were 13. Both runs accept at the start
  with `_lastImplicitForcedUpdateSkipped` set, take the terminal thin-phase path (phase C below
  the width 0.31) and stop there. All C32 criteria pass.
- *Decision.* The change from 13 to 15 halvings is accepted as an intended consequence of the
  stricter normalization. C32 stays on the new default; it is not pinned to 1.0 and its retry
  settings are unchanged. This authorizes the behavior change, not the reference approval.
- *Regression test* (`4ad05c4`,
  `test_c32_terminal_thin_phase_run_halves_until_the_start_meets_the_tolerance`): the exact
  attempt sequence and final timestep; that only the last attempt passes and one halving earlier
  would not; the terminal thin-phase record and stop; one committed step with histories equal
  to the live state, no pending candidate and moles conserved within `2 * tol * scale`; a
  fresh model repeating the run bit for bit; and the earlier 13 halvings with an explicit scale
  of 1.0. `reset` is not part of the case: `maxIterations = 1` is below the validated minimum
  and can only be set after `setup`, which the test also checks.

**Review follow-ups (2026-11-15).** Both defects were first reproduced on the committed code
(`b2b3021`): the new tests failed there (6 capture tests, 16 scale tests) and pass after the fixes.

- *Private captures left behind when the capture itself fails* (`5362b6a`).
  `acceptance.check_case_settings` called `capture_example_settings` before its cleanup
  `try/finally`, so an O04 capture stayed when the later O05 capture failed. The capture now
  runs inside the cleanup, and the private capture files are removed by their known names
  (`<directory>/<case>.settings.json`), complete or partial. The original exception propagates,
  also an interruption, and an approval promotes nothing. Adopted settings, public captures and
  unrelated files are not touched. Tests on synthetic temporary projects: a completed O04
  capture followed by a failing or interrupted O05 capture, directly and through `approve` in
  the private-settings and the layout mode.
- *A private capture that cannot be removed* (second review round, 2026-11-15). The cleanup
  added in `5362b6a` caught `OSError`, printed a warning and continued, so a synthetic approval
  succeeded and promoted while both private capture files remained after a `PermissionError`.
  Reproduced on `d7bddce`: seven new tests failed there (the approval returned success in the
  private-settings and the layout mode with one or both files locked; the leftover file was
  neither attached to a propagating exception nor reported beside a setting difference). Now
  (`acceptance._remove_private_captures`): removal is attempted for every private capture even
  when one fails, and a file counts as removed only when it is gone. If the proof otherwise
  completed, each remaining file is a proof failure that names it, so the approval promotes
  nothing. If an exception is already propagating, including an interruption, it is re-raised
  unchanged, and what remains is printed and attached to it as a note. Only the capture files
  of this operation are touched. Tests on synthetic temporary projects, in both modes where
  relevant: equal settings with a deletion failure; a capture failure or an interruption with
  a deletion failure; one locked file not preventing removal of the other; a setting
  difference and a deletion failure both reported; the normal path still removing the
  private captures and promoting. Focused approval and layout files: 157 passed. No numerical
  code, reference, criterion or assertion changed, so the numerical gates were not rerun.
- *Valid explicit scales assigned after construction* (`8a9fef4`). A list or tuple assigned to
  `interfaceResidualCompositionScale` passed validation and then raised `TypeError` in both
  residual-scale calculations. Both now use a local validated `float64` pair
  (`shared._explicit_residual_composition_scale`, also used by the validator). The attribute is
  never replaced and no attribute is added, so captured settings are unchanged; the explicit
  expression including `1e-300` is unchanged bit for bit. Tests: scalar, list, tuple, array,
  NumPy scalar and subnormal assignments give the scale and the step of the same value passed
  to the constructor, in both families; invalid values and shapes assigned later are refused
  before `setup`.
- *Validation after the last code change* (`4ad05c4`): the candidate results above; focused
  approval, layout and normalization files 315 passed before the C32 test was added.

**Benchmark (one periodic comparison; advisory).**

- *First attempt* (`b2b3021`, run `20261115T054814Z-53c0ee79`): the tool refused the comparison
  as incompatible (`case.workload_sha256`, `harness_sha256`). The clean clone had been checked
  out with CRLF line endings, which changed the file hashes, and the result had not been
  exported. Its raw measurements are kept locally; it is not counted as a comparison.
- *Completed comparison* (`4ad05c4`, run `20261115T152212Z-0c6ddec8`): standard suite from a
  clean clone with LF line endings, exported with `benchmarks/baselines/export_result.py` and
  compared with `2026-11-13-windows-standard` by `tools/performance.py compare`. All nine cases
  are comparable; the tool reports no environment and no protocol difference. Worker suite
  366 s.
- *Work counts (deterministic).* Newton iterations per trajectory: `ternary-constant-large-off`
  3201 to 3213, `ternary-lagged-large-record` 3202 to 3214, `three-lagged-large-off` 3213 to
  3218; the three binary cases and the other three ternary and three-phase cases are unchanged.
  Step retries are 0 in every case. Inventory drift in the three changed cases fell from
  6.0e-10, 6.0e-10 and 3.0e-10 to 1.8e-10, 1.8e-10 and 9.1e-11.
- *Runtime (separate from the work counts).* Median solve-time ratios, candidate over baseline,
  are 0.76 to 0.98 in every case; no case is flagged for solve time (one advisory flag on the
  sub-millisecond `reset` of `binary-implicit-large-record`).
- *Limitations.* The baseline was measured on another day from a dirty tree at `6beb341`. The
  binary cases, whose solver code and work counts did not change, ran 7% to 24% faster, so
  session-to-session variation is larger than any R13 effect, and the ratios are not evidence of
  a speed-up. What the comparison supports is that no slowdown beyond the tool's thresholds was
  observed and that the extra work is 5 to 12 iterations in about 3200. Acceptance cases at
  tighter tolerances were not timed. The recorded dirty flag of the candidate comes from
  untracked log files at the clone root; no tracked file was modified.

**Adoption (2026-11-15, user-authorized).** The proposed set was approved unchanged.

- *Approval* `20261115T181200Z-6caa07b-1` (commit `e1bb66d`, manifest and force-added report
  only): `approve --tier full --verify-private-settings` for exactly runs C09, C10, C11, C12,
  C16, C17, C18, C19, C20, C21, C32, F11, F12, O01, O02, O03b, O04, O05, O06 and contract C27, in
  a short-path clean clone of `6caa07b` (full history, LF line endings, fresh targets) with the
  local inputs copied and hash-verified (318 files). Reason "LEDGERS R13: ternary and
  three-phase interface residuals normalized by component inventory (default
  interface_residual_composition_scale; C32 takes 15 halvings instead of 13, user decision
  2026-11-15)", 1021 s. `6caa07b` changes no solver code after `4ad05c4` (acceptance tooling,
  its tests and docs only).
  - Gates, all unchanged: 40 run cases (live O06) each repeated bit-identically;
    `approval_criteria` for every case; whole suite on the installed wheel 1738 passed, nothing
    skipped, 1 deselected; contract assertion groups for C25-C29; all 25 unlisted cases and
    contracts within public tolerance; nothing skipped. C20 and C21 are within public tolerance,
    the other 18 listed cases differ, as recorded above.
  - Private settings: O04 and O05 `verified-equal` against a fresh capture and carried forward
    (`private-carried-forward`); both private capture files were removed. The committed report
    holds verdicts only for O01, O02, O03b, O04, O05 and O06, plus O05's conservation value and
    limit; no settings content, settings hash or private digest.
  - Agreement with the reviewed results: every array of every candidate (40 runs, 5 contracts)
    is bit-identical to the reviewed `4ad05c4` candidates. The only differences are per-run
    paths (`adapter_origin`; the O06 cache directory, normalized by the comparison). C32 ends at
    `2e-6 / 2^15 = 6.1e-11` (15 halvings) as documented; all C32 criteria pass.
  - Artifacts: the 45 approved files (`.verification/acceptance/reference-approved/20261115T181200Z-6caa07b-1/`,
    relative to `packages/moving-boundary`; no settings file for O04/O05) were copied back to the
    original worktree and match the manifest hashes, as does the committed report. `reference/`,
    the four earlier approval folders, `example_settings/`, `inputs/` (including
    `private-inputs.json`) and `pilot/` are byte-identical to before the approval.
- *Gates after the approval* (from `e1bb66d`, same clone, fresh targets, no special flags, not
  `--record`ed): exact full 40/40 runs and 5 contracts bit-identical against the selected
  references, live O06, 34 repeat pairs with no differences, 504 s; examples E01-E10 pass with 68
  tests (E11 a pending fixture), 148 s. The 363 local input, reference and approval files were
  byte-identical before and after these gates.
- *Other checkouts.* `.verification/` is git-ignored, so the approved artifacts are not in Git.
  Any other checkout or clone must copy in `.verification/acceptance/reference-approved/` with
  all five approval folders (plus the other local inputs listed in R10_HANDOFF.md section 3) and
  verify them against the manifest hashes; without them exact quick/full acceptance fails
  for the R13 cases.

### R14: default Thermo-Calc kinetics settings can produce wrong surrogate samples (open)

**Status:** open; recorded 2026-11-17 from the Nb-Ni-Ti (TCHEA5/MOBHEA3, 1473 K, LIQUID#1 |
BCC_B2#2) surrogate investigation. No package code has been changed. **Scope:**
`src/moving_boundary/providers/thermocalc.py` with the default `kinetics_*` settings
(`ThermoCalcConfig` docstring :209-232, `_force_kinetics_phase` :1250,
`_validate_single_kinetics_phase` :1268, `_configure_global_minimization` :1543), and every
surrogate built from forced-phase TC kinetics (`TernaryMovingBoundaryThermodynamicsSurrogate.from_database`).
Paths are relative to `packages/moving-boundary`.

With the defaults (`kinetics_constrain_single_composition_set=False`,
`kinetics_disable_positive_definite_hessian=False`, `kinetics_disable_global_minimization=False`),
a "forced" kinetics calculation suspends only the configured phase names. Two silent failures
were found:

1. *Forced calculations are not single-phase.* Global minimization re-creates composition sets
   (`BCC_B2#3`, `BCC_A2#3`) that are not suspended. In the low-Nb build, 1,877 of 2,083 forced
   `LIQUID#1` samples converged to LIQUID + B2 or B2 only (liquid composition up to 0.09-0.12
   mole fraction from the request; 740 of them outside the LIQUID + B2 field), and 171 forced
   `BCC_B2#2` samples split into two bcc sets (offsets up to 0.38). The surrogate stores every
   one at the *requested* composition. The adapter checks the stable sets and phase
   composition only in constrained mode, so nothing warns.
2. *Path-dependent B2 Hessian (disordered state).* Partway through a long build, the session can
   switch into a state in which the reported BCC_B2 thermodynamic factors, and hence `DC`, are
   those of *disordered* B2. A fresh session with local minimization and the positive-definite
   Hessian on reproduces them. It converges to identical site fractions on both sublattices, with
   G 5.3-5.7 kJ/mol above the ordered equilibrium (Nb 0.002-0.16), and Ni<->Ti `DC` about 110x
   larger at Nb/Ni/Ti 0.002/0.499/0.499. Yet the stored kinetics diagnostics of the affected
   records list the *ordered* site fractions, identical to a correct record, so a site-fraction
   check on stored records does not detect the failure (not explained). At Ni 0.4975 / Ti 0.5005 the
   curvature of G along Ni<->Ti is 2.4654e6 J/mol (finite differences of the basis-free
   chemical potentials, and TC's Hessian in a fresh session, in every reference basis);
   affected bundles stored 2.0793e5, and `DC` differed by up to about 60x in the Ni<->Ti
   exchange term. In calculation order the switch happened once, during the B2 tie-line-end
   stage, and persisted (old NB build: 11 good points, then 756 bad; old TI/NI builds: none at
   low Nb; repeat builds with the positive-definite Hessian off still switched). All four
   bundles also carried bad values at some Nb >= 0.05 points. Local minimization with the
   forced positive-definite Hessian on reproduces the bad values to every printed digit; the
   other three on/off combinations give the correct value. A single disordered or split
   calculation did not trigger the switch in a fresh session, so the exact trigger is not
   identified. The docstring already notes that reused TC calculation objects "can retain a
   path-dependent phase-Hessian state", but only to justify fresh multistart candidates.

Consequence: in this system the B2 diffusivities depended on the build history and looked
like a reference-element dependence (NB- vs TI/NI-reference bundles differed by 5-60x in B2
fluxes and by 60-210 um in Illingworth interface positions). TC itself is basis-consistent
when given clean single-set calculations. Liquid samples were consistently wrong in every
bundle.

**Validated configuration (downsampled).** `kinetics_constrain_single_composition_set=True`
with `kinetics_disable_positive_definite_hessian=True` (the WTiFe bond example's choice) on
every 5th point of the low-Nb grid, tie-line stage included (401 bulk points, 176 s): all 538
liquid and 538 B2 records had one stable set at the requested composition (deviation
<= 2e-13), and the B2 Hessian matched finite differences of MU at 40 spread points along
Ni<->Ti and Nb<->Ni (median relative difference 4.5e-8, maximum 8.3e-4 on the dilute-Nb
direction).

**Full-grid confirmation.** Builds of both full grids with this configuration (1,635 and 2,002 bulk
points, 3.5-4.2 min instead of 9-16 min) gave single-set records at the requested composition for
every sample (1,772 / 2,139 per phase). The B2 Hessian matched fresh finite differences at 29
points per build spread over the build order (median relative difference 4e-8 / 7e-8). The same
check on the old bundles failed at 52-53 of 58 directions. Each build rejected 50 B2 samples at
liquid-box compositions (Ni <= 0.40) with a non-positive diffusivity spectrum, where B2 is not
stable; the old bundles had accepted them as multi-set calculations. Rerunning the 98 Illingworth
runs that had used the old bundles changed the results qualitatively. Thick-liquid cases moved
-187 um instead of +64 to +76 um, solidification times changed by up to 30x, and a step-start case
that had failed with negative Nb now completes with no Nb dip.

**DICTRA uses the same disordered state.** Diffusivities extracted with TC-Python's
post-processor (`DC`, `TF`, `MUR` along the profile) from short runs of the same cases show:

- DICTRA's sharp-interface solver matches the disordered-state values (median relative difference
  2e-8 in `DC`), with identical chemical potentials and G.
- The homogenization solver matches the corrected configuration (2e-7).
- `Options().enable_forced_starting_values_in_equilibrium_calculations()` does not change the
  sharp-interface result.

Agreement with sharp-interface DICTRA therefore does not validate ordered-phase kinetics data; it
was how the old surrogate's errors stayed hidden in this study.

**Possibly affected elsewhere.** The CuNiTi and FeCrNi Lee1996 research examples use local
kinetics minimization with the positive-definite Hessian on for their non-multistart phase;
the NiTiNb three-phase TC example uses the defaults. Disordered phases may be unaffected, but
this has not been checked.

**Investigate:**

- *Defaults.* Whether to default to single-set kinetics with the positive-definite Hessian off,
  or to require an explicit choice, at least for ordered phases. Check the cost (the validated
  run was faster than the default build) and the effect on the existing acceptance and
  example references before changing anything.
- *Documentation.* Describe what each `kinetics_*` option does, the failure each default
  allows, and the recommended combination for ordered phases, in the `ThermoCalcConfig`
  docstring and the README's Thermo-Calc section.
- *Validation.* Check every kinetics result, not only in constrained mode: stable sets equal
  `[requested phase]` and phase composition equals the request. Warn, raise, or at minimum
  record the mismatch in kinetics diagnostics and surrogate metadata; a build-level summary
  (count of mismatched samples) would have exposed failure 1 at once.
- *Hessian check.* An optional diagnostic comparing TC's thermodynamic factors with finite
  differences of chemical potentials at a few sampled points (two extra equilibria per
  direction), or a cross-basis consistency check, would expose failure 2. Cheaper still for
  ordered phases: compare G (or the chemical potentials) of the kinetics state with a
  global-minimization equilibrium of the same phase at the same composition. The disordered
  state was 5.3-5.7 kJ/mol higher here, while stored site fractions did not show it.
- *Tests.* The research fake-backend tests (`kawin/tests/test_thermocalc_examples.py`, from the
  extraction root) cover the switches' mechanics only and were not ported; no test checks
  the numerical result. Any fix needs a licence-free test for the validation and, where
  licensing allows, a live TC check.

**Evidence.** Scripts, logs, outputs and the six examined bundles (with their kinetics
diagnostics) are in `../ternaryTLP/R14_tc_kinetics_evidence/` (from the extraction root; outside
this repository, not version-controlled); its `README.md` lists every file. Key items:
`basis_diag.py`, `basis_order.py`, `basis_hessian.py` (stored-diagnostics comparisons),
`hess_probe.py` (finite-difference curvature), `order_probe.py`, `option_probe.py`,
`flip_probe.py` (query order, option combinations, trigger tests), `sequence_check.py`,
`split_check2.py`, `liquid_split_check.py` (switch position and set splitting in the stored
bundles) and `mini_fix.py` / `logs/mini_fix.log` (validated configuration). The study context
is in `../ternaryTLP/NbNiTi_Illingworth_and_DICTRA_followup.md`; its section 6 attributes the B2
difference to the reference element and is superseded by this entry. The full-grid builds, the
reruns and the DICTRA diffusivity extraction are reported in
`../ternaryTLP/NbNiTi_corrected_surrogate_rerun.md` (sections 2-5 and 9), with the corrected bundles,
run outputs and scripts in `../ternaryTLP/NbNiTi_corrected_surrogate_runs/`. Key items there:
`verify_ss.py` (full-grid check), `dc_compare.py` (DICTRA vs surrogate and TC diffusivities),
`state_probe.py` (G and site fractions per mode) and `dictra_dc/` (extraction worker, saved DICTRA
results).

## Two-phase ternary analytic acceptance cases (C33/F33)

**Status:** active since `65ce32c` (2026-11-15): C33 runs in quick, F33 in full (`full_drift`).
Registered inactive in `a26d099`; baselines captured as `20261115T224052Z-a26d099-1`; validated
and retrospectively compared with pre-R13 code on 2026-11-15. Paths below are relative to
`packages/moving-boundary`.

**Gap closed.** Before these cases the two-phase ternary Illingworth solver
(`MovingBoundaryIllingworthTernaryFD1DModel`) was compared with the constant-matrix similarity
solution only in pytest (`tests/test_illingworth_ternary_similarity.py`,
`tests/test_analytic.py::test_two_phase_solver_meets_source_bounds`), which assert bounds but record
no values. Acceptance gated two-phase ternary cases on reference parity and conservation only.
C33/F33 add recorded analytic-error metrics with fixed `approval_criteria`, measured on the
solver's own two-phase Landau grids, interface Stefan/tie-line solve and R13 residual scale. The
other ternary cases (C05-C08, C16-C19, C24, O01, O03a, O04-O06) still record no exact-solution error
(`approval_criteria.not_covered.ternary_analytic`).

**Problem and frozen configuration** (`tools/two_phase_similarity_cases.py`; constants as in
`tests/test_analytic.py`, source research `test_illingworth_ternary_similarity.py` at `06dc7b4`):
constant 2x2 interdiffusivity matrices per phase, a linear tie-line closure, far fields
`[0.38, 0.02]` / `[0.20, 0.15]`, elements `Z, X, Y`. Domain `L = 0.6`, `S0 = 0.3 + 1e-12`;
`two_phase_landau_grids(31, 31, "linear", spacing_ratio=0.1)`; start from the exact similarity
profile at similarity time `T0 = 0.1` on a 2401-node physical mesh; similarity end 0.4 (C33) and
1.0 (F33); `semi_log`, `semiLog_dt = 0.05`, `semiLogT0 = 1e-4`; `tolerance = 1e-11`,
`max_iterations = 50`, default (R13) residual scale; `initial_eta_guess` = analytic eta (a seed
only: setup estimates eta itself and pins the interface nodes to that tie line, so the start is not
exactly analytic). Metrics (`exact_arrays`): interface error at every recorded time (m), eta error,
final-time per-component profile errors (each phase against its own formula, so the composition
jump never enters a difference), and the relative error of `k` fitted over `t >= 2 T0`.

**Stage 0 evidence and qualifications** (scratch evidence, not committed; configuration and limits
frozen before any capture):

- `T0` rule passed (interface-resolved start time 0.0747 <= 0.1); 2401 physical nodes met the
  start-projection rule.
- *Approved rule revisions (user, after reviewing the results; not an accuracy certification).*
  R1: the setup's discrete eta estimate counts as discretization under test (it converges at about
  second order with the transformed grid: -5.69e-5, -1.57e-5, -4.13e-6, -1.05e-6 at 31/61/121/241
  nodes); the 10% nuisance budget covers physical-mesh initialization plus nonlinear sensitivity
  only (measured <= 0.41% for every metric). R2: the matched-inner-cell domain/grid change must stay
  within 10% of each proposed headroom (C33: interface 5.3%, k 4.3%, profiles 2.0%/1.2%, eta 0.4%;
  F33: interface 9.2%, k 6.8%, profiles 3.2%/2.9%, eta 0.4%). Limits were not enlarged to pass.
- *Failed original rules, kept on record:* the initialization budget as written fails for eta
  (12.3%, setup-eta share 11.9%); the original 1% domain screen fails (unmatched 24-32%, matched
  0.3-9.4%).
- Time convergence is approximately first order (0.98-1.02). Spatial errors decrease with observed
  orders trending toward, but below, one at the finest tested pair (k, s, eta 0.87-0.96; profiles
  0.81-0.86). The coarse joint-sweep cancellation flags (k, s) and the pre-asymptotic space profile-X
  flag (0.547) stay documented with their explanations (additive space/time reconstruction within
  5.7%; finer-triple orders 0.75-0.87).
- Profile convergence orders carry an interpolation qualification: the exact-profile reconstruction
  term exceeds the 10% check except at the finest pairs, while linear and PCHIP interpolation agree
  within about 7%.
- The domain/grid study establishes a sensitivity, not negligible boundary effects or
  grid-independent accuracy. The cases are fixed regression gates against the analytic solution;
  their error interpretation carries these qualifications and they do not detect every regression.

**Limits** (existing rules: analytic 2x current, conservation 10x current, floor 1e-13; all within
the independent pytest source bounds: eta 5e-3, profile 1e-3, |k_rel| 1e-2, interface 1e-2 x
displacement, final inventory drift 1e-9 m):

| Criterion | C33 | F33 |
|---|---|---|
| interface (m) | 8.7e-6 | 1.5e-5 |
| eta | 9.3e-4 | 9.3e-4 |
| profile X / Y | 2.5e-4 / 2.1e-4 | 2.6e-4 / 2.2e-4 |
| k (relative) | 1.5e-3 | 1.8e-3 |
| composition inventory (relative normalized drift) | 1.3e-9 | 1.3e-9 |

The inventory criterion is the maximum over time and components of `|I_k(t) - I_k(0)| / max_k |I_k(0)|`
and is distinct from the absolute source bound checked at capture.

**Baseline capture and provenance.** `acceptance.py baseline --cases C33 F33` in a fresh, clean LF
clone of `a26d099` with fresh targets: two unseeded package runs per case bit-identical (26/26
arrays), every independent source bound passed, criteria preview equal to the frozen limits. The
capture record holds the commit, environment, the installed wheel's name and SHA-256, a digest of
the installed `moving_boundary` tree, per-run timings, check values, the diagnostic
residual-tolerance budget D (reported, never gated) and each case's `scientific_sha256`. The
fixtures were installed into `.verification/acceptance/reference/` with the documented exclusive,
hash-checked copy (`tools/acceptance_baseline.install_capture`). **The baseline files are stored
locally, not committed:** Git holds their SHA-256 hashes and provenance in
`tools/acceptance_manifest.json` (`expected_artifacts`, `captured_role: "package"`, `capture_id`,
`scientific_sha256`, `capture_record`). Another checkout must copy in `reference/C33.*` and
`reference/F33.*` (and, for provenance review, the capture folder
`.verification/acceptance/baseline-staging/20261115T224052Z-a26d099-1/` with its original
`capture.json`) and verify them against the manifest; `baseline --verify --cases C33 F33`
regenerates and compares without writing.

**Activation** (`65ce32c`): tiers, expected artifacts and `approval_criteria` equal to the preview;
`propose` from the hash-validated baselines reproduces every adopted criterion exactly.

**Validation of `65ce32c`** (fresh clean LF clones, fresh targets, local artifacts copied and
hash-verified):

- `baseline --verify --cases C33 F33`: both pass (same keys, no differing arrays or retained report
  fields, `scientific_sha256` equal, npz bytes equal).
- Full package test suite: 1772 passed, 5 skipped, 1 deselected. The 5 skips are the C25-C29
  contract tests, which need the state that acceptance quick configures (the quick and full gates
  ran those contracts); the deselection is the `slow` test `tests/test_examples.py:397`.
- Exact quick: 31 runs (C33 included) and 5 contracts bit-identical, 34 repeat pairs, 145 gate
  tests, 414 s. Exact full: 42 runs (C33, F33 and live O06 included) and 5 contracts bit-identical,
  35 repeat pairs, 531 s. **No acceptance case was skipped** in either gate. C33/F33 meet every
  criterion (6/6 each), with candidates byte-identical to the baselines.
- **Exact full needs a sufficiently short clone path for O06.** A first exact full attempt from a
  long scratch path failed at the live O06 availability probe after 22 s: Thermo-Calc reported
  `ERROR IN MLOPEN : PERMISSION TO ACCESS FILE DENIED` for a 268-character cache path, beyond the
  Windows 260-character limit. No case ran and the attempt was not counted. The same commit passed
  from a clone directly under the user temporary directory (equivalent cache path 195 characters).
- **F33's repeatability evidence** is its two bit-identical capture runs plus matching regeneration
  (`baseline --verify`) and matching exact full acceptance; the full gate does not repeat
  `full_drift` cases, as for F04/F11/F12.

### C33/F33 retrospective comparison: pre- and post-R13

**Protocol.** Clean LF clones of `9db54aa` (before R13) and `65ce32c`, each with its own fresh wheel
target in the test environment. The only commits between them that touch `src/` are the R13
commits `cdde760` (documentation), `b2b3021` and `8a9fef4`. One driver loaded the frozen
`tools/two_phase_similarity_cases.py` and `acceptance_cases.collect` of `65ce32c` by path for both
revisions and solved as the acceptance worker does. Six configurations: C33 and F33 at case level
(31 nodes, `semiLog_dt` 0.05, 2401 physical nodes) and the joint levels (61, 0.025, 2401) and
(121, 0.0125, 6001) for both windows. Materiality: |change| >= 1% of the pre-R13 value. Single run
per configuration.

**Checks.** `65ce32c` reproduces the committed C33/F33 baselines bit for bit (26/26 arrays).
**Explicit-scale cross-check:** `65ce32c` with `interface_residual_composition_scale=1.0`
reproduces `9db54aa` bit for bit in all six configurations (26/26 arrays).

**Numerical identity.** In every configuration the accepted times are identical and 10 of 26 arrays
are bit-identical (time histories, final time, grids, initial inventory). The state histories
differ: the largest eta-history difference is about 3.51e-8 (C33 joint level 121; F33 joint level
121 has the same value) and the largest interface-history difference about 4.26e-10 m (F33). These
history differences are distinct from the changes in the scalar maximum-error metrics below.

**Analytic accuracy** (scalar maximum-error metrics; reported separately from conservation):

| Metric | C33 pre -> post | F33 pre -> post | All six configurations |
|---|---|---|---|
| interface max (m) | 4.30899e-6 -> 4.30896e-6 | 7.42618e-6 -> 7.42576e-6 | slightly smaller (relative -5.7e-5 to -5.6e-6) |
| \|k_rel\| | 7.15824e-4 -> 7.15823e-4 | 8.88855e-4 -> 8.88828e-4 | slightly smaller (-3.1e-5 to -1.8e-6) |
| profile X / Y max | relative -2.0e-8 / -1.5e-8 | relative -1.2e-8 / -1.4e-7 | slightly smaller (<= 2.5e-7) |
| eta max | +1.2e-15 | +1.2e-15 | slightly larger (up to +5.1e-10, relative <= 3.2e-6) |
| final eta error (signed) | -4.57440e-4 -> -4.57443e-4 | -3.95108e-4 -> -3.95123e-4 | magnitude slightly larger (<= 3.8e-5 relative) |

Every change is immaterial. The largest relative change, 5.73e-5 (F33 interface maximum), is
0.00573%, about 175 times smaller than the 1% materiality threshold. Interface, profile and k errors
improve slightly while eta errors worsen slightly; there is no meaningful overall accuracy
improvement, and none is inferred from the conservation change.

**Conservation.** The relative normalized drift falls by about 21-91% across the six configurations
(C33 1.565e-10 -> 1.236e-10, -21%; F33 6.14e-10 -> 1.24e-10, -80%; joint level 61 -64% / -31%;
joint level 121 -91% / -74%, C33 / F33 windows). The final inventory drift falls for both
components everywhere (F33 X 1.07e-10 -> 2.15e-11 m); in F33 joint level 121 the Y drift changes
sign (-1.07e-12 -> +6.6e-14 m). The diagnostic allowance D tightens about 3.4x (X) and 11.8x (Y),
because the per-component scale `max(|I_k|, L * minComposition)` replaces `L = 0.6` with the
inventories (about 0.174 and 0.051). D is an approximate diagnostic, not a bound; smaller observed
drift is not a guarantee.

**Nonlinear work.** Accepted steps are identical. R13 needs slightly more nonlinear work: C33 +3
Newton iterations (+0.9%), F33 +25 (+6.8%, with +75 candidate and +25 Jacobian evaluations), +1 at
each joint level; no retries in either revision. Single-run wall times (0.5-2 s per run, the first
solve per target including a one-time Numba compile) are not a performance benchmark or a
reliable speed comparison.

**Scope.** The conclusion is limited to these six configurations of one problem: R13 reduces the
relative normalized conservation drift by about 21-91%, changes analytic accuracy negligibly under
the 1% materiality rule, and requires slightly more nonlinear work. All Stage 0 qualifications
above apply; this is not an accuracy certification.

**Corrected reporting.** The first Stage 5 report quoted history differences from the C33
case-level run only (eta about 3.9e-9, interface about 3.4e-11 m) and described the largest
relative analytic change as "four or more orders of magnitude" below materiality. The figures above
were recomputed from the saved run outputs without new simulations; the original outputs are kept
with the scratch evidence.

## Example acceptance (pending tier approval)

See [ACCEPTANCE_CANDIDATES.md](ACCEPTANCE_CANDIDATES.md#proposed-example-cases-pending-approval)
and `tools/example_manifest.json`. These claims are kept separate: parity shows matching
behaviour with the frozen references and is not an accuracy result; analytic comparisons
assess accuracy within their assumptions; conservation, reload and live-workflow checks each
assess one named property. Frozen-matrix FeCrNi doesn't validate composition-dependent
physics, and no check here establishes accuracy for the lagged workflow.

Known gaps: O04, O05 and O06 use frozen matrices, so lagged composition-dependent parity (E11)
needs a new fixture that isn't captured yet. Its shared-bundle design won't compare grid
sampling between hosts. Database-dependent tests skip in reviewer installs until a database
is cleared for redistribution. Linux and other tested-version behaviour hasn't been checked,
so no thermodynamic golden is frozen. `binary_tlp_bonding` has no E case or fixture; only its
tests check it. Three Fe-Cr-Ni references (Lee 1993, Hoglund et al. 1990, Fridberg et al. 1969)
are cited only as listed by Lee and Oh, not checked against the originals.

An unused local `tieTriangle_dict` with hard-coded compositions remains in
`from_database`'s seed-point scan, with its other uses commented out. It's inert and shared
with 06dc7b4; removal belongs on the research branch.

## Historical evidence and narrow fix integration

verification.json, assessment.json and git-audit.txt preserve the original scoped
binary/host audits and their actual source hashes. They are historical evidence,
not statements that public compatibility was already established then. Whole
binary/ternary/three-phase relocation commits are 8233ad2/e08eed0/693595d, each
with a passing exact quick gate. acceptance.json retains each stage separately.

The approved debugger-fix branch was integrated into local main. The final
5f4ad57 commit removed only obsolete test stubs (12 lines), with two retry tests
passing; production numerics were unchanged. The temporary runtime pin followed
that fix until the explicit switch to public 0.5.0. No research branch was merged
and the frozen extraction was not merged forward. Live TC is accepted only
within the separately authorized O06 scope.

## Redistribution and scientific maintenance

NOTICE retains upstream MIT text and separately records the MAP provenance and
published redistribution conditions for the modified Python adaptation of
ConservativeIFF by T.C. Illingworth and I.O. Golosnoy. MAP permits modification and
redistribution with attribution in documentation/source, identification of changes
and no charge for distribution; NOTICE links the authoritative terms and disclaimer.
The terms were checked on November 14, 2026; no external contact was made. Source
headers credit MAP and the originators and identify the adaptation. This supersedes
the earlier statement that MAP's redistribution permission was unresolved; it does
not grant rights in separately authored local additions or select a package-wide
license (see LICENSE). No solver behavior, references or acceptance criteria changed.
No PDF, C++ source, digitized figure data, database, trained model or TC-derived
reference is packaged or committed.

Package name, audience/distribution, release/maintenance ownership, pushing,
upstream contribution, optional MAP outreach and ternary prompt-default changes remain
user decisions. Scientific repairs have one authoritative source and deliberate
port records; baseline refreshes are separate reviewed changes. Do not maintain
independent scientific fixes or regenerate fixtures to hide a discrepancy.
