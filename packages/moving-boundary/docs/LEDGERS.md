# Capability and dependency ledgers

## Current capability inventory

Quick: 30 run comparisons and 88 pytest tests passed in 427.34 s. Full: 40 run comparisons and 88 pytest tests passed in 578.02 s.
These are regression claims for the approved frozen inputs, not a full release or
physical convergence audit. See ACCEPTANCE_CANDIDATES.md for exact input variants.

| Capability | Evidence and supported scope |
|---|---|
| Binary constant, lagged and implicit diffusivity | C01-C04, C13-C15, C30-C31 and legacy 33 histories/12 bulk probes |
| Ternary two-phase and motion directions | C05-C08, interstitial x/u C16-C19, saved surrogate C24 |
| Three-phase, moving geometry and molar volume | C09-C12, C20-C23; equal/unequal explicit volumes; binary/two-phase unequal volumes unsupported |
| Exact/similarity diagnostics and drift | C04/C11/C12 and full F04/F11/F12; analytic errors recorded independently of reference |
| Lifecycle, rollback, failures and terminal stop | C25-C29 package pytest main/package contracts; C32 configured continue stop using real retries; no topology transition |
| Cached material/provider workflows | O01/O02/O03a/O03b/O04/O07 run whenever present; private TC-derived inputs/references remain ignored |
| pycalphad generation (Stage 2B relocation complete within O05/O08) | Installed providers/pycalphad.py; real restricted O05 and default-phase O08 sampling, seed 0, package examples import/read guard; O08 has two valid tie lines and one invalid probe, bitwise fixture/metadata gate |
| TC-Python live generation | Stage 2C complete within full-only O06: installed batch-safe adapter, real guarded sampling, approved suffixed local input, six steps and unseeded repeat pair; exact arrays, only metadata cache path normalized |
| Unaccepted workflows | Arbitrary plotting/results/restart/coupling, Numba fallback, other kawin versions, alternative moving-boundary solvers and curvature/precipitation APIs are outside this gate |

## Ownership and provenance

Original whole-module solver/numerics/mesh/equilibrium/provider/domain copies map
to 06dc7b4 in relocation-map.json and BINARY_SLICE.md. Stage 2B generation maps
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

**Status:** open; high priority before release. **Scope:** binary and two-phase
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

### R08: binary convergence tolerance depends on length units

**Status:** open; high priority before release. **Scope:** binary fixed-point
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

### R09: sparse recording mixes live state with recorded snapshots

**Status:** open; medium priority. **Scope:** binary history classes, shared
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

### R10: failed initialization or output preparation can leave partial state

**Status:** open; medium priority. **Scope:** all three setup methods; binary and
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

### R11: inherited deserialization does not restore the solver state

**Status:** open; medium priority. **Scope:** binary and two-phase ternary
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

### R12: ternary numerical controls lack consistent validation

**Status:** open; medium priority. **Scope:** both ternary configuration validators;
compare binary's existing checks and three-phase solve-argument checks.

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

NOTICE retains upstream MIT text. Repository history classifies local additions
without proving independent authorship; the binary code identifies a MAP C++ port.
Local/MAP licensing, outreach and clearance remain unresolved. No PDF, C++ source,
digitized figure data, database, trained model or TC-derived reference is packaged
or committed. No external legal terms were reverified or external contact made.

Package name, audience/distribution, release/maintenance ownership, pushing,
upstream contribution, MAP outreach and ternary prompt-default changes remain
user decisions. Scientific repairs have one authoritative source and deliberate
port records; baseline refreshes are separate reviewed changes. Do not maintain
independent scientific fixes or regenerate fixtures to hide a discrepancy.
