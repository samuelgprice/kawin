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

## Research numerical issue R06: binary interface iteration stalls above its tolerance (open, not investigated)

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
fine-grid iteration floor is related context, but not established as the same cause.

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
