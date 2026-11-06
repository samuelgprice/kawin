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
