# Capability and dependency ledgers

## Current capability inventory

Quick: 29 run comparisons and 41 pytest tests passed in 435.73 s. Full: 38 run comparisons and 41 pytest tests passed in 495.45 s.
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
| pycalphad generation | O05 real seeded sampling each worker, 4.94-second pilot; raw samples and solver outputs compared |
| TC-Python live generation | O06 disabled; examples/ThermoCalc/tc_python_adapter.py still imports/calls debugInPlace. Remove at Stage 2C before promotion |
| Unaccepted workflows | Arbitrary plotting/results/restart/coupling, Numba fallback, other kawin versions, alternative moving-boundary solvers and curvature/precipitation APIs are outside this gate |

## Ownership and provenance

Every local solver/numerics/mesh/equilibrium/provider/domain file is mapped to its
06dc7b4 source path in relocation-map.json and BINARY_SLICE.md. Shared private
helpers live in shared.py. No fork solver inheritance remains. Public unmodified
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
at ae9fb29. Inputs and tolerances were frozen before comparisons. Only O05
needs the narrow hash exception described in R01 below. The original fixtures
remain unchanged; NumPy seeding has been removed from every worker.

## Research reproducibility issue R01 (open)

Only O05 differed in this suite: 42 other cases (33 quick runs/contracts plus nine
full additions) each matched in two fresh processes with neither seed override.
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
seed alone failed to restore repeatability. Only PYTHONHASHSEED=0 remains for O05;
no worker calls np.random.seed. Other cases explicitly require no hash override.

Affected research workflow at main 06dc7b4: surrogate generation in
kawin/diffusion/MovingBoundarySurrogates.py:1810/:887 and the restricted provider
factory examples/ternaryExamples/pycalphad_default_phase_adapter.py:86 delegates
to these third-party models. Rebuilding pycalphad-backed training data can propagate
roundoff into sampled endpoints/matrices and then solver histories. Analytic,
saved-surrogate and cached-table solves do not invoke that model generation.
Default-phase generation shares the dependency but was not separately diagnosed.

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
and the frozen extraction was not merged forward. Live TC remains disabled.

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
