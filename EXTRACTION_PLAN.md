# Incremental extraction plan for the Illingworth moving-boundary package

The current execution list is [NEXT_SESSIONS.md](packages/moving-boundary/docs/NEXT_SESSIONS.md).
The source-backed Stage 1A bridge and bounded substitutional baseline are verified.
Stages 0 and 1B remain open; installed-fork import fails on missing examples.
See the package handoff for historical results.

**Recommended sequence:** establish a baseline from local `main`; package the existing implementation with kawin required; make that dependency work against a genuine upstream release; separate surrogate construction and providers; extract numerical state and stepping; replace the host driver and results only after parity is demonstrated. Decide whether to make kawin optional after those steps.

The goal is one installable scientific package that exposes the supported solver, material-data generation, surrogate construction, diagnostics, and result workflows. Internal separation must not reduce the intended user-facing functionality.

## 1. Starting point and scope

### Verified repository state

| Reference | Observed commit | Meaning |
|---|---|---|
| Local `main` | `ae9fb2971a9fd49132f71ddc883a5bd5406644b6` | Requested starting point; October 16, 2026 merge of the interstitial branch |
| Current research branch, `feature/DICTRA-comparison` | `da398e3294860abaea22d3e15f381feac27c353e` | Later research and comparison work |
| Cached `upstream/main` | `f0c347967fcfc18f6615cec205f584573606e100` | Public kawin snapshot used in the preceding audit; not a fresh remote-head guarantee |
| Shared ancestor of the research development and that upstream snapshot | `279cedfe0d5f8c12f69beb8c8579f30ac5c41b08` | Separates local development from subsequent upstream changes |

At preparation time, local `main` is an ancestor of `feature/DICTRA-comparison`. The diff from `main` to that branch contains 47 changed files, primarily DICTRA comparison additions and interstitial/ternary example and test changes. The inspected production directories `kawin/diffusion`, `kawin/thermo`, and `kawin/solver`, together with `GenericModel.py`, `Constants.py`, `pyproject.toml`, and `.github`, have no differences between those two commits.

Consequently, starting from `main` preserves the core implementation examined in the dependency assessment. Later comparison material should be brought over selectively if it becomes part of the extraction acceptance suite. Do not merge the entire research branch merely to obtain one reference calculation.

The upstream baseline remains [materialsgenomefoundation/kawin](https://github.com/materialsgenomefoundation/kawin). Local `main` is a research-fork baseline, not upstream kawin. Files located under the package directory are not thereby upstream functionality.

The current checkout contains untracked files, including local configuration, research documents, generated artifacts, and DICTRA reference JSON files. A new branch/worktree starts from committed content. Review required untracked inputs individually; do not automatically add all of them to the extraction branch.

### Scientific functionality to preserve

Build a capability inventory before moving code. Initially include:

- Binary, ternary two-phase, and ternary three-phase Illingworth solvers.
- Transformed grids, conservative interface balances, nonlinear solves, retry behavior, and supported timestep schedules.
- Phase-uniform, composition-dependent lagged, and composition-dependent implicit diffusivity modes where supported.
- Supported interstitial/u-fraction behavior and supported phase-molar-volume behavior, with their existing restrictions.
- Tie-line closures, sampling, parameterization, diffusivity sampling, surrogate construction, interpolation/domain policies, and surrogate merging.
- Open pycalphad workflows and optional licensed TC-Python workflows for generating the material data.
- Surrogate persistence, artifact verification, provenance, diagnostics, plotting, and benchmarks.
- Initialization, reset, repeated setup, continued solves, recording, saved-result loading, and supported coupling behavior.

Do not assume that every combination of features is supported merely because individual features exist. Record permitted combinations explicitly. In particular, keep the current geometry, boundary, temperature, composition-basis, and molar-volume assumptions visible. A terminal thin-phase stop is not phase-disappearance/topology-transition support.

Other local solvers, including the alternative moving-boundary and Olaye implementations, are outside the initial Illingworth extraction unless explicitly included in the capability inventory. Document this scope decision; do not accidentally promise or discard those methods through broad imports.

## 2. Git workflow and baseline management

Use a branch from **local `main`**, preferably in a separate worktree. The following commands are a proposed procedure; creating this document does not execute them or create a branch.

```powershell
git status --short
git log -1 --format="%H %s" main
git worktree list
git worktree add -b refactor/package-extraction '..\moving-boundary-extraction' main
```

Before running the final command, check that the branch and destination do not already exist. If `main` has advanced, explicitly decide whether to use the new tip or the recorded baseline commit above. Record the selected SHA in the extraction ledger. Do not silently update the baseline as development proceeds.

Keep kawin/, required example imports, fixtures and packaging inputs frozen at
the recorded reference in this extraction worktree. Package-only commits must
remain verifiable. A selected maintained-fork fix gets a separate source reference;
do not silently alter the baseline. Numerical fixes are authored once on the
active research branch and deliberately ported with source commit, rationale and
tests recorded. Baseline refreshes are separately reviewed scientific changes.
See NEXT_SESSIONS.md for the concurrent-research policy.

Keep the research checkout available for reference. The new worktree has its own branch and working files but shares Git history. Untracked `AGENTS.md` and local configuration do not automatically follow it: deliberately carry the relevant development instructions into the new workspace. This plan is also initially untracked; transfer or commit this file deliberately when establishing the extraction branch.

Use a separate baseline worktree or a fixed reference artifact when comparing old and new behavior. Avoid editing the baseline implementation to make a comparison pass. If a baseline bug is discovered, record a separate scientific change, establish an independently justified expected result, and decide whether to fix both implementations in an explicit follow-up.

Prefer commits with one reviewable purpose:

1. Characterize an existing contract or reference calculation.
2. Move or isolate an implementation while preserving behavior.
3. Change a dependency or interface.
4. Change numerical behavior only when that change has its own rationale and validation.

Git infers renames from similarity. `git mv` helps perform moves, but does not create a special permanent rename record. Keep mechanical moves separate from substantial edits when feasible. A temporary extracted copy may be needed while the original remains the reference; record its origin and make one side authoritative rather than developing two competing implementations.

Do not rewrite the existing repository history. When a separate public repository is ready, perform any selective history extraction in a disposable clone. Preserve attribution and a mapping to original paths/commits even if the final history is curated.

## 3. Packaging strategy during the transition

Use a distinct distribution name and import namespace. The examples below use `moving_boundary` only as a placeholder; package-name availability and the final project identity are not decided.

Keep the existing root project intact initially. Add a separately buildable project:

```text
packages/moving-boundary/
  pyproject.toml
  README.md
  LICENSE
  NOTICE
  src/moving_boundary/
    __init__.py
    solvers/
    numerics/
    surrogates/
    sampling/
    providers/
    diagnostics/
    results/
    compatibility/
  tests/
  examples/
  docs/
```

Configure explicit package discovery under that project's `src/` directory. Its wheel must not include or overwrite the `kawin` namespace, publish under the distribution name `kawin`, or accidentally include the surrounding research checkout. Build this project explicitly; running a build at the repository root would still build the forked kawin project.

Maintain one release stream initially. Backend adapters can be modules in the same distribution, with optional dependencies later. A second repository or independent adapter release should solve a demonstrated ownership or compatibility problem, not be an initial requirement.

Distinguish three installation states:

| State | Meaning | Release claim permitted |
|---|---|---|
| Development bridge | New namespace forwards to or imports functionality from the local fork | Internal extraction checkpoint only; explicitly requires the identified fork |
| Upstream-dependent package | New package owns its functionality and works with a tested upstream kawin wheel plus declared dependencies | Can be released once the other scientific/provenance gates pass |
| Optional-kawin package | Analytic, saved-surrogate, and TC-Python paths run without kawin; the pycalphad provider may still require it | Claim independence only for the workflows actually tested without kawin |

A development bridge that imports `kawin.diffusion.MovingBoundaryIllingworth...` does not work with public kawin merely because its metadata says `dependencies = ["kawin"]`. Public kawin does not contain those local classes. Do not distribute such a bridge as an ordinary upstream-compatible package.

## 4. Stage 0: freeze the baseline and define acceptance evidence

**Purpose:** make extraction measurable before changing architecture.

**Work**

- Record the selected local `main` SHA, upstream comparison SHA, environment versions, numerical acceleration status, and relevant database/input hashes.
- Create a capability ledger with columns for feature, current implementation, representative case, supported assumptions, target API, and completion status.
- Create a dependency ledger separating unchanged upstream code, upstream-derived local modifications, local additions, and third-party/reference implementations. Git addition status alone is not evidence of independent authorship.
- Select small deterministic cases for all three solver configurations, supported diffusivity modes, and supported interstitial/volume variants.
- Establish one open pycalphad material-generation workflow and one TC-Python workflow. Use fake/provider-contract tests for ordinary CI; identify a licensed environment for live TC-Python acceptance checks.
- Inventory provenance restrictions before redistributing reference sources, PDFs, digitized data, databases, or trained material models.
- Identify the few later DICTRA comparison files needed from `feature/DICTRA-comparison`, if any, and record deliberate cherry-picks or adaptations. The untracked reference bundles are not part of either committed baseline.

**Useful existing tests**

Start from `test_illingworth_substitutional_baseline.py`, `test_illingworth_setup_and_diagnostics.py`, the binary-variable-diffusivity tests, ternary and three-phase tests, similarity/convergence tests, interstitial tests, `test_transformed_grids.py`, `test_mesh_data.py`, `test_solver.py`, surrogate artifact/domain tests, and provider tests. Review actual assumptions rather than copying every test unchanged.

The existing substitutional baseline test skips when its fixture is missing. Required release evidence must fail clearly when a required open fixture is absent. A skipped test is not evidence of parity.

Before numerical moves, record relevant interstitial tests and at least one
deliberate rejected-step rollback/retry case. A happy-path run does not establish
failure behavior.

**Exit gate**

- Every promised capability has an acceptance case or an explicit validation gap.
- At least one reference run per solver completes and produces a recorded comparison artifact.
- Required fixtures are available from a clean checkout or reproducible open recipe.
- Baseline failures are documented rather than hidden by regenerating expected results.
- The environment and provenance ledger identify the source of every release-critical input.

**Suggested commits:** `docs: record extraction baseline and capability ledger`; `test: capture small solver and provider reference cases`.

## 5. Stage 1: package the existing implementation with kawin required

**Purpose:** establish package identity and installation without simultaneously replacing the host framework.

### Stage 1A: development bridge and package skeleton

- Add the nested `src/` project, explicit build configuration, minimal documentation, and a small public API.
- A short-lived facade may delegate to the local implementation to establish user-facing examples. Label this development-only and record the exact required fork baseline.
- Move or extract the three solver classes and their directly required local helpers into the new namespace in small steps. Preserve their current numerical equations and defaults.
- Continue using kawin's model/mesh/solver infrastructure during this stage.
- Decide explicitly how to provide the locally added `CartesianFD1D` and transformed-grid helpers; public kawin does not supply these additions.
- Replace imports between the extracted local modules with imports in the new namespace. Keep provider imports out of package initialization when they are not needed by the requested operation.
- Remove production dependencies on `examples.debugInPlace`, debugger waits, and import-time debugger reloads. Use structured exceptions and optional logging. Keep intentional interactive policies explicit and provide batch-safe configurations.

Do not retain a broad import of the local `kawin.diffusion` export list as a shortcut. It imports unrelated local functionality and obscures which dependencies the new package actually needs.

**Stage 1A gate:** the new project builds a wheel and runs a representative development workflow under its own import namespace. The documentation accurately states any remaining dependence on the local fork. This is not yet upstream compatibility.

Before choosing the next code move, perform a bounded host-dependency assessment.
Compare a narrow maintained fork fix, extension-owned helpers, released-kawin
compatibility and a minimal driver. Removing production debugger dependencies is
required: cover both import-time dependencies and ternary/three-phase retry
exhaustion, exception propagation and accepted-state preservation. The choice is
where to implement the removal, not whether to remove it. Consider upstream
contribution without assuming acceptance or copying all host infrastructure.

### Stage 1B: replace fork dependence with released-kawin compatibility

Prove the extracted package works with an unmodified, explicitly selected upstream kawin release. The earlier audit found public release 0.5.0, but the supported version range must be chosen and tested at implementation time rather than assumed from that observation.

Use a small `compatibility/` layer for missing host behavior while retaining the numerical implementation:

| Local behavior currently relied upon | Transitional action |
|---|---|
| Changed timestep minimum/final-remainder policy | Supply an extension-owned solve wrapper or driver compatibility implementation; do not assume upstream `GenericModel.solve()` accepts a custom solver class |
| Changed initial composition validation and floor | Override or wrap initialization; pass explicit constraints; preserve the interstitial reset correction |
| `MeshData` preallocation and time lookup changes | Provide a compatible recorder or narrow extension; preserve exact-time restrictions where needed |
| `MultiTherm.returnMeta` | Use provider-owned equilibrium metadata extraction or an explicitly maintained extension |
| Same-state kinetics diagnostic capture | Seek a supported upstream hook; otherwise maintain a narrow, attributed, version-tested compatibility implementation |
| Upstream surrogate forwarding changes | Include only where an inventoried workflow actually needs those surrogate types |
| Different gas constants | Record the convention and assess affected outputs; do not silently patch upstream globals |

The generic model's local change is documentation, not a reason to copy that whole class. Your custom ternary surrogate does not inherit upstream's general surrogate class, so do not copy that subsystem by default either.

Prefer composition and narrow adapters. If a method must be copied, preserve notices, record the exact upstream source and local changes, identify an owner, and create a check for upstream changes affecting that method. Do not represent an attributed copy as an independent implementation. Avoid process-wide monkeypatching of kawin internals.

**Stage 1B gate**

- Built-wheel installation succeeds with a genuine upstream kawin installation and declared dependencies.
- None of the package's imports require the local fork's additions to `kawin`.
- Small binary, ternary two-phase, and ternary three-phase runs meet the parity criteria.
- Representative surrogate construction, loading, and evaluation work under the same installation.
- Public-kawin compatibility is documented separately from known remaining private-API dependencies.
- Provenance and license obligations for extracted code are recorded.

**Suggested commits:** `build: add standalone package skeleton`; `refactor: move local solver modules into package namespace`; `compat: isolate fork-specific host behavior`; `test: verify wheel against released kawin`.

If Stage 1B proves disproportionately difficult, record which dependency causes that difficulty and advance the corresponding later extraction stage. Do not broaden the claimed compatibility or silently substitute the fork for public kawin.

## 6. Stage 2: separate providers, sampling, and surrogate models

**Purpose:** preserve the complete material-generation workflow while removing backend assumptions from the reusable surrogate implementation.

### Stage 2A: define explicit provider contracts

Define separate contracts for equilibrium sampling, transport sampling, and optional detailed diagnostics. Avoid one large provider class requiring every backend to implement unrelated operations.

Contracts must specify:

- Elements in caller order, dependent component, and substitutional/interstitial roles.
- Phase identifiers and allowed phase universe; distinguish requested interface phases from all phases allowed during equilibrium.
- Temperature and other supported state variables, with explicit units.
- Independent versus full composition arrays and mole-fraction versus u-fraction basis.
- The diffusion matrix's row/column order, reference component, flux frame, and basis.
- Stable/global equilibrium versus phase-local/metastable calculations.
- Phase-labelled endpoint metadata, invalid-region behavior, duplicate composition-set handling, and numerical failure behavior.
- Batch shapes, cache controls, and optional diagnostic capabilities.

The material-data provider's equilibrium query is different from the solver's eta-to-endpoints closure. Keep these interfaces distinct. Do not adapt them by guessing from a method name such as `getInterfacialComposition`.

### Stage 2B: retain kawin for the pycalphad transport engine

Move the local default-phase adapter into the package. Reuse kawin for model construction, local equilibrium, mobility evaluation, constrained thermodynamic derivatives, and diffusion-matrix construction.

Keep full-phase validity checks and phase-labelled metadata in the adapter. Do not simply assign the requested labels to two arrays without checking the equilibrium state. The existing `PycalphadDefaultPhaseThermodynamics` code is a useful starting point because it already constructs metadata from composition sets.

Preserve kinetics diagnostics from the same equilibrium state used for the diffusion matrix. A second equilibrium calculation is not an equivalent implementation of that guarantee. If upstream does not supply the needed hook, retain a clearly identified compatibility implementation and its version-specific tests.

Do not independently rebuild `MobilityModel`, `FreeEnergyHessian`, or local-equilibrium machinery as an incidental part of this extraction. That is a separate scientific/maintenance decision.

### Stage 2C: move the TC-Python provider

Promote `examples/ThermoCalc/tc_python_adapter.py` and the reusable sampling/checkpointing functionality into package modules. Replace its three direct upstream utility dependencies with the package's input normalization and phase conventions, or retain small attributed copies where justified.

Keep TC-Python imports lazy. The adapter is distributed with your package; Thermo-Calc, its SDK, licenses, and databases are supplied separately by the user. Clearly report unavailable backend capabilities.

Preserve backend session lifecycle, cache behavior, retries, phase selection, kinetic conventions, and diagnostic records. An adapter with no direct kawin imports is not sufficient proof that its complete workflow is kawin-independent; the builder and solver paths must also be checked.

### Stage 2D: extract the surrogate workflow itself

Separate these operations while retaining convenient end-to-end helpers:

1. Sample equilibrium and transport data from a provider.
2. Validate and store the sampled dataset with conventions and provenance.
3. Build tie-line and diffusivity surrogates from sampled data.
4. Evaluate saved surrogates without constructing the original provider.
5. Validate against explicitly supplied ground truth when requested.

Move the local interpolation schemes, domain checks, spectral checks, merging behavior, diagnostics, and artifact system. Remove automatic backend construction from generic evaluation/loading paths. A pycalphad-specific convenience factory may still create a kawin-backed provider.

Replace private `_backend.totalNumCalcs` access and synthetic backend attributes with optional statistics or instrumentation owned by the sampling layer. Preserve meaningful measurements; the numerical builder should not require a particular backend's private counters.

Do not equate moving data between backends with validating their physical conventions. Keep source metadata and conversion decisions attached to each dataset.

**Exit gate**

- Open pycalphad and optional live TC-Python workflows each generate data, build a surrogate, validate it, save/load it, and drive a representative solver.
- Synthetic provider tests cover phase ordering, extra phases, invalid regions, batch conventions, and failure propagation.
- Saved-surrogate evaluation does not need a live backend session or a thermodynamic database.
- Detailed diagnostics remain available where supported; unsupported fields are explicit rather than fabricated.
- No production package module imports helpers from the research `examples/` directory.
- Existing surrogate behavior and artifact checks pass the agreed parity and round-trip criteria.

**Suggested commits:** `feat: define material-provider contracts`; `refactor: move backend adapters and sampling`; `refactor: separate surrogate fitting from backend construction`; `test: validate provider-to-solver workflows`.

## 7. Stage 3: extract numerical state and a complete-step interface

**Purpose:** separate the numerical method from the host without changing the outer driver yet.

Define a state representation covering phase profiles, interface positions, tie-line coordinates, time, previous-step information, and geometry required by the formulation. Distinguish accepted state, trial candidate, and diagnostic output.

Identify every piece of history used by the numerical method. Previous interface displacements, extrapolation predictors, retry state, and schedule state may affect the next step. A state object containing only current profiles is not necessarily sufficient for equivalent continuation or restart.

Extract an interface conceptually equivalent to:

```text
attempt_step(accepted_state, requested_dt, problem, material)
    -> candidate, actual_dt, diagnostics
    -> or structured failure

commit_step(candidate)
    -> next accepted state
```

This is illustrative, not a finalized public API. Decide whether retry control belongs in `attempt_step` or the driver, then give it one owner. The accepted timestep and candidate must remain consistent. No generic host may rescale an already-solved implicit increment to a different timestep.

Continue adapting this step interface to the legacy kawin lifecycle during this stage. Move common utilities out of the ternary model so binary and three-phase solvers do not depend on its private class organization.

**Exit gate**

- The existing host executes the extracted step implementation with matching results.
- Rejected trials leave accepted state, histories, and callbacks unchanged.
- Accepted phase profiles, interface state, fluxes, and inventory diagnostics all describe the same candidate.
- Nonlinear initialization, motion direction, retries, and terminal thin-phase handling retain their documented semantics.
- Public APIs and numerically subtle internal helpers have current docstrings.

**Suggested commits:** `refactor: introduce explicit moving-boundary state`; `refactor: extract complete implicit step`; `test: verify accepted-state and rejection invariants`.

## 8. Stage 4: replace the host driver and results after parity

**Purpose:** own the simulation services that need moving-boundary-specific behavior, without copying the complete kawin framework.

### Stage 4A: independent driver

Implement initialization, time limits, timestep scheduling, retries if assigned here, accepted-step commits, stopping, progress callbacks, and finalization around the step interface from Stage 3.

Preserve or explicitly migrate:

- Whether solve time is a duration or an absolute final time.
- Repeated `setup()`, reset, and continued solves.
- Exact final-time handling and ordinary minimum-timestep failures.
- Semi-log schedules and any existing rounding/no-op behavior.
- Terminal thin-phase stop status and diagnostic details.
- Coupling notifications and observer order.
- Batch-safe failure behavior.

Running a generic Runge-Kutta or SciPy ODE routine is not a drop-in replacement for the current implicit recurrence. The driver should execute complete steps directly.

Keep the legacy-host adapter for comparison during the transition. Any change to legacy behavior should be a separately reviewed numerical/API decision.

### Stage 4B: independent inputs and results

Replace the subset of `DiffusionModel`, mesh, and `MeshData` services the solvers actually need:

- Explicit physical coordinates, initial profiles, phase order, and interface positions.
- Supported boundary-condition descriptions, independent of kawin class identity.
- Temperature inputs and numerical constraints with explicit units/defaults.
- Unified histories for scalars, arrays, phase-local profiles, geometry, and diagnostics.
- Physical-profile reconstruction and documented interpolation semantics.
- Versioned saved results and restart/checkpoint contracts where supported.

Use the existing algebraic composition conversion helpers after auditing their provenance and assumptions. Make component roles explicit while preserving compatibility defaults. Do not copy all of `Mobility.py` to obtain a short conversion or element list.

Retain kawin mesh/model adapters where they provide value to existing users. Preserve the ability to read old results where promised, or provide an explicit migration tool. Do not imply that interpolated display profiles preserve the same inventories as the underlying phase-local state.

**Exit gate**

- All three solver configurations execute with the independent driver.
- Comparing independent and legacy drivers on the same material inputs meets parity criteria.
- Continued solves, resets, accepted/rejected steps, terminal stops, and result round trips are verified.
- Required coupling behavior is tested through an adapter or explicitly marked as a remaining migration item.
- Open analytic and saved-surrogate cases run without importing kawin through any solver, result, utility, or initialization module.
- Unsupported combinations remain rejected rather than silently accepted by a more general-looking input API.

**Suggested commits:** `feat: add moving-boundary simulation driver`; `refactor: introduce independent inputs and results`; `compat: adapt kawin inputs and legacy result files`; `test: verify independent-driver parity`.

## 9. Stage 5: make dependency optionality an evidence-based choice

The package can be useful and releasable with kawin required. Optionality is a later gate, not a prerequisite that justifies a rushed rewrite.

Once Stages 2–4 pass, test the following proposed dependency arrangement:

| Workflow | Intended dependencies |
|---|---|
| Analytic material inputs and numerical solves | Core numerical dependencies |
| Build/evaluate surrogates from user-supplied arrays | Core numerical and interpolation dependencies |
| Load a surrogate and solve | Core dependencies, no live thermodynamic backend |
| Generate data with TC-Python, build surrogate, solve | Core plus separately installed licensed TC-Python environment |
| Generate data through the pycalphad provider | Core plus upstream kawin and its dependencies |
| Kawin model/mesh integration | Core plus upstream kawin |
| Plotting or acceleration | Appropriate optional visualization/Numba dependencies |

An extra named `pycalphad` may legitimately install kawin because kawin supplies the transport implementation for that workflow. State this clearly. A provider interface does not imply a kawin-free implementation behind it.

Audit direct dependencies such as `tqdm`, scikit-learn where actually used, plotting libraries, and Numba. Do not rely on transitive dependencies remaining installed accidentally. Keep package initialization minimal and test import failures with backends absent.

**Exit gate:** installation and documented examples are verified for each supported dependency combination in clean environments. A package metadata edit alone is not evidence of optionality.

## 10. Stage 6: public release and repository transition

A first public release can happen after Stage 2 if Stage 1B and all release/provenance gates pass. It may still require kawin. The independent driver can follow in a later release. Do not delay useful distribution solely to achieve dependency independence.

Before release:

- Confirm the package name, supported Python/platform versions, upstream kawin compatibility range, and optional-backend requirements.
- Complete the capability ledger or clearly document deferred features. Do not advertise full migration while essential diagnostics or material-generation workflows remain available only in the research checkout.
- Publish wheel/sdist installation instructions and small end-to-end examples for the open backend and TC-Python users.
- Run build/install tests outside the repository root and inspect wheel/sdist contents.
- Resolve applicable permissions for MAP-derived code and reference materials; preserve upstream notices for copied code.
- Separate software licensing from database, generated-data, and surrogate redistribution permissions.
- Provide citation metadata, contribution instructions, release notes, and named maintenance ownership.
- Archive the release and reproducibility materials, recording exact versions and input provenance.

Create the final standalone repository once the extracted package can be built without relying on the surrounding checkout. Transfer the relevant history or a curated history with source mappings. Keep the original research repository available as the development/provenance record; do not make its entire contents part of the new distribution.

Once released, designate the new package as the authoritative implementation. The research fork should consume it through an adapter or remain a frozen reference. Avoid implementing new solver fixes independently in both places.

## 11. Parity and scientific validation policy

Use separate checks for extraction fidelity and mathematical correctness. Matching the old implementation can preserve an old bug; matching one commercial simulation does not prove correctness either.

| Comparison | Required evidence |
|---|---|
| Pure relocation/import changes | Aim for exact outputs in the same environment; investigate differences before accepting them |
| Driver/state refactor | Profiles, interfaces, eta histories, inventories, accepted times, stop status, and retry behavior; explain any intended differences |
| Backend compatibility change | Sampled endpoints/matrices and diagnostics first, then surrogate and solver outputs |
| Surrogate refactor | Fit/evaluation behavior, domain rejection, interpolation modes, merging, and serialization round trips |
| Platform or accelerated/unaccelerated implementation | Explicit numerical tolerances and convergence behavior; do not assume universal bitwise reproducibility |
| Scientific correctness | Independent analytic/similarity cases, grid/time refinement, flux balance, conservation, and limiting cases |

Choose tolerances by quantity and scale, including units and dilute-component behavior. Record absolute and relative tolerances before reviewing the result. Preserve exact baseline tests where they are valid, but do not demand bitwise identity across unrelated BLAS/compiler/backend versions.

Minimum acceptance dimensions include:

- All three solver configurations and supported diffusivity modes.
- Supported substitutional/interstitial and molar-volume cases, with unsupported combinations tested for rejection.
- Stationary, positive, and negative interface motion where valid.
- Coupled cross-diffusion and disparate component scales.
- Failed nonlinear steps, rollback, timestep reduction, final remainders, and terminal thin-phase behavior.
- Repeated setup, reset, continued solves, and supported checkpoint/restart behavior.
- Phase identity, component ordering, x/u conventions, and transport reference-frame contracts.
- Surrogate domain errors, save/load, artifact integrity, and provider diagnostics.

Classify tests into a fast per-commit suite, a focused stage-acceptance suite, slower scheduled/release verification, and optional live licensed-backend tests. Target fast jobs under roughly 5–10 minutes and ordinary stage suites under 30 minutes. These are budgets to measure, not claims about current runtimes. Partition or explicitly justify longer studies before running them.

Keep small open reference cases mandatory. Licensed or externally restricted cases may be opt-in, but their absence must be reported. Fake-backend tests establish interface behavior, not equivalence to a real Thermo-Calc calculation.

## 12. Local execution, installation isolation, and timeouts

Follow the repository's Windows Conda activation instructions for every Python-based command. Never invoke the environment's `python.exe` directly to bypass activation. Native DLL configuration matters on this machine.

Example of the required activation form:

```powershell
cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && python -V'
```

Use repo-local pytest temporary directories and batch-safe solver options. Long-running suites must have an actual wall-time limit; a tool returning a session ID is not a timeout.

For a first bounded reference check, the following example activates the required environment and gives the pytest subprocess a ten-minute timeout. Its expected runtime must still be assessed; it is an example command, not a test already run for this document.

```powershell
cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && python -c "import subprocess,sys; subprocess.run([sys.executable, ''-m'', ''pytest'', ''kawin/tests/test_illingworth_substitutional_baseline.py'', ''--basetemp'', ''.pytest_tmp/extraction_baseline''], check=True, timeout=600)"'
```

Audit test helpers that spawn external processes and apply their own timeouts. A timeout on a parent process does not necessarily terminate an entire process tree. Avoid interactive prompts, debugger attachment waits, or unattended backend sessions in acceptance jobs.

Worktrees share the same machine environments. Reinstalling a different editable `kawin` into the shared Conda environment can change the behavior of the research checkout. Record the reference environment first and use controlled environments/CI for upstream-wheel compatibility experiments rather than casually overwriting the research installation.

For installed-package acceptance:

- Build the nested extraction project, not the repository root.
- Execute from a directory outside both source trees.
- Use tests configured to avoid source-tree path injection; `--import-mode=importlib` can help but is not by itself a complete isolation guarantee.
- Assert and record `kawin.__file__`, the new package's `__file__`, and distribution versions. Confirm that kawin comes from the intended upstream installation.
- Run reference and extracted implementations in separate processes when their imports would otherwise collide.
- Test optional-backend absence in genuinely clean environments, not by assuming an unused installed dependency has no effect.

The local activation rule remains in force for local Python commands. CI should have its own explicit, reproducible environment setup. Do not assume the checked-in Windows environment export is a portable release specification.

## 13. Provenance, ownership, and maintenance gates

Maintain a file/function-level ledger for material copied or adapted from upstream and third parties. Record source URL, commit, original path, applicable license, modifications, and update owner. Keep scientific-method citations alongside code provenance; neither replaces the other.

Kawin's MIT license permits reuse subject to retaining its notices. MAP reference code has separate published conditions, and the binary solver describes itself as a port. Resolve the status of affected implementation before applying a blanket permissive license. Removing a reference-code folder does not itself resolve the provenance of a port.

TC-Python adapter source, licensed databases, exported sampled data, and trained surrogates have different redistribution questions. Check applicable terms and permissions before publishing generated artifacts. Prefer open analytic or openly licensed material inputs for mandatory reproducibility examples.

For every retained compatibility copy, specify who watches upstream changes and how compatibility is tested. Favor upstream contributions for generally useful fixes or observation hooks. Do not assume those proposals will be accepted or block the entire extraction indefinitely on them.

Every implementation stage must keep docstrings synchronized with behavior, assumptions, arguments, failure modes, and numerical conventions. Preserve existing line endings and keep unrelated edits out of extraction commits.

## 14. Suggested milestone schedule and decision points

Calendar estimates are planning ranges, not commitments. Advance on the gates above, not elapsed time.

| Milestone | Indicative period | Review decision |
|---|---|---|
| M0: baseline and ledgers | Weeks 1–2 | Is the intended functionality reproducible and its provenance understood? |
| M1: own namespace and development bridge | Weeks 2–4 | Does the package boundary work without numerical changes? |
| M2: real upstream compatibility | Months 1–3 | Are compatibility patches narrow enough to maintain? |
| M3: provider/surrogate workflow | Months 2–4 | Can both backends support the complete documented workflow? |
| M4: first usable release, kawin required if needed | Months 3–5 | Are installation, scientific, documentation, and rights gates satisfied? |
| M5: state/step and driver/results extraction | Months 4–8 | Does independent execution preserve behavior and justify its maintenance cost? |
| M6: optional dependency arrangement and broader adoption | Months 6–12 | Which workflows genuinely benefit from removing kawin? |

Reconsider the plan if maintaining private thermodynamic hooks becomes extensive, if upstream offers active co-maintenance, or if new goals require fundamentally different physics or execution models. Do not expand into a complete independent thermodynamic framework merely because a small host abstraction was easy to replace.

## 15. First implementation session

The first coding session should aim to complete a small, reviewable slice:

1. Create the worktree/branch from the selected local `main` baseline and bring over the development instructions and this plan.
2. Record the baseline, capability inventory, and direct dependencies of one representative workflow.
3. Run a bounded baseline check and record import origins/environment versions.
4. Add the nested package skeleton and one public entry point under the new namespace.
5. Run the same small calculation through that entry point with the local fork still explicitly required.
6. Build and inspect the first wheel, documenting its development-only status.
7. Commit the package skeleton separately from any implementation movement.

The next session should identify and remove the first fork-only dependency needed for that workflow. Do not start by copying the entire `kawin` directory, replacing the thermodynamic engine, or rewriting all three solvers at once.

## Source and implementation references

These references support the preceding assessment; they are not claims that the future extraction has been tested.

- [Public kawin repository](https://github.com/materialsgenomefoundation/kawin) and [MIT license](https://github.com/materialsgenomefoundation/kawin/blob/main/LICENSE.txt).
- [Public kawin release information](https://pypi.org/project/kawin/).
- [MAP program provenance](https://www.phase-trans.msm.cam.ac.uk/map/kinetics/programs/ConservativeIFF.html) and [MAP redistribution conditions](https://www.phase-trans.msm.cam.ac.uk/map/donor/advice.html).
- [TC-Python product information](https://thermocalc.com/products/software-development-kits/tc-python/) and [Thermo-Calc guidance on generated datasets/surrogates](https://thermocalc.com/applications/applications-for-machine-learning/).
- Local model/host code: `kawin/GenericModel.py`, `kawin/solver/Solver.py`, `kawin/diffusion/Diffusion.py`, `kawin/diffusion/mesh/MeshBase.py`, and the three `MovingBoundaryIllingworth*FDM.py` modules.
- Local material workflow: `kawin/diffusion/MovingBoundarySurrogates.py`, `MovingBoundaryEquilibrium.py`, `SurrogateArtifacts.py`, `SurrogateDiagnostics.py`, `examples/ThermoCalc/`, and `examples/ternaryExamples/pycalphad_default_phase_adapter.py`.
- Local thermodynamic extensions: `kawin/thermo/Thermodynamics.py`, `MultiTherm.py`, and `Surrogate.py`, compared to the shared ancestor and upstream snapshot listed above.
