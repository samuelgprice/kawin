# Definition of done

The end state: my Illingworth solvers (binary, ternary two-phase, ternary three-phase,
with interstitial and molar-volume support) live in this package. The package depends
on UNMODIFIED public kawin (a chosen release, currently 0.5.0) only for what it
really needs: the model/solver loop, mesh and recording, TemperatureParameters, and
the pycalphad/thermo side. Fork-only host behavior lives in the package's own
compatibility layer. A single command shows that the runs I actually use still give
the same results as main.

## Current checkpoint

The intermediate binary subclass and scalar kernel are committed in 945dff3;
the separate verification checkpoint is c970f85. The recorded run passed 15
package tests and exact installed-wheel parity for 33 binary history arrays and
12 nonuniform-grid bulk probes. All nine frozen source cases match their fixture.
Only EOF blank lines changed while checkpointing; verification.json preserves the
actual tested hashes and separately records those formatting-only adjustments.
These are prior results, not a newly executed acceptance suite.

The source reference remains ae9fb2971a9fd49132f71ddc883a5bd5406644b6 and the
installed dependency is 5f4ad574f2b7348ee2f818b1a18b75c0c0325c2e.
The approved-run suite will use main at 06dc7b4 as its reference. Do not silently
replace the earlier reference fixture. Public-kawin compatibility is unproven.

## Approval gate and implementation order

The concrete candidate list is ACCEPTANCE_CANDIDATES.md (review item 4a):
The list is approved with quick/full tiers, enabled-when-present local caches,
conditional default pycalphad sampling, detached 06dc7b4 references, independent
analytic drift metrics and lifecycle/failure contracts in package pytest.

Approval is recorded in ACCEPTANCE_CANDIDATES.md. Implement batch guards and
cleanup first, then freeze tolerances/inputs and capture detached references
before relocating solvers or changing the dependency. Item 4e requires
that suite for every subsequent solver move and the public-kawin switch.

After approval, write per-quantity tolerances before examining new results, then
capture outputs from a clean detached 06dc7b4 worktree and build the isolated acceptance runner.
Use exact comparison for mechanical relocation and installed-wheel processes.

### Whole-module relocation (review item 2; not yet implemented)

Replace the method-by-method subclass strategy with complete local modules.
The current subclass is an intermediate checkpoint, not the intended architecture.
Use mechanical copies with import changes, preserving operation order, numerical
docstrings, units and exceptions. Identify and document any unavoidable change
separately; do not hide a numerical repair in a relocation.

1. Binary commit: relocate MovingBoundaryIllingworthFDM.py and the helper modules
   it needs. The resulting class must derive from host DiffusionModel/GenericModel,
   never the fork's Illingworth class. Include the complete binary mesh/kernel
   helper module, TransformedGrids.py, the needed FDM1D.py CartesianFD1D additions,
   _illingworth_interstitials.py and dependency-graph-required local helpers.
   Keep host mesh base classes and MeshData supplied by kawin at this stage.
2. Ternary two-phase commit: relocate MovingBoundaryIllingworthTernaryFDM.py and
   its complete numerical/helper modules, including _spectral_validation.py and
   required MovingBoundaryEquilibrium services.
3. Three-phase commit: relocate MovingBoundaryIllingworthTernaryThreePhaseFDM.py
   and any remaining local helpers. Do not introduce a subclass of a fork solver.

Move shared private helpers to a package-level module rather than importing from
the ternary solver: diffusivity-mode constants/coercion for binary; scalar/array
histories, finite-difference perturbation, eta bounds, matrix/stack validation,
log schedules and shared constants for three-phase. Determine the exact closure
from imports and call sites, rather than copying all of kawin.

For each commit, record every source path and commit in BINARY_SLICE.md/LEDGERS.md.
Select debugger-free solver sources explicitly at 06dc7b4 where needed; retain
ae9fb29 as the older frozen fixture source and document their narrow difference.
The approved main-reference suite, existing fixture and bulk probes must pass
exactly in separate processes against the privately installed package wheel.
Retain the host model/solver loop, mesh/recording and thermodynamic engine.

### Stage 1B: unmodified public kawin (review item 3; not yet implemented)

After all three whole-module relocations pass the approved main-reference suite,
implement the bounded compatibility/ layer described in PUBLIC_KAWIN_MILESTONE.md.
The first chosen public release is kawin 0.5.0. Retain its model/solver loop,
mesh/recording, TemperatureParameters and pycalphad/thermo services; put required
fork-only behavior in package-owned adapters without monkeypatching kawin.

Acceptance requires a privately installed wheel with stock kawin 0.5.0 in a
separate environment, no fork-only imports, and every approved run/quantity
passing its pre-recorded tolerance (explicit opt-in skips remain visible).
No new environment, compatibility code or dependency switch has been built yet.
The shared kawin Conda environment is unchanged.

## Open decisions reserved for the user

- Final package name.
- Intended audience and distribution/maintenance model.
- Pushing branches or publishing artifacts.
- MAP licensing outreach.
- Upstream contribution.
- Changing either ternary solver's thin-phase prompt default.

No push, research-branch merge, upstream contact or MAP-author contact is authorized.
The shared kawin Conda environment must not be repackaged or reinstalled.

## Baseline and concurrent research policy

- The reference is commit ae9fb2971a9fd49132f71ddc883a5bd5406644b6.
  Keep kawin/, its required example imports, fixtures and packaging inputs frozen
  in this extraction worktree. New verification and API work lives in packages/.
- This is not a universal requirement to copy every dependency. If a maintained
  fork fix is selected, implement it separately with its own SHA and evidence;
  retain the original reference for comparison and update dependency metadata
  explicitly. Do not silently patch the reference or use import monkeypatches.
- During extraction, scientific fixes are authored once on the active research
  branch (currently feature/DICTRA-comparison, or its designated successor).
  Port required fixes deliberately to extracted code. Record source commit,
  affected functions, reason, tests and expected numerical changes in a fix
  ledger. Package metadata/tooling fixes belong to extraction. Fork packaging
  fixes such as debugger removal live on a narrow branch from the recorded
  reference, are tested and committed there, and are merged into local main.
  Pin that exact commit without merging it into the frozen extraction source.
  Existing research branches receive the shared fix through a deliberate merge
  from main; they are not rewritten automatically. Never automatically merge
  research HEAD or regenerate fixtures to hide a numerical change.
- A baseline refresh is a separate reviewed scientific change. After release,
  explicitly transfer numerical authority to the new package and adapt research
  consumers; avoid indefinite independent implementations.
- Preserve floating-point update order (including derivative/step round trips),
  gas constant 8.3145 where used, and composition/flux conventions. Temperature
  arrays use hours; do not generalize that to every callback's time argument.
- Investigate MAP-port provenance before publication. No permissions, legal
  clearance, package-name availability or upstream acceptance is assumed.