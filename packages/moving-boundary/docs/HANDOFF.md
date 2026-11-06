# Extraction handoff

## Current state

- Extraction branch: refactor/package-extraction. The intermediate binary slice
  is committed in 945dff3 and its prior verification in c970f85.
- Approved quick/full inputs and implementation order are in
  ACCEPTANCE_CANDIDATES.md. Approval is complete; do not ask again to implement it.
- Definition of done and whole-module/public-kawin milestones are in
  NEXT_SESSIONS.md and PUBLIC_KAWIN_MILESTONE.md.
- Frozen older numerical source/fixture: ae9fb2971a9fd49132f71ddc883a5bd5406644b6.
  The extraction checkout's kawin/, required examples and root packaging are frozen.
- Approved reference: clean detached worktree at
  06dc7b4ab56f67226ceabfde41ebf2690c088841. Each reference worker must assert
  kawin.__file__; do not import from the user's main checkout.
- Temporary package dependency and local main:
  5f4ad574f2b7348ee2f818b1a18b75c0c0325c2e. The separate fix branch removed
  only the two unused debugger test stubs (12 lines); production numerical code is
  unchanged. It was fast-forwarded into main; research branches were not merged.
- The package still subclasses the fork binary model and overrides two bulk
  methods. Complete binary, ternary two-phase and three-phase relocation remain
  required, in separate parity-checked commits. Keep host loop/mesh/recording and
  thermo services; shared local helpers belong at package level.

## Verification evidence

- Prior numerical verification: 15 package tests, 33 binary history arrays plus
  12 bulk probes exact in separate installed -I -S processes; all nine old source
  cases match the committed fixture. See verification.json for the actual tested
  source/dependency hashes. EOF-only checkpoint changes are explicitly recorded.
- Batch guard/tooling tests: 12 passed in 1.19 s. Workers prohibit input() before
  imports and explicitly inject raise/continue in case constructor ASTs, without
  modifying frozen sources or production module globals. Prompt defaults persist.
- Narrow test-stub cleanup: two retry tests passed in 5.81 s without stubs.
- These are scoped checks, not completion of the approved suite or Stage 1B.
  No new main-reference fixture capture or public-kawin acceptance is claimed yet.

## Approved suite implementation

tools/acceptance.py quick passes 29 separate run comparisons and all 22 package
pytest tests (including C25–C29) in 142.58 s with existing private wheels.
Full passes 38 run comparisons and the same 22 tests in 197.80 s. O06 is skipped
explicitly because the TC adapter remains unsafe. Enabled caches were all present.
All output quantities, original binary fixtures and bulk probes match exactly.
See acceptance.json for sanitized evidence; TC-derived reference arrays stay local.

Real pycalphad sampling took 4.94 s and runs afresh in O05. Initial unseeded
reference-vs-reference sampling varied; fixed Python hashing and NumPy seed 0
restored exact repeatability. The old fixture is archived locally, no tolerance
was widened, and the detached source is unchanged. Workers use -P -S with Python
path variables removed so the hash seed is honored while retaining safe paths and
no editable/site hooks. Every reference worker asserts kawin.__file__.

The regular/irregular binary, opposite/drift similarity, interstitial and volume
runs have reference fixtures. F04/F11/F12 retain longer drift and independent
analytic errors. C32 reaches configured terminal stop through actual retries,
with continue explicit and no fake solve callback. It is a terminal-retry stress
setup; phase-disappearance support remains outside scope. Explicit unequal molar
volumes in binary and ternary two-phase remain unsupported.

## Next execution

1. Relocate complete binary, ternary two-phase and three-phase local modules,
   each as an import-only mechanical move with a separate passing quick gate and
   commit. Shared private helpers belong in a package-level module; remove the
   intermediate inheritance from the fork's binary Illingworth class.
2. Add bounded package compatibility and a separate stock-kawin 0.5.0 environment,
   preserving host loop/mesh/recording and thermo services without monkeypatches.
   Require no fork-only imports and the preregistered public tolerances.
3. Run full including cached research cases and analytic drift after the public
   switch; report the runtime estimate first and preserve all private fixtures.

All Python, pytest, package-inspection and setup commands inherit the required
Conda activation wrapper and use actual subprocess timeouts, noninteractive stdin
and local temp directories. No shared editable reinstall, dependency upgrade,
push, publication, research-branch merge or external contact is authorized.
AGENTS.md remains untracked. Source mappings/attribution and numerical docstrings
must stay synchronized. Existing mixed tooling line endings are preserved;
the README/handoff have been rewritten as requested using their dominant CRLF.

## Open decisions and limitations

Package name/audience, pushing, MAP outreach, upstream contribution and changing
the terminal prompt default remain user decisions. No rights clearance is implied
by copying local/MAP-derived code. The TC adapter still imports/calls debugInPlace
and must lose that dependency on promotion in Stage 2C. Stock host/provider parity,
the full capability matrix, live TC availability and complete fallback acceptance
remain unproven until separately tested.
