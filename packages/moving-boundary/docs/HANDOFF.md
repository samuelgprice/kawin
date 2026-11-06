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

## Next execution

1. Freeze exact input variants and per-quantity tolerances before comparisons.
2. Create/assert the clean detached 06dc7b4 reference worktree and capture approved
   quick/full references there. No generated licensed/data-derived outputs may be
   committed. Keep the original research checkout's untracked files untouched.
3. Put lifecycle/failure C25–C29 into package pytest with separate main/package
   workers. Add irregular-grid, supported terminal-continue and independent
   exact/similarity errors. Explicitly document unsupported volume combinations.
4. Pilot real O05 sampling: quick/default if it fits about a minute, otherwise
   full. Full enables O01/O02/O03/O04/O07 when caches exist, skipping only missing
   caches. O06 stays disabled pending a batch-safe promoted adapter.
5. Capture/compare installed wheels after each whole-module move (binary, ternary
   two-phase, then three-phase). Quick including pytest must stay under ten minutes.
6. Build the package compatibility layer and verify stock kawin 0.5.0 in a separate
   environment. Keep 8.3145 explicit, minComposition 1e-10, and required host
   minimum/final-step, recording and initialization semantics without monkeypatches.

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
