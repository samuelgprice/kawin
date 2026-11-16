# Extraction handoff

## Current checkpoint

- **Last pushed revision:** `522c695` on `origin/refactor/package-extraction` (pushed 2026-11-15), the
  pre-transfer checkpoint. C33 (quick) and F33 (full_drift) are active, and five approvals are selected.
- **Local commits after it** (not pushed as of writing):
  - `9e9464b` adds `tools/acceptance_transfer.py`, a hash-checked export/check/restore of the git-ignored
    local acceptance artifacts. Validated with full `tests/` (1847 passed, 5 skipped, 1 deselected), a
    real export/check/restore smoke check, and exact quick in the restored clone (31 runs and 5 contracts
    bit-identical).
  - `b2528a6` keeps digests and private contents out of transfer refusals, and refuses a boolean or
    negative bundle-entry `size`. Validated with 196 focused tests; the full suite and numerical gates
    were not repeated.
  - `fa22e73` and `f78a13a` port surrogate diagnostics and saved-surrogate comparison
    ([FUTURE_PORTS.md](FUTURE_PORTS.md) entries 1-2) with an optional Plotly `[diagnostics]` extra.
    Validated with oracle parity (bit-identical), fresh-target full `tests/` (1918 passed, 5 skipped,
    1 deselected), and exact quick and exact full acceptance (bit-identical).
- **Details, current gate counts and open follow-ups:** [NEXT_SESSIONS.md](NEXT_SESSIONS.md) "Current
  checkpoint" and "Next scoped work", and [LEDGERS.md](LEDGERS.md).

## State at the public-kawin milestone (historical, last edited 2026-11-10)

This section and the ones after it record that time. Their acceptance figures, the skipped O06 and
"No push" no longer describe the current state.

The extraction branch is refactor/package-extraction. Whole binary, two-phase
ternary and three-phase modules are package-owned, committed separately as
8233ad2, e08eed0 and 693595d after exact quick gates (29 runs plus 22 tests each).
The public-kawin milestone is implemented and accepted by the frozen quick/full
suite. Runtime metadata requires stock kawin==0.5.0, not the research fork.

The separate offline environment is `mb-public-050` beside the extraction worktree.
It holds stock kawin and the tested dependencies only; the package itself is never installed
there (its editable worktree install was removed on 2026-11-11, see README "Local acceptance").
All 73 stock files match the cached 0.5.0 wheel byte for byte. Its LAPACK check
passed. The package wheel installs privately beside that host. The original shared
kawin environment was not reinstalled or upgraded. Every Python operation uses
the required activation wrapper and bounded, noninteractive subprocesses.

Small compatibility subclasses preserve fork-only policy while inheriting public
iteration, reset, recording, temperature handling and equilibrium/interdiffusivity.
Provider modules are complete local additions; metadata and explicit-R tracer
wrappers are bounded adaptations, not a thermodynamic engine copy. See
PUBLIC_KAWIN_MILESTONE.md and relocation-map.json. No monkeypatches are used.

## Acceptance evidence

Quick: 29 run comparisons and 41 pytest tests passed in 435.73 s. Full: 38 run comparisons and 41 pytest tests passed in 495.45 s.

All cached inputs were present and enabled. Only O06 was SKIPPED: the TC adapter
still imports/calls debugInPlace and is disabled until batch-safe. Exact/similarity
errors are saved for C04/C11/C12 and F04/F11/F12, plus irregular Fig. 4 C31.
C32 reaches a configured terminal retry stop under explicit continue with real
retries, not a fake callback. Binary/two-phase unequal volumes are unsupported;
three-phase volumes and all supported interstitial families are covered.

References and C25-C29 contract outputs come from clean detached 7af24f4 (moved from
06dc7b4 via 90700be and 720d122 on 2026-11-09/10; the O08 fixture stays from 06dc7b4, the O06 fixture was
recaptured from 720d122; see LEDGERS.md R06);
every reference worker asserts its origin. The older ae9fb29 binary fixture and
12 bulk probes remain checked. Source inputs/tolerances were frozen first. Real
O05 sampling runs afresh with only its Python hash seed fixed; third-party
pycalphad ideal-mixing construction/CSE causes hash-order-dependent roundoff.
NumPy seeding is removed. All other quick cases/contracts repeat twice unseeded;
see LEDGERS.md R01 for exact cause, controls and the proposed research-branch fix.
Raw reference/provider outputs and their hashes remain ignored and private.
acceptance.json records sanitized results and separate relocation-stage evidence.

## Repository boundaries

Local main and fix/remove-debugger-imports are at 5f4ad57. That narrow test cleanup
removed 12 obsolete stub lines and passed two retry tests before the authorized
fast-forward; production numerics did not change. It was never merged into this
extraction's frozen kawin/examples/root packaging at ae9fb29. The reference
worktree remains detached and clean. Original untracked research work is preserved;
AGENTS.md stays untracked. No push, publication, research-branch merge or external
contact occurred. Attribution is retained; no licensing clearance is claimed.

The final diff passes git diff --check. Nine inherited whitespace-only lines in
new copies were trimmed with a numerical-AST equality assertion. The substantially
rewritten verify.py now uses CRLF instead of mixed endings; other localized edits
preserve existing mixed endings. No broad whitespace cleanup was applied.

## Next work and open decisions

The authorized extraction/public-host acceptance work is complete. Future work
requires its own scope: promote a batch-safe TC adapter (Stage 2C), broaden
provider/fallback/plotting/restart coverage, and review scientific convergence
separately from regression parity. The package has no topology transition feature.
Name, audience/distribution, release ownership, pushing, MAP licensing outreach,
upstream contribution and the ternary prompt default remain user decisions.
README contains the current single quick/full command. Keep numerical fixes
authoritative in one place and record deliberate ports; never refresh a frozen
reference to conceal a discrepancy.
