# Extraction handoff

## Current status

- Numerical reference: ae9fb2971a9fd49132f71ddc883a5bd5406644b6; frozen source/fixtures unchanged.
- Dependency fix: 06dc7b4ab56f67226ceabfde41ebf2690c088841 on fix/remove-debugger-imports,
  based directly on the reference and fast-forwarded into local main.
- Package metadata/runtime pin updated on refactor/package-extraction.
- Installed fork imports and installed bridge/fork binary parity pass without
  checkout paths; the nine-case reference capture still matches the fixture.
- 76 focused fork tests passed in 26.33 seconds; 3 fake-provider tests passed
  in 3.20 seconds. New public retry exhaustion tests cover both ternary families,
  exception chaining, reduced timesteps and unchanged accepted state/histories.
- Provider tests initially failed on an absent generated output directory in the
  clean worktree; the affected test now uses pytest tmp_path. No production
  numerical workaround or expected-result update was made.
- Scope: removed production debugger imports/calls/reload. Existing exceptions,
  fallback behavior and explicit thin-phase prompt policies remain. Debugger-only
  legacy branches retain their previous batch no-op behavior; this patch does
  not repair unrelated numerical or legacy exception-handling behavior.
- Local main's untracked-file list was verified unchanged. Other research branches
  were not automatically merged. AGENTS.md is untracked in both new worktrees.
- Nothing pushed. The dependency commit is local; remote installation cannot be
  claimed until it is made available remotely. Local Git-archive wheel testing
  does not require publication or reinstalling the shared Conda environment.

The following sections preserve the historical checkpoint and review record;
statements about earlier uncommitted state or an open import gate describe those
earlier runs, not the current state.

## First extraction checkpoint — November 4, 2026

## State and reproducibility

- Branch: refactor/package-extraction; all additions remain uncommitted.
- Worktree: C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\moving-boundary-extraction.
- Starting local main and current HEAD: ae9fb2971a9fd49132f71ddc883a5bd5406644b6.
- Cached public upstream/main: f0c347967fcfc18f6615cec205f584573606e100.
- Shared ancestor: 279cedfe0d5f8c12f69beb8c8579f30ac5c41b08.
- origin: https://github.com/samuelgprice/kawin.git.
- upstream: https://github.com/materialsgenomefoundation/kawin.git.
- Research feature/DICTRA-comparison: da398e3294860abaea22d3e15f381feac27c353e.
- The original checkout was on main, unlike the plan's earlier snapshot.
  No tracked modifications were present. Its pre-existing untracked files were
  preserved. Only AGENTS.md and EXTRACTION_PLAN.md were copied byte-for-byte.
- No fetch, merge, rebase, commit, push, publication or later research import.
  Upstream statements refer to the cached commit, not today's remote or PyPI.
  The production-path comparison with the research branch agrees with the plan.
- Only root AGENTS.md was found in the repository. Its activated Conda rule
  was used for every Python command, including child processes inheriting it.
- NumPy 2.4.3, SciPy 1.17.1, Python 3.11.15, pycalphad 0.11.1,
  Numba 0.65.1; compiled binary and block-tridiagonal functions available.
  Full versions, origins, hashes and array shapes are in verification.json.
  The kawin distribution reports 0.4.0; that version does not identify this fork.
  The explicit source path and Git SHA identify the implementation actually used.

## Changes and supported operation

Only packages/moving-boundary and the two deliberately copied documents were
added. No existing tracked source, tests, tolerances, fixtures or line endings
were changed.

- pyproject.toml: separate src-layout distribution; exact fork Git dependency;
  explicit discovery excludes kawin. No shared editable reinstall.
- src/moving_boundary: minimal lazy namespace and one binary class facade.
  The class is an alias to the existing class, retaining its API/docstrings.
- examples/binary_constant.py: standalone analytic input builder and short solve.
- tests/test_bridge.py: exact histories/state parity, lazy import, missing-fork
  diagnostic. The missing-backend test simulates absence; it is not evidence of
  a clean environment without installed backends.
- tools/verify.py: bounded build/install/check workflow, isolated target,
  outside-checkout execution and asserted import origins.
- README.md, LICENSE, NOTICE, docs/: scope, attribution, ledgers and evidence.
- .gitignore: excludes local build/install/log artifacts.

The wheel is independently buildable. This is a development bridge supplied
with the exact local fork via explicit PYTHONPATH, not upstream-wheel
compatibility, dependency independence, a public release, or an end-to-end
material-generation package. Requesting the class still triggers the legacy
package initializer. The public facade has no broad export-list import itself.
Reserved architecture directories are deferred until they contain real APIs.

## Commands and outcomes

All commands were run in the activated kawin environment with:

    cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && <command>'

From the extraction root, baseline command:

    python -c "import subprocess,sys; subprocess.run([sys.executable, '-m', 'pytest', 'kawin/tests/test_illingworth_substitutional_baseline.py', 'kawin/tests/test_illingworth_setup_and_diagnostics.py', '--basetemp', '.pytest_tmp/extraction_baseline', '-q'], check=True, timeout=600)"

Result: 17 passed, 6 warnings, 15.42 seconds. Warnings were existing pyparsing
deprecations and invalid escape sequences in legacy docstrings.
No skipped fixture and no baseline failure.

Then:

    python packages/moving-boundary/tools/verify.py

Successful subprocess times: build 2.89 s, install 0.74 s, fresh baseline capture
5.58 s, parity pytest 3.97 s (3 passed in 3.07 s), example 3.30 s.
The first attempt failed only in the new LAPACK sanity check because literal
0.2 and 0.6 differed from the computed values by up to 1.11e-16. That check now
uses absolute tolerance 1e-15, relative tolerance zero. Solver fixture and
facade comparisons remain exact; no expected numerical results were regenerated.

The outside-directory run used the installed wheel target and an explicit
source dependency path to this worktree. It did not depend on current-directory
imports and did not install the fork. No upstream wheel or remote Git install
was tested. Build isolation was deliberately disabled to reuse installed tools
without upgrades or network access. Final SPDX licensing metadata is deferred until rights are resolved.

The wheel listing and SHA256, fixture SHA256 and every captured array's shape
and byte hash are in verification.json. Fresh baseline arrays and full logs are
in ignored .verification/. The committed legacy fixture is the durable numeric
reference, not a newly generated expected result. The verification script fails
if it is missing. Working wheel:
.verification/moving_boundary-0.0.1.dev0-py3-none-any.whl.

Parity compares separate model instances with identical analytic inputs:
accepted times, mesh and transformed profiles, interface history, concentration
history, initial inventory and final compositions, with no tolerance.
This establishes extraction fidelity for one workflow, not independent
mathematical correctness of the inherited method.

## Gates and next step

- Small Stage 1A gate satisfied: nested wheel builds, representative namespace
  workflow runs, and exact fork dependency is explicit.
- Broader Stage 1A work remains: own numerical modules, ternary facades and
  removal of transitive legacy initialization/debugger dependencies.
- Stage 0 incomplete: nine substitutional baseline runs cover all three families,
  but the complete capability acceptance matrix, interstitial checks, convergence,
  rollback/terminal/coupling/restart cases, provider workflows and provenance
  clearance remain open. See LEDGERS.md for explicit coverage and gaps.
- Stage 1B and all later gates remain open. No upstream compatibility claim.
- Live TC-Python and database-dependent checks were not run; a licensed
  environment was not established. No live pycalphad material-generation
  workflow was tested. Installed pycalphad does not establish provider parity.

Next small milestone: characterize the binary solver's transitive helper imports,
then extract the minimal binary solver/transformed-grid helper slice with source
mapping and attribution, preserving this exact baseline. Keep the host driver
and results unchanged. Resolve MAP-port redistribution status before publishing
that extracted code. Inspect/remove debugger paths before exposing ternary
failure handling. Do not use a host-framework rewrite to force optional kawin.

## Review follow-up — November 5, 2026

Final review verification: 6 tests passed in 10.82 seconds (two existing
pyparsing deprecation warnings); all nine fresh substitutional cases matched
the committed fixture exactly. Build, private installation and source-backed
outside-directory example passed. The separate installed-fork gate remains
blocked on missing examples. The first new temporary-directory test run failed
because .pytest_tmp did not exist; verification now creates that parent.
The real extraction branch and frozen reference remain uncommitted/unchanged
respectively. No dependency upgrades, source patches, fixture updates or
maintainer messages were made.
The original checkpoint below is historical. NEXT_SESSIONS.md supersedes its
next-step recommendation. No numerical source or fixture has changed.
tools/probe_fork.py builds the pinned Git snapshot and installs its wheel in a
private target. An isolated -I -S process excludes source paths and editable
hooks. This confirms an open packaging blocker: examples/ is absent, while
MovingBoundaryFDM.py imports examples.debugInPlace through package initialization.
The source bridge remains usable; the pinned dependency alone is not installable
as a working solver workflow. Removing/isolating this dependency is a prerequisite
for an installed workflow, with the implementation strategy chosen after the
bounded host assessment. The baseline stays frozen.

tools/verify.py now compares baseline content, records baseline/current commits
separately, derives counts from JUnit and normalizes durable path fields. Local
.verification/verification-full.json retains diagnostic paths. New guard tests
use disposable repositories only: package-only commits pass; modified tracked
solver files and untracked source additions fail. No real worktree commit is made.
Binary reference and facade now run in separate subprocesses, and both compare
exactly with the committed fixture. The implementation remains an alias.

The updated plan records research-fix authority/ports, upstream contribution as
an option, and interstitial plus rollback evidence as prerequisites to numerical
moves. These additional scientific baseline tests and the host assessment are
next-session work, not claimed completed in this verification repair.


## Verification cleanup and bounded assessment

README and extraction plan now describe current behavior in place. Verification
discovers a single wheel in a fresh output directory and inspects its metadata;
no wheel filename/version is hard-coded. The final console summary prints
OPEN GATE: installed-fork import fails (examples). --require-installed-fork
requests exit status 2 for that gate. The strict integration run exited nonzero
after 11 tests passed in 11.12 seconds and the nine-case exact baseline passed;
unit tests cover both modes and changed-version wheel discovery.

The additional interstitial/retry/rollback selection passed 39 tests in 28.52
seconds. Public 0.5.0 host source was compared without installing it. The findings
and bounded-scope recommendation are integrated into LEDGERS.md and NEXT_SESSIONS.md.
No numerical code was moved or changed; debugger removal itself remains the next
implementation change against a separately identified source, not this reference.
All real-worktree changes remain uncommitted.
