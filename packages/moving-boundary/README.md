# moving-boundary: extraction checkpoint

The installable development package currently exposes one binary solver:

    from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel

It is an intermediate subclass with package-owned planar bulk solves and a scalar
Thomas kernel. Remaining solver/driver, mesh, recording and thermodynamic services
come from the research fork. Whole-module relocation is the authorized next
implementation, not continued method-by-method inheritance. See
docs/BINARY_SLICE.md for source mappings and docs/NEXT_SESSIONS.md for the target.

The temporary dependency is samuelgprice/kawin at
5f4ad574f2b7348ee2f818b1a18b75c0c0325c2e. It removes production debugger
dependencies and two obsolete test stubs, and is integrated into local main.
Nothing is pushed: the Git pin is not remotely installable from this work.
Verification builds local Git objects into private targets; shared packages
must not be upgraded or reinstalled. Stock kawin 0.5.0 compatibility is planned,
not established; see docs/PUBLIC_KAWIN_MILESTONE.md.

## Reference and acceptance

The older source/fixture oracle remains ae9fb29. The approved quick/full suite
will capture from a clean detached 06dc7b4 worktree with asserted import origins,
never from the user's main checkout. See docs/ACCEPTANCE_CANDIDATES.md for the
approved inputs, budgets, cache policy, analytic drift metrics and contract tests.
TC-derived fixtures stay local/git-ignored; their availability is not permission
to redistribute them. O06 is disabled until the TC adapter is batch-safe.

Prior installed verification passed 15 tests, 33 binary history arrays and 12
nonuniform-grid bulk probes exactly, with all nine frozen fixture cases exact.
That evidence is docs/verification.json and identifies its actual tested pin and
source hashes. The current repin/guard changes are not falsely represented as a
new run of that numerical verification. New results will record their own inputs.

## Local verification

Every Python-based command must use the activated kawin Conda environment:

    cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && python packages/moving-boundary/tools/verify.py --require-installed-fork'

The existing command builds the nested project and pinned local fork privately,
checks the frozen fixture and separate installed imports/parity, and runs package
pytest. Each subprocess has an actual timeout (30–600 seconds). Batch workers
forbid input() and frozen case loading supplies explicit ternary terminal policies;
production prompt defaults are unchanged. No source/editable paths are used by
the installed -I -S probe. Logs/builds/caches stay under ignored .verification/.

The approved quick/full command is implemented after input/tolerance manifests
are frozen; its default quick budget is under ten minutes. The full tier adds
present cached research cases and long drift checks, with an estimate reported
before execution and a cap well below 30 minutes. C25–C29 live in pytest.

No independent driver/results or thermodynamic rewrite is intended. Unequal
per-phase molar volumes are supported by the three-phase solver only, not binary
or ternary two-phase. New namespace initialization stays lazy. See LICENSE/NOTICE
for unresolved MAP/local provenance; no blanket redistribution clearance exists.
Package name, audience, publication, outreach, upstream contribution and changing
the prompt default remain user decisions.
