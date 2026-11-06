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

    cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && python packages/moving-boundary/tools/acceptance.py quick'

The command builds the nested package and pinned local fork in private targets,
runs outside source roots, compares every recorded quantity with detached 06dc7b4,
then runs package pytest including C25–C29. Quick passes 29 run comparisons plus
22 tests in about 143 seconds using existing targets; fresh private builds add
roughly 20 seconds. Real O05 pycalphad sampling is enabled after its five-second
pilot. C31 uses the irregular Fig. 4 grid and C32 reaches configured terminal stop
under continue with real solver retries. Binary/two-phase unequal volumes are
explicitly unsupported.

Use `python packages/moving-boundary/tools/acceptance.py full` through the same
activation wrapper for quick plus present cached runs and longer analytic drift
checks. Full passes 38 runs and 22 tests in about 198 seconds; O06 is explicitly
SKIPPED pending a batch-safe TC adapter. Missing cached inputs are the only cache
skip; corrupt data or enabled-case errors fail. Full has a 20-minute cap and a
runtime estimate is reported before execution.

Numerical fixtures, datasets, logs, build/install targets and private input hashes
remain under ignored .verification/; no TC-derived reference output is committed.
docs/acceptance.json contains only sanitized counts/status and provenance. Input
sources and tolerances are frozen in tools/acceptance_manifest.json. Workers use
safe-path/no-site startup, cleared Python path variables and deterministic hash/
NumPy seeds; import origins and the detached source's cleanliness are asserted.
The previous non-repeatable O05 fixture is preserved locally. Tolerances did not
change. Exact/similarity errors are recorded as independent quantities.

The older verify.py --require-installed-fork remains a scoped frozen-fixture
verification tool for the temporary Git-pinned checkpoint. The quick/full command
is the approved suite and includes the retained 33 legacy arrays and 12 probes.

No independent driver/results or thermodynamic rewrite is intended. Unequal
per-phase molar volumes are supported by the three-phase solver only, not binary
or ternary two-phase. New namespace initialization stays lazy. See LICENSE/NOTICE
for unresolved MAP/local provenance; no blanket redistribution clearance exists.
Package name, audience, publication, outreach, upstream contribution and changing
the prompt default remain user decisions.
