# moving-boundary: development bridge

This separately buildable checkpoint exposes one binary Illingworth solver:

    from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel

It returns the original class, retaining its docstring, inputs and results.
See examples/binary_constant.py for a complete analytic example.

**Requires the research fork samuelgprice/kawin at
06dc7b4ab56f67226ceabfde41ebf2690c088841.** Public kawin is not sufficient.
This commit removes production debugger dependencies. It is on the local branch
fix/remove-debugger-imports and has been fast-forwarded into local main.
Nothing has been pushed: the Git URL in metadata identifies the intended source,
but a remote install of this new commit is not yet available from this work.
Verification builds the fork from the local Git object and installs both wheels
into isolated targets, without changing the shared environment.

The numerical reference remains ae9fb2971a9fd49132f71ddc883a5bd5406644b6.
It is separate from the dependency pin; the extraction worktree's kawin/ source
and fixtures stay frozen at that reference. The facade does not authenticate an
arbitrary installed fork at runtime.

## Isolated verification on this Windows development machine

From the extraction worktree, run in PowerShell:

    cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && python packages/moving-boundary/tools/verify.py --require-installed-fork'

This uses existing build tools and dependencies without installing or upgrading
anything in the shared environment. It builds this nested project with
pip wheel --no-deps --no-build-isolation --no-index, installs its wheel with
pip install --no-deps --target into a private directory, and runs outside both
checkouts. The reference run explicitly supplies the frozen source tree through
PYTHONPATH. A separate -I -S process imports the installed bridge and pinned fork,
with no source checkout or editable-install hooks, and compares its binary
histories against that captured reference. Import origins are asserted.
Each subprocess has a 30-600 second timeout; expected total runtime is under
five minutes. The committed open fixture is mandatory.

Verification prints separate results for the reference and installed-fork gates.
Both currently pass. If installed imports regress, the summary prints an
OPEN GATE line; --require-installed-fork returns exit status 2 for the known
missing-module gate. Unexpected verification failures are nonzero in either mode.

The fork probe builds the exact Git snapshot and excludes source paths and
editable-install hooks. The bridge build uses a fresh wheel directory and reads
the artifact's metadata rather than assuming its filename or version.
Frozen source content is checked independently of HEAD, so package-only commits
are permitted. Legacy and bridge cases run in separate processes against the
fixed fixture. Test counts come from JUnit XML. Durable evidence uses path roles;
full diagnostic paths remain in ignored local logs.

Generated logs, fresh numerical arrays and installation files stay in
.verification/ (ignored); docs/verification.json is the reviewable evidence.

This is an independently buildable development bridge, not a tested public
release or proof of upstream compatibility. A bare namespace import is lazy;
requesting the solver still executes the fork's broad kawin.diffusion
initialization, with transitive thermodynamic and unrelated solver imports.
CartesianFD1D, transformed grids, host driver and results remain fork-owned.
No thermodynamic provider, surrogate, ternary solver or numerical infrastructure
has been extracted. Future numerics/, providers/, sampling/, surrogates/,
diagnostics/, results/ and compatibility/ namespaces are reserved in the plan,
not implemented APIs in this wheel.

Only the existing closed planar substitutional binary example is exposed and
verified here: constant temperature, zero-flux outer boundaries, fixed interface
compositions, positive phase-uniform diffusivity. The original class retains
other capabilities; this milestone does not validate every combination.
No database or live TC-Python session is needed. Review LICENSE, NOTICE and
docs/HANDOFF.md before considering any redistribution.

See docs/NEXT_SESSIONS.md for the current execution sequence.
