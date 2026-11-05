# moving-boundary: development bridge

This separately buildable checkpoint exposes one binary Illingworth solver:

    from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel

It returns the original class, retaining its docstring, inputs and results.
See examples/binary_constant.py for a complete analytic example.

**Requires the research fork samuelgprice/kawin at
ae9fb2971a9fd49132f71ddc883a5bd5406644b6.** Public kawin is not sufficient.
The direct Git dependency identifies the exact source. A wheel built from that
local Git snapshot builds and installs successfully, but importing kawin.diffusion
fails because it imports examples.debugInPlace and the wheel omits examples/.
Fetching and installing directly from the remote Git URL was not tested.
The working development setup explicitly supplies the fork source tree.
The facade does not authenticate an arbitrary installed fork at runtime.

## Isolated verification on this Windows development machine

From the extraction worktree, run in PowerShell:

    cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && python packages/moving-boundary/tools/verify.py'

This uses existing build tools and dependencies without installing or upgrading
anything in the shared environment. It builds this nested project with
pip wheel --no-deps --no-build-isolation --no-index, installs its wheel with
pip install --no-deps --target into a private directory, and runs outside both
checkouts. PYTHONPATH explicitly supplies that installed target and the pinned
fork worktree. Import origins are asserted, not inferred from the current folder.
Each subprocess has a 30–600 second timeout; expected total runtime is under
five minutes. It requires the committed open baseline fixture and fails if absent.
Verification prints separate results for the source bridge and installed fork.
The known import blocker appears as:
OPEN GATE: installed-fork import fails (examples).
Add --require-installed-fork to the command to return exit status 2 when this
gate remains open; ordinary development verification can succeed with this gate
open. Unexpected verification failures return nonzero in either mode.

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
