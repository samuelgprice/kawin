# First binary numerical extraction

November 5, 2026. Numerical reference remains
ae9fb2971a9fd49132f71ddc883a5bd5406644b6; dependency remains
06dc7b4ab56f67226ceabfde41ebf2690c088841. No host source or fixture is changed.

## Dependency review and boundary

This is the retained intermediate checkpoint. The accepted next strategy is
whole-module relocation, with shared helpers at package level and no inheritance
from the fork's Illingworth class. See NEXT_SESSIONS.md. Further relocation waits
for approval and implementation of the main-reference acceptance suite.

The original binary class imports DiffusionModel/GenericModel, interstitial
helpers, diffusivity-mode helpers from the ternary module, CartesianFD1D and
boundary classes, four transformed-profile/kernel helpers, explicitEulerIterator,
and NumPy. Importing its submodule executes kawin.diffusion initialization.
The public 0.5.0 host assessment still favors retaining the fork host.

This slice needs only NumPy, the extracted scalar kernel, and inherited
fields: _u_grid/_v_grid, _R and interfaceCompositions. The package class subclasses
the pinned fork class and overrides precisely the two bulk-concentration methods.
Inherited Picard iteration and phase-uniform/lagged paths dispatch to those
overrides naturally. No import monkeypatch, global replacement or driver copy
is used. NumPy is declared directly; optional Numba behavior is preserved.
The exported class now has its own identity and module; it is no longer
an alias, although its constructor signature and remaining behavior are inherited.
Previously pickled host-class objects remain host-class objects; no migration or
restart guarantee is introduced here.

Interface balance, retry/rollback, derivative/step round trips, timestep limits,
transformed grid validation, x/u conversions, temperature inputs, recording,
physical-profile reconstruction, inventories and results remain fork-owned.
No gas constant is used in the extracted slice; inherited conventions, including
8.3145 where relevant and temperature-array times in hours, are retained.

## Function-level source and ownership mapping

Source repository: https://github.com/samuelgprice/kawin
Source commit for every copied function: ae9fb2971a9fd49132f71ddc883a5bd5406644b6.

| Package path / functions | Original path / functions | Changes |
|---|---|---|
| solvers/binary.py, MovingBoundaryIllingworthFD1DModel._new_concentration_left_planar and _new_concentration_right_planar | kawin/diffusion/MovingBoundaryIllingworthFDM.py, same class/method names | Bodies and numerical docstrings unchanged; inherited host with package-local kernel import |
| numerics/tridiagonal.py, _solve_illingworth_tridiagonal_python, _solve_illingworth_tridiagonal_numba, solve_illingworth_tridiagonal | kawin/diffusion/mesh/MovingBoundaryIllingworthFD1D.py, same function names | Bodies unchanged; compiled helper gains a docstring; optional Numba import preserved |
| tools/binary_cases.py, run_cases | kawin/tests/illingworth_baseline_cases.py, _binary_model, _BinaryThermodynamics and _collect | Open inputs adapted to injected class and direct history capture; no expected outputs copied |
| tools/binary_cases.py, run_bulk_cases | New probes | Nonuniform grids, three interface directions, two diffusivity forms, two phase sides |

The fork explicitly identifies the scheme as a port of Illingworth and Golosnoy's
MAP C++ implementation. Existing upstream MIT text remains verbatim in NOTICE;
it does not establish permission for all local or MAP-derived material. LICENSE
continues to withhold a blanket grant. No reference C++, PDFs, databases, digitized
data or surrogate artifacts are included in the wheel. No clearance/outreach is
claimed, and nothing is published.

The frozen fork remains the numerical authority during this comparison slice.
Any scientific correction must have one authoritative research commit and an
explicit recorded port; do not evolve two implementations independently. The
package maintainer role owns this temporary copy and checks it against the
recorded source before accepting ports; a named release/update owner is unresolved.

## Verification contract

tools/verify.py --require-installed-fork retains the nine-case frozen-fixture
capture and runs the new class from a privately installed wheel against a privately
installed dependency wheel built from the exact pinned Git object. The installed
worker uses -I -S, excludes source/editable paths and research/debugger imports,
and asserts the solver and kernel originate in the wheel target.

All 33 binary history arrays (three modes, eleven arrays each) must match the
committed fixture exactly. Twelve additional bulk outputs must match a separate
frozen-source process exactly: advancing, stationary and receding interface,
scalar or face-valued diffusivity, left and right phase, nonuniform grids.
The probes establish extraction fidelity, not independent physical correctness.

Kernel tests compare with an independent dense solve, preserve input arrays and
existing length/zero-pivot errors, compare accelerated/fallback outputs, and
exercise import without kawin or Numba in a fresh process. This blocked-import
test is not a clean dependency-free environment claim. Full fallback solver parity,
the Stage 0 matrix, upstream compatibility, ternary extraction, licensed provider
workflows and scientific convergence studies remain open.

Each verification subprocess is bounded (30-600 seconds); normal total runtime
is expected under five minutes. Shared packages are neither upgraded nor installed.
Durable measured results are in verification.json; full logs stay in .verification/.
