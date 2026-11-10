# moving-boundary: local extraction checkpoint

Binary, ternary two-phase and ternary three-phase Illingworth solvers now live in
this package, with their local meshes, numerical helpers, interface closures and
surrogate providers. The wheel depends on unmodified public `kawin==0.5.0`.

    from moving_boundary.solvers import (
        MovingBoundaryIllingworthFD1DModel,
        MovingBoundaryIllingworthTernaryFD1DModel,
        MovingBoundaryIllingworthTernaryThreePhaseFD1DModel,
    )

Solver families live in their own subpackages under `solvers/` (currently only
`solvers/illingworth/`), so other method families can be added beside them. The
classes above are re-exported from `moving_boundary.solvers`, so this import does
not depend on the file layout. Module map from the research fork (full detail,
including symbols, in docs/relocation-map.json):

| Package module | Research fork module (06dc7b4) |
|---|---|
| solvers/illingworth/binary.py | kawin/diffusion/MovingBoundaryIllingworthFDM.py |
| solvers/illingworth/ternary.py | kawin/diffusion/MovingBoundaryIllingworthTernaryFDM.py (class) |
| solvers/illingworth/shared.py | kawin/diffusion/MovingBoundaryIllingworthTernaryFDM.py (helpers) |
| solvers/illingworth/three_phase.py | kawin/diffusion/MovingBoundaryIllingworthTernaryThreePhaseFDM.py |
| solvers/illingworth/_interstitials.py | kawin/diffusion/_illingworth_interstitials.py |
| solvers/illingworth/numerics/binary.py | kawin/diffusion/mesh/MovingBoundaryIllingworthFD1D.py |
| solvers/illingworth/numerics/ternary.py | kawin/diffusion/mesh/MovingBoundaryIllingworthTernaryFD1D.py |
| mesh/FDM1D.py, mesh/TransformedGrids.py | kawin/diffusion/mesh/ (same names) |
| equilibrium.py | kawin/diffusion/MovingBoundaryEquilibrium.py |
| surrogates.py | kawin/diffusion/MovingBoundarySurrogates.py |
| surrogate_artifacts.py | kawin/diffusion/SurrogateArtifacts.py |
| _diffusivity_domain.py, _spectral_validation.py | kawin/diffusion/ (same names) |
| providers/pycalphad.py | examples/ternaryExamples/pycalphad_default_phase_adapter.py |
| providers/thermocalc.py | examples/ThermoCalc/tc_python_adapter.py |

Public kawin still owns the model/solver loop, mesh bases, recording machinery,
TemperatureParameters and pycalphad/thermodynamic engine. Small subclasses in
compatibility/ preserve minimum-step/final-remainder policy, valid initial
compositions, the 1e-10 floor, preallocation/exact-time lookup and provider endpoint
metadata. The local tracer wrapper explicitly uses R=8.3145. No kawin module,
constant or class global is patched. Curvature/precipitation APIs retain stock
conventions and are outside the accepted moving-boundary provider boundary.

All three families retain tested interstitial x/u conventions. Explicit unequal
molar volumes are supported by three-phase only; binary and ternary two-phase
unequal-volume combinations are explicitly unsupported. Thin-phase stopping is
not a topology transition. Both ternary prompt defaults are unchanged; all batch
cases pass an explicit policy and input() is prohibited before imports.

C32 is a configured retry stress case. A small natural physical stop run has
not been established.

Stage 2B adapter relocation is complete for accepted O05 restricted-phase generation and O08 default-phase endpoint/equilibrium sampling. See [docs/NEXT_SESSIONS.md](docs/NEXT_SESSIONS.md#installed-solving-and-research-owned-generation) for remaining research script orchestration, the open broader Stage 2 split and research-branch R01.

Stage 2C is complete within O06: real TC generation uses the installed
`moving_boundary.providers.thermocalc` facade with separately installed
TC-Python and a valid licence; no TC pip dependency is added. See
[NEXT_SESSIONS.md](docs/NEXT_SESSIONS.md#stage-2c-complete-within-o06) for
the accepted scope, remaining research script orchestration and uncovered workflows.

## Building surrogates

Build Thermo-Calc or pycalphad surrogates with the installed APIs; no research
checkout is needed. Sketch (`...` and `sampling_arguments` are system-specific):

    from moving_boundary.providers.thermocalc import TCPythonThermodynamics
    from moving_boundary.surrogates import TernaryMovingBoundaryThermodynamicsSurrogate as Surrogate
    with TCPythonThermodynamics(...) as source:
        surrogate = Surrogate.from_database(thermodynamics=source, **sampling_arguments)

For pycalphad, pass a source from
`moving_boundary.providers.pycalphad.create_pycalphad_thermodynamics_source(...)`
instead. Optionally put the build in a callback for
`moving_boundary.surrogate_artifacts.prepare_surrogate_artifact` to create a
verified, reloadable bundle; the callback returns `(surrogate, members)`.
`moving_boundary.surrogates.merge_phase_diffusivity_surrogates` combines one
phase's diffusivity samples from two surrogates. See
[research orchestration scope](docs/NEXT_SESSIONS.md#installed-solving-and-research-owned-generation).

## Examples and exact solutions

[examples/README.md](examples/README.md) has worked examples, including a full Fe-Cr-Ni
workflow and what each one does and doesn't validate. `moving_boundary.analytic` provides exact
solutions and comparison helpers for checking your own runs. Only its `__all__` is supported API.

## Local acceptance

From the extraction root, run:

    cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && python packages/moving-boundary/tools/acceptance.py quick'

This selects the separate `mb-public-050` Conda clone, builds/installs a fresh
package wheel into a private target, asserts package/stock import origins and
checks the host's 73 files against the cached stock wheel. Workers run outside
source checkouts with no editable/site hooks. Shared kawin is unchanged.
The local environment pointer is ignored `.verification/public-environment.json`;
this is a configured local verification command, not a portable bootstrap.

Quick: 30 run comparisons and 88 pytest tests passed in 427.34 s. Full: 40 run comparisons and 88 pytest tests passed in 578.02 s.

Quick includes C01-C24/C30-C32, the older binary fixture/bulk probes and freshly
sampled O05 (4.94-second sampling pilot) and O08 (two valid default-phase tie lines
and one FCC-only invalid probe). O08 was explicitly approved for quick/full after
a 3.83-second worker pilot. C25-C29 are pytest lifecycle/failure
contracts, with independent main/package workers. Full adds enabled cached
O01/O02/O03a/O03b/O04/O07 and F04/F11/F12 drift checks. Full-only O06 samples
real TC data and completes six accepted steps with the approved suffixed phase
names, fresh separate caches and an unseeded reference repeat pair. It skips
with a clear reason if local inputs, TC-Python/runtime or licence are missing.
All available cached cases ran. Missing caches are
the only cache skip; invalid or changed existing inputs fail. Reference-independent
exact/similarity errors are recorded in both short and long variants.

Use the same wrapper with `acceptance.py full` for full. Announce its runtime
estimate first (allow about 11 minutes including the O06 availability check and repeat pair); quick has a 600-second budget and
full a 1200-second cap, well below 30 minutes. `verify.py` delegates to quick;
`probe_fork.py` is explicitly historical. Separate environment creation is an
offline setup operation, outside the quick budget. Use tools/setup_public_environment.py
through the same activation wrapper to rebuild that offline clone from the cached wheel.

References are captured from the clean detached `mb-ref-7af24f4` worktree at
7af24f4b0a41b588824ecff22a9d664475602fcc (research main with the R05 and binary, two-phase
ternary and three-phase R06 merges; moved from 06dc7b4 via 90700be and 720d122 on 2026-11-09/10,
see docs/LEDGERS.md R06). The
frozen O08 fixture remains from detached 06dc7b4; the frozen O06 fixture was deliberately
recaptured from 720d122 (R06). Every reference worker asserts its
kawin import origin. The extraction's older kawin/examples/root packaging remain
frozen at ae9fb29; legacy fixtures are also checked. Tolerances/inputs are frozen
in tools/acceptance_manifest.json; relocation gates were exact, public gates use
the preregistered quantity tolerances without widening. O06/O08 require byte
identity and frozen fixture hashes. O06 compares every numerical array exactly;
only metadata_json.thermocalc_config.cache_dir is normalized, retaining all
other metadata text. O08 also checks ordered phase metadata. Only O05/O08 use a fixed Python hash
seed for the diagnosed pycalphad symbolic-ordering issue (LEDGERS.md R01). No
NumPy seed is imposed. All other quick runs/contracts are repeated twice in fresh
unseeded reference processes and must match byte for byte; full also repeats
O06. All 89 older reference files are unchanged; only the new O06 pair was captured.

Numeric reference outputs, TC inputs, logs, build targets and environment
state stay local/git-ignored. Only approved O06 input/fixture provenance hashes
enter the manifest; cached TC data hashes remain local. Committed acceptance.json contains only
sanitized counts/provenance. docs/acceptance-deviations.json records per-run and
per-quantity deviations, actual tolerances and bit identity, with TC metrics
redacted to ignored output. Runs write only ignored local reports by default.
Use `quick --record` or `full --record` to replace the compact committed report
only after a complete passing public-tolerance gate; per-array detail stays local.
See docs/BINARY_SLICE.md, LEDGERS.md and
PUBLIC_KAWIN_MILESTONE.md for ownership, scope and method-level source mapping.

Package identity/audience, publication, pushing, MAP outreach, upstream contribution
and changing the prompt default remain user decisions. LICENSE/NOTICE preserve
attribution without claiming unresolved local/MAP redistribution rights are cleared.
