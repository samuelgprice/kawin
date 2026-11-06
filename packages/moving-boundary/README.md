# moving-boundary: local extraction checkpoint

Binary, ternary two-phase and ternary three-phase Illingworth solvers now live in
this package, with their local meshes, numerical helpers, interface closures and
surrogate providers. The wheel depends on unmodified public `kawin==0.5.0`.

    from moving_boundary.solvers import (
        MovingBoundaryIllingworthFD1DModel,
        MovingBoundaryIllingworthTernaryFD1DModel,
        MovingBoundaryIllingworthTernaryThreePhaseFD1DModel,
    )

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

Stage 2B is complete: pycalphad data generation uses the installed package. See [docs/NEXT_SESSIONS.md](docs/NEXT_SESSIONS.md#installed-solving-and-research-owned-generation) for the remaining research builders/assembly dependency, open broader Stage 2 split and research-branch R01.

## Local acceptance

From the extraction root, run:

    cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && python packages/moving-boundary/tools/acceptance.py quick'

This selects the separate `mb-public-050` Conda clone, builds/installs a fresh
package wheel into a private target, asserts package/stock import origins and
checks the host's 73 files against the cached stock wheel. Workers run outside
source checkouts with no editable/site hooks. Shared kawin is unchanged.
The local environment pointer is ignored `.verification/public-environment.json`;
this is a configured local verification command, not a portable bootstrap.

Quick: 29 run comparisons and 41 pytest tests passed in 435.73 s. Full: 38 run comparisons and 41 pytest tests passed in 495.45 s.

Quick includes C01-C24/C30-C32, the older binary fixture/bulk probes and freshly
sampled O05 (4.94-second sampling pilot). C25-C29 are pytest lifecycle/failure
contracts, with independent main/package workers. Full adds enabled cached
O01/O02/O03a/O03b/O04/O07 and F04/F11/F12 drift checks. O06 reports SKIPPED until
the TC adapter is batch-safe. All available cached cases ran. Missing caches are
the only cache skip; invalid or changed existing inputs fail. Reference-independent
exact/similarity errors are recorded in both short and long variants.

Use the same wrapper with `acceptance.py full` for full. Announce its runtime
estimate first (currently about 8-11 minutes including unseeded reference pairs); quick has a 600-second budget and
full a 1200-second cap, well below 30 minutes. `verify.py` delegates to quick;
`probe_fork.py` is explicitly historical. Separate environment creation is an
offline setup operation, outside the quick budget. Use tools/setup_public_environment.py
through the same activation wrapper to rebuild that offline clone from the cached wheel.

References were captured from the clean detached `mb-ref-06dc7b4` worktree at
06dc7b4ab56f67226ceabfde41ebf2690c088841. Every reference worker asserts its
kawin import origin. The extraction's older kawin/examples/root packaging remain
frozen at ae9fb29; legacy fixtures are also checked. Tolerances/inputs are frozen
in tools/acceptance_manifest.json; relocation gates were exact, public gates use
the preregistered quantity tolerances without widening. Only O05 uses a fixed Python hash
seed for the diagnosed pycalphad symbolic-ordering issue (LEDGERS.md R01). No
NumPy seed is imposed. All other quick runs/contracts are repeated twice in fresh
unseeded reference processes and must match byte for byte; fixtures are unchanged.

Numeric reference outputs, TC inputs, private hashes, logs, build targets and
environment state stay local/git-ignored. Committed acceptance.json contains only
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
