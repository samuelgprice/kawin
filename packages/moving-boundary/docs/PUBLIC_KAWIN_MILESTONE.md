# Stage 1B: stock kawin 0.5.0

Status: planned, not implemented or accepted. This is review item 3, after the
whole-module binary, ternary two-phase and ternary three-phase moves in item 2.
All implementation waits for approval and creation of the run suite in item 4.

## Boundary

Use unmodified public kawin 0.5.0 for the host model/solver loop, mesh/recording,
TemperatureParameters and pycalphad/thermodynamic services. The package owns its
Illingworth solvers, local numerical modules and narrowly required compatibility
behavior. The completed static assessment is in LEDGERS.md/assessment.json;
it is not runtime evidence of public-kawin compatibility.

| Fork-only behavior | Package compatibility responsibility and required checks |
|---|---|
| Solver minimum-timestep rejection | Explicit adapter must reject an ordinary reduced trial below the outer minimum before acceptance; never enlarge dt after solving the candidate |
| Solver final remainder | Allow the final remainder below the ordinary minimum and retain synchronized recorded times/state, including tiny remainders and continuation |
| MeshData preallocation and exact-time lookup | Local recorder adapter provides capacity growth without changing data and exact recorded-time selection/range errors where required; keep upstream mesh bases |
| DiffusionModel initial composition adjustment | Local model adapter retains conditional adjustment only for genuine below-floor violations during initialization/reset; preserve valid inputs |
| minComposition | Pass 1e-10 explicitly through package-owned configuration; never change an upstream module default |
| Gas constant | Use 8.3145 explicitly at relevant package call sites/constants; do not assign to kawin.Constants or patch upstream code |
| MultiTherm.returnMeta | Implement only if an approved provider actually requires endpoint/phase metadata; preserve phase and element ordering and validate sampled outputs before solving |

Select adapter mechanisms from the actual release APIs. GenericModel.solve
constructs DESolver directly; a package entry point must explicitly select the
compatible host-loop adapter if required. Do not replace class/module globals or
copy a complete driver/results/thermodynamic framework to avoid a small mismatch.
Constructor signatures, docstrings, x/u conversions, temperature-array hours and
exception/stop contracts remain documented and tested.

Audit all imports, including material/surrogate paths exercised by the approved
suite. A module merely renamed compatibility is not an acceptable fork dependency.
Any local provider/surrogate evaluation required by that suite must be explicitly
relocated or adapted before claiming this gate; fake-provider tests cannot replace
approved pycalphad/live-provider runs. Keep live licensed runs opt-in with reasons.

## Separate environment and acceptance

After approval, create a disposable environment under ignored .verification/,
without shared site packages or editable hooks. Install the moving-boundary wheel,
the stock kawin 0.5.0 artifact and recorded dependencies there. The existing local
wheel has SHA256 9f7a1497c5dff4314536860ac6fd0b96a68243dd681db8eb106dbe33fe25cbaf;
verify identity before using it. Do not reinstall, upgrade or remove packages in
the shared kawin Conda environment. All local Python-based setup and subprocesses
must inherit the required Conda activation and have actual timeouts.

Before examining new results, record per-quantity tolerances for the public-host
comparison in a versioned manifest; pure relocation remains exact. Gas-constant
and composition-floor differences are possible causes to investigate, not a reason
to auto-widen tolerances or regenerate reference fixtures.

Definition of done:

- The built wheel installs with stock kawin 0.5.0 in the separate environment.
- Runs execute outside source checkouts; origins, versions, wheel/source hashes
  and dependency setup are asserted and recorded.
- No execution path imports a fork-only symbol, including saved-surrogate and
  provider paths. No upstream module is monkeypatched or edited.
- The approved main-06dc7b4 acceptance suite passes every required quantity within
  its recorded tolerance and prints a per-run/per-quantity pass/fail/skip table.
  Every opt-in skip has a reason; skipped provider coverage is not provider parity.
- The single acceptance command stays within its measured approximately ten-minute
  budget, or a longer run is explained and approved before execution.

Nothing here authorizes pushing, publication, upstream contribution, MAP outreach,
a new package identity, or a change to the thin-phase prompt default.
