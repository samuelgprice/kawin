# Examples

These examples show how to set up realistic moving-boundary problems with the installed
package. Each script runs on its own and is meant to be copied and adapted.

| Script | Question | Needs | Time* |
|---|---|---|---|
| `binary_planar_growth.py` | How fast does a planar interface grow under diffusion control? | Nothing | Under 1 s |
| `ternary_interstitial_similarity.py` | How do a fast interstitial and a slow substitutional solute move an interface? | Nothing | About 1 s |
| `surrogate_lifecycle.py` | How do I save, verify, reload and solve with a surrogate? | Nothing | Under 1 s |
| `three_phase_similarity.py` | How do two independent interfaces move as a middle layer grows? | Nothing | About 5 s |
| `ternary_fecrni_lee1996.py` | How does a Fe-Cr-Ni couple evolve with composition-dependent diffusivity? | A Fe-Cr-Ni TDB | About 40 s |
| `binary_constant.py` | The smallest possible solve | Nothing | Under 1 s |

*Solve and sampling time on one Windows workstation, excluding Python start-up.

Start with `binary_planar_growth.py`. `ternary_fecrni_lee1996.py` shows the full workflow on
real thermodynamic data. Each file opens with notes on its assumptions, units, ordering and
limitations.

**In a notebook.** Each file is split into `# %%` cells. Open it in VS Code or Jupyter and run
the cells in order in an Interactive window: edit the Settings cell first, then run the cells
below it. Run cells only execute when the file is run directly, so tests and other code can
import its functions without starting a solve.

**From a terminal.** `python <example>.py`, adding `--fast` for shorter settings, or `--help`
for other options.

## Conventions worth knowing

- **Ordering.** The first element is the dependent one, and the rest are independent. Phases
  are listed left to right.
- **Composition basis.** Without interstitials, compositions are mole fractions. With them,
  the solver state is in u-fractions, while `getCompositions` and `getPhysicalPhaseProfiles`
  return mole fractions. `getTransformedState` returns the state itself.
- **Units.** Use any consistent set; the Fe-Cr-Ni example uses SI.
- **Thin phases.** Ternary models need an explicit `terminal_thin_phase_policy`, because the
  default asks interactively. `"raise"` stops at the limit and `"continue"` proceeds past it.
  Neither changes the phase topology.
- **Molar volumes.** Binary and two-phase models assume equal constant volumes. Only the
  three-phase model accepts explicit unequal volumes.

## What these results do and don't show

Each check supports a specific claim, and passing one doesn't imply the others.

| Check | What it shows | What it doesn't show |
|---|---|---|
| Comparison with an exact solution | Numerical accuracy, within that solution's assumptions | That real materials behave this way |
| Solute conservation | Inventories are accounted for | Accuracy |
| Fresh vs reloaded surrogate | The saved bundle preserved the data | That the data is correct |
| Bundle integrity checks | Damage or changed settings are detected | Surrogate quality |
| Live lagged workflow | The bulk path ran, stayed inside its sampled data, and used composition-dependent matrices | Accuracy |
| Domain enlargement | The walls have little influence on the result | Anything about grid convergence in general |

In particular:

- `FROZEN_ANALYTIC` in the Fe-Cr-Ni example checks numerics with real data and closure. It does
  not validate composition-dependent Fe-Cr-Ni physics.
- `LAGGED_DEMO` has no exact solution, so it makes no accuracy claim. It is also deliberately
  coarser than a production run. `PRODUCTION_B` has the research case B settings but is
  untested here, takes roughly 10 minutes, and makes no accuracy claim either.
- `semi_infinite_validity_time` is a screening estimate of a wall-tail amplitude, not an error
  bound. When it matters, repeat the run on a larger domain and compare directly.

Numerical limits appear in each example with the exact settings, normalization and time window
they apply to. They don't carry over to other settings.

## Fe-Cr-Ni database

`ternary_fecrni_lee1996.py` needs a TDB with FE, CR and NI, BCC_A2 and FCC_A1, and mobilities.
Pass it with `--tdb PATH` or set `MB_EXAMPLE_TDB`. **No database is included yet**, because
redistribution of the one used in development hasn't been cleared. It was assembled from
Lee (1993) with mobilities from Lee & Oh (1996). The sampled probe point and compositions in the
example suit that database at 1373 K, and may need changing for another.

Surrogate bundles are written to `~/moving_boundary_example_outputs/<example>/`, wherever you run
from, unless you set `OUTPUT_DIR` (or pass `--output-dir`). They never go into the source tree. Existing compatible bundles are reloaded, incompatible
ones stop with an error, and `--rebuild` replaces them. A Fe-Cr-Ni bundle records the
`moving-boundary` version, so a new release won't reload an old bundle; within one version,
such as in an editable install, rebuild after changing surrogate code.

## Running the tests

The tests need the source distribution or repository, not just the wheel. From the package
directory, using the tested dependency versions:

    python -m pip install ".[test]" -c examples/requirements-tested.txt
    python -m pytest tests/test_analytic.py tests/test_examples.py

Without a database, tests that need one skip with a reason. To include them, set `MB_EXAMPLE_TDB`.
Slower refinement studies are deselected by default; run them with `-m slow`.
