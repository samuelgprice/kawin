# Examples

These examples show how to set up realistic moving-boundary problems with the installed
package. Each script can be run directly and adapted. Keep the adjacent `advanced/`
directory when copying scripts; it holds detailed validation and compatibility workflows.

| Script | Question | Needs | Time* |
|---|---|---|---|
| `binary_planar_growth.py` | How fast does a planar interface grow under diffusion control? | Nothing | Under 1 s |
| `binary_tlp_bonding.py` | How does a liquid interlayer widen and then solidify when bonding nickel? | Nothing | About 2 s |
| `ternary_interstitial_similarity.py` | How do a fast interstitial and a slow substitutional solute move an interface? | Nothing | About 1 s |
| `surrogate_lifecycle.py` | How do I save, verify, reload and solve with a surrogate? | Nothing | Under 1 s |
| `three_phase_similarity.py` | How do two independent interfaces move as a middle layer grows? | Nothing | About 5 s |
| `ternary_fecrni_lee1996.py` | How does a Fe-Cr-Ni couple evolve with composition-dependent diffusivity? | A Fe-Cr-Ni TDB | About 40 s |
| `surrogate_workflow_fecrni_open.py` | How do I check, compare and use Fe-Cr-Ni surrogates built from an open database? | Nothing (bundled open TDB); Thermo-Calc optional | About 15 s |
| `binary_constant.py` | The smallest possible solve | Nothing | Under 1 s |

*Solve and sampling time on one Windows workstation, excluding Python start-up.

Start with `binary_planar_growth.py`. `binary_tlp_bonding.py` applies the same binary solver to
a real process, the Ni-P bonding case from the algorithm's paper, with exact checks at early
times and on the liquid width. `ternary_fecrni_lee1996.py` shows the full workflow on
real thermodynamic data. Each file opens with notes on its assumptions, units, ordering and
limitations.

**In a notebook.** Each file is split into `# %%` cells. Open it in VS Code or Jupyter and run
the cells in order in an Interactive window: edit the Settings cell first, then run the cells
below it. Run cells only execute when the file is run directly, so tests and other code can
import its functions without starting a solve.

**Notebook working directory:** start the kernel at the **extraction repository
root** (the directory containing `packages/`). Check it with `Path.cwd()`. Each
example's first import cell adds `packages/moving-boundary/examples` to `sys.path`
using that working directory; it does not search for the repository or change directories.
If your kernel starts elsewhere, change its working directory to the extraction root
before running the cells. Keep `advanced/` beside copied scripts and adjust the setup
path when using a different layout. The open Fe-Cr-Ni example locates its bundled
database through its loaded companion, so notebook cells do not need `__file__`.

**From a terminal at the extraction repository root.** Run
`python packages/moving-boundary/examples/<example>.py`, adding `--fast` for shorter
settings, or `--help` for other options.

## A shorter introductory path

The seven structured examples expose `run_core(...)`: prepare inputs, build the model,
solve once and inspect basic results. Physical inputs and model construction remain in
the main file. Existing `run(...)` calls and terminal commands retain the full validation
workflow; detailed checks and orchestration live in `advanced/<example>.py`.

Starting at the **extraction repository root**, in an interactive Python session with
the package installed and the repository's `kawin` environment activated:

```python
from pathlib import Path
import sys

sys.path.append(str(Path("packages/moving-boundary/examples").resolve()))

from binary_planar_growth import FAST, run_core
from moving_boundary.plotting import plot_interface_history, plot_composition_profiles

result = run_core(FAST)
print(result.metrics)
plot_interface_history(result.model)
plot_composition_profiles(result.model)
```

For a real database-backed case, import `WORKFLOW` and `run_core` from
`surrogate_workflow_fecrni_open`. This uses the bundled open database and pycalphad,
without starting Thermo-Calc or running the advanced surrogate comparisons.
Use the package plotting helpers for core results; the example-specific `plot`/`figures`
functions can require the exact solutions, metrics and reports returned by the full `run`.

See [Reusable workflows](../docs/example-workflows.md) for the package APIs, strict
bundle reuse, inventory normalization, plotting options and advanced entry points.

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

## Citing sources

Every example cites the papers its values, equations, cases and comparison data come from.
This applies to new examples and to changes to existing ones:

- **In-line, numbered.** Cite where the information is used, in the notes, in a comment beside
  the value or in a docstring, with a number in square brackets: `[1]`, or `[2, Table 3]` with
  a table, figure, equation or page where it helps.
- **A References cell at the end** of the file (`# %% [markdown]` then `## References`), as a
  numbered list in order of first citation. Give authors, title, journal, volume, year and
  pages, and a DOI when one is confirmed, followed by a short note of what the example takes
  from that source.
- **Check each reference against the source itself** or another reliable record, never from
  memory. A source known only through another paper's reference list is marked "as cited in
  [n]", and a detail that couldn't be confirmed is left out rather than guessed.
- **Say when nothing comes from the literature.** A synthetic example states that its numbers
  are synthetic, and cites only the sources of any method or exact solution it relies on.

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

Numerical limits are documented beside the advanced checks with their exact settings, normalization and time window
they apply to. They don't carry over to other settings.

## Fe-Cr-Ni database

`ternary_fecrni_lee1996.py` needs a TDB with FE, CR and NI, BCC_A2 and FCC_A1, and mobilities.
Pass it with `--tdb PATH` or set `MB_EXAMPLE_TDB`. **Its database is not included**, because
redistribution of the one used in development hasn't been cleared. It was assembled from
Lee (1993) with mobilities from Lee & Oh (1996). The sampled probe point and compositions in the
example suit that database at 1373 K, and may need changing for another. The open database
below is a different database.

## Open Fe-Cr-Ni database and surrogate workflow

`FeCrNi_MatCalc_open.tdb`, used by `surrogate_workflow_fecrni_open.py`, is an existing adapted
excerpt of the MatCalc open steel databases, copied unchanged from the kawin repository's
`examples/FeCrNi.tdb` (upstream kawin, added 2022; sha256 `f713f965…`). Its header notices
are kept as they are. It ships in the source distribution, not in the wheel.

- **Thermodynamic parameters:** MatCalc thermodynamic steel database mc_fe, release 2.060
  (`mc_fe_v2.060.tdb`), by Erwin Povoden-Karadeniz, TU Wien.
- **Mobility parameters:** MatCalc diffusion mobility database for steel (mc_fe), by Erwin
  Povoden-Karadeniz, TU Wien. They match releases 2.012, 2.015 and 2.016 alike, so no single
  release is identified; 2.012 is only a possible historical source.
- **Licence:** both are made available under the
  [Open Database License 1.0](http://opendatacommons.org/licenses/odbl/1.0/); any rights in
  individual contents are licensed under the
  [Database Contents License 1.0](http://opendatacommons.org/licenses/dbcl/1.0/). This excerpt,
  a derived database, is redistributed under the same terms. Source:
  [MatCalc open databases](https://www.matcalc.at/index.php/databases/open-databases).
- **Adaptations already in the excerpt** (not made by this package): one upper temperature
  limit written `6000.00.00` in the releases is corrected to `6000.00` (`G(SIGMA,FE:CR:FE;0)`);
  twelve BCC mobility entries spell out the releases' "any solvent" entries
  `MQ/MF(BCC_A2&X,*)` as explicit Fe and Ni entries with the same values; and
  `MQ(FCC_A1&FE,FE:*)` writes out the release's helper functions (QFE_FCC = 286000,
  D0FE_FCC = 7.0E-5) as `-286000+R*T*LN(7.0E-5)`.
- **Header notes:** the header names release 2.060 for the parameters and 2.059 in its licence
  sentence; the official 2.060 file carries the same 2.059 sentence. The header credits only the
  thermodynamic database; the mobility source is credited here and in `NOTICE`. The reference
  key `jac17` is undefined in the MatCalc reference lists; it may refer to Aurélie Jacob, but
  that is unconfirmed.

The workflow builds a pycalphad bundle (and, optionally, a Thermo-Calc one), runs the
saved-bundle and fresh-query diagnostics, compares the two bundles and solves a 1 h lagged
couple from each. With `--thermocalc saved` a Thermo-Calc bundle made elsewhere is reused
without `tc_python`; only `--thermocalc build` and `--tc-fresh-queries` need TC-Python and a
licence. The Thermo-Calc cache goes to a short folder of its own (`--tc-cache-dir` or
`MB_TC_CACHE_DIR`), because Thermo-Calc truncates long workspace paths. Thermo-Calc labels the
phases `BCC_A2#1` and `FCC_A1#1`, so the comparison uses an explicit `phase_mapping`. Fresh-query
discrepancies are reported separately for queries that coincide with training samples (which
only shows the stored data are reproduced) and held-out ones; none of the results is an accuracy
claim for the Fe-Cr-Ni physics, and similar interface motion from both backends doesn't show
that either surrogate or profile is accurate.

The open Fe-Cr-Ni example also plots sampled diffusivity values, separately from
surrogate predictions. New bundles store genuine calculation records for either backend
in an optional, hash-checked `kinetics_diagnostics` JSONL member. The plot shows the stored
matrices and adds mobilities, thermodynamic factors and sublattice details when available.
Older bundles without this member remain usable and show sample values without those
details. Reloading never regenerates records; use `--rebuild` explicitly to capture them.
Thermo-Calc-derived records remain local in the output bundle.

Surrogate bundles are written to `~/moving_boundary_example_outputs/<example>/`, wherever you run
from, unless you set `OUTPUT_DIR` (or pass `--output-dir`). They never go into the source tree. Existing compatible bundles are reloaded, incompatible
or damaged ones stop with an error, as does a bundle folder with files but no `manifest.json`,
and `--rebuild` replaces them. A Fe-Cr-Ni bundle records the
`moving-boundary` version, so a new release won't reload an old bundle; within one version,
such as in an editable install, rebuild after changing surrogate code.

## Running the tests

The tests need the source distribution or repository, not just the wheel. From the package
directory, using the tested dependency versions:

    python -m pip install ".[test]" -c examples/requirements-tested.txt
    python -m pytest tests/test_analytic.py tests/test_examples.py

Without a database, tests that need one skip with a reason. To include them, set `MB_EXAMPLE_TDB`.
The open Fe-Cr-Ni workflow tests use the bundled database and need no Thermo-Calc licence.
Slower refinement studies are deselected by default; run them with `-m slow`.
