# Reusing the example workflows

The main scripts in `examples/` keep physical inputs, units, component ordering,
initial profiles, grids and model construction visible. Their `run_core` functions
prepare inputs and solve once. The original `run` functions and command-line
options retain the detailed checks, implemented in adjacent `examples/advanced/`
companions. Keep that directory beside the scripts when copying examples.

## Notebook tutorials

For direct, cell-by-cell teaching examples, see the
[notebook collection](../examples/README.md#notebook-tutorials). It covers binary bonding,
Fe-Cr-Ni simulation, verified surrogate reuse, assessment, resolution studies,
interstitial composition bases, three phases and Thermo-Calc. Each notebook uses
package APIs without importing example modules. Kernels may start at the extraction
repository root or in `packages/moving-boundary/examples`; database-backed tutorials
handle either location. The script-import workflow below still requires the root.

## A short first run

Start at the **extraction repository root**, with the package installed and the
repository's `kawin` environment activated. The notebook kernel's working directory
must be that root; check it with `Path.cwd()`. The example import cells use this
assumption directly, without searching parent directories. In an interactive Python session:

```python
from pathlib import Path
import sys

sys.path.append(str(Path("packages/moving-boundary/examples").resolve()))

from binary_planar_growth import FAST, run_core
from moving_boundary.plotting import plot_interface_history, plot_composition_profiles

result = run_core(FAST)
print(result.metrics)
interfaces = plot_interface_history(result.model)
profiles = plot_composition_profiles(result.model)
```

The plotting helpers return Matplotlib axes and never display implicitly. Call
`matplotlib.pyplot.show()` when ready. `ax=` accepts existing axes. Interface
coordinates are `(time + time_offset) * time_scale` and
`(position - position_offset) * position_scale`. Profile `elements=` selects
component names; the default selects all independent components. Physical profiles
are mole fractions, including when the solver state uses interstitial u-fractions.

For a real database-backed run, after the same repository-root path setup:

```python
from surrogate_workflow_fecrni_open import WORKFLOW, run_core

result = run_core(WORKFLOW)
print(result.metrics)
```

This builds or reloads one pycalphad bundle, solves once and checks recorded
inventory drift and final-state sample coverage. It does not start Thermo-Calc,
run leave-one-out validation or make fresh backend queries on reload. Bundle
outputs use the example's existing output directory; pass `output_dir=` to change
it. `rebuild=True` explicitly replaces an existing bundle.

The Lee example's `run_core` accepts a supplied `surrogate=` and retains the
configuration's frozen/lagged construction. Its basic inventory check compares
final with initial inventory. The analytic examples also retain their physical
setup but leave detailed accuracy evaluation to `run`. The lifecycle core omits
corruption demonstrations and fresh-versus-reloaded twin solves.

## Installed package capabilities

These APIs need no example imports and are available from a wheel:

| API | Contract |
|---|---|
| `providers.constant.ConstantDiffusivity(values)` | Positive finite scalar per phase; ignores composition and temperature; preserves temperature-array return shapes. |
| `sampling.ternary_simplex_points(spacing, floor=1e-5)` | Ordered independent-component mole fractions; spacing is the reciprocal of an integer. Buffered projection and ten-decimal rounding preserve the original example grids. |
| `surrogate.get_diffusivity_samples(phase, query_context="general")` | Copies of retained usable composition and matrix arrays, in fit order. Also accepts `"interface"`; excludes dropped/unusable and duplicate samples. |
| `run_diagnostics.inventory_drift(values, reduction="max")` | `(absolute, relative)` maxima across independent components. Rows are times; a 1D input is a scalar history. `"max"` includes excursions; `"final"` compares only the last row. Zero initial inventory gives zero if unchanged, infinity otherwise. |
| `run_diagnostics.evaluate_bulk_coverage(surrogate, compositions_by_phase)` | Per-phase membership mask, query count, outside count and retained-sample count. Inputs use the surrogate's component basis and order. Checks supplied states only. |
| `plotting.plot_interface_history(model, ...)` | All recorded interfaces; optional offsets, scaling and existing axes. |
| `plotting.plot_composition_profiles(model, ...)` | Physical mole fractions; optional component selection, time, position scaling and existing axes. Historical ternary profiles require recorded transformed states. |

Coverage uses a Delaunay hull for full-rank samples. With no two-dimensional hull,
only retained sample matches within the documented floating-point tolerance count
as supported. Nonfinite queries count as outside. Coverage is not an accuracy test.

## Strict bundle reuse

`surrogate_artifacts.load_or_build_surrogate_artifact(path, spec, builder, ...)`
returns `(bundle, "built" | "reloaded")`. Its zero-argument builder returns
`(surrogate, extra_members)`. A missing or empty destination is built; a compatible
bundle is reloaded. Incompatible or damaged bundles, and nonempty directories
missing their manifest, raise without calling the builder. Only `rebuild=True`
permits replacement. `required_members=` is enforced before publication and on
reload. The older `prepare_surrogate_artifact` API keeps its original defaults.

`load_or_build_diagnostic_surrogate(path, spec, builder, ...)` adds the standard
diagnostic sidecars. Its callback is `builder(records) -> surrogate`; `records` is
`None` unless `capture_kinetics=True`, when it is a list for genuine captured backend
calculations. `report_roles=` selects `failed_bulk_points` and/or
`invalid_bulk_points` from surrogate metadata. Capture requires `backend=` and
`elements=`. Reload never invokes the callback or regenerates records; older
bundles lacking optional kinetics records remain usable.

For custom publication flows, `write_surrogate_diagnostic_members(directory,
surrogate, ...)` returns member descriptors. The caller owns that staging directory
and must keep it alive until publication copies its files. This helper never
queries a backend or evaluates fitted predictions.

## Detailed checks

Existing commands and `run(...)` calls retain their configurations, result fields,
metrics and checks. Detailed implementation lives in the matching advanced
companion, which receives explicit case inputs and callbacks and never imports
the main script. Its scientific assumptions and references remain in the main
example, with numerical details beside the checks.

The open Fe-Cr-Ni workflow retains saved-only Thermo-Calc operation without
`tc_python`, explicit backend phase mapping, short-cache requirements and early
failure for unavailable live requests. Neither backend is treated as ground truth.
Accuracy bounds remain specific to the original named configurations; inventory
conservation, coverage and successful reload do not establish accuracy.
