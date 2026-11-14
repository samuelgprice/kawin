# Manual package performance benchmarks

This is a developer measurement tool, independent of pytest, acceptance runs,
reference selection and approvals. No solver code, defaults or criteria are
changed. Run it periodically on the same machine under similar background load.
Close other CPU-heavy programs and keep power/thermal settings consistent.

From the extraction root in PowerShell:

```powershell
# Standard suite: one warm-up plus seven measured samples of each case.
cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && python packages/moving-boundary/tools/performance.py run'

# Smoke: identical meshes, three scheduled steps, one warm-up and two samples.
cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && python packages/moving-boundary/tools/performance.py run --mode smoke --timeout 180'

# Substitute two saved results.json paths chosen explicitly by you.
cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && python packages/moving-boundary/tools/performance.py compare "BASELINE/results.json" "NEW/results.json"'
```

Each run reserves a unique UTC/UUID directory under the package's ignored
`.verification/benchmarks/`. It holds `request.json`, `results.json`, build,
installation and worker logs, a fresh private package install, and a private
Numba cache. `wall-time.json` records total elapsed time including installation.
Comparisons reserve their own unique folder under `benchmarks/comparisons/`,
containing both `comparison.json` and `comparison.txt`. They never write either
input or automatically select/promote a baseline. Timing review flags do not
produce a failing exit status; invalid simulations and execution failures do.

Curated, tracked milestone results are kept in `benchmarks/baselines/`; see the
[saved baseline instructions](../benchmarks/baselines/README.md). Export local
results with the documented helper before comparing them with a saved baseline,
so local path normalization is consistent on both sides.

The launcher builds an offline wheel and installs it into a **new empty private
target on every run**, using the activated `kawin` environment for pip. It then
activates the selected public Conda environment for an isolated `python -S`
worker. Only the fresh target, host site-packages and package tools are added to
its import path. Editable `.pth` hooks are not executed; both package and kawin
import origins are checked. Nothing is installed into `mb-public-050` or any
shared environment. Pip needs setuptools/wheel already available; there is no
network access or dependency installation. Build and install each have a
120-second deadline, and the worker defaults to 1,200 seconds with process-tree
termination on timeout. stdin is closed and unexpected `input()` calls raise.

The default host comes from `.verification/public-environment.json` when present,
otherwise the active environment. To choose the sibling `mb-public-050` explicitly,
append the following to a `run` command launched from the extraction repository root:

```powershell
--environment "..\mb-public-050"
```

The argument is resolved from the command's working directory. If your environment
lives elsewhere, substitute its location locally or use the ignored environment
pointer; keep machine-specific absolute paths out of committed documentation.

Use a host with **unmodified public kawin==0.5.0**, NumPy, SciPy and the package's
other dependencies. The host environment pointer is optional; no private
acceptance input, database, license or research checkout is needed. An export
without Git works with `--environment`; Git provenance is then marked unavailable.
The Windows activation script path follows this repository's environment rules.

## Versioned workloads

`tools/performance_cases.json` fixes profile sizes, timesteps, durations (step
count times timestep), recording policy and repetitions. `performance_cases.py`
owns the deterministic analytic providers and constructors. Configuration and
provider hashes, constructor defaults, effective numerical controls and solver
constraints are saved. A changed workload or effective control is incompatible
with an old result, even if the case ID matches. Intentional workload changes
should increment the configuration version and be reviewed as code changes;
there is no automatic calibration or retuning.

| Case | Diffusivity | Phase nodes | Physical nodes | Recording |
|---|---|---:|---:|---|
| binary-constant-small-record | constant, phase uniform | 21/21 | 101 | on |
| binary-lagged-large-off | composition dependent, lagged | 161/241 | 801 | off |
| binary-implicit-large-record | composition dependent, Picard | 161/241 | 801 | on |
| ternary-constant-small-record | constant coupled matrices | 21/21 | 101 | on |
| ternary-constant-large-off | constant coupled matrices | 161/241 | 801 | off |
| ternary-lagged-large-record | composition dependent, lagged | 161/241 | 801 | on |
| ternary-implicit-small-off | composition dependent, Picard | 21/21 | 101 | off |
| three-constant-small-record | constant coupled matrices | 9/9/9 | 101 | on |
| three-lagged-large-off | composition dependent, lagged | 65/65/65 | 401 | off |

Standard runs have 1,600 scheduled fixed steps per trajectory. All models use
closed planar domains, substitutional compositions, 1,000 K, and uniform
transformed grids. Three-phase uses equal explicit molar volumes, allowing a
physical molar-conservation check. This compact selection exercises scalar and
block solves, interface startup, nonlinear bulk iteration, reconstruction and
history storage without an exhaustive parameter product. It does not isolate
recording overhead through matched pairs or cover all supported physics.

The provider/geometry ideas come from package-owned open substitutional fixtures
and `benchmarks/benchmark_illingworth_ternary_refactor.py`. They are independent
copies, not imports of research scripts, test builders, or acceptance definitions.
Existing acceptance wall times include installation/reference checks and are
unsuitable as a solver-performance baseline. The surrogate research benchmark
framework is not ported. Three-phase implicit bulk mode is supported but omitted
from this compact default set; binary and two-phase cases exercise Picard
iteration. Unequal molar volumes and interstitial systems are also outside this
initial performance suite.

## Measurement and validity

Each case warms up once, then uses fresh model, mesh and provider instances for
every measured trajectory. The process, compiled/JIT code and operating-system
caches are warm; analytic providers have no mutable caches. A new run uses a new
process and private JIT directory, so first-use compilation belongs to the
discarded warm-up. Garbage collection of prior instance wrapper cycles occurs
before construction, outside timings. Normal Python GC remains enabled during
measurement. `time.perf_counter()` is the monotonic high-resolution clock.

Four metrics retain every individual sample, median, minimum, maximum, median
absolute deviation (MAD) and relative MAD:

- `setup`: first `model.setup()` nested inside the public solve, excluding
  construction and instrumentation. This overlaps `solve`.
- `solve`: full public `model.solve()` on a fresh uninitialized model, including
  first setup, numerical work, recording, first finalization and the
  lightweight counting wrappers. Imports, construction, validity checks and
  result serialization are excluded.
- `finalize`: first `postSolve()` inside that solve. This overlaps `solve`; it
  measures normal finalization, not a second already-finalized no-op.
- `reset`: one `reset()` on the completed, validated model, excluding re-setup.

The postProcess wrapper counts successful accepted time advances. Near-final
binary/two-phase clock-only commits are recorded separately as
`final_noop_commits` and do not double-count nonlinear work. Available last-step
interface iteration, candidate/Jacobian, Picard, diffusivity-evaluation and retry
counters are summed over accepted steps, with individual per-sample totals.
These diagnostics have solver-specific meanings: some refer only to the final
successful candidate/attempt; they are **not** totals of every rejected solve.
Unavailable counters are `null`. Work changes are reported independently of
timing, and unequal sample counts alone are not a work change.

Every warm-up and sample must reach the requested final time, have enough
accepted steps, finite physical phase profiles within the substitutional
simplex, ordered interfaces and coordinates, and correct recorded times when
recording is enabled. Existing solver conservation diagnostics must remain
finite and below the versioned absolute inventory bound (`1e-7` for these closed,
nondimensional synthetic cases). Three-phase uses all-component physical molar
conservation. Reset must return to time zero. These are validity guards, not a
replacement for the package's numerical tests or an analytic accuracy study.
Failures retain their tracebacks and are never presented as speed improvements.

## Reading comparisons

Ratios are **candidate median / baseline median** for each compatible case and
metric. Missing cases/metrics, failed runs, changed configurations and changed
measurement protocols are explicit. Incompatible pairs have no ratio. Python,
library versions, platform, machine, NumPy/BLAS configuration and thread-variable
differences are listed; ratios under differing environments are marked
`environment_differs` and are confounded. Package source fingerprints, wheel
hash, optional Git revision/dirty status, actual target/host origins and creation
time remain in the result provenance. A dirty Git revision alone does not
identify installed code; use the saved per-file fingerprints as well.

Default advisory review requires **both** an absolute median increase over 2 ms
and a relative increase above `max(20%, 6 * max(relative MADs))`. Early local
samples showed several percent variability, motivating this deliberately
conservative starting policy. This heuristic is not a confidence interval or a
significance test. Tiny setup/reset/finalization changes may sit below the
absolute threshold; their raw timings and ratios are still reported.
Configure with `--slowdown 0.20 --noise-multiplier 6 --minimum-seconds 0.002`.
Repeat a flagged run under matched conditions and inspect work counts before
attributing it to a revision. Seven samples do not characterize long-term noise.

For a shorter follow-up at unchanged standard settings, use `run --samples 2`;
this changes repetition count only and remains comparable, though less reliable.
`--case ID` can be repeated to measure a subset; missing cases stay explicit.
Smoke versus standard trajectories are incompatible because their step counts
differ. The first local measurement alone establishes **no historical regression**.

## Opt-in harness checks

```powershell
cmd.exe /d /c 'call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3\envs\kawin" && python -m pytest packages/moving-boundary/tools/performance_checks.py --basetemp .pytest_tmp/performance-checks -q'
```

These focused tests use synthetic clocks, models and timing data. Their filename
is outside ordinary pytest discovery. They assert no machine speed and execute
no solver benchmark. Neither ordinary tests nor acceptance launches this tool.
