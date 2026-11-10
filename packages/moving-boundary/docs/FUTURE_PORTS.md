# Future ports: research functionality deferred after the authority switch

Status: **backlog only. Every entry is deferred and not implemented.** Recorded 2026-11-10 at
extraction `518162b`. Nothing here is part of the authority-switch work
([AUTHORITY_SWITCH_PLAN.md](AUTHORITY_SWITCH_PLAN.md), batches M0-M10), and no entry may expand it.

- **Provenance:** research main frozen at `7af24f4` (clean oracle worktree `mb-ref-7af24f4`).
  Each port records the source path and `7af24f4` and changes only the package. Porting these
  tools needs no further changes to research main.
- **Sequence:** authority switch (items 14-18 and 7) → R07/R08 (items 1-2) → entries 1-2
  (diagnostics, comparison) → entries 3-6 in order of workflow priority.
- **Reminder: revisit this backlog after R07/R08 are adopted**, before choosing the next work. Set
  the priority of entries 3-6 then, and confirm the entry 6 decisions with the user.
- **Rules for any port:** no numerical references or tolerances change; tests are ported with the
  feature (see [Deferred tests](#deferred-tests)); and for code that currently affects solver
  outputs, the port is a relocation with proof (bit-identical gates), not an approval.

## Does current acceptance need any of this?

**No.** No package tool, test, example or source module executes any of the entries below (checked
2026-11-10). The only links are historical:

| Link | Where | Normal acceptance? |
|---|---|---|
| `examples/ThermoCalc/training_data.py` (`build_moving_boundary_surrogate`) | `tools/acceptance_thermocalc.py` **reference role** for O06 | **No.** Package role uses the tooling copy `_installed_surrogate()`, which reproduces `training_data.py:187-223` through installed APIs. Under the authority-switch plan the reference role is historical-only. |
| `IllingworthTernaryConvergence.py`, `IllingworthTernaryPlotly.py`, `IllingworthTernaryThreePhaseStable.py`, `kawin/tests/test_ternary_plotly_helper.py`, `training_data.py` | `tools/acceptance_manifest.json` `source_sha256` (research-checkout integrity hashes) | **No.** Only read when the research checkout is validated, which becomes historical-only (plan §2). |

Entry 5 doesn't port `build_moving_boundary_surrogate` (it is a demo wrapper; see entry 5), so the
O06 tooling copy stays. Any later change to how O06 generates its surrogate is a relocation inside
O06's byte gate: it must stay bit-identical and needs live Thermo-Calc to prove.

## 1. Surrogate diagnostics (`SurrogateDiagnostics.py`)

**Status: deferred, not implemented.** Priority: first after R07/R08 (with entry 2).

- **Research source:** `kawin/diffusion/SurrogateDiagnostics.py` (3239 lines). Public API:
  - evaluate: `evaluate_tieline_diagnostics`, `evaluate_interface_bulk_diffusivity_consistency`,
    `evaluate_diffusivity_leave_one_out`, `evaluate_diffusivity_diagnostics`,
    `evaluate_surrogate_construction_diagnostics`;
  - plot / report: `plot_tieline_diagnostics`, `plot_interface_diffusivity_diagnostics`,
    `plot_bulk_diffusivity_diagnostics`, `plot_diffusivity_leave_one_out`,
    `plot_surrogate_construction_diagnostics`, `plot_selected_diffusivity_calculations`,
    `plot_surrogate_diagnostics` (interactive HTML/plotly);
  - site fractions: `load_calculation_site_fractions`, `plot_calculation_site_fractions`.
- **Research tests:**
  - `kawin/tests/test_surrogate_diagnostics.py` (36);
  - the 3 construction-diagnostic tests in `kawin/tests/test_illingworth_ternary.py` (see
    [Deferred tests](#deferred-tests));
  - `kawin/tests/test_no_debugger_dependencies.py` (4; imports the diagnostics module, port the
    relevant check).
- **Research users:** `examples/ternaryExamples/FeCrNi/IllingworthTernaryTwoPhase_FeCrNi_Lee1996Cases.py`,
  `IllingworthTernaryTwoVsThreePhaseSymmetry_copy.py`, `examplesForZhaoxi/*` (3 scripts).
- **Already in the package:** everything the module imports exists in
  `moving_boundary.surrogates`: `TernaryMovingBoundaryThermodynamicsSurrogate`,
  `MergedPhaseDiffusivitySurrogate`, `_BulkDiffusivitySimplexLinear2D`,
  `_BulkDiffusivitySimplexPositive2x2`, `_matrix_validity_diagnostics`. The surrogate stores and
  round-trips `construction_records` (same handling as research). `scipy` and `tqdm` are declared
  dependencies.
- **Missing:** the whole diagnostics module (evaluation, leave-one-out, consistency checks, reports)
  and a declared `plotly` dependency (not in `pyproject.toml`; it should be an optional extra, not
  a core requirement).
- **Approach:**
  1. Relocate as `moving_boundary.diagnostics` (or `surrogate_diagnostics`) with import changes
     only, plus a provenance header.
  2. Keep plotly imports lazy and add a `[diagnostics]` optional extra.
  3. Port the 36 + 3 + relevant debugger-guard tests in the same change.
  4. Validate on surrogates with known values; the evaluation functions are pure numerics on saved
     surrogates. Report figures are checked structurally (trace and button labels), as the
     research tests do.

## 2. Surrogate comparison (`SurrogateComparison.py`)

**Status: deferred, not implemented.** Priority: first after R07/R08 (with entry 1).

- **Research source:** `kawin/diffusion/SurrogateComparison.py` (741 lines):
  `evaluate_saved_surrogate_comparison`, `plot_saved_surrogate_comparison`, `compare_saved_surrogates`.
- **Research tests:** `kawin/tests/test_surrogate_comparison.py` (10).
- **Already in the package:** its only internal dependency, `SurrogateArtifactBundle`
  (`moving_boundary.surrogate_artifacts`), with hash-validated saved bundles.
- **Missing:** the numerical comparison of two saved bundles and its visual report.
- **Approach:** relocate next to `surrogate_artifacts` with import changes only; plotting under the
  same optional extra as entry 1; port the 10 tests. Validate against pairs of bundles built from
  identical and deliberately different specs (zero and known deviations).

## 3. Surrogate benchmarking

**Status: deferred, not implemented.** Priority: set at the post-R07/R08 review.

- **Research source** (all under `kawin/diffusion/`):
  - `SurrogateBenchmark.py` (700 lines): `SpectralCriteria`, `SpectralRecord`, `classify_matrix`,
    `DiffusivityDataset`, validation splits and plans (`make_loo_plan`, `make_blocked_plan`,
    `make_independent_holdout_plan`), predictor protocols, `pointwise_metrics`,
    `run_paired_benchmark`, `result_rows`;
  - `SurrogateBenchmarkPhase2.py` (461): flux error, refinement study, `IllingworthRun` capture and
    comparison, `run_illingworth_benchmark`;
  - `SurrogateBenchmarkExperiment.py` (467): `load_thermocalc_surrogate_dataset`,
    `experiment_schemes`, `build_experiment_splits`, `run_experiment` (uses `subprocess`);
  - `SurrogateBenchmarkAnalysis.py` (270): analysis loading, failure breakdowns, simplex diagnostics.
- **Research tests:** `test_surrogate_benchmark.py` (18), `test_surrogate_benchmark_phase2.py` (10),
  `test_surrogate_benchmark_experiment.py` (12), `test_surrogate_benchmark_analysis.py` (6), and
  possibly `test_benchmark_wtife_surrogate_runtime.py` (5; classify when porting).
- **Reusable supporting tools:** `examples/ternaryExamples/benchmark_diffusivity_surrogates.py`,
  `examples/ternaryExamples/analyze_diffusivity_benchmark.py`.
- **Already in the package:** `MergedPhaseDiffusivitySurrogate` and `_signed_cuberoot`, plus only
  the two spectral *constants* in `_spectral_validation.py`.
- **Missing:** all four modules. `SpectralCriteria` and `classify_matrix` don't exist in the package.
- **Approach:**
  1. Port in dependency order: `SurrogateBenchmark` → `Phase2` → `Experiment` → `Analysis`, each
     with its tests.
  2. Phase 2 drives Illingworth solves, so the runs use the package solvers. Validate by running
     the research tests unchanged apart from imports.
  3. Experiment and analysis read TC-derived datasets: these stay local inputs (never committed),
     and the tests use synthetic datasets.
  4. Decide then whether the benchmark tools belong in the library or in a `tools/`/`benchmarks/`
     area.

## 4. Visualization and convergence tools

**Status: deferred, not implemented.** Priority: set at the post-R07/R08 review.

| Source | Content | Tests | Notes |
|---|---|---|---|
| `examples/ternaryExamples/IllingworthTernaryPlotly.py` (1456 lines) | `plot_two_phase_composition_profile`, `plot_three_phase_composition_profile` (interactive, with tie-line overlays) | `kawin/tests/test_ternary_plotly_helper.py` (19) | Imports `evaluate_tieline_diagnostics`, so it **depends on entry 1** |
| `examples/ternaryExamples/PlotlyHtmlToMp4.py` | Plotly HTML → MP4 export (subprocess-based) | `kawin/tests/test_plotly_html_to_mp4.py` (6) | **Neither the converter nor its tests are tracked at `7af24f4`.** Both exist only untracked in the research working tree. Before porting, capture provenance for both (e.g. record their sha256 and date in this file). The user keeps the test file out of commits, so how provenance is captured is the user's decision; don't commit either file to research without asking. |
| `examples/ternaryExamples/IllingworthTernaryConvergence.py` (560 lines) | `default_convergence_config`, `run_convergence_sweep`, `summarize_convergence`, plots, `save_convergence_results` | `kawin/tests/test_illingworth_ternary_convergence.py` (5) | Uses `importlib` to load example modules: replace with package imports |
| `plotMovingBoundaryState` in `kawin/diffusion/Plot.py` (l.247) | state plot for moving-boundary models and meshes | covered in `kawin/tests/test_plotting.py` (9 tests in total, only some relevant) | Research callers are the Lee-Oh/generic moving-boundary models (`MovingBoundary.py`, `MovingBoundaryFDM.py`, mesh `MovingBoundary*FD1D.py`), **not** the Illingworth solvers. Port only the Illingworth-relevant parts. |

- **Already in the package:** matplotlib plot helpers on the solvers
  (`plot_latestCompProfile`, `plot_phaseWidths_vs_time`, `plot_eta_vs_time` in `ternary.py`,
  `plot_latestCompProfile` in `three_phase.py`) and the transformed-grid plot in
  `mesh/TransformedGrids.py`. No plotly, MP4 or convergence-sweep tooling.
- **Approach:**
  1. Port after entry 1, under the same optional plotting extra.
  2. Port the convergence sweep as a tool that drives the package solvers. Its 5 tests come with it.
  3. Validate with the ported tests, plus a structural check of the figures. Animations and MP4
     stay manual checks; CI must not need `ffmpeg`.

## 5. Training-data workflow utilities (`examples/ThermoCalc/training_data.py`)

**Status: deferred, not implemented.** Priority: set at the post-R07/R08 review.

- **Research source:** `examples/ThermoCalc/training_data.py` (373 lines).
- **Research tests:** `kawin/tests/test_thermocalc_examples.py` (61; it also covers
  `tc_python_adapter`, which is already ported, so split them when porting).
- **Reusable utilities vs demo-specific wrappers:**

| Reusable workflow (candidate for the package) | Demo-specific (stays a demo/example) |
|---|---|
| `initialize_dataset`, `sample_training_data` (sampling loop with checkpoint/resume and failure reporting), `save_training_dataset`, `load_training_dataset` (dataset export/load) | `build_moving_boundary_surrogate` (Fe-Cr-Ni defaults, calls `make_fecrni_demo_grid`); `make_fecrni_demo_grid` (Fe-Cr-Ni demo grid); `examples/ThermoCalc/FeCrNi_TC_Python.py` (driver script) |

- **`build_moving_boundary_surrogate` stays outside the library.** This keeps the existing decision
  in [NEXT_SESSIONS.md](NEXT_SESSIONS.md) ("do not move `training_data.py:build_moving_boundary_surrogate`
  into the package"; it is a thin Fe-Cr-Ni demo wrapper). A general builder (session handling and
  provenance metadata only, no system defaults) would be a **separate design decision**, not part
  of this entry. Real workflows already call the installed `from_database`, usually inside
  `prepare_surrogate_artifact`.

- **Already in the package:** the TC adapter itself (`moving_boundary.providers.thermocalc`:
  `ThermoCalcConfig`, `TCPythonThermodynamics`, composition helpers, error classes), and the
  acceptance tooling copy of the sampling call (`tools/acceptance_thermocalc._installed_surrogate`,
  `training_data.py:187-223`), which is not a library API.
- **Missing:** a library sampling workflow with checkpoint/resume, failure reports and dataset
  export/load. (Not the demo builder; see above.)
- **Dependencies:** TC-Python and licence (optional, local only); TC-derived datasets are never
  committed.
- **Related:** the TC caches in [Cache rebuilding gaps](#cache-rebuilding-gaps) are built by study
  scripts (entry 6), not by `training_data.py`. These utilities may help port those builders, but
  porting entry 5 alone does not close the gaps.
- **Approach:**
  1. Port the reusable functions into `moving_boundary.providers` (or a `training` module) without
     changing their arithmetic.
  2. Unit-test checkpoint/resume and failure reporting with a fake thermodynamics source, so no
     licence is needed.
  3. O06's package role keeps its tooling copy of the sampling call (`_installed_surrogate`), since
     the builder isn't ported. Any later change to how O06 generates its surrogate is a relocation
     inside O06's byte gate: it must be bit-identical to the frozen fixture, proven on live TC.

## 6. Active material-specific research workflows

**Status: deferred, not implemented.** This is a tracking list, not a commitment to put study
scripts into the library. Four of these scripts generate private acceptance caches, so porting
them also closes the [cache rebuilding gaps](#cache-rebuilding-gaps). For each script the user decides: **migrate to package imports**
(script stays a study, imports `moving_boundary`), **package example** (becomes a maintained
example), or **keep frozen** (stays on research `7af24f4`). Nothing is decided yet.

Illingworth-using scripts at `7af24f4` (excluding the Lee 1999, Olaye 2020 and comparison-code
folders, whose solvers aren't in the package):

| Script (`examples/…`) | Package counterpart today | Decision |
|---|---|---|
| `ternaryExamples/FeCrNi/IllingworthTernaryTwoPhase_FeCrNi_Lee1996Cases.py` | `examples/ternary_fecrni_lee1996.py` (partial) | open |
| `ternaryExamples/FeCrNi/IllingworthTernaryTwoPhase_FeCrNi_AnalyticComparison.py` | builders become acceptance definitions (plan M4) | open |
| `ThermoCalc/FeCrNi_TC_Python.py` | — (depends on entry 5) | open |
| `Interstitials/fe_c_exb1a.py`, `fe_mn_c_ferrite.py`, `fe_mn_c_long_time.py`, `fe_ni_c_exh2_le.py`, `ternary_ferrite_growth_le.py`, `three_phase_symmetric_check.py`, `convergence_study.py`, `check_tc_interstitial_frame.py` | `examples/ternary_interstitial_similarity.py` (related only) | open |
| `ternaryExamples/exb5_gamma_alpha_gamma.py` | builders become acceptance definitions (plan M4) | open |
| `ternaryExamples/IllingworthTernaryExamples.py`, `IllingworthTernaryThreePhaseStable.py`, `IllingworthTernaryThreePhaseNiTiNb_TC.py` | `examples/three_phase_similarity.py` (related only) | open |
| `ternaryExamples/IllingworthTernaryTwoVsThreePhaseSymmetry.py` and `_copy`, `_copyNiTiNb` | — | open |
| `ternaryExamples/examplesForZhaoxi/*` (CuNiTi solidification, CuVTi and WTiFe bonding) | — (uses entries 1 and 4) | open |
| `Illingworth2005/replicate_illingworth2005_fig4_5_6_planar.py`, `compare_illingworth2005_planar.py`, `compare_saved_planar_runs.py` | `examples/binary_planar_growth.py`, `moving_boundary.analytic` | open |
| `ternaryExamples/ternaryTwoPhaseSimilarity.py`, `ternaryThreePhaseSimilarity.py` | `moving_boundary.analytic` (similarity solutions already ported) | likely superseded; confirm |
| DICTRA comparisons (`examples/DICTRAComparisons/`) | — | open. **Not at `7af24f4`:** it lives on `feature/DICTRA-comparison` and is untracked on research main, so its provenance is that branch. |

- **Approach for "migrate to package imports":**
  1. Change imports to `moving_boundary` (plus entry 1/4/5 APIs where needed).
  2. Run once on frozen research `7af24f4` and once on the package, at the script's own settings.
  3. Compare outputs. Expect bit-identical results until R07/R08; after they are adopted, expect
     the approved differences, documented per script.

## Cache rebuilding gaps

Four private acceptance inputs (O01-O04, O07) have **no package generation path**. They are
adopted as hash-validated local inputs, so acceptance works, but the package can't rebuild them
yet. The authority-switch regeneration check (plan §3a, M5b) reports them as known gaps,
**not verified**. Each gap closes only when its generator is ported, the cache is regenerated into
scratch, and the result is compared with the adopted file (same rules as M5b: adopted files and
hashes are never replaced). **Status: deferred, not implemented.**

| Cache (`.verification/acceptance/inputs/`) | Cases | Research generator at `7af24f4` (source recorded in `private-inputs.json`) | Backend | Entry |
|---|---|---|---|---|
| `exb5.json` | O01, O02 | `examples/ternaryExamples/exb5_gamma_alpha_gamma.py` `build_tc_inputs()` (l.151) → `outputs/exb5/tc_inputs.json` | Thermo-Calc | 6 (that script); TC sampling utilities in 5 are related |
| `femnc.json` | O03a, O03b | `examples/Interstitials/ternary_ferrite_growth_le.py` `build_tc_inputs(case)` (l.243), with `case = fe_mn_c_ferrite.CASE` → `outputs/fe_mn_c_ferrite_le_tc_inputs.json` | Thermo-Calc | 6 |
| `fec.json` | O07 | `examples/Interstitials/fe_c_exb1a.py` `build_tc_inputs("exb1a", n_austenite)` (l.178) → `outputs/fe_c_exb1a_tc_inputs_v2.json` | Thermo-Calc | 6 |
| `fecrni.surrogate/` | O04, E06 | `examples/ternaryExamples/FeCrNi/IllingworthTernaryTwoPhase_FeCrNi_Lee1996Cases.py` (bundle at l.1224, stem from `_diffusivity_artifact_stem()`, l.~720) → `ThermoCalc/outputs/FeCrNi_pycalphad_1373K_default_phases_0_g_offset_0p0_bulk_0p01.surrogate` | pycalphad + Lee TDB | 6. The package example `ternary_fecrni_lee1996.py` might be able to build an equivalent bundle; **unverified** |

- **Inputs stay local and private.** TC-derived caches and Lee TDB-derived bundles are never
  committed, and their hashes stay in the local `private-inputs.json`.
- **Rebuild checks need the licensed backend** (TC licence, or the Lee TDB), so they are optional
  checks: report them as `unavailable` when the resource is missing, never as verified.
- **Line numbers** are from `7af24f4` and should be re-checked when porting.

## Deferred tests

The authority-switch test port (plan §6, item 16) **does not port** research tests that need
functionality from this backlog. They are listed here with their feature. They are not skipped in
place, not counted as ported, and get ported in the same change as their feature.

| Research test | Count | Needs | Entry |
|---|---|---|---|
| `test_illingworth_ternary.py::test_surrogate_construction_provenance_records_line_and_bulk_samples` | 1 | `evaluate_surrogate_construction_diagnostics` | 1 |
| `test_illingworth_ternary.py::test_surrogate_construction_dashboard_has_phase_tabs` | 1 | `evaluate_…` + `plot_surrogate_construction_diagnostics` (plotly) | 1 |
| `test_illingworth_ternary.py::test_surrogate_construction_report_includes_thermocalc_metadata` | 1 | `evaluate_surrogate_construction_diagnostics` | 1 |
| `test_illingworth_ternary_convergence.py` (whole file) | 5 | `IllingworthTernaryConvergence` | 4 |
| `test_three_phase_examples.py` (whole file) | 3 | the research script `IllingworthTernaryThreePhaseStable` | 6 (decision per script) |

Feature test files outside item 16 that are also deferred: `test_surrogate_diagnostics.py` (36),
`test_no_debugger_dependencies.py` (4, the diagnostics part), `test_surrogate_comparison.py` (10),
`test_surrogate_benchmark*.py` (18 + 10 + 12 + 6), `test_benchmark_wtife_surrogate_runtime.py` (5),
`test_ternary_plotly_helper.py` (19), `test_plotly_html_to_mp4.py` (6; untracked, provenance to be
captured first), the relevant parts of `test_plotting.py` (9), the
`training_data` part of `test_thermocalc_examples.py` (61, shared with the already-ported adapter),
and `test_fecrni_surrogate_artifact_recipe.py` (11; it checks the research FeCrNi recipe and Lee-Oh
figure-9 study scripts through their source, so it goes with entry 6; classified in M0).

Effect on the item 16 counts:

- `test_illingworth_ternary.py`: 152 of 155 tests are ported and 3 are deferred.
  `test_illingworth_ternary_convergence.py` (5) and `test_three_phase_examples.py` (3) are deferred
  whole.
- The other similarity-group files *are* portable: `FixedMatrixTernaryDiffusivity`,
  `solve_similarity_roots`, `compare_model_to_similarity`, `semi_infinite_validity_time`,
  `solve_three_phase_similarity_roots`, `SimilarityInitialProfile`, `tie_line_crossing_etas` and
  `compare_three_phase_model_to_similarity` already exist in `moving_boundary.analytic`.
- **Accounting rule for the port:** the generator excludes the deferred tests *and* their
  module-level imports (here `evaluate_surrogate_construction_diagnostics` and
  `plot_surrogate_construction_diagnostics`). Each ported file's docstring names its excluded tests
  and points here. The port report gives ported / deferred / total per file, and every research
  test function must be either ported or listed in this table.
