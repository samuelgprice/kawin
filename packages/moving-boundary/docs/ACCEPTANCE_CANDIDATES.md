# Approved acceptance runs

The user approved this list with the tier, cache, analytic-error and reference
isolation rules below. Implementation is authorized. Inputs and tolerances must
be frozen before comparison; changes are recorded before reference capture.

## Execution and time budget

- Quick: C01–C24, C30, and the irregular-grid binary C31. C25–C29 are package
  pytest contracts with separate main/package workers, not comparison-table runs.
  The combined quick run plus contracts must remain below ten minutes and run
  after every numerical code move.
- Full: quick, enabled O-runs, and long drift F04/F11/F12. Estimate 8–18 minutes
  before measurement; report the updated estimate before each materially changed
  full run. Use a 1200-second overall cap, well below 30 minutes. Longer jobs
  require a new explanation before execution, never an unnoticed timeout extension.
- Workers have actual timeouts, noninteractive stdin and an input() prohibition
  installed before imports. Ternary constructors explicitly pass raise, except
  approved continuation/terminal-stop cases which explicitly pass continue.
- O01/O02/O03/O04/O07 are enabled in full whenever cached inputs exist; SKIPPED
  only when missing. Invalid existing data, construction errors and timeouts FAIL.
  Their TC-derived reference outputs and input copies stay local/git-ignored;
  never commit them or any derived numeric values/hashes that expose their data.
- O05 pilots real pycalphad sampling with a roughly 60-second budget. If successful
  inside that budget it is enabled by default in quick and full; otherwise retain
  it in full with a separately reported bounded sampling estimate. No fake-provider
  substitution. O06 is disabled until the TC adapter is batch-safe.

Capture every reference from a clean DETACHED worktree at
06dc7b4ab56f67226ceabfde41ebf2690c088841. Never use the user's main checkout as
the reference import path. Assert kawin.__file__ in every reference worker and
assert installed-package/host origins in every package worker. Keep reference
source clean (external caches and bytecode disabled), verifying Git state before
and after reference capture. The ae9fb29 fixture remains a separate older oracle.
Each worker runs outside source checkouts; case input definitions may be read
from the detached source without adding research checkout paths to installed runs.

## Mandatory self-contained candidates

All files are relative to the repository root at main 06dc7b4. No database,
licence, digitized figure data, reference C++ executable or pre-existing generated
dataset is needed for the self-contained C cases. Runtime targets include worker startup and are
initial estimates to measure. Modified short setups are approved
new input variants with new main-reference fixtures, not replacements for the
older committed fixture.

| IDs | Source file / builder or test | Proposed shortened setup | Runtime target per ID |
|---|---|---|---|
| C01 | kawin/tests/illingworth_baseline_cases.py::_binary_model | Existing phase_uniform case, 501 physical nodes, 100/100 transformed nodes, dt=0.1, duration 0.5; retain exact existing binary fixture check as well | 5–10 s |
| C02–C03 | kawin/tests/test_illingworth_binary_variable_diffusivity.py::_model and StrongVariableBinaryThermodynamics | Lagged and implicit D=D0(1+20x²), FAST settings: 40/60 transformed nodes, dt=0.05, duration 0.5; exercise appreciable composition dependence rather than only near-constant drift | 5–15 s |
| C04 | examples/Illingworth2005/replicate_illingworth2005_fig4_5_6_planar.py::PlanarValidationCase, PlanarExactProfile and build_python_planar_validation_model | One Fig. 4-style regular-grid analytic start; 21/21 transformed nodes, 401 physical nodes; absolute time 1e-4 to 2e-4 with dt=2e-5. Python model only; no CSV, plots or MAP compilation | 5–15 s |
| C05–C08 | kawin/tests/illingworth_baseline_cases.py::_ternary_model and run_all_cases | Keep its four cases: uniform right, uniform left, lagged right, implicit right. Original 21 physical nodes, mesh-derived phase node counts, duration 2e-3 and dt=1e-4. Verify actual motion sign | 5–10 s |
| C09–C10 | kawin/tests/illingworth_baseline_cases.py::_three_phase_model | Original uniform and lagged three-phase cases: 31 physical nodes, 9/9/9 phase nodes, volumes (2,3,1.5), dt=1e-4, duration 2e-3 | 5–10 s |
| C11–C12 | kawin/tests/test_illingworth_ternary_three_phase_similarity.py::_solution, _build_model; examples/ternaryExamples/ternaryThreePhaseSimilarity.py | Both opposite and drift solutions, original matrices/closures and T0=0.01; 21 nodes/phase, physical mesh shortened to 401 nodes, semiLog_dt=0.05, absolute end time 0.05. Preserve root-selection rule; do not run the 21/41/81 refinement sweep | 15–45 s |
| C13–C15 | kawin/tests/test_illingworth_interstitials.py::test_binary_interstitial_queries_receive_mole_fractions and _binary_model | All three binary diffusivity modes; original Fe–C u-fraction/mole-fraction inputs, 60/120 phase nodes, dt=0.05, duration 0.1. Record initialization/reset and provider-query basis | 5–10 s |
| C16–C19 | kawin/tests/test_illingworth_interstitials_ternary.py::_model and test_ternary_interstitial_plumbing | Uniform/lagged × u/x closure basis (four runs). Original 21/21 transformed nodes, 400 physical nodes, diffusivity ratio 1e-2, semiLog_dt=0.05, duration 0.05. No unsupported combination is implied | 5–15 s |
| C20–C21 | kawin/tests/test_illingworth_interstitials_three_phase.py::_model and test_three_phase_interstitial_state_queries_and_conservation | Uniform and lagged Fe–C–Mn runs; 31 physical nodes, 9/9/9 phase nodes, volumes (2,3,1.5), dt=1e-4, duration 2e-3. Check Fe+Mn substitutional closure separately from carbon | 5–10 s |
| C22 | kawin/tests/test_illingworth_ternary_three_phase.py::_make_nonstationary_unequal_volume_model and test_three_phase_public_nonstationary_unequal_volume_solve_commits_moving_geometry | Exact existing unequal-volume moving-boundary setup: 61 physical nodes, 9/9/9 phase nodes, volumes (2,3,1.5), dt=duration=2e-6. Record both interfaces and the moving right boundary | 5–10 s |
| C23 | kawin/tests/test_illingworth_ternary_three_phase.py::_make_eta_varying_three_phase_model and test_three_phase_explicit_equal_volumes_match_legacy_nonlinear_step_exactly | Explicit equal volumes (7.2e-6 each) with the eta-varying builder's 31 physical/7+7+7 phase nodes and dt=duration=1e-5; capture a public accepted step and exactly fixed right boundary. This is a new short full-solve variant of the nonlinear-step test | 5–10 s |
| C24 | kawin/tests/test_surrogate_artifacts.py::_surrogate and test_bundle_roundtrip_checks_model_and_sidecar; kawin/tests/test_illingworth_ternary.py solver assembly | Construct the source's tiny analytic two-endpoint surrogate, save it in the reference-preparation job, then require a fresh worker to load that artifact and drive a 21-node physical/9+9-node two-phase solve, dt=1e-4, duration 2e-3. Use ALPHA/BETA bulk compositions from the eta=0 endpoints; do not rebuild the surrogate during acceptance | 5–15 s |
| C30 | kawin/tests/test_illingworth_binary_variable_diffusivity.py::test_time_varying_temperature_re_evaluates_phase_uniform_diffusivities_each_step | Original FAST grid, dt=0.05, duration 0.2; TemperatureParameters times [0,1] hours and [1000,1360] K. Capture per-step temperatures, query counts and state, so the hours convention is explicit | 5–10 s |

| C31 | examples/Illingworth2005/replicate_illingworth2005_fig4_5_6_planar.py::_clustered_v_grid and Fig. 4 irregular PlanarValidationCase | Retain the example's 3/10-node irregular transformed grid, analytic start at 1e-4; quick end 2e-4, dt=2e-5, physical mesh 401. This is the example's chosen irregular grid, not a claim that the paper specified its exact nodes | 5–15 s |

### Package pytest contracts (C25–C29)

These retain main-vs-package comparisons in separate workers and run with the
quick tier. They are excluded from the run-comparison table. Test-only failure
injection never changes production kawin globals.

| IDs | Source file / builder or test | Approved setup | Initial estimate |
|---|---|---|---|
| C25–C27 | kawin/tests/test_illingworth_setup_and_diagnostics.py::_build, test_setup_is_idempotent and test_continued_solve_keeps_reference_inventory_and_history | Binary, ternary two-phase, three-phase lifecycle cases. Original small builders/durations: 0.5, 2e-3, 2e-3. Check repeated setup, split continued solve, reset and reference inventory/history retention, comparing the same operation sequence between implementations | 5–15 s |
| C28 | kawin/tests/test_illingworth_ternary_three_phase.py::test_three_phase_final_remainder_below_outer_minimum_is_solved_and_accepted and test_three_phase_tiny_final_remainder_commits_all_histories_and_allows_continuation | Two short schedules in one case worker: dt=1e-4 to 1.2e-4 with minDtFrac=0.25; then dt+5e-15 and continued solve. Use original stationary unequal-volume builder. Read accepted histories directly; do not patch production methods to run the acceptance case | 5–15 s |
| C29 | kawin/tests/test_illingworth_binary_variable_diffusivity.py::test_failed_step_is_retried_with_smaller_dt and test_retry_exhaustion_reraises_last_error; kawin/tests/test_illingworth_ternary_three_phase.py::test_three_phase_failed_retries_do_not_mutate_accepted_boundary_or_histories and test_three_phase_retry_below_outer_minimum_is_rejected_before_wrapper | Small failure-contract group, with provider-level scripted errors or a test-only injected failing subclass where internal failure is required. Preserve production globals; compare failure class/chaining, reduced dt, no accepted-state/history mutation, and ordinary below-minimum rejection. Use original source test sizes and at most one accepted step | 5–15 s |

C23's equal-volume setup is proposed as a full accepted-step variant of the
existing nonlinear-step test; C24 is a new saved-surrogate solve built from its
existing open synthetic arrays. Neither is represented as a previously passing
end-to-end run. Their buildability and numerical contracts will be checked only
after approval. Likewise, shortening the similarity mesh/time is a proposed
acceptance variant, not a claim that the original convergence study was rerun.

## Cached research/data and provider candidates (full unless O05 pilots fast)

These source files also come from main 06dc7b4. The listed existing inputs were
found in the original checkout, not copied or staged. File existence alone is not
integrity validation, backend availability, redistribution clearance or a pass.
Validate only the approved inputs; keep TC-derived fixtures and evidence local.
Do not invoke full interactive example modules or their fallback cache builders.
Adapt their constructors/settings into explicit short builders without modifying
research sources, module globals, or installed kawin.

| ID | Source file | Short setup, target below one minute | Required input / default skip reason |
|---|---|---|---|
| O01 | examples/ternaryExamples/exb5_gamma_alpha_gamma.py::run_kawin2 | exb5 symmetric half couple, original 1373 K/compositions/geometry, duration 3600 s, alpha/gamma 21/41 nodes, physical mesh 401, semiLog_dt=0.05, semiLogT0=1e-2. No DICTRA solve | Enabled when present, tables: examples/ternaryExamples/outputs/exb5/tc_inputs.json (present). Generated from FEDEMO/MFEDEMO; loading needs no live licence. Only a missing cache is SKIPPED; invalid existing data FAIL; never auto-generate |
| O02 | examples/ternaryExamples/exb5_gamma_alpha_gamma.py::run_kawin3 | Same shortened horizon/settings on full gamma/alpha/gamma geometry, phase nodes 41/21/41, physical mesh 801; capture symmetric interfaces and etas | Same exb5 table input and enabled-when-present policy as O01 |
| O03 | examples/Interstitials/three_phase_symmetric_check.py; examples/Interstitials/fe_mn_c_ferrite.py; examples/Interstitials/ternary_ferrite_growth_le.py::run_kawin | Pair the two-phase Fe–Mn–C growth and symmetric three-phase check in separate workers: duration 0.02 s, ferrite/austenite nodes 21/81, physical meshes 401/801, semiLog_dt=0.05, power=3.0. Compare each with main and include their symmetry relation | examples/Interstitials/outputs/fe_mn_c_ferrite_le_tc_inputs.json (present); cached FEDEMO/MFEDEMO tie-lines/DC data. No live licence when loaded; only an absent cache is SKIPPED |
| O04 | examples/ternaryExamples/FeCrNi/IllingworthTernaryTwoPhase_FeCrNi_AnalyticComparison.py::interface_matrices_at_eta and model constructor | Saved-surrogate Lee case B at 1373 K: reload the existing bundle, retain its initial-eta matrix selection and closure, 21/41 phase nodes, 401 physical nodes, dt=600 s, duration 3600 s. No sampling, plots or refinement; preserve actual bundle identity | examples/ThermoCalc/outputs/FeCrNi_pycalphad_1373K_default_phases_0_g_offset_0p0_bulk_0p01.surrogate (present). Enabled-when-present local bundle, no live database/licence. Its build spec, members and hashes must validate; only absent input is SKIPPED |
| O05 | examples/ternaryExamples/FeCrNi/IllingworthTernaryTwoPhase_FeCrNi_Lee1996Cases.py::build_source_thermodynamics/build_tieline_surrogate; examples/ternaryExamples/pycalphad_default_phase_adapter.py | Real pycalphad BCC_A2/FCC_A1 sampling at 1373 K: source seed point [0.45,0.21], at most five eta samples and a small explicitly recorded bulk sample set, then 21/41 phase nodes, 401 physical nodes, dt=600 s, duration 3600 s. Preflight sample validity; no silent analytic replacement | Default-enabled after a <=60-second sampling pilot, with examples/FeCrNi_Lee1993_L_style_ternary_checked_withMobility.tdb (present). No TC licence. Database identity, kinetic coverage and sample success required; unavailable database is SKIPPED; slow sampling moves to full. Do not import the interactive Lee script to execute its cells |
| O06 | examples/ThermoCalc/FeCrNi_TC_Python.py; examples/ThermoCalc/training_data.py; examples/ThermoCalc/tc_python_adapter.py | Explicitly enabled live TC-Python: 1373 K BCC_A2/FCC_A1, three eta samples on the example's valid probe line [0.25,0.068]–[0.45,0.242], minimal recorded bulk samples, then the same 21/41-node six-step solve. No DICTRA invocation or full demo grid | Thermo-Calc/TC-Python runtime, valid TCFE9 and MOBFE4 access and licence. Backend availability is unverified. Adapter still imports/calls debugInPlace; a batch-safe promoted path is required before enabling it. Disabled/missing resources report SKIPPED explicitly; fake tests do not satisfy this row |
| O07 | examples/Interstitials/fe_c_exb1a.py::run_kawin | Physical Fe–C interstitial exb1a: load its existing LE/DC tables, lagged D, duration 0.1 s, ferrite/austenite 21/81 nodes, physical mesh 401, semiLog_dt=0.05, original geometry/compositions | examples/Interstitials/outputs/fe_c_exb1a_tc_inputs_v2.json (present), generated TC inputs; no live licence for loading. Enabled when present, SKIPPED only if absent. Never invoke build_tc_inputs if the cache is missing |

O03 uses separate two-phase and three-phase workers. No cache builder is allowed
to silently launch Thermo-Calc. Source paths above identify case provenance;
original interactive scripts are not executed wholesale.

## Full-tier drift and support checks

| ID | Source / setup | Independent quantity |
|---|---|---|
| F04 | C04 analytic case, 21/21 phase nodes, absolute end 1e-2 (start 1e-4), dt=2e-5 | Error in interface and physical phase profiles against the exact planar solution |
| F11/F12 | C11/C12 opposite/drift similarity cases, absolute end 1.0, 21 nodes/phase and semiLog_dt=0.05 | Interface rates/positions, eta and phase-profile error against the exact similarity solution |
| C32 if feasible | Small natural terminal-thin-phase setup under continue; pilot before freezing | Reached terminal stop, accepted state/history and explicit stop status. A forced-retry test alone is classified as a pytest contract, not a natural physical stop run |

C04, C11/C12 and C31 also record their exact/similarity errors in quick runs,
independently of main. Freeze values/acceptance limits before package comparisons.
Long variants protect against slow numerical drift at the public-kawin stage.

Binary and ternary two-phase solvers assume equal constant substitutional molar
volume and do not accept phase_molar_volumes. Unequal per-phase volumes for those
solvers are explicitly UNSUPPORTED. Equal-volume conventions remain covered by
their substitutional/interstitial runs; only three-phase supports explicit unequal
volumes. Do not invent a binary/two-phase volume option to fill the matrix.

## Quantities, tolerances and outcomes

For pure relocation, every applicable array/scalar below must match main exactly
in the same environment: atol=rtol=0, identical shapes and recorded-time order,
with NaNs accepted only at the same predefined masked positions. Extra/missing
records, changed finite/NaN masks, phase/component order or a changed stop/failure
classification are failures. Retain the older exact fixture/probe checks alongside
the new main-reference suite rather than silently regenerating them.

| Quantity | What each applicable run records | Relocation rule |
|---|---|---|
| Interface position | Full accepted interface history; both interfaces and right-boundary history for moving-volume cases; coordinates/units | Exact |
| Phase profiles | Phase-local transformed profiles and grids, reconstructed physical profiles, full x-fraction outputs and explicit u-fraction state where relevant | Exact with identical predefined masks |
| Eta | Accepted eta histories and initial branch/eta selection for ternary solvers; explicitly N/A for binary | Exact |
| Inventories/conservation | Initial and accepted length inventories, component-mole inventories and substitutional totals/closure where supported; distinguish them under unequal volumes/interstitials | Exact reference quantities; retain separately documented source-test conservation bounds |
| Accepted times | All synchronized history times, counts, final time, and schedule/continuation segments | Exact |
| Stop status | Completion vs explicit terminal-thin-phase failure, exception class/chaining, retry diagnostics and absence of accepted-state mutation on rejected steps | Exact structural match |

Public-kawin per-quantity numerical tolerances are a separate preregistration:
choose and commit explicit dimensional atol/rtol values by run and quantity before any public-host/reference comparison. No new result
has been examined and no public-host tolerance has been selected here. Do not
infer tolerances from observed errors; gas-constant/composition-floor differences
must be investigated against this preregistration, not silently excused. Optional
provider outputs (phase endpoints, element order, matrices, temperature and build
metadata) must be checked before downstream solver comparisons.

Every applicable quantity gets its own table entry. N/A must be explicit for
unsupported quantities; N/A is not a pass or skip of a required case. All mandatory
cases must run; skipped opt-ins are visible gaps in coverage, never evidence of
provider or database parity.

## Batch safety required before implementation

Before any new fixture capture or acceptance/verification execution, implement
review item 5 in its own commit: each ternary construction gets an explicit
terminal_thin_phase_policy, normally raise. Use continue only for O01–O03 if the
approved example deliberately requires continuation after the existing thin-phase
threshold; record that exception and reason. Other cases use raise unless an
approved short-case preflight demonstrates that their intended behavior needs
continue, in which case report the required change before reference capture.

A worker-start guard must make builtins.input raise immediately, covering imports,
model construction and solve; stdin is noninteractive and subprocess timeouts are
real. Its forbidden-prompt check is a failure, not a licence/data skip. Do not
change either production prompt default. A timeout/session poll alone is not a
prompt guard. The guard and runner implementation are tracked in the corresponding commits.

## Authorized implementation order

Implement item 5 batch guards, then item 6 cleanup (current-state documentation,
test-stub removal on its fix branch, fast-forward main, explicit package repin).
Then freeze input/tolerance manifests and capture main references from the clean
detached 06dc7b4 worktree. Implement and verify complete binary, ternary two-phase,
then three-phase modules in separate commits (item 2), followed by stock kawin
0.5.0 with package-owned compatibility in a separate environment (item 3).

No publication, push, research-branch merge or external contact is authorized.
Package name/audience, MAP outreach, upstream contribution and changing the
thin-phase prompt default remain reserved for the user. Approval of this run list
is complete; no repeated confirmation is required for its implementation.
