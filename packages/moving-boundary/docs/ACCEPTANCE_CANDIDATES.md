# Proposed acceptance runs — awaiting approval

This is the candidate list requested in review item 4a, not an implemented suite.
No new runs, fixtures, sampling, environment or wheel builds were performed while
preparing it. Source files were read at local main 06dc7b4; the extraction tree
and its older ae9fb29 fixture remain separate. Approval is required before building
the suite or recording reference outputs. Changes to the shortened setups below
must be recorded before any reference capture, not selected after comparing results.

## Execution and time budget

Propose 30 mandatory self-contained runs (C01–C30) plus the explicit opt-ins below.
Each listed ID is a separate worker/process, except the paired optional symmetry
entry, whose two solvers also get separate workers. Reuse no imported solver state
between the main reference and the installed package. Independent lifecycle
operations intentionally occur inside their own single case worker.

The default suite targets approximately 4–8 minutes including interpreter startup,
with a 60-second per-worker limit and a 600-second overall limit. These are
estimates/budgets, not new measurements; existing evidence is nine baseline cases
in a roughly five-second capture and 39 focused tests in 28.52 seconds. Pilot and
measure the approved short setups before freezing fixtures. If a scientifically
useful setup cannot fit, report it before executing a longer run; do not silently
shrink the approved run, omit it, or treat timeout as SKIPPED. Optional sampling
also targets less than one minute; full grids, full 1000-hour studies, DICTRA
reference solves, plotting and refinement sweeps are excluded from this fast list.

The intended single installed-package command will run all IDs, printing a
per-run/per-quantity PASS/FAIL/SKIPPED table. Its name/interface and executable
implementation are not created before approval. Database/licensed/cache cases
are opt-in and remain in the default table as SKIPPED with precise reasons.
Missing required inputs are FAIL for mandatory cases; missing opt-in resources
are explicit SKIPPED. A present, enabled case that errors or times out is FAIL.
One-time environment installation and reference capture are separate bounded jobs,
not hidden parts of a claimed sub-ten-minute repeated acceptance run.

## Mandatory self-contained candidates

All files are relative to the repository root at main 06dc7b4. No database,
licence, digitized figure data, reference C++ executable or pre-existing generated
dataset is needed for C01–C30. Runtime targets include worker startup and are
unmeasured estimates, not acceptance promises. Modified short setups are proposed
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
| C25–C27 | kawin/tests/test_illingworth_setup_and_diagnostics.py::_build, test_setup_is_idempotent and test_continued_solve_keeps_reference_inventory_and_history | Binary, ternary two-phase, three-phase lifecycle cases. Original small builders/durations: 0.5, 2e-3, 2e-3. Check repeated setup, split continued solve, reset and reference inventory/history retention, comparing the same operation sequence between implementations | 5–15 s |
| C28 | kawin/tests/test_illingworth_ternary_three_phase.py::test_three_phase_final_remainder_below_outer_minimum_is_solved_and_accepted and test_three_phase_tiny_final_remainder_commits_all_histories_and_allows_continuation | Two short schedules in one case worker: dt=1e-4 to 1.2e-4 with minDtFrac=0.25; then dt+5e-15 and continued solve. Use original stationary unequal-volume builder. Read accepted histories directly; do not patch production methods to run the acceptance case | 5–15 s |
| C29 | kawin/tests/test_illingworth_binary_variable_diffusivity.py::test_failed_step_is_retried_with_smaller_dt and test_retry_exhaustion_reraises_last_error; kawin/tests/test_illingworth_ternary_three_phase.py::test_three_phase_failed_retries_do_not_mutate_accepted_boundary_or_histories and test_three_phase_retry_below_outer_minimum_is_rejected_before_wrapper | Small failure-contract group, with provider-level scripted errors or a test-only injected failing subclass where internal failure is required. Preserve production globals; compare failure class/chaining, reduced dt, no accepted-state/history mutation, and ordinary below-minimum rejection. Use original source test sizes and at most one accepted step | 5–15 s |
| C30 | kawin/tests/test_illingworth_binary_variable_diffusivity.py::test_time_varying_temperature_re_evaluates_phase_uniform_diffusivities_each_step | Original FAST grid, dt=0.05, duration 0.2; TemperatureParameters times [0,1] hours and [1000,1360] K. Capture per-step temperatures, query counts and state, so the hours convention is explicit | 5–10 s |

C23's equal-volume setup is proposed as a full accepted-step variant of the
existing nonlinear-step test; C24 is a new saved-surrogate solve built from its
existing open synthetic arrays. Neither is represented as a previously passing
end-to-end run. Their buildability and numerical contracts will be checked only
after approval. Likewise, shortening the similarity mesh/time is a proposed
acceptance variant, not a claim that the original convergence study was rerun.

## Explicit opt-in research/data/provider candidates

These source files also come from main 06dc7b4. The listed existing inputs were
found in the original checkout, not copied or staged. File existence alone is not
integrity validation, backend availability, redistribution clearance or a pass.
After approval, fingerprint and validate only the specifically selected inputs.
Do not invoke full interactive example modules or their fallback cache builders.
Adapt their constructors/settings into explicit short builders without modifying
research sources, module globals, or installed kawin.

| ID | Source file | Short setup, target below one minute | Required input / default skip reason |
|---|---|---|---|
| O01 | examples/ternaryExamples/exb5_gamma_alpha_gamma.py::run_kawin2 | exb5 symmetric half couple, original 1373 K/compositions/geometry, duration 3600 s, alpha/gamma 21/41 nodes, physical mesh 401, semiLog_dt=0.05, semiLogT0=1e-2. No DICTRA solve | Opt-in tables: examples/ternaryExamples/outputs/exb5/tc_inputs.json (present). Generated from FEDEMO/MFEDEMO; loading needs no live licence. Missing/not-selected/invalid cache is explicit SKIPPED; never auto-generate |
| O02 | examples/ternaryExamples/exb5_gamma_alpha_gamma.py::run_kawin3 | Same shortened horizon/settings on full gamma/alpha/gamma geometry, phase nodes 41/21/41, physical mesh 801; capture symmetric interfaces and etas | Same exb5 table input and opt-in policy as O01 |
| O03 | examples/Interstitials/three_phase_symmetric_check.py; examples/Interstitials/fe_mn_c_ferrite.py; examples/Interstitials/ternary_ferrite_growth_le.py::run_kawin | Pair the two-phase Fe–Mn–C growth and symmetric three-phase check in separate workers: duration 0.02 s, ferrite/austenite nodes 21/81, physical meshes 401/801, semiLog_dt=0.05, power=3.0. Compare each with main and include their symmetry relation | examples/Interstitials/outputs/fe_mn_c_ferrite_le_tc_inputs.json (present); cached FEDEMO/MFEDEMO tie-lines/DC data. No live licence when loaded; absent/disabled cache is SKIPPED |
| O04 | examples/ternaryExamples/FeCrNi/IllingworthTernaryTwoPhase_FeCrNi_AnalyticComparison.py::interface_matrices_at_eta and model constructor | Saved-surrogate Lee case B at 1373 K: reload the existing bundle, retain its initial-eta matrix selection and closure, 21/41 phase nodes, 401 physical nodes, dt=600 s, duration 3600 s. No sampling, plots or refinement; preserve actual bundle identity | examples/ThermoCalc/outputs/FeCrNi_pycalphad_1373K_default_phases_0_g_offset_0p0_bulk_0p01.surrogate (present). Opt-in local bundle, no live database/licence. Its build spec, members and hashes must validate; absent/disabled is SKIPPED |
| O05 | examples/ternaryExamples/FeCrNi/IllingworthTernaryTwoPhase_FeCrNi_Lee1996Cases.py::build_source_thermodynamics/build_tieline_surrogate; examples/ternaryExamples/pycalphad_default_phase_adapter.py | Real pycalphad BCC_A2/FCC_A1 sampling at 1373 K: source seed point [0.45,0.21], at most five eta samples and a small explicitly recorded bulk sample set, then 21/41 phase nodes, 401 physical nodes, dt=600 s, duration 3600 s. Preflight sample validity; no silent analytic replacement | Opt-in pycalphad plus examples/FeCrNi_Lee1993_L_style_ternary_checked_withMobility.tdb (present). No TC licence. Database identity, kinetic coverage and sample success required; disabled/unavailable database is SKIPPED. Do not import the interactive Lee script to execute its cells |
| O06 | examples/ThermoCalc/FeCrNi_TC_Python.py; examples/ThermoCalc/training_data.py; examples/ThermoCalc/tc_python_adapter.py | Explicitly enabled live TC-Python: 1373 K BCC_A2/FCC_A1, three eta samples on the example's valid probe line [0.25,0.068]–[0.45,0.242], minimal recorded bulk samples, then the same 21/41-node six-step solve. No DICTRA invocation or full demo grid | Thermo-Calc/TC-Python runtime, valid TCFE9 and MOBFE4 access and licence. Backend availability is unverified. Adapter still imports/calls debugInPlace; a batch-safe promoted path is required before enabling it. Disabled/missing resources report SKIPPED explicitly; fake tests do not satisfy this row |
| O07 | examples/Interstitials/fe_c_exb1a.py::run_kawin | Physical Fe–C interstitial exb1a: load its existing LE/DC tables, lagged D, duration 0.1 s, ferrite/austenite 21/81 nodes, physical mesh 401, semiLog_dt=0.05, original geometry/compositions | examples/Interstitials/outputs/fe_c_exb1a_tc_inputs_v2.json (present), generated TC inputs; no live licence for loading. Opt-in, SKIPPED if absent/disabled. Never invoke build_tc_inputs if the cache is missing |

Optional O03 adds two processes, so the seven optional entries represent eight
run processes. All enabled cases share the overall budget; measure approved
sampling before claiming the fully enabled suite fits ten minutes. If a provider
cannot complete meaningful sampling in its minute budget, propose a separately
bounded slow opt-in before running it. No new licence or cached-data permission
is assumed by selecting a candidate.

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
choose and commit explicit dimensional atol/rtol values by run and quantity after
run-list approval and before any public-host/reference comparison. No new result
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
prompt guard. No guard or new runner has been implemented in this proposal.

## Approval requested and remaining review work

Approve or revise the mandatory list and shortened setups, and identify which
optional local-input/provider rows should be enabled by default for your runs.
The selection belongs to the user. Absence of a reply is not approval.

After approval: implement the batch guard (item 5), perform item 6's separate
current-state documentation cleanup and narrow test-stub removal/explicit repin,
then freeze tolerances/inputs and capture approved references from main 06dc7b4
(item 4b–e). Preserve that reference even if the later test-only cleanup advances
main. Only then proceed through the three separate whole-module moves (item 2)
and the isolated stock-kawin gate (item 3).

Item 6 remains pending, including rewriting HANDOFF.md rather than appending,
removing obsolete debugInPlace test stubs on fix/remove-debugger-imports in a
separate commit fast-forwarded to main, repinning the dependency, and recording
the adapter's remaining debugger dependency in LEDGERS.md for Stage 2C.
No research branch merge, push, publication or external contact has occurred.
