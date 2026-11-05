# Capability and dependency ledgers

All paths below are relative to the fork at
ae9fb2971a9fd49132f71ddc883a5bd5406644b6 unless marked otherwise.
These are scoped first-session records, not complete release audits.

## Capability inventory

| Feature / implementation | Representative evidence | Assumptions / status / target API |
|---|---|---|
| Binary, MovingBoundaryIllingworthFDM.py | Three baseline modes; new constant-D parity | Closed planar, substitutional, constant T, fixed endpoints; public moving_boundary.solvers.MovingBoundaryIllingworthFD1DModel |
| Ternary two-phase, MovingBoundaryIllingworthTernaryFDM.py | Uniform right/left motion, lagged and implicit coupled-matrix cases | Analytic curved tie-lines, zero flux, isothermal; legacy only, future solver API |
| Ternary three-phase, MovingBoundaryIllingworthTernaryThreePhaseFDM.py | Uniform and lagged cases, phase volumes (2,3,1.5) | Two analytic interface closures, nine nodes/phase, substitutional; legacy only |
| Transformed profiles, interface positions, eta, inventories, accepted times | Full nine-case fixture comparison, fresh array hashes | Exact same-environment regression; independent correctness/convergence work remains |
| Repeated setup and continued solves | Existing setup/diagnostics suite | All three families; reference inventory/history retained |
| Outer boundary constraints | Existing binary rejection/zero-flux tests | Fixed composition, mixed, periodic and nonzero-flux boundaries rejected |
| Interstitial/u-fraction behavior | All three test_illingworth_interstitials*.py modules passed; see assessment below | Allowed component roles/order and supported volume combinations require explicit acceptance; deferred |
| Three-phase implicit diffusion and other volume combinations | Not covered by chosen cases | Do not infer support from two-phase modes; validation gap |
| Nonuniform grids, similarity, refinement and flux conservation | Existing transformed-grid/similarity/convergence tests inventoried, not run | Scientific correctness gate open |
| Retry/rollback, thin-phase stop, schedules, reset, restart and coupling | Selected binary retry and three-phase rollback tests passed; remaining coverage open | Thin-phase termination is not topology transition; lifecycle gate open |
| Sampling, tie-line/diffusivity surrogates, merging and persistence | Local MovingBoundarySurrogates.py, SurrogateArtifacts.py, diagnostics modules | No new namespace/API; material workflow and artifact acceptance deferred |
| pycalphad generation | examples/ternaryExamples/pycalphad_default_phase_adapter.py | Not run; no database hashes because no database inputs used here |
| TC-Python generation | examples/ThermoCalc/tc_python_adapter.py | Live license/runtime not established; fake-provider and live acceptance deferred |
| Plotting, saved-result loading and diagnostics | Existing fork APIs only | No extraction or new compatibility promise |
| Alternative moving-boundary and Olaye solvers | Outside Illingworth scope | Not exposed; remain incidental legacy initializer dependencies |

The deterministic fixture/builders are kawin/tests/illingworth_baseline_cases.py
and kawin/tests/data/illingworth_substitutional_baseline.npz. They contain analytic
providers and no licensed database data. Their hashes are recorded in the
verification/audit artifacts. No later DICTRA inputs were needed or copied.

## Dependency/provenance mapping

Public source: https://github.com/materialsgenomefoundation/kawin at cached
f0c347967fcfc18f6615cec205f584573606e100. Local source:
https://github.com/samuelgprice/kawin at the fork commit above.
Comparison ancestor: 279cedfe0d5f8c12f69beb8c8579f30ac5c41b08.
Git addition status establishes repository history, not independent authorship.

| Paths / functions | Classification and required action |
|---|---|
| GenericModel.py | Upstream-derived host; local diff from ancestor is solve docstring only. Retain dependency, do not copy framework |
| solver/Solver.py | Upstream-derived local changes to minimum/final timestep behavior; required fork policy pending characterization |
| diffusion/Diffusion.py, DiffusionParameters.py | Upstream-derived modifications including initial composition handling; retain fork |
| diffusion/mesh/MeshBase.py | Upstream-derived modifications to recording/preallocation/time lookup; results remain fork-owned |
| Constants.py | Upstream-derived gas constant changed from 8.314 to 8.3145 relative to ancestor; preserve convention |
| thermo/Thermodynamics.py, MultiTherm.py, Surrogate.py, Mobility.py | Upstream-derived local modifications; metadata/diagnostics/provider differences remain. Binary analytic case does not validate thermodynamic equivalence |
| thermo/LocalEquilibrium.py | Differs against cached upstream but unchanged relative to common ancestor; do not mislabel subsequent upstream development as local authorship |
| diffusion/MovingBoundaryIllingworth*FDM.py | Local additions, absent from cached upstream; binary class explicitly describes a port of authors' MAP C++ code. No solver source copied into new wheel |
| diffusion/mesh/FDM1D.py (CartesianFD1D), TransformedGrids.py, MovingBoundaryIllingworth*FD1D.py | Local additions absent upstream; transformed grids and scalar/block tridiagonal kernels remain supplied by pinned fork |
| diffusion/_illingworth_interstitials.py, _spectral_validation.py, MovingBoundaryEquilibrium.py | Local helpers retained transitively, not extracted |
| diffusion/MovingBoundarySurrogates.py, SurrogateArtifacts.py, SurrogateDiagnostics.py and related benchmark/domain modules | Local additions; provider/surrogate stage deferred |
| diffusion/__init__.py and mesh/__init__.py | Local expanded export lists; targeted submodule import still executes initializer and imports unrelated machinery |
| New facade __getattr__ | New lazy alias only; no numerical code, global patches or compatibility copies |
| New example build_model and ConstantDiffusivity | Adapted from _binary_model and _BinaryThermodynamics in baseline builder at fork SHA; original source mapping retained here |
| New parity tests | Same-environment comparisons; no copied expected values |
| LICENSE / NOTICE | License status explicitly unresolved; NOTICE reproduces upstream LICENSE.txt verbatim |

Direct binary imports include GenericModel, DiffusionModel, interstitial helpers,
three private diffusivity-mode helpers from the ternary module, CartesianFD1D
and boundary types, transformed-profile integration/reconstruction/flattening
and tridiagonal solve, NumPy, and explicitEulerIterator. Thus even an analytic
binary case still requires considerably more than the facade's one import.
Numba acceleration is active in this environment; fallback parity is not tested.

Ternary retry exhaustion imports examples.debugInPlace at lines 3141–3142 of
the two-phase module and 2311–2312 of the three-phase module. These paths are
not part of the new binary API and were not reached in the baseline. They must
be removed/characterized before robust ternary extraction. Their presence is
documented, not hidden by changing the frozen reference.

## Redistribution and maintenance gaps

The upstream MIT notice is retained verbatim, but does not establish rights to
local additions. The MAP port and reference source conditions require a separate
review before a blanket software license or publication. Scientific citations
do not replace code permission. No PDFs, digitized data, databases, trained
surrogates or reference C++ files were packaged.

TC-Python adapter code, SDK/runtime, databases, generated data and trained models
have distinct provenance/terms; none was cleared or redistributed here.
No external legal terms were reverified in this session. A release owner and
upstream-update maintenance owner still need designation. There are no
compatibility code copies to maintain at this checkpoint; the fork remains
authoritative for every numerical implementation.

See git-audit.txt for file-level addition/modification evidence and input hashes.

## Bounded host assessment (public release 0.5.0)

Compared the local frozen source with the published
[kawin 0.5.0 wheel](https://pypi.org/project/kawin/0.5.0/), downloaded without
installation. SHA256:
9f7a1497c5dff4314536860ac6fd0b96a68243dd681db8eb106dbe33fe25cbaf.
This selects a concrete release; it is not a claim about the latest release.
The downloaded wheel and line-ending-normalized comparison are retained locally
in .verification/kawin-0.5.0-py3-none-any.whl and host-release.diff.
Only the comparison normalized line endings, not source files.

| Host dependency | Difference from released wheel | Implication |
|---|---|---|
| GenericModel | solve docstring only | Reuse host; copying entire class is unjustified. solve constructs DESolver directly, so a custom driver needs an explicit wrapper |
| Solver | 20 added/5 removed lines including documentation; below-minimum rejection and final-remainder handling | Small code surface but essential semantics: do not enlarge dt after an implicit candidate has been solved |
| MeshData | 39 added/7 removed; preallocation and valid recorded-time range, optional exact-time lookup | Narrow recorder compatibility is plausible; three-phase moving-domain output uses the interpolation switch |
| DiffusionModel | Conditional rather than unconditional composition adjustment; status formatting | Characterize initialization/reset and preserve starting compositions |
| DiffusionParameters | minComposition 1e-10 versus 1e-8; added alternative moving-boundary constraints | Pass constraints explicitly; do not port unrelated constraint fields blindly |
| MultiTherm | returnMeta endpoint/phase metadata extension | Provider concern, not required by analytic binary kinetics; defer to provider boundary |
| Constants | gas constant 8.3145 versus 8.314 | Preserve convention where used; do not globally patch upstream constants |
| TemperatureParameters | No change to array-time conversion | Hours for temperature arrays remain a contract; callbacks retain their own contract |

These are static differences, not successful upstream runtime parity. Provider
thermodynamic extensions and all driver lifecycle behavior were not exhaustively
audited. A small diff does not prove an independent driver is cheaper.

**Recommendation:** retain the host for the next binary milestone. First remove
production debugger dependencies in a separately identified, narrow fork
packaging change while preserving this reference, then assess extraction of the
binary class/helpers with targeted host compatibility. Do not copy the host
framework or accelerate a complete driver/results rewrite based on this audit.
Upstream contribution remains an alternative destination, not assumed acceptance.

### Debugger failure surface

The import chain includes MovingBoundaryFDM, MovingBoundarySurrogates,
MovingBoundaryOlayeFDM and SurrogateDiagnostics; the latter reloads the helper at
import time. MovingBoundaryOlayeFDM_rework imports the helper through Olaye.
Thus removing only the first import would reveal later import failures.
Ternary and three-phase getdXdt retry exhaustion imports/calls debugInPlace before
raising RuntimeError from the last nonlinear error. In a notebook the helper
can wait for debugger attachment; without examples the import can mask that error.

The required fix covers all reachable production imports/reloads and calls, with
explicit exceptions or existing optional diagnostics. It must preserve exception
chaining and accepted state; merely relocating the helper or making it a no-op
does not establish appropriate behavior at every call site. Public retry exhaustion without a debugger now passes dedicated tests for both
ternary families in the separate debugger-fix commit; see the integration below.
The focused rollback tests below exercise internal failure/commit paths; they
do not prove that all public debugger paths are removed.

### Additional baseline evidence

Activated kawin environment, repo-local .pytest_tmp/host_assessment, subprocess
timeout 600 seconds. 39 passed in 28.52 seconds; 1425 pre-existing pyparsing
deprecation warnings, no skips. Inputs are committed analytic examples/test
fixtures; no live TC-Python session or external licensed database was used.

Pytest selection (with -q and --junitxml .verification/assessment.xml, located
under the nested package):
- kawin/tests/test_illingworth_interstitials.py
- kawin/tests/test_illingworth_interstitials_ternary.py
- kawin/tests/test_illingworth_interstitials_three_phase.py
- kawin/tests/test_illingworth_binary_variable_diffusivity.py::test_failed_step_is_retried_with_smaller_dt
- kawin/tests/test_illingworth_binary_variable_diffusivity.py::test_retry_exhaustion_reraises_last_error
- kawin/tests/test_illingworth_ternary_three_phase.py::test_three_phase_failed_retries_do_not_mutate_accepted_boundary_or_histories
- kawin/tests/test_illingworth_ternary_three_phase.py::test_three_phase_atomic_history_failure_rolls_back_every_history

This satisfies the selected interstitial and retry/rollback prerequisites, not
the entire Stage 0 matrix or independent mathematical validation.

## Fork packaging integration

The baseline mapping above remains at ae9fb29. The installable dependency now
pins 06dc7b4ab56f67226ceabfde41ebf2690c088841, from fix/remove-debugger-imports
based directly on ae9fb29 and fast-forwarded into local main. No solver/host code
was copied into the extension and the extraction reference was not merged forward.
The change removes production debugger dependencies and adds batch-import and
public retry-exhaustion tests. An existing fake-provider test now checks absence
of the debugger symbol and uses tmp_path for clean-checkout isolation.

Checks: 76 focused tests (new tests, substitutional baseline, setup/diagnostics,
surrogate diagnostics and positive-matrix tests), then 3 affected fake-provider
cases. No licensed provider run. The installed fork/bridge binary example matches
the 11 reference arrays exactly using -I -S and no source/import hooks.
Source/fork origins and separate baseline/dependency SHAs are in verification.json.
The Git URL pin is not remotely installable from this work until the local commit
is pushed; no publication or push was performed.

For fork packaging fixes, main integration is explicitly required as well as
package repinning. Research feature branches take that same fix via deliberate
main merges; numerical reference refresh is a separate decision.
