# Stage 1B: stock kawin 0.5.0

Status: implemented and acceptance passed after all three exact module relocations.
Quick: 30 run comparisons and 88 pytest tests passed in 427.34 s. Full: 40 run comparisons and 88 pytest tests passed in 578.02 s.

Stage 2B adapter relocation is complete for O05 restricted-phase generation and O08 default-phase endpoint/equilibrium sampling; these need no research checkout. See [NEXT_SESSIONS.md](NEXT_SESSIONS.md#installed-solving-and-research-owned-generation) for remaining research script orchestration, the open broader Stage 2 split and research-branch R01.

## Ownership and method-level source mapping

The host distribution is unmodified kawin==0.5.0. The 73 installed stock files are
checked byte for byte against wheel SHA256
9f7a1497c5dff4314536860ac6fd0b96a68243dd681db8eb106dbe33fe25cbaf.
No dependency module/global is patched and no Illingworth fork class is inherited.

Policy methods below derive from samuelgprice/kawin at
06dc7b4ab56f67226ceabfde41ebf2690c088841. Whole local additions are mapped in
relocation-map.json and BINARY_SLICE.md; upstream MIT notice remains in NOTICE.

| Package adapter | Frozen source / behavior retained | Host services inherited |
|---|---|---|
| compatibility/solver.py::DESolver._getdXdt | solver/Solver.py; below-minimum rejection and exact final-remainder exception. Remaining time is computed at the iterator callback from the requested final time | Complete integration loop, callbacks and corrections |
| compatibility/solver.py::GenericModel.solve | GenericModel.py::solve; selects the local DESolver adapter only | Model state, flattening and time bookkeeping |
| compatibility/recording.py::MeshData.preallocate/y | diffusion/mesh/MeshBase.py; allocation, valid recorded range, optional exact-time lookup | reset/record/finalize and underlying storage conventions |
| compatibility/diffusion.py::DiffusionModel._validateMeshComposition | diffusion/Diffusion.py; adjust only floor violations, preserve valid inputs and x-to-u conversion | Construction, reset and diffusion lifecycle |
| compatibility/diffusion.py::DiffusionConstraints | diffusion/DiffusionParameters.py; local minComposition=1e-10 and moving-boundary fields | Host constraint defaults and initialization |
| compatibility/thermo.py::MulticomponentThermodynamics.getInterfacialComposition/_interfacialComposition | thermo/MultiTherm.py; returnMeta required by real O05 sampling, phase-labelled endpoints in input element order | Equilibrium engine and interdiffusivity |
| compatibility/thermo.py::_tracerDiffusivitySingle/tracer_diffusivity | thermo/Thermodynamics.py tracer wrapper plus explicit package GAS_CONSTANT=8.3145 and host mobility evaluation | Local equilibrium and mobility engine |

### Stage 2B generation source map

All names below map one-to-one from examples/ternaryExamples/pycalphad_default_phase_adapter.py
at 06dc7b4ab56f67226ceabfde41ebf2690c088841 to providers/pycalphad.py.
Frozen source-file SHA256: a373d63a29c0b975a6d6b90d15c54869a3a7f03fd2af98a65f7dfe8fe0fe6bef.
The module docstring also records every source function/method and its line.

| Package function/method | Frozen source lines / retained behavior |
|---|---|
| resolve_pycalphad_equilibrium_phases | 15; explicit phases or database component filter, vacancy inclusion |
| order_pycalphad_phase_universe | 30; ordered deduplication of resolved universe, interface membership |
| _validate_g_offset; create_pycalphad_thermodynamics_source | 51; 59; finite offset, restricted/default source selection and unchanged defaults |
| PycalphadDefaultPhaseThermodynamics.__init__/__enter__/__exit__ | 106/135/139; full-phase engine setup and context protocol |
| clearCache; captureKineticsDiagnostics | 143; 147; wrapped-engine delegation |
| getEquilibriumData; getInterfacialComposition; getInterdiffusivity | 151; 169; 199; active phase amounts/compositions, planar endpoints, phase-specific diffusivity |
| _active_composition_sets; _metadata_from_composition_sets | 203; 210; threshold filtering and requested endpoint order |
| _full_composition_from_composition_set; _independent_composition_from_composition_set; _nonvacant_elements | 231; 243; 246; restore input component order, omit reference component/vacancy |

The complete import closure is __future__.annotations, pathlib.Path, NumPy, pycalphad.Database,
pycalphad.core.utils.filter_phases and kawin.thermo.MulticomponentThermodynamics,
plus the conditional unused examples.debugInPlace import at source line 190.
That small helper imports os/sys and conditionally debugpy; it imports no further
examples modules. Its adapter call is commented out, so the package omits the
unused import and copies no debugger code. The package binds its existing
compatibility.thermo adapter and imports the thermodynamic engine lazily. Stock
kawin 0.5.0 already requires pycalphad>=0.11; no optional extra is added.

Executable numerical AST matches the frozen adapter after excluding imports and
docstrings. Phase/g_offset/endpoint conventions, arithmetic and iteration order are
unchanged, including R01 hash sensitivity. O05 uses real sampling with hash seed 0
and rejects examples imports/source reads only during package-side generation.
O08 uses the same real FeCrNi database with use_default_phases=True: two BCC/FCC
tie lines and one FCC-only invalid probe at 1373 K, comparing endpoints, ordered
phase labels, phase amounts/compositions and chemical potentials bitwise. Hash
0/1 changes five of its seven arrays; hash 0 repeats exactly, so O08 shares O05's
narrow hash exception. Its new JSON/NPZ reference came from clean detached 06dc7b4
and is frozen by hashes in acceptance_manifest.json; older fixtures are unchanged.
The installed worker uses the existing generation guard; the reference worker
imports the original adapter, including its unused invalid-branch debugger import.
Surrogate building uses installed APIs (README's Building surrogates); research
script orchestration stays in the checkout. Acceptance is bounded to O05/O08;
arbitrary default-phase workflows and delegated methods are not established.
The broader Stage 2 split and R01 research-branch repair remain open.

### Stage 2C TC generation source map

providers/thermocalc.py maps all 87 classes/functions/methods one-to-one, with
unchanged names, to examples/ThermoCalc/tc_python_adapter.py at
06dc7b4ab56f67226ceabfde41ebf2690c088841. Source file SHA256:
4f8886bbee01dc3b96752951a05a5d8fc5c605572b185470fc9003c0df931389.
The module docstring lists every definition's original qualified name and line.

| Package ownership | Frozen source / changes permitted |
|---|---|
| ThermoCalcConfig, TCPythonThermodynamics and all error/helper/backend definitions | Complete adapter copy; same arithmetic, defaults, composition/phase conventions, grid points, calls and retry order |
| _TCPythonBackend.start/get_runtime_version | Original lazy tc_python imports; separately installed/licensed runtime, no pip dependency |
| kawin.thermo.utils private helper imports | _process_xT_arrays:13, _getMatrixPhase:59, _getPrecipitatePhase:67 are byte-identical in stock 0.5.0; no compatibility copies |
| Debugger dependency removal | Omit source import :21; replace only the live :1289 call with pass, retaining its conditional, print, exception/message/from-cause chain and retries |
| Acceptance's installed builder call | tools/acceptance_thermocalc.py::_installed_surrogate maps training_data.py:187-223 without the research import; same conversions, from_database arguments/order and metadata |

The complete adapter closure is __future__.annotations, contextlib.contextmanager,
dataclasses.asdict/dataclass/field, pathlib.Path, re, typing.Any, NumPy, the three
stock helpers and lazy tc_python/TCPython. No examples/debugger helper is copied.
Executable adapter AST matches the frozen source after the two removals and
module provenance docstring. Main remains unchanged (LEDGERS.md R03/R04).

Stage 2C is complete within full-only O06: real installed TC generation under
the examples import/read guard, three tie lines/two bulk points, six accepted
steps and an unseeded reference repeat pair. The approved suffixed phase names
are local input, not an adapter normalization change. All numerical arrays are
bitwise-only; only metadata_json.thermocalc_config.cache_dir is normalized, and
all other metadata text remains exact. The new reference is from detached
06dc7b4, refuses overwrite and is frozen by manifest hashes. Inputs/databases,
TC-derived values and metrics remain ignored; only counts/hashes are committed.
Thermo-Calc and pycalphad surrogate building use installed providers/from_database,
prepare_surrogate_artifact bundles and merge_phase_diffusivity_surrogates; see
[README's pattern](../README.md#building-surrogates). The research checkout retains
script orchestration: the FeCrNi demo wrapper, training_data.py dataset
sampling/export/resume, case assembly, plotting, and scripts' probe-point search
settings, failure/diagnostic reports and bundle build specs. The demo wrapper will
not move because its defaults are demo-specific and no real research workflow uses
it; a general session/provenance-only helper without system defaults is possible
but not planned. See NEXT_SESSIONS.md for that decision and unaccepted TC workflows.
O06 follows the FeCrNi demo; a future real-usage TC case (Lee 1996 probe-point search plus a prepare_surrogate_artifact bundle) remains a user decision, not an approved addition.
The broader Stage 2 split remains open; no scientific fix was made.

The model constructs a local MeshData subclass and transfers the host's initial
record state; it never replaces host class/module globals. Host mesh bases and
TemperatureParameters remain public (array times in hours). The local provider
boundary covers moving-boundary equilibrium, interdiffusivity and tracer queries;
unrelated precipitation/curvature APIs retain stock conventions and are untested.
No independent solver/results/thermodynamic framework was copied.

The two inherited trailing-whitespace lines in surrogates.py and seven in
compatibility methods were removed without changing numerical AST. C32 is a
configured retry stress variant; a small natural physical stop remains unestablished.

## Environment and acceptance boundary

`mb-public-050` is an offline Conda clone in a separate sibling directory. Stock
kawin was installed only there, then the built package wheel installed privately.
The shared kawin environment is unchanged. The ignored environment pointer and
cached wheel configure the local one-command runner described in README; portable
environment bootstrap is not claimed. All operations have subprocess timeouts.

The installed workers assert stock version and package/host origins and use safe
paths/no site hooks. Reference workers assert the clean detached 06dc7b4 origin.
Frozen public tolerances are unchanged. Exact relocation evidence is separate in
acceptance.json. C25-C29 compare reference/package contract outputs inside pytest.
Full covers cached O-runs, longer analytic drift and real O06, with its own
unseeded repeat pair. O06 skips explicitly when runtime/licence/local inputs
are unavailable. TC-derived values and metrics remain ignored; only sanitized
counts/hashes are committed. Binary/two-phase unequal molar volumes remain
unsupported; three-phase volume and supported x/u cases pass.

Regression acceptance does not establish untested physical models, topology
transitions, fallback parity, arbitrary stock versions or licensing clearance.
Package identity, audience, pushing/publication, outreach, upstream contribution
and changing the prompt default remain user decisions.
