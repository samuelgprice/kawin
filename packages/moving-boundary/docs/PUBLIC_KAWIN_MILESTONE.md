# Stage 1B: stock kawin 0.5.0

Status: implemented and acceptance passed after all three exact module relocations.
Quick: 29 run comparisons and 65 pytest tests passed in 437.06 s. Full: 38 run comparisons and 65 pytest tests passed in 517.25 s.

Stage 2B is complete: installed pycalphad generation needs no research checkout. See [NEXT_SESSIONS.md](NEXT_SESSIONS.md#installed-solving-and-research-owned-generation) for remaining research builders/assembly, the open broader Stage 2 split and research-branch R01.

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
and rejects examples imports/source reads only during package-side generation;
the original reference adapter and research assembly remain in the checkout.
The broader Stage 2 split and R01 research-branch repair remain open.

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
Full covers cached O-runs and longer analytic drift. Only live O06 is SKIPPED,
pending a batch-safe TC adapter; cached cases ran when present. TC-derived values
and hashes are never committed. Binary/two-phase unequal molar volumes remain
unsupported; three-phase volume and supported x/u cases pass.

Regression acceptance does not establish untested physical models, topology
transitions, fallback parity, arbitrary stock versions or licensing clearance.
Package identity, audience, pushing/publication, outreach, upstream contribution
and changing the prompt default remain user decisions.
