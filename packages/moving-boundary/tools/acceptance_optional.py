"""Approved cached and real-provider inputs; never auto-build a missing TC cache."""
from pathlib import Path
from contextlib import contextmanager
import dataclasses
import json
import numpy as np

CACHES = {
    "O01":"exb5.json", "O02":"exb5.json", "O03a":"femnc.json", "O03b":"femnc.json",
    "O04":"fecrni.surrogate", "O07":"fec.json",
}


def build_optional(inputs, case):
    """Construct shortened research cases from existing immutable private inputs."""
    root = inputs.root
    artifacts = Path(inputs.artifacts)
    inputs.policy = "continue" if case in {"O01", "O02", "O03a", "O03b"} else "raise"
    if case in {"O01", "O02"}:
        fn = "run_kawin2" if case == "O01" else "run_kawin3"
        m = inputs.load(root / "examples/ternaryExamples/exb5_gamma_alpha_gamma.py", [fn],
            constants={4001:401, 8001:801, 3.6e6:3600.}, constructors=[fn])
        data = json.loads((artifacts / CACHES[case]).read_text())
        kwargs = dict(n_alpha=21, n_gamma=41, semi_log_dt=.05, semi_log_t0=1e-2)
        return getattr(m, fn)(data, **kwargs), 3600., None
    if case == "O07":
        m = inputs.load(root / "examples/Interstitials/fe_c_exb1a.py", ["run_kawin"],
                        constants={4001:401}, constructors=["run_kawin"])
        data = json.loads((artifacts / CACHES[case]).read_text())
        return m.run_kawin(data, .1, n_left=21, n_right=81, semi_log_dt=.05), .1, None
    if case in {"O03a", "O03b"}:
        m = inputs.load(root / "examples/Interstitials/ternary_ferrite_growth_le.py",
            ["run_kawin", "tie_line_closure", "TabulatedTernaryDiffusivity"],
            constants={2001:401}, constructors=["run_kawin"])
        config = inputs.load(root / "examples/Interstitials/fe_mn_c_ferrite.py", ["CASE"])
        case_config = dataclasses.replace(config.CASE, sim_time=.02)
        data = json.loads((artifacts / CACHES[case]).read_text())
        if case == "O03a":
            return m.run_kawin(case_config, data, n_left=21, n_right=81, semi_log_dt=.05), .02, None
        wrapper = inputs.load(root / "examples/Interstitials/three_phase_symmetric_check.py", ["_AusteniteFerriteClosure"])
        mesh_cls = inputs.symbol("kawin.diffusion.mesh", "CartesianFD1D")
        profile = inputs.symbol("kawin.diffusion.mesh", "ProfileBuilder")
        step = inputs.symbol("kawin.diffusion.mesh", "StepProfile1D")
        grid = inputs.symbol("kawin.diffusion.mesh.TransformedGrids", "landau_grid")
        half, width = case_config.ferrite_width, case_config.austenite_width
        length = 2*width + 2*half
        x = np.asarray(data["x_alloy"])
        mesh = mesh_cls(["C",case_config.solute], [0.,length], 801)
        mesh.setResponseProfile(profile([(step(.5*length,x,x), ["C",case_config.solute])]))
        closure = m.tie_line_closure(data)
        model = inputs.model("MovingBoundaryIllingworthTernaryThreePhaseFD1DModel")(
            mesh=mesh, elements=["FE","C",case_config.solute], phases=["FCC_A1","BCC_A2","FCC_A1"],
            thermodynamics=m.TabulatedTernaryDiffusivity(data["tables"]), temperature=case_config.temperature,
            interfacePositions=(width,width+2*half), interface_equilibria=(wrapper._AusteniteFerriteClosure(closure), closure),
            time_step=.02, dt_mode="semi_log", semiLog_dt=.05, semiLogT0=1e-7,
            bulk_diffusivity_mode="composition_dependent_lagged", tolerance=1e-10, max_iterations=50,
            max_step_retries=12, terminal_thin_phase_width=1e-12, terminal_thin_phase_policy="continue", record=True,
            transformed_grids=(grid(81,"power",phase="left",exponent=3.),
                grid(21,"linear",phase="middle",spacing_ratio=.2), grid(81,"power",phase="right",exponent=3.)))
        return model, .02, None
    if case in {"O04", "O05"}:
        if case == "O04":
            bundle_cls = inputs.symbol("kawin.diffusion", "SurrogateArtifactBundle")
            surrogate = bundle_cls.load_published(artifacts / "fecrni.surrogate").surrogate
        else:
            surrogate = sample_pycalphad(inputs)
        m = inputs.load(root / "examples/ternaryExamples/FeCrNi/IllingworthTernaryTwoPhase_FeCrNi_AnalyticComparison.py",
            ["build_comparison_model", "comparison_geometry", "interface_matrices_at_eta"],
            overrides=dict(surrogate=surrogate, PHASE_A="BCC_A2", PHASE_B="FCC_A1", PHYSICAL_NODES=401))
        matrices = m.interface_matrices_at_eta(surrogate, .5)
        s0, length = m.comparison_geometry(matrices)
        probe = m.build_comparison_model(surrogate, s0, length, 21, 41, .05)
        probe.setup()
        matrices = m.interface_matrices_at_eta(surrogate, probe.getInterfaceEta())
        s0, length = m.comparison_geometry(matrices)
        therm_cls = inputs.load(root / "examples/ternaryExamples/ternaryTwoPhaseSimilarity.py", ["FixedMatrixTernaryDiffusivity"]).FixedMatrixTernaryDiffusivity
        model = m.build_comparison_model(therm_cls(matrices,["BCC_A2","FCC_A1"]), s0, length, 21, 41, .05)
        model.dtMode, model.timeStep = "fixed", 600.
        model._acceptance_provider_surrogate = surrogate
        return model, 3600., None
    raise KeyError(case)


@contextmanager
def guard_package_generation():
    """Reject examples/debugger imports and every examples source read while sampling.

    An audit hook also catches file reads that bypass Python import machinery.
    Audit hooks cannot be removed, so the hook becomes inert on context exit.
    Research assembly after generation is deliberately outside this boundary.
    """
    import os
    import sys
    active = True
    audit = dict(guarded=True,examples_imported=False,examples_read=False)
    def reject_import(name):
        """Fail for research/debugger roots, including any preloaded module."""
        if name.split(".")[0] in {"examples","debugpy"}:
            raise AssertionError("Research/debugger import during package generation: " + name)
    class NoResearch:
        """Reject research/debugger modules before their loaders can read source."""
        def find_spec(self, fullname, path=None, target=None):
            """Allow other finders to resolve every permitted module."""
            reject_import(fullname)
    def audit_event(event, args):
        """Catch direct reads and import events only while this context is active."""
        if not active:
            return
        if event == "import":
            reject_import(args[0])
        elif event == "open" and isinstance(args[0],(str,bytes,os.PathLike)):
            parts = os.fsdecode(args[0]).replace("\\","/").lower().split("/")
            if "examples" in parts:
                raise AssertionError("Research examples source read during package generation: " + os.fsdecode(args[0]))
    for name in sys.modules:
        reject_import(name)
    finder = NoResearch()
    sys.meta_path.insert(0,finder)
    sys.addaudithook(audit_event)
    try:
        yield audit
        for name in sys.modules:
            reject_import(name)
    finally:
        active = False
        sys.meta_path.remove(finder)


def sample_pycalphad(inputs):
    """Sample real tie lines; package generation must use installed code without examples.

    Reference generation continues to load the frozen 06dc7b4 adapter. The
    scoped package guard ends before research case assembly begins.
    """
    if inputs.role == "package":
        with guard_package_generation() as audit:
            from moving_boundary.providers import pycalphad as source
            import moving_boundary
            assert Path(source.__file__).resolve().is_relative_to(Path(moving_boundary.__file__).resolve().parent)
            surrogate = _sample_pycalphad(inputs,source)
            audit.update(adapter_module=source.__name__,adapter_origin=source.__file__)
            inputs.generation_audit = audit
            return surrogate
    source = inputs.load(inputs.root / "examples/ternaryExamples/pycalphad_default_phase_adapter.py",
                         ["create_pycalphad_thermodynamics_source"])
    return _sample_pycalphad(inputs,source)


def _sample_pycalphad(inputs, source):
    """Keep the approved five-tie-line real sampling inputs and arithmetic unchanged."""
    database = Path(inputs.artifacts) / "FeCrNi.tdb"
    therm = source.create_pycalphad_thermodynamics_source(database, ["FE","CR","NI"],
        ["BCC_A2","FCC_A1"], use_default_phases=False, g_offset=0.)
    cls = inputs.symbol("kawin.diffusion", "TernaryMovingBoundaryThermodynamicsSurrogate")
    surrogate = cls.from_database(thermodynamics=therm, elements=["FE","CR","NI"],
        phases=["BCC_A2","FCC_A1"], tieline_phases=["BCC_A2","FCC_A1"], temperature=1373.,
        probe_point=[.45,.21], probe_samples_per_side=2, precipitate_phase="FCC_A1",
        diffusivity_bulk_points=np.array([[.39696,.0001],[.139297,.142387]]),
        validity_policy="legacy", min_composition=1e-10)
    surrogate.save(Path(inputs.artifacts) / ("pycalphad-" + inputs.role + ".npz"))
    return surrogate
