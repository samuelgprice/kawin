"""Approved scientific inputs, independent of reference/installed solver selection."""
from pathlib import Path
import numpy as np


def build_case(inputs, case_id, scratch):
    """Return model, duration, and optional exact-solution diagnostics for one ID."""
    root = inputs.root
    tests = root / "kawin/tests"
    analytic = None
    if case_id == "C01":
        m = inputs.load(tests / "illingworth_baseline_cases.py", ["_binary_model"])
        model, duration = m._binary_model("phase_uniform", False), .5
    elif case_id in {"C02", "C03", "C30"}:
        names = ["_model", "StrongVariableBinaryThermodynamics", "CountingVariableThermodynamics"]
        m = inputs.load(tests / "test_illingworth_binary_variable_diffusivity.py", names)
        mode = "composition_dependent_lagged" if case_id == "C02" else "composition_dependent_implicit"
        model, duration = m._model(mode, therm_class=m.StrongVariableBinaryThermodynamics), .5
        if case_id == "C30":
            model, duration = m._model("phase_uniform", therm_class=m.CountingVariableThermodynamics), .2
            temperature = inputs.symbol("kawin.diffusion", "TemperatureParameters")
            model.temperatureParameters = temperature([0., 1.], [1000., 1360.])
    elif case_id in {"C04", "C31", "F04"}:
        file = root / "examples/Illingworth2005/replicate_illingworth2005_fig4_5_6_planar.py"
        names = ["PlanarValidationCase", "build_python_planar_validation_model", "_clustered_v_grid",
                 "planar_exact_interface", "planar_exact_profile", "VALIDATION_PARAMS"]
        m = inputs.load(file, names)
        irregular = case_id == "C31"
        dt, start, end = 2e-5, 1e-4, (1e-2 if case_id == "F04" else 2e-4)
        grid = m._clustered_v_grid(10) if irregular else None
        case = m.PlanarValidationCase("4", case_id, 3 if irregular else 21,
                                     10 if irregular else 21, dt, transformed_v_grid=grid)
        config = dict(t_init=start, t_abs_end=end, mesh_nodes=401, validation_params=dict(t_init=start),
                      preallocate_recordings=True)
        model, duration = m.build_python_planar_validation_model(case, config), end - start
        analytic = ("binary", m, start)
    elif case_id in {"C05", "C06", "C07", "C08", "C09", "C10"}:
        names = ["_ternary_model", "_three_phase_model", "_CoupledTernaryThermodynamics", "_SmoothBulkTernaryThermodynamics"]
        m = inputs.load(tests / "illingworth_baseline_cases.py", names)
        if case_id in {"C09", "C10"}:
            model = m._three_phase_model("phase_uniform" if case_id == "C09" else "composition_dependent_lagged")
        else:
            mode = {"C05":"phase_uniform", "C06":"phase_uniform", "C07":"composition_dependent_lagged", "C08":"composition_dependent_implicit"}[case_id]
            left, right = ([.16, .12], [.56, .25]) if case_id == "C06" else ([.16, .06], [.3261538461538461, .193])
            therm = m._CoupledTernaryThermodynamics() if case_id in {"C05", "C06"} else m._SmoothBulkTernaryThermodynamics()
            model = m._ternary_model(mode, therm, np.array(left), np.array(right))
        duration = 2e-3
    elif case_id in {"C11", "C12", "F11", "F12"}:
        m = inputs.load(tests / "test_illingworth_ternary_three_phase_similarity.py", ["_solution", "_build_model"], constants={6001:401})
        solution = m._solution("opposite" if case_id.endswith("11") else "drift")
        model = m._build_model(solution, 21, semi_log_dt=.05)
        duration = (.05 if case_id.startswith("C") else 1.) - .01
        analytic = ("three_phase", solution, .01)
    elif case_id in {"C13", "C14", "C15"}:
        m = inputs.load(tests / "test_illingworth_interstitials.py", ["_binary_model", "_VariableBinaryThermodynamics", "StepProfile1D"])
        mode = ["phase_uniform", "composition_dependent_lagged", "composition_dependent_implicit"][int(case_id[-2:]) - 13]
        model = m._binary_model(["FE", "C"], m._VariableBinaryThermodynamics({"ALPHA":1., "BETA":.5}),
            m.StepProfile1D(1., .002, .03), (.001, .02), bulk_diffusivity_mode=mode)
        duration = .1
    elif case_id in {"C16", "C17", "C18", "C19"}:
        m = inputs.load(tests / "test_illingworth_interstitials_ternary.py", ["_model", "_closure", "_diffusivities", "_SpyTernaryDiffusivity"])
        index = int(case_id[1:]) - 16
        left, right = m._diffusivities(1e-2)
        therm = m._SpyTernaryDiffusivity({"ALPHA":left, "BETA":right}, ["ALPHA", "BETA"])
        model = m._model(therm, m._closure("u" if index % 2 == 0 else "x"),
                         mode="phase_uniform" if index < 2 else "composition_dependent_lagged")
        duration = .05
    elif case_id in {"C20", "C21"}:
        m = inputs.load(tests / "test_illingworth_interstitials_three_phase.py", ["_model"])
        model, _ = m._model("phase_uniform" if case_id == "C20" else "composition_dependent_lagged")
        duration = 2e-3
    elif case_id in {"C22", "C23"}:
        m = inputs.load(tests / "test_illingworth_ternary_three_phase.py", ["_make_nonstationary_unequal_volume_model", "_make_eta_varying_three_phase_model"])
        if case_id == "C22":
            model, duration = m._make_nonstationary_unequal_volume_model(), 2e-6
        else:
            model, duration = m._make_eta_varying_three_phase_model(phase_molar_volumes=(7.2e-6,)*3), 1e-5
    elif case_id == "C24":
        m = inputs.load(tests / "test_surrogate_artifacts.py", ["_surrogate"])
        saved = Path(inputs.artifacts) / "open-surrogate.npz"
        cls = inputs.symbol("kawin.diffusion", "TernaryMovingBoundaryThermodynamicsSurrogate")
        surrogate = cls.load(saved)
        mesh_cls = inputs.symbol("kawin.diffusion.mesh", "CartesianFD1D")
        profile = inputs.symbol("kawin.diffusion.mesh", "ProfileBuilder")
        step = inputs.symbol("kawin.diffusion.mesh", "StepProfile1D")
        mesh = mesh_cls(["X", "Y"], [0., 1.], 21)
        mesh.setResponseProfile(profile([(step(.45, [.2,.1], [.1,.2]), ["X","Y"])]))
        model = inputs.model("MovingBoundaryIllingworthTernaryFD1DModel")(
            mesh, ["Z","X","Y"], ["ALPHA","BETA"], surrogate, 1000., .45,
            interface_equilibrium=surrogate, time_step=1e-4, phase_a_nodes=9, phase_b_nodes=9,
            initial_eta_bracket=(0.,1.), record=True, terminal_thin_phase_policy="raise")
        duration = 2e-3
    elif case_id == "C32":
        inputs.policy = "continue"
        m = inputs.load(tests / "test_illingworth_ternary_three_phase.py", ["_make_nonstationary_unequal_volume_model"])
        model = m._make_nonstationary_unequal_volume_model()
        model.setup()
        model.maxIterations = 1
        model.maxStepRetries = 1
        model.terminalThinPhaseWidth = .31
        model.terminalThinPhaseExtraRetries = 25
        duration = 2e-6
    else:
        raise KeyError(case_id)
    return model, duration, analytic


def collect(model, analytic=None):
    """Capture accepted histories, physical/transformed profiles and exact errors.

    Quantities keep phase/component ordering and existing NaN masks. Unsupported
    APIs are absent and reported N/A; no reconstructed display integral is treated
    as the conservative inventory. Analytic errors are computed independently.
    """
    arrays = {}
    histories = ("data", "interfaceData", "concData", "etaData", "inventoryData", "pData", "qData",
        "profileData", "rightBoundaryData", "molarInventoryData", "totalMolesData", "dependentMoleClosureErrorData")
    for name in histories:
        value = getattr(model, name, None)
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        for index, history in enumerate(values):
            prefix = name + (str(index) if isinstance(value, (list,tuple)) else "")
            n = history.N + 1
            arrays[prefix + "/time"] = np.array(history._time[:n], copy=True)
            arrays[prefix + "/values"] = np.array(history._y[:n], copy=True)
    arrays["profiles/compositions"] = np.array(model.getCompositions(), copy=True)
    arrays["accepted/final_time"] = np.asarray(model.currentTime)
    if hasattr(model, "getPhysicalPhaseProfiles"):
        for i, (z, c) in enumerate(model.getPhysicalPhaseProfiles()):
            arrays[f"profiles/phase{i}/z"] = np.array(z, copy=True)
            arrays[f"profiles/phase{i}/x"] = np.array(c, copy=True)
    for name in ("_u_grid", "_v_grid", "_grids", "_initialInventory", "_initialMolarInventory"):
        value = getattr(model, name, None)
        if value is not None:
            if isinstance(value, (list,tuple)):
                for index, part in enumerate(value):
                    arrays[f"state/{name}/{index}"] = np.asarray(part).copy()
            else:
                arrays["state/" + name] = np.asarray(value).copy()
    if analytic:
        kind, exact, start = analytic
        times = arrays["interfaceData/time"] + start
        if kind == "binary":
            params = exact.VALIDATION_PARAMS
            arrays["exact/interface_error"] = arrays["interfaceData/values"] - exact.planar_exact_interface(times, params)
            z = np.ravel(model.mesh.z)
            exact_profile = exact.planar_exact_profile(z, model.currentTime + start, params)
            arrays["exact/profile_error"] = np.ravel(model.data.currentY) - np.ravel(exact_profile)
        else:
            arrays["exact/interface_error"] = arrays["interfaceData/values"] - np.asarray([exact.interface_positions(t) for t in times])
            arrays["exact/eta_error"] = arrays["etaData/values"] - np.asarray(exact.etas)
            for i, (z, c) in enumerate(model.getPhysicalPhaseProfiles()):
                arrays[f"exact/profile_error{i}"] = np.asarray(c)[:,1:] - exact.profile(z, model.currentTime + start, i)
    return arrays
