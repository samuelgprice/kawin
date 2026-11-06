"""Open binary parity inputs shared by source and isolated installed workers.

Case inputs/provider follow kawin/tests/illingworth_baseline_cases.py at
ae9fb2971a9fd49132f71ddc883a5bd5406644b6. No expected outputs are generated here.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def run_cases(model_class, example_path):
    """Collect all binary fixture histories in each of the three diffusivity modes."""
    spec = importlib.util.spec_from_file_location("binary_example", example_path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    arrays = {}
    for mode in ("phase_uniform", "composition_dependent_lagged",
                 "composition_dependent_implicit"):
        class Provider:
            def clearCache(self):
                pass

            def getInterdiffusivity(self, x, T, removeCache=True, phase=None,
                                   query_context=None):
                base = float({"ALPHA": 1.0e-7, "BETA": 1.0e-5}[phase])
                if mode != "phase_uniform":
                    composition = np.asarray(x, dtype=np.float64)
                    return np.squeeze(base * (1.0 + 0.5 * composition))
                return np.squeeze(np.ones(np.atleast_1d(T).shape, dtype=np.float64) * base)

        class CaseModel(model_class):
            def __init__(self, *args, **kwargs):
                kwargs.update(thermodynamics=Provider(), bulk_diffusivity_mode=mode)
                super().__init__(*args, **kwargs)

        model = example.build_model(CaseModel)
        model.solve(0.5, minDtFrac=1e-14)
        assert model.currentTime == 0.5
        prefix = f"binary_{mode}__"
        arrays[prefix + "compositions"] = model.getCompositions()
        for name in ("data", "interfaceData", "concData", "pData", "qData"):
            history = getattr(model, name)
            n = history.N + 1
            arrays[prefix + name + "_time"] = history._time[:n].copy()
            arrays[prefix + name + "_y"] = history._y[:n].copy()
    return arrays


def run_bulk_cases(model_class):
    """Exercise both upwind branches and stationary motion on nonuniform grids.

    Scalar and face-valued diffusivities use independent planar phase states.
    Calls only require grids, domain length and fixed interface concentrations;
    no mesh, driver, thermodynamic engine or global import patch is involved.
    """
    state = SimpleNamespace(_u_grid=np.array([0., .08, .3, .7, 1.]),
                            _v_grid=np.array([0., .2, .55, .9, 1.]),
                            _R=5., interfaceCompositions=(.6, .1))
    arrays = {}
    for future_s in (.9, 1., 1.1):
        for kind, diffusion in (("scalar", 1.e-3),
                                ("faces", np.array([1.e-3, 2.e-3, 4.e-3, 3.e-3]))):
            for side, profile in (("left", np.array([.8, .78, .75, .7, .6])),
                                  ("right", np.array([.1, .2, .3, .35, .4]))):
                method = getattr(model_class, f"_new_concentration_{side}_planar")
                arrays[f"bulk__{future_s}_{kind}_{side}"] = method(
                    state, profile, 1., future_s, .1, diffusion)
    return arrays


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "legacy":
        from kawin.diffusion.MovingBoundaryIllingworthFDM import MovingBoundaryIllingworthFD1DModel as Model
    else:
        from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel as Model
        assert Model._new_concentration_left_planar.__module__ == "moving_boundary.solvers.binary"
        assert Model._new_concentration_right_planar.__module__ == "moving_boundary.solvers.binary"
    arrays = run_cases(Model, Path(__file__).parents[1] / "examples/binary_constant.py")
    arrays.update(run_bulk_cases(Model))
    np.savez(sys.argv[2], **arrays)
