"""Small analytic, closed planar binary example; no database is needed."""

import numpy as np
from moving_boundary.solvers import MovingBoundaryIllingworthFD1DModel
from kawin.diffusion.DiffusionParameters import TemperatureParameters
from kawin.diffusion.mesh import CartesianFD1D, ProfileBuilder, StepProfile1D


class ConstantDiffusivity:
    """Supply positive phase-uniform scalar diffusivities in solver units."""

    def clearCache(self):
        """No cache is retained by this analytic provider."""

    def getInterdiffusivity(self, x, T, removeCache=True, phase=None,
                           query_context=None):
        """Return a scalar or temperature-shaped array for the requested phase."""
        return np.squeeze(
            np.ones(np.atleast_1d(T).shape) *
            {"ALPHA": 1.0e-7, "BETA": 1.0e-5}[phase]
        )


def build_model(model_class=MovingBoundaryIllingworthFD1DModel):
    """Build a substitutional, constant-temperature, zero-flux planar case.

    Uses the existing baseline's grid, fixed interface compositions and timestep.
    The injected class enables independent legacy/facade runs without patching
    globals or changing the numerical inputs.
    """
    mesh = CartesianFD1D(["CR"], [0.0, 5.0], 501)
    mesh.setResponseProfile(ProfileBuilder([(StepProfile1D(1.0, 0.8, 0.4), "CR")]))
    return model_class(
        mesh, ["NI", "CR"], ["ALPHA", "BETA"],
        thermodynamics=ConstantDiffusivity(),
        temperature=TemperatureParameters(1000.0),
        interfacePosition=1.0, interface_compositions=(0.6, 0.0),
        time_step=0.1, phase_a_nodes=100, phase_b_nodes=100,
        tolerance=1.0e-8, record=True, bulk_diffusivity_mode="phase_uniform",
    )


if __name__ == "__main__":
    model = build_model()
    model.solve(0.5, minDtFrac=1e-14)
    print(f"time={model.currentTime}, interface={model._s_curr}")
