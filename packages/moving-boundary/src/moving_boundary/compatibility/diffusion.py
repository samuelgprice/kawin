"""Initial composition and defaults without replacing the host diffusion model.

Policy source: 06dc7b4 kawin/diffusion/{Diffusion,DiffusionParameters}.py.
"""
import numpy as np
from kawin.diffusion.Diffusion import DiffusionModel as _HostDiffusion
from kawin.diffusion.DiffusionParameters import DiffusionConstraints as _HostConstraints
from kawin.thermo.Mobility import expand_x_frac, x_to_u_frac, interstitials
from .solver import GenericModel
from .recording import MeshData


class DiffusionConstraints(_HostConstraints):
    """Preserve research composition floor and moving-boundary constraint defaults."""
    def reset(self):
        """Reset host constraints, then apply the documented local defaults."""
        super().reset()
        self.minComposition = 1e-10
        self.movingBoundaryThreshold = .25
        self.movingBoundaryMassTolerance = np.inf
        self.movingBoundaryMassAction = 'warn'


class DiffusionModel(_HostDiffusion, GenericModel):
    """Retain valid initial compositions; adapt only recording and solve policy.

    Explicit user constraints are preserved. Host construction/reset, temperature
    handling, time bookkeeping and composition conversion remain authoritative.
    """
    def __init__(self, mesh, elements, phases, thermodynamics, temperature,
                 constraints=None, record=False):
        """Construct the host model with local defaults and transfer its initial record."""
        if constraints is None:
            constraints = DiffusionConstraints()
        super().__init__(mesh, elements, phases, thermodynamics, temperature,
                         constraints=constraints, record=record)
        data = MeshData(mesh, record)
        data.__dict__.update(self.data.__dict__)
        self.data = data

    def _validateMeshComposition(self):
        '''
        Checks that composition along mesh will be between min/max limits (set by DiffusionConstraints.minComposition)
        Since the mesh can store y in an arbitrary shape, we use the flatten array and unflatten it to store back in the mesh
        '''
        yFlat = self.mesh.flattenResponse(self.mesh.y)
        ySum = np.sum(yFlat, axis=1)
        if np.any(ySum > 1) or np.any(ySum < 0):
            raise Exception('Some compositions sum up to below 0 or above 1')
        # We a scaled minimum composition to account for multicomponent system
        # So for a 3 component system, if 2 components at at the min comp, then the third is as 1-2*minComp
        scaledMinComp = len(self.allElements)*self.constraints.minComposition
        if ((1-ySum)<scaledMinComp).any() or (yFlat<scaledMinComp).any():
            yFlat[yFlat > scaledMinComp] = yFlat[yFlat > scaledMinComp] - scaledMinComp
            yFlat[yFlat < scaledMinComp] = self.constraints.minComposition

        # Convert composition to u-fraction and store into mesh
        yFull = expand_x_frac(yFlat)
        uFull = x_to_u_frac(yFull, self.allElements, interstitials, return_usum=False)
        self.data.setResponseAtN(uFull[:,1:])
        self.data.currentY = uFull[:,1:]
