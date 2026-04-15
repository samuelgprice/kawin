import numpy as np

from kawin.diffusion.mesh.FVM1D import MixedBoundary1D, PeriodicBoundary1D
from kawin.diffusion.mesh.MeshBase import (
    FiniteVolumeGrid,
    DiffusionPair,
    arithmeticMean,
    noChangeAtNode,
)


class FiniteDifference1D(FiniteVolumeGrid):
    """
    1D node-centered finite-difference mesh on a uniform Cartesian grid.

    The response variables are stored at the grid nodes, including the domain
    boundaries. Interface fluxes are still reported on an ``N+1`` edge layout so
    that plotting and downstream code can use the same shape as the finite-volume
    implementation.
    """

    def __init__(self, responses, zlim, N):
        if N < 2:
            raise ValueError("FiniteDifference1D requires at least two nodes.")
        super().__init__(responses, [zlim], [N], 1)

    def defineZCoordinates(self):
        z = np.linspace(self.zlim[0][0], self.zlim[0][1], self.Ns[0], dtype=np.float64)
        self.z = np.reshape(z, (self.Ns[0], 1))
        self.dzs = [self.z[1] - self.z[0]]
        self.dz = float(self.dzs[0][0])

        z_edge = np.empty(self.Ns[0] + 1, dtype=np.float64)
        z_edge[0] = z[0]
        z_edge[-1] = z[-1]
        z_edge[1:-1] = 0.5 * (z[:-1] + z[1:])
        self.zEdge = np.reshape(z_edge, (self.Ns[0] + 1, 1))

    def setResponseProfile(self, profileBuilder, boundaryConditions=None):
        if boundaryConditions is None:
            boundaryConditions = MixedBoundary1D(self.numResponses)
        self.boundaryConditions = boundaryConditions
        super().setResponseProfile(profileBuilder, self.boundaryConditions)

    def _getControlVolumeWidths(self):
        widths = np.ones(self.Ns[0], dtype=np.float64) * self.dz
        widths[0] *= 0.5
        widths[-1] *= 0.5
        return widths[:, np.newaxis]

    def getDiffusivityCoordinates(self, y):
        return y, self.z

    def computeFluxes(self, pairs: list[DiffusionPair]):
        fluxes = np.zeros((self.Ns[0] + 1, self.numResponses), dtype=np.float64)
        isPeriodic = isinstance(self.boundaryConditions, PeriodicBoundary1D)

        for p in pairs:
            D = p.diffusivity
            r = p.response
            avgFunc = arithmeticMean if p.averageFunction is None else p.averageFunction
            atNodeFunc = noChangeAtNode if p.atNodeFunction is None else p.atNodeFunction

            D_mid = np.asarray(avgFunc([D[:-1], D[1:]]), dtype=np.float64)
            response_diff = np.asarray(r[1:] - r[:-1], dtype=np.float64)
            if D_mid.ndim < response_diff.ndim:
                D_mid = np.expand_dims(D_mid, axis=-1)
            fluxes[1:-1] += -D_mid * response_diff / self.dz

            if isPeriodic:
                end_D = np.asarray(
                    avgFunc([D[-1][np.newaxis, ...], D[0][np.newaxis, ...]]),
                    dtype=np.float64,
                )[0]
                end_response_diff = np.asarray(r[0] - r[-1], dtype=np.float64)
                if np.ndim(end_D) < np.ndim(end_response_diff):
                    end_D = np.expand_dims(end_D, axis=-1)
                end_flux = -end_D * end_response_diff / self.dz
                fluxes[0] += end_flux
                fluxes[-1] += end_flux
            else:
                left_response_diff = np.asarray(r[1] - r[0], dtype=np.float64)
                right_response_diff = np.asarray(r[-1] - r[-2], dtype=np.float64)
                left_D = np.asarray(atNodeFunc(D[0]), dtype=np.float64) ##Note: I'm not sure this is the right way to do BCs for finite difference> It may make more sense to still use D_mid[0] (Actually this is what happens because fluxes[0] gets set to 0 by adjustFluxes())
                right_D = np.asarray(atNodeFunc(D[-1]), dtype=np.float64)
                if np.ndim(left_D) < np.ndim(left_response_diff):
                    left_D = np.expand_dims(left_D, axis=-1)
                if np.ndim(right_D) < np.ndim(right_response_diff):
                    right_D = np.expand_dims(right_D, axis=-1)
                fluxes[0] += -left_D * left_response_diff / self.dz
                fluxes[-1] += -right_D * right_response_diff / self.dz

        self.boundaryConditions.adjustFluxes(fluxes)
        return fluxes

    def computedXdt(self, pairs: list[DiffusionPair]):
        fluxes = self.computeFluxes(pairs)
        widths = self._getControlVolumeWidths()
        dXdt = -(fluxes[1:] - fluxes[:-1]) / widths
        self.boundaryConditions.adjustdXdt(dXdt)
        return dXdt


class CartesianFD1D(FiniteDifference1D):
    """Uniform 1D Cartesian finite-difference mesh."""
