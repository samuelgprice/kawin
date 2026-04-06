import numpy as np

from kawin.GenericModel import GenericModel
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.diffusion.mesh.FVM1D import Cartesian1D, MixedBoundary1D, PeriodicBoundary1D
from kawin.diffusion.mesh.MovingBoundary1D import (
    get_control_volume_widths,
    get_moving_boundary_geometry,
    integrate_binary_profile,
)
from kawin.thermo.Mobility import interstitials


class _ScalarHistory:
    def __init__(self, record: bool | int = False):
        if isinstance(record, bool):
            self.recordInterval = 1 if record else -1
        else:
            self.recordInterval = record
        self.batchSize = 1000
        self.reset()

    def reset(self):
        self._y = np.zeros(self.batchSize, dtype=np.float64)
        self._time = np.zeros(self.batchSize, dtype=np.float64)
        self.currentIndex = 0
        self.currentY = 0.0
        self.currentTime = 0.0
        self.N = 0

    def record(self, time, y, force: bool = False):
        if self.recordInterval > 0:
            if self.currentIndex % self.recordInterval == 0 or force:
                self.N = int(self.currentIndex / self.recordInterval)
                if self.N >= self._time.shape[0]:
                    self._y = np.pad(self._y, (0, self.batchSize))
                    self._time = np.pad(self._time, (0, self.batchSize))
                self._y[self.N] = y
                self._time[self.N] = time
            self.currentIndex += 1
        else:
            self._y[self.N] = y
            self._time[self.N] = time

        self.currentY = float(y)
        self.currentTime = float(time)

    def finalize(self):
        self.record(self.currentTime, self.currentY, force=True)
        self._y = self._y[: self.N + 1]
        self._time = self._time[: self.N + 1]

    def y(self, time=None):
        if time is None:
            return float(self._y[self.N])

        if self.recordInterval > 0:
            if time <= self._time[0]:
                return float(self._y[0])
            if time >= self._time[self.N]:
                return float(self._y[self.N])
            upper = int(np.argmax(self._time[: self.N + 1] > time))
            lower = upper - 1
            y0 = self._y[lower]
            y1 = self._y[upper]
            t0 = self._time[lower]
            t1 = self._time[upper]
            return float((y1 - y0) * (time - t0) / (t1 - t0) + y0)
        return float(self._y[0])


class MovingBoundary1DModel(DiffusionModel):
    """
    Binary 1D moving-boundary diffusion model on a fixed Cartesian grid.

    The composition field is stored on the existing 1D mesh cell centers,
    while the planar interface position is evolved as an explicit scalar state.
    """

    def __init__(
        self,
        mesh,
        elements,
        phases,
        thermodynamics,
        temperature,
        interfacePosition,
        constraints=None,
        record=False,
    ):
        self.initialInterfacePosition = float(interfacePosition)
        self.interfaceData = _ScalarHistory(record)
        self._initialInventory = None
        self._lastFluxes = None
        self._lastInterfaceFluxes = (0.0, 0.0)
        self._lastInterfaceVelocity = 0.0
        self._currdt = np.inf
        super().__init__(
            mesh=mesh,
            elements=elements,
            phases=phases,
            thermodynamics=thermodynamics,
            temperature=temperature,
            constraints=constraints,
            record=record,
        )
        self._validateMovingBoundaryModel()
        self.interfaceData.currentY = self.initialInterfacePosition
        self.interfaceData._y[0] = self.initialInterfacePosition
        self._initialInventory = self.getTotalMass()

    def _validateMovingBoundaryModel(self):
        if not isinstance(self.mesh, Cartesian1D):
            raise TypeError("MovingBoundary1DModel requires a Cartesian1D mesh.")
        if len(self.allElements) != 2 or self.mesh.numResponses != 1:
            raise ValueError("MovingBoundary1DModel currently supports only binary systems.")
        if any(e in interstitials for e in self.allElements):
            raise ValueError("MovingBoundary1DModel currently supports only substitutional binary systems.")
        if len(self.phases) != 2:
            raise ValueError("MovingBoundary1DModel requires exactly two explicit phases.")
        if not hasattr(self.therm, "getInterfacialComposition"):
            raise TypeError("Thermodynamics object must implement getInterfacialComposition for MovingBoundary1DModel.")
        if not hasattr(self.therm, "getInterdiffusivity"):
            raise TypeError("Thermodynamics object must implement getInterdiffusivity for MovingBoundary1DModel.")
        if isinstance(getattr(self.mesh, "boundaryConditions", None), PeriodicBoundary1D):
            raise ValueError("Periodic boundary conditions are not supported for MovingBoundary1DModel.")
        self._clipInterfacePosition(self.initialInterfacePosition, strict=True)

    def reset(self):
        super().reset()
        self.interfaceData.reset()
        self.interfaceData.record(0, self.initialInterfacePosition)
        self._lastFluxes = None
        self._lastInterfaceFluxes = (0.0, 0.0)
        self._lastInterfaceVelocity = 0.0
        self._currdt = np.inf
        if hasattr(self, "mesh") and self.mesh is not None and hasattr(self, "constraints"):
            self._validateMeshComposition()
        if hasattr(self, "mesh") and self.mesh is not None:
            self._initialInventory = integrate_binary_profile(
                self.mesh, np.asarray(self.data.currentY).reshape(-1), self.initialInterfacePosition
            )

    def toDict(self):
        data = super().toDict()
        data.update(
            {
                "interface_position": self.interfaceData._y,
                "interface_time": self.interfaceData._time,
                "interface_interval": self.interfaceData.recordInterval,
                "interface_index": self.interfaceData.N,
            }
        )
        return data

    def fromDict(self, data):
        super().fromDict(data)
        self.interfaceData.recordInterval = int(data["interface_interval"])
        self.interfaceData.N = int(data["interface_index"])
        self.interfaceData._y = np.array(data["interface_position"], dtype=np.float64)
        self.interfaceData._time = np.array(data["interface_time"], dtype=np.float64)
        self.interfaceData.currentY = float(self.interfaceData._y[-1])
        self.interfaceData.currentTime = float(self.interfaceData._time[-1])
        self.interfaceData.currentIndex = self.interfaceData.N
        self._initialInventory = self.getTotalMass(0)

    def setup(self):
        super().setup()
        self._validateMovingBoundaryModel()

    def getCurrentX(self):
        return [self.data.currentY, self.interfaceData.currentY]

    def flattenX(self, X):
        return np.concatenate((np.asarray(X[0], dtype=np.float64).reshape(-1), [float(X[1])]))

    def unflattenX(self, X_flat, X_ref):
        comp_shape = np.asarray(X_ref[0]).shape
        n_comp = int(np.prod(comp_shape))
        composition = np.reshape(X_flat[:n_comp], comp_shape)
        interface_position = float(X_flat[n_comp])
        return [composition, interface_position]

    def _getBoundaryConditions(self):
        boundary_conditions = getattr(self.mesh, "boundaryConditions", None)
        if boundary_conditions is None:
            boundary_conditions = MixedBoundary1D(self.mesh.responses)
            self.mesh.boundaryConditions = boundary_conditions
        return boundary_conditions

    def _clipInterfacePosition(self, interface_position: float, strict: bool = False) -> float:
        z = np.ravel(self.mesh.z)
        eps = max(float(self.mesh.dz) * 1e-8, 1e-14)
        lower = float(z[0] + eps)
        upper = float(z[-1] - eps)
        if strict and not (lower < interface_position < upper):
            raise ValueError("Interface position must lie strictly within the cell-center domain.")
        return float(np.clip(interface_position, lower, upper))

    def _getInterfaceState(self, t, composition, interface_position):
        geometry = get_moving_boundary_geometry(self.mesh, interface_position)
        T_interface = float(self.temperatureParameters(np.array([[interface_position]]), t)[0])
        c_left_int, c_right_int = self.therm.getInterfacialComposition(T_interface, 0, precPhase=self.phases[1])
        c_left_int = float(np.squeeze(c_left_int))
        c_right_int = float(np.squeeze(c_right_int))
        if c_left_int < 0 or c_right_int < 0:
            raise ValueError("Interface local equilibrium could not be determined for the requested phase pair.")
        c_left_int = float(np.clip(c_left_int, self.constraints.minComposition, 1 - self.constraints.minComposition))
        c_right_int = float(np.clip(c_right_int, self.constraints.minComposition, 1 - self.constraints.minComposition))
        D_left_int = float(np.squeeze(self.therm.getInterdiffusivity(c_left_int, T_interface, phase=self.phases[0])))
        D_right_int = float(np.squeeze(self.therm.getInterdiffusivity(c_right_int, T_interface, phase=self.phases[1])))
        return geometry, T_interface, c_left_int, c_right_int, D_left_int, D_right_int

    def _computeBulkFluxes(self, composition, t, geometry):
        comp = np.asarray(composition, dtype=np.float64).reshape(-1)
        z = np.ravel(self.mesh.z)
        face_fluxes = np.zeros(len(comp) + 1, dtype=np.float64)
        left_bc = self._getBoundaryConditions()
        max_diffusivity = 0.0

        if geometry.left_index > 0:
            left_comp = comp[: geometry.left_index + 1]
            left_t = self.temperatureParameters(self.mesh.z[: geometry.left_index + 1], t)
            left_D = np.asarray(
                self.therm.getInterdiffusivity(left_comp, left_t, phase=self.phases[0]), dtype=np.float64
            ).reshape(-1)
            if len(left_D) > 0:
                max_diffusivity = max(max_diffusivity, float(np.max(np.abs(left_D))))
            if len(left_D) > 1:
                left_D_face = 0.5 * (left_D[:-1] + left_D[1:])
                face_fluxes[1 : geometry.left_index + 1] = -left_D_face * np.diff(left_comp) / np.diff(z[: geometry.left_index + 1])

        if geometry.right_index < len(comp) - 1:
            right_comp = comp[geometry.right_index:]
            right_t = self.temperatureParameters(self.mesh.z[geometry.right_index:], t)
            right_D = np.asarray(
                self.therm.getInterdiffusivity(right_comp, right_t, phase=self.phases[1]), dtype=np.float64
            ).reshape(-1)
            if len(right_D) > 0:
                max_diffusivity = max(max_diffusivity, float(np.max(np.abs(right_D))))
            if len(right_D) > 1:
                right_D_face = 0.5 * (right_D[:-1] + right_D[1:])
                start = geometry.right_index + 1
                stop = len(comp)
                face_fluxes[start:stop] = -right_D_face * np.diff(right_comp) / np.diff(z[geometry.right_index:])

        fluxes_2d = face_fluxes[:, np.newaxis]
        left_bc.adjustFluxes(fluxes_2d)
        return fluxes_2d[:, 0], max_diffusivity

    def _computeState(self, t, xCurr):
        composition = np.asarray(xCurr[0], dtype=np.float64).reshape(-1)
        interface_position = self._clipInterfacePosition(float(xCurr[1]))
        geometry, _, c_left_int, c_right_int, D_left_int, D_right_int = self._getInterfaceState(
            t, composition, interface_position
        )
        face_fluxes, max_bulk_diffusivity = self._computeBulkFluxes(composition, t, geometry)
        J_int_left = -D_left_int * (c_left_int - composition[geometry.left_index]) / geometry.left_distance
        J_int_right = -D_right_int * (composition[geometry.right_index] - c_right_int) / geometry.right_distance
        velocity = (J_int_right - J_int_left) / (c_right_int - c_left_int)

        dcdt = np.zeros_like(composition, dtype=np.float64)
        widths = get_control_volume_widths(self.mesh, interface_position)
        for i in range(len(composition)):
            if i == geometry.left_index:
                left_flux = face_fluxes[i]
                right_flux = J_int_left
            elif i == geometry.right_index:
                left_flux = J_int_right
                right_flux = face_fluxes[i + 1]
            else:
                left_flux = face_fluxes[i]
                right_flux = face_fluxes[i + 1]
            dcdt[i] = -(right_flux - left_flux) / widths[i]

        bc = self._getBoundaryConditions()
        bc.adjustdXdt(dcdt[:, np.newaxis])
        face_fluxes[geometry.right_index] = 0.5 * (J_int_left + J_int_right)

        diff_candidates = [max_bulk_diffusivity]
        if np.isfinite(D_left_int):
            diff_candidates.append(abs(D_left_int))
        if np.isfinite(D_right_int):
            diff_candidates.append(abs(D_right_int))
        D_scale = max(diff_candidates)
        min_width = float(np.min(widths))
        if D_scale > 0:
            dt_diff = self.constraints.vonNeumannThreshold * min_width**2 / D_scale
        else:
            dt_diff = np.inf
        if abs(velocity) > 0:
            dt_move = 0.45 * min(geometry.left_distance, geometry.right_distance) / abs(velocity)
        else:
            dt_move = np.inf
        self._currdt = min(dt_diff, dt_move)
        self._lastFluxes = face_fluxes
        self._lastInterfaceFluxes = (float(J_int_left), float(J_int_right))
        self._lastInterfaceVelocity = float(velocity)
        return dcdt[:, np.newaxis], float(velocity)

    def getFluxes(self, t, xCurr):
        self._computeState(t, xCurr)
        return self._lastFluxes[:, np.newaxis]

    def getdXdt(self, t, xCurr):
        dcdt, velocity = self._computeState(t, xCurr)
        return [dcdt, velocity]

    def getDt(self, dXdt):
        if np.isfinite(self._currdt) and self._currdt > 0:
            return self._currdt
        return self.deltaTime

    def _isClosedSystem(self):
        bc = self._getBoundaryConditions()
        if isinstance(bc, PeriodicBoundary1D):
            return False
        if not isinstance(bc, MixedBoundary1D):
            return True
        return (
            np.all(bc.LBCtype == MixedBoundary1D.NEUMANN)
            and np.all(bc.RBCtype == MixedBoundary1D.NEUMANN)
            and np.allclose(bc.LBCvalue, 0)
            and np.allclose(bc.RBCvalue, 0)
        )

    def _correctInterfaceMass(self, composition, interface_position):
        if not self._isClosedSystem():
            return interface_position

        geometry = get_moving_boundary_geometry(self.mesh, interface_position)
        current_mass = integrate_binary_profile(self.mesh, composition, interface_position)
        delta_mass = self._initialInventory - current_mass
        denominator = composition[geometry.left_index] - composition[geometry.right_index]
        if abs(denominator) < 1e-14:
            return interface_position
        corrected = interface_position + delta_mass / denominator
        corrected = float(np.clip(corrected, geometry.left_center + 1e-14, geometry.right_center - 1e-14))
        return corrected

    def postProcess(self, time, x):
        GenericModel.postProcess(self, time, x)
        composition = np.asarray(x[0], dtype=np.float64)
        composition[:, 0] = np.clip(
            composition[:, 0], self.constraints.minComposition, 1 - self.constraints.minComposition
        )
        interface_position = self._clipInterfacePosition(float(x[1]))
        interface_position = self._correctInterfaceMass(composition[:, 0], interface_position)
        self.data.record(time, composition)
        self.interfaceData.record(time, interface_position)
        self.updateCoupledModels()
        return [composition, interface_position], False

    def postSolve(self):
        self.data.finalize()
        self.interfaceData.finalize()

    def getInterfacePosition(self, time=None):
        return self.interfaceData.y(time)

    def getTotalMass(self, time=None):
        composition = np.asarray(self.data.y(time), dtype=np.float64).reshape(-1)
        interface_position = self.getInterfacePosition(time)
        return integrate_binary_profile(self.mesh, composition, interface_position)
