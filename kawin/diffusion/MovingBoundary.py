import warnings

import numpy as np

from kawin.GenericModel import GenericModel
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.diffusion.mesh.FVM1D import Cartesian1D, MixedBoundary1D, PeriodicBoundary1D
from kawin.diffusion.mesh.MovingBoundary1D import (
    debug_moving_boundary_state,
    get_control_volume_widths,
    get_moving_boundary_geometry,
    integrate_binary_profile,
    summarize_moving_boundary_state,
)
from kawin.thermo.Mobility import interstitials


class _ScalarHistory:
    def __init__(self, record: bool | int = False):
        if isinstance(record, bool):
            if record:
                self.recordInterval = 1
            else:
                self.recordInterval = -1
        else:
            self.recordInterval = record

        self.batchSize = 1000
        self.reset()

    def reset(self):
        '''
        Resets arrays
        '''
        self._y = np.zeros(self.batchSize, dtype=np.float64)
        self._time = np.zeros(self.batchSize, dtype=np.float64)
        self.currentIndex = 0
        self.currentY = 0.0
        self.currentTime = 0.0
        self.N = 0

    def record(self, time, y, force: bool = False):
        '''
        Stores current state of time and scalar variable
        '''
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
        '''
        Removes extra padding
        '''
        self.record(self.currentTime, self.currentY, force=True)
        self._y = self._y[:self.N+1]
        self._time = self._time[:self.N+1]

    def y(self, time = None):
        '''
        Returns scalar variable at time

        If recording is disabled, then this will return the current state
        '''
        if time is None:
            return float(self._y[self.N])

        if self.recordInterval > 0:
            if time <= self._time[0]:
                return float(self._y[0])
            if time >= self._time[self.N]:
                return float(self._y[self.N])
            else:
                uind = np.argmax(self._time > time)
                lind = uind - 1

                uy, utime = self._y[uind], self._time[uind]
                ly, ltime = self._y[lind], self._time[lind]

                return float((uy - ly) * (time - ltime) / (utime - ltime) + ly)
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
        if self.constraints.movingBoundaryThreshold <= 0 or self.constraints.movingBoundaryThreshold >= 0.5:
            raise ValueError("movingBoundaryThreshold must be in the range (0, 0.5) for MovingBoundary1DModel.")
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
            self._initialInventory = integrate_binary_profile(self.mesh, np.asarray(self.data.currentY).reshape(-1), self.initialInterfacePosition) ##XXX: This seems suspect that it uses self.initialInterfacePosition

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
        '''
        Returns boundary conditions and creates a default zero-flux condition if needed
        '''
        bc = getattr(self.mesh, "boundaryConditions", None)
        if bc is None:
            bc = MixedBoundary1D(self.mesh.responses)
            self.mesh.boundaryConditions = bc
        return bc

    def _clipInterfacePosition(self, interface_position: float, strict: bool = True) -> float:
        '''
        Bounds interface position to the valid cell-center domain

        The moving-boundary geometry assumes the interface lies strictly between
        adjacent cell centers, so this prevents invalid cut-cell widths
        and one-sided interface distances.
        '''
        z = np.ravel(self.mesh.z)
        eps = max(float(self.mesh.dz) * 1e-8, 1e-14)
        lower = float(z[0] + eps)
        upper = float(z[-1] - eps)
        if strict and not (lower < interface_position < upper):
            raise ValueError("Interface position must lie strictly within the cell-center domain.")
        return float(np.clip(interface_position, lower, upper))

    def _getInterfaceState(self, t, composition, interface_position):
        '''
        Returns geometry, local-equilibrium interface compositions and interface diffusivities
        '''
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
        '''
        Computes bulk finite-volume face fluxes away from the moving interface

        The interface-adjacent one-sided fluxes are handled separately in
        _computeState using the local-equilibrium interface compositions.
        '''
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
                left_D_face, _ = self.mesh.midPointCalculator.getDMid(left_D, isPeriodic=False)
                face_fluxes[1 : geometry.left_index + 1] = self.mesh._diffusiveFlux(
                    left_D_face, left_comp[1:], left_comp[:-1], self.mesh.dzs[0]
                )

        if geometry.right_index < len(comp) - 1:
            right_comp = comp[geometry.right_index:]
            right_t = self.temperatureParameters(self.mesh.z[geometry.right_index:], t)
            right_D = np.asarray(
                self.therm.getInterdiffusivity(right_comp, right_t, phase=self.phases[1]), dtype=np.float64
            ).reshape(-1)
            if len(right_D) > 0:
                max_diffusivity = max(max_diffusivity, float(np.max(np.abs(right_D))))
            if len(right_D) > 1:
                right_D_face, _ = self.mesh.midPointCalculator.getDMid(right_D, isPeriodic=False)
                face_fluxes[geometry.right_index + 1 : len(comp)] = self.mesh._diffusiveFlux(
                    right_D_face, right_comp[1:], right_comp[:-1], self.mesh.dzs[0]
                )

        fluxes_2d = face_fluxes[:, np.newaxis]
        left_bc.adjustFluxes(fluxes_2d)
        return fluxes_2d[:, 0], max_diffusivity

    def _computeState(self, t, xCurr):
        '''
        Computes composition rates and interface velocity for the current state

        This evaluates one-sided interface fluxes, applies the Stefan balance
        for the moving boundary, updates the interface-adjacent cut cells using
        their actual control-volume widths, and stores a stable time step estimate.
        '''
        composition = np.asarray(xCurr[0], dtype=np.float64).reshape(-1)
        interface_position = self._clipInterfacePosition(float(xCurr[1]))
        geometry, _, c_left_int, c_right_int, D_left_int, D_right_int = self._getInterfaceState(
            t, composition, interface_position
        )
        # debug_moving_boundary_state(self.mesh, composition=composition, interface_position=interface_position, interface_compositions=self._getInterfaceState(t, composition, interface_position)[2:4], zoom_cells=2, distance_multiplier=1e6, annotate_positions=True, annotate_interface_distances=True, annotate_interface_compositions=True)
        face_fluxes, max_bulk_diffusivity = self._computeBulkFluxes(composition, t, geometry)
        J_int_left = -D_left_int * (c_left_int - composition[geometry.left_index]) / geometry.left_distance
        J_int_right = -D_right_int * (composition[geometry.right_index] - c_right_int) / geometry.right_distance
        velocity = (J_int_right - J_int_left) / (c_right_int - c_left_int) ##XXX This does not use the compositions at the interface-adjacent cells (Lee and Oh does) so this may need to be revisited

        dcdt = self.mesh._fluxTodXdt(face_fluxes[:,np.newaxis])[:,0]
        widths = get_control_volume_widths(self.mesh, interface_position)
        dcdt[geometry.left_index] = -(J_int_left - face_fluxes[geometry.left_index]) / widths[geometry.left_index] ##NOTE This breaks the conservation property of the update
        dcdt[geometry.right_index] = -(face_fluxes[geometry.right_index + 1] - J_int_right) / widths[geometry.right_index]

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
            dt_diff = self.constraints.vonNeumannThreshold * min_width**2 / D_scale ##NOTE This DOES take account for the interface-adjacent cells having very small widths because get_control_volume_widths() uses geom.left_volume and geom.right_volume
        else:
            dt_diff = np.inf
        if abs(velocity) > 0:
            dt_move = self.constraints.movingBoundaryThreshold * float(self.mesh.dz) / abs(velocity)
        else:
            dt_move = np.inf
        self._currdt = min(dt_diff, dt_move)
        self._lastFluxes = face_fluxes
        self._lastInterfaceFluxes = (float(J_int_left), float(J_int_right))
        self._lastInterfaceVelocity = float(velocity)
        return dcdt[:, np.newaxis], float(velocity)

    def getFluxes(self, t, xCurr):
        '''
        Returns total face fluxes for the current moving-boundary state
        '''
        self._computeState(t, xCurr)
        return self._lastFluxes[:, np.newaxis]

    def getdXdt(self, t, xCurr):
        '''
        Returns composition rates and interface-position rate
        '''
        dcdt, velocity = self._computeState(t, xCurr)
        return [dcdt, velocity]

    def getDt(self, dXdt):
        '''
        Returns the most recent diffusion/interface-motion time step limit
        '''
        if np.isfinite(self._currdt) and self._currdt > 0:
            return self._currdt
        return self.deltaTime

    def _isClosedSystem(self):
        '''
        Checks whether both external boundaries are zero-flux
        '''
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
        '''
        Applies a small interface-position correction to preserve total inventory

        The bulk update is conservative by construction, but this correction
        reduces drift from explicit stepping and interface-position clipping.
        '''
        if not self._isClosedSystem():
            raise ValueError("Mass correction is only valid for closed systems.")
            # return interface_position

        geometry = get_moving_boundary_geometry(self.mesh, interface_position)
        current_mass = integrate_binary_profile(self.mesh, composition, interface_position)
        delta_mass = self._initialInventory - current_mass
        denominator = composition[geometry.left_index] - composition[geometry.right_index] ##XXX This does not use the interface compositions (Lee and Oh does) so this may need to be revisited [Actually, it may have to be like this because of how the mass integral works it only considers the cell compositions and not the interface ones]
        if abs(denominator) < 1e-14:
            return interface_position
        corrected = interface_position + delta_mass / denominator
        corrected = float(np.clip(corrected, geometry.left_center + 1e-14, geometry.right_center - 1e-14)) 
        return corrected

    def _checkMassCorrection(self, composition, interface_position):
        '''
        Checks the residual inventory error after mass correction

        If a finite tolerance is supplied, this can ignore, warn or raise
        depending on the configured moving-boundary mass action.
        '''
        if not self._isClosedSystem() or self._initialInventory is None:
            return

        tolerance = self.constraints.movingBoundaryMassTolerance
        if tolerance is None or not np.isfinite(tolerance):
            return

        current_mass = integrate_binary_profile(self.mesh, composition, interface_position)
        residual = abs(self._initialInventory - current_mass)
        if residual <= tolerance:
            return

        action = str(self.constraints.movingBoundaryMassAction).lower()
        message = (
            f'MovingBoundary1DModel mass correction residual {residual:.3e} exceeded '
            f'tolerance {tolerance:.3e} at t = {self.currentTime:.3e}.'
        )
        if action == 'ignore':
            return
        if action == 'warn':
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return
        if action == 'raise':
            raise ValueError(message)
        raise ValueError(
            "movingBoundaryMassAction must be one of ['ignore', 'warn', 'raise']."
        )

    def postProcess(self, time, x):
        '''
        Clips the updated state, records composition and interface position, and updates coupled models
        '''
        GenericModel.postProcess(self, time, x)
        composition = np.asarray(x[0], dtype=np.float64)
        composition[:, 0] = np.clip(
            composition[:, 0], self.constraints.minComposition, 1 - self.constraints.minComposition
        )
        interface_position = self._clipInterfacePosition(float(x[1]))
        (composition==self.mesh.y).all()
        # debug_moving_boundary_state(self.mesh, composition=composition, interface_position=interface_position, interface_compositions=self._getInterfaceState(time, composition, interface_position)[2:4], zoom_cells=2, distance_multiplier=1e6, annotate_positions=True, annotate_interface_distances=True, annotate_interface_compositions=True)
        interface_position = self._correctInterfaceMass(composition[:, 0], interface_position)
        self._checkMassCorrection(composition[:, 0], interface_position)
        self.data.record(time, composition)
        self.interfaceData.record(time, interface_position)
        self.updateCoupledModels()
        return [composition, interface_position], False

    def postSolve(self):
        '''
        Finalizes recorded composition and interface-position histories
        '''
        self.data.finalize()
        self.interfaceData.finalize()

    def getInterfacePosition(self, time = None):
        '''
        Returns interface position at time
        '''
        return self.interfaceData.y(time)

    def getTotalMass(self, time = None):
        '''
        Returns total solute inventory using the current moving-boundary geometry
        '''
        composition = np.asarray(self.data.y(time), dtype=np.float64).reshape(-1)
        interface_position = self.getInterfacePosition(time)
        return integrate_binary_profile(self.mesh, composition, interface_position)

    def describeMeshState(self, time = None, window: int = 2, precision: int = 9, distance_multiplier: float = 1.0):
        '''
        Returns a compact text summary of the current moving-boundary mesh state.

        This is mainly intended for debugger use.
        '''
        composition = np.asarray(self.data.y(time), dtype=np.float64).reshape(-1)
        interface_position = self.getInterfacePosition(time)
        summary = summarize_moving_boundary_state(
            self.mesh,
            composition,
            interface_position,
            window=window,
            precision=precision,
            distance_multiplier=distance_multiplier,
        )
        return f"time = {self.currentTime:.6g}\n{summary}" if time is None else f"time = {time:.6g}\n{summary}"

    def plotMeshState(self, time = None, ax = None, **kwargs):
        '''
        Plots the current moving-boundary mesh state.

        This is intended as a convenience wrapper so the state can be visualized
        directly from a debugger or notebook.
        '''
        from kawin.diffusion.Plot import plotMovingBoundaryState

        return plotMovingBoundaryState(self, time=time, ax=ax, **kwargs)

    def debugMeshState(self, time = None, ax = None, show: bool = True, **kwargs):
        '''
        Prints and plots the current moving-boundary mesh state.

        This is intended as the most direct debugger convenience entry point.
        '''
        from kawin.diffusion.Plot import plotMovingBoundaryState

        composition = np.asarray(self.data.y(time), dtype=np.float64).reshape(-1)
        interface_position = self.getInterfacePosition(time)
        summary = summarize_moving_boundary_state(
            self.mesh,
            composition,
            interface_position,
            window=kwargs.pop('window', 2),
            precision=kwargs.pop('precision', 9),
            distance_multiplier=kwargs.get('distance_multiplier', 1.0),
        )
        print_summary = kwargs.pop('print_summary', True)
        if print_summary:
            time_label = self.currentTime if time is None else time
            print(f"time = {time_label:.6g}\n{summary}")

        ax = plotMovingBoundaryState(self, time=time, ax=ax, **kwargs)
        if show:
            import matplotlib.pyplot as plt

            plt.show()
        return summary, ax
