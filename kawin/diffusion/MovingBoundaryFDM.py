import warnings

import numpy as np

from kawin.GenericModel import GenericModel
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.diffusion.mesh import CartesianFD1D, MixedBoundary1D, PeriodicBoundary1D
from kawin.diffusion.mesh.MeshBase import DiffusionPair, arithmeticMean
from kawin.diffusion.mesh.MovingBoundaryFD1D import (
    augment_profile_with_interface_compositions,
    debug_moving_boundary_fd_state,
    get_moving_boundary_fd_geometry,
    integrate_binary_fd_profile,
    quad_fit_derivs,
    interpolate_previous_ignored_composition,
    summarize_moving_boundary_fd_state,
)
from kawin.solver import explicitEulerIterator
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


class MovingBoundaryFD1DModel(DiffusionModel):
    """
    Binary 1D moving-boundary diffusion model on a node-centered FDM mesh.

    The composition field evolves on a fixed ``CartesianFD1D`` grid while the
    planar interface position is tracked as a separate scalar state. Interface
    motion uses an explicit Lee/Oh-style interpolation treatment with either a
    basic Stefan update or the corrected ``lee_oh_corrected`` update. Bulk
    nodes away from the interface can use either the legacy ``D c_xx`` update
    or a flux-form finite-difference update that matches ``CartesianFD1D``.
    In ``flux_form`` mode, the near-interface nodes also use conservative
    cut-cell balances. In ``legacy`` mode, they retain the historical
    quadratic ``D_i * c_xx`` update.

    Parameters
    ----------
    bulkUpdateScheme : {"legacy", "flux_form"}
        Bulk diffusion update used for non-interface nodes. ``"legacy"``
        reproduces the historical ``D_i * c_xx`` treatment, while
        ``"flux_form"`` uses the node-centered flux-form discretization from
        the fixed-grid FDM implementation. This argument is required so that
        comparisons between the two schemes are explicit.
    fluxGradientMode : {"pre_diffusion", "post_diffusion"}, optional
        Selects which interface gradients are used when computing the
        interfacial fluxes for the Stefan update. ``"pre_diffusion"`` uses
        gradients reconstructed from the profile before the bulk diffusion
        update, while ``"post_diffusion"`` uses gradients from the profile
        after that explicit diffusion stage. The default is
        ``"post_diffusion"``.
    """

    def __init__(
        self,
        mesh,
        elements,
        phases,
        thermodynamics,
        temperature,
        interfacePosition,
        bulkUpdateScheme: str,
        constraints=None,
        record=False,
        interfaceUpdate: str = "basic",
        pstar: float = 0.5,
        integrationMode: str = "weighted",
        fluxGradientMode: str = "post_diffusion",
    ):
        self.initialInterfacePosition = float(interfacePosition)
        self.interfaceData = _ScalarHistory(record)
        self.interfaceUpdate = str(interfaceUpdate)
        self.pstar = float(pstar)
        self.integrationMode = str(integrationMode)
        self.fluxGradientMode = str(fluxGradientMode)
        self.bulkUpdateScheme = str(bulkUpdateScheme)
        self._initialInventory = None
        self._currdt = np.inf
        self._lastFluxes = None
        self._lastInterfaceFluxes = (0.0, 0.0)
        self._lastInterfaceVelocity = 0.0
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
        '''
        Validates mesh, thermodynamic, and algorithm assumptions for the model
        '''
        if not isinstance(self.mesh, CartesianFD1D):
            raise TypeError("MovingBoundaryFD1DModel requires a CartesianFD1D mesh.")
        if len(self.allElements) != 2 or self.mesh.numResponses != 1:
            raise ValueError("MovingBoundaryFD1DModel currently supports only binary systems.")
        if any(e in interstitials for e in self.allElements):
            raise ValueError("MovingBoundaryFD1DModel currently supports only substitutional binary systems.")
        if len(self.phases) != 2:
            raise ValueError("MovingBoundaryFD1DModel requires exactly two explicit phases.")
        if not hasattr(self.therm, "getInterfacialComposition") or not hasattr(self.therm, "getInterdiffusivity"):
            raise TypeError("Thermodynamics object must implement interface composition and interdiffusivity methods.")
        if isinstance(getattr(self.mesh, "boundaryConditions", None), PeriodicBoundary1D):
            raise ValueError("Periodic boundary conditions are not supported for MovingBoundaryFD1DModel.")
        if self.interfaceUpdate not in {"basic", "lee_oh_corrected"}:
            raise ValueError("interfaceUpdate must be one of ['basic', 'lee_oh_corrected'].")
        if self.integrationMode not in {"ignore", "noIgnore", "weighted"}:
            raise ValueError("integrationMode must be one of ['ignore', 'noIgnore', 'weighted'].")
        if self.fluxGradientMode not in {"pre_diffusion", "post_diffusion"}:
            raise ValueError("fluxGradientMode must be one of ['pre_diffusion', 'post_diffusion'].")
        if self.bulkUpdateScheme not in {"legacy", "flux_form"}:
            raise ValueError("bulkUpdateScheme must be one of ['legacy', 'flux_form'].")
        if not (0 < self.pstar < 1):
            raise ValueError("pstar must lie strictly between 0 and 1.")
        self._clipInterfacePosition(self.initialInterfacePosition, strict=True)

    def reset(self):
        super().reset()
        self.interfaceData.reset()
        self.interfaceData.record(0, self.initialInterfacePosition)
        self._lastFluxes = None
        self._lastInterfaceFluxes = (0.0, 0.0)
        self._lastInterfaceVelocity = 0.0
        self._currdt = np.inf
        if hasattr(self, "mesh") and self.mesh is not None:
            self._initialInventory = self.getTotalMass()

    def toDict(self):
        data = super().toDict()
        data.update(
            {
                "interface_position": self.interfaceData._y,
                "interface_time": self.interfaceData._time,
                "interface_interval": self.interfaceData.recordInterval,
                "interface_index": self.interfaceData.N,
                "flux_gradient_mode": self.fluxGradientMode,
                "bulk_update_scheme": self.bulkUpdateScheme,
            }
        )
        return data

    def fromDict(self, data):
        super().fromDict(data)
        self.fluxGradientMode = str(data.get("flux_gradient_mode", "post_diffusion"))
        self.bulkUpdateScheme = str(data["bulk_update_scheme"])
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
        Clips the interface to the open domain and nudges it off exact node locations
        '''
        return interface_position ##NOTE: Bypassing this for now as it is unlikely that the interface will need to be nudged and this function is slow (mostly due to calling np.isclose() on entire array)
        z = np.ravel(self.mesh.z)
        eps = max(float(self.mesh.dz) * 1e-8, 1e-14)
        lower = float(z[0] + eps)
        upper = float(z[-1] - eps)
        if strict and not (lower < interface_position < upper):
            raise ValueError("Interface position must lie strictly inside the FDM node domain.")
        clipped = float(np.clip(interface_position, lower, upper))
        if np.any(np.isclose(z, clipped, atol=eps, rtol=0.0)):
            clipped = float(np.clip(clipped + eps, lower, upper))
        return clipped

    def _getInterfaceState(self, t, composition, interface_position):
        '''
        Returns geometry, interface compositions, and interfacial diffusivities
        '''
        geometry = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar)
        T_interface = float(self.temperatureParameters(np.array([[interface_position]]), t)[0])
        c_left_int, c_right_int = self.therm.getInterfacialComposition(T_interface, 0, precPhase=self.phases[1])
        c_left_int = float(np.clip(np.squeeze(c_left_int), self.constraints.minComposition, 1 - self.constraints.minComposition))
        c_right_int = float(np.clip(np.squeeze(c_right_int), self.constraints.minComposition, 1 - self.constraints.minComposition))
        D_left_int = float(np.squeeze(self.therm.getInterdiffusivity(c_left_int, T_interface, phase=self.phases[0])))
        D_right_int = float(np.squeeze(self.therm.getInterdiffusivity(c_right_int, T_interface, phase=self.phases[1])))
        return geometry, T_interface, c_left_int, c_right_int, D_left_int, D_right_int

    def _bulk_diffusivity_nodes(self, composition, t, geometry):
        '''
        Evaluates phase-appropriate node diffusivities on each side of the interface
        '''
        comp = np.asarray(composition, dtype=np.float64).reshape(-1)
        temperatures = self.temperatureParameters(self.mesh.z, t)
        D = np.zeros_like(comp, dtype=np.float64)
        left_slice = slice(0, geometry.right_index)
        right_slice = slice(geometry.right_index, len(comp))
        if geometry.right_index > 0:
            D[left_slice] = np.asarray(
                self.therm.getInterdiffusivity(comp[left_slice], temperatures[left_slice], phase=self.phases[0]),
                dtype=np.float64,
            ).reshape(-1)
        if geometry.right_index < len(comp):
            D[right_slice] = np.asarray(
                self.therm.getInterdiffusivity(comp[right_slice], temperatures[right_slice], phase=self.phases[1]),
                dtype=np.float64,
            ).reshape(-1)
        return D

    def _neumann_laplacian_uniform(self, c, i):
        '''
        Computes a centered second derivative with zero-flux end treatment
        '''
        if i == 0:
            return 2.0 * (c[1] - c[0]) / (self.mesh.dz**2)
        if i == len(c) - 1:
            return 2.0 * (c[-2] - c[-1]) / (self.mesh.dz**2)
        return (c[i + 1] - 2.0 * c[i] + c[i - 1]) / (self.mesh.dz**2)

    def _bulk_dcdt_legacy(self, composition, diffusivity_nodes):
        '''
        Returns the legacy bulk-node rate using ``D_i * c_xx`` on the FDM grid.
        '''
        c = np.asarray(composition, dtype=np.float64).reshape(-1)
        d = np.asarray(diffusivity_nodes, dtype=np.float64).reshape(-1)
        laplacian = np.empty_like(c)
        laplacian[0] = 2.0 * (c[1] - c[0]) / (self.mesh.dz**2)
        laplacian[-1] = 2.0 * (c[-2] - c[-1]) / (self.mesh.dz**2)
        laplacian[1:-1] = (c[2:] - 2.0 * c[1:-1] + c[:-2]) / (self.mesh.dz**2)
        return d * laplacian

    def _bulk_dcdt_flux_form(self, composition, diffusivity_nodes):
        '''
        Returns the flux-form bulk-node rate consistent with ``CartesianFD1D``.
        '''
        c = np.asarray(composition, dtype=np.float64).reshape(-1)
        d = np.asarray(diffusivity_nodes, dtype=np.float64).reshape(-1)
        pairs = [
            DiffusionPair(
                diffusivity=d[:, np.newaxis],
                response=c[:, np.newaxis],
                averageFunction=arithmeticMean,
            )
        ]
        return self.mesh.computedXdt(pairs)[:, 0]

    def _bulk_dcdt(self, composition, diffusivity_nodes):
        '''
        Returns the selected bulk diffusion operator for non-interface nodes.
        '''
        if self.bulkUpdateScheme == "legacy":
            return self._bulk_dcdt_legacy(composition, diffusivity_nodes)
        if self.bulkUpdateScheme == "flux_form":
            return self._bulk_dcdt_flux_form(composition, diffusivity_nodes)
        raise ValueError("bulkUpdateScheme must be one of ['legacy', 'flux_form'].")

    def _update_near_interface_node(self, c_old, c_new, idx, s, side, interface_compositions, diffusivity):
        '''
        Updates an interface-adjacent node using the legacy quadratic stencil.
        '''
        z = np.ravel(self.mesh.z)
        if side == "A":
            xq = np.array([z[idx - 1], z[idx], s], dtype=np.float64)
            yq = np.array([c_old[idx - 1], c_old[idx], interface_compositions[0]], dtype=np.float64)
        else:
            xq = np.array([s, z[idx], z[idx + 1]], dtype=np.float64)
            yq = np.array([interface_compositions[1], c_old[idx], c_old[idx + 1]], dtype=np.float64)
        _, d2 = quad_fit_derivs(xq, yq, z[idx])
        c_new[idx] = c_old[idx] + self._currdt * diffusivity * d2

    def _near_interface_dcdt_flux_form(
        self,
        composition,
        diffusivity_nodes,
        geometry,
        side,
        interface_compositions,
        interface_diffusivities,
    ):
        '''
        Returns a conservative cut-cell update for one near-interface node.

        The node-centered FDM is interpreted on the dual mesh. Near the moving
        interface, the dual cell is truncated on one side, so the update is the
        divergence of one bulk face flux and one cut-face flux divided by the
        asymmetric dual-cell width.
        '''
        c = np.asarray(composition, dtype=np.float64).reshape(-1)
        d = np.asarray(diffusivity_nodes, dtype=np.float64).reshape(-1)
        z = np.ravel(self.mesh.z)
        h = float(self.mesh.dz)
        avg = lambda a, b: float(np.asarray(arithmeticMean([a, b]), dtype=np.float64))

        if side == "A":
            idx = int(geometry.left_near_index)
            if idx < 1:
                raise ValueError("Left near-interface flux-form update requires a left neighbor.")
            dA = float(geometry.interface_position - z[idx])
            if dA <= 0:
                raise ValueError("Left near-interface distance must be positive.")
            D_AL = avg(d[idx - 1], d[idx])
            D_AR = avg(d[idx], interface_diffusivities[0])
            J_AL = -D_AL * (c[idx] - c[idx - 1]) / h
            J_AR = -D_AR * (interface_compositions[0] - c[idx]) / dA
            deltaA = 0.5 * (h + dA)
            return idx, float(-(J_AR - J_AL) / deltaA)

        if side == "B":
            idx = int(geometry.right_near_index)
            if idx + 1 >= len(c):
                raise ValueError("Right near-interface flux-form update requires a right neighbor.")
            dB = float(z[idx] - geometry.interface_position)
            if dB <= 0:
                raise ValueError("Right near-interface distance must be positive.")
            D_BL = avg(interface_diffusivities[1], d[idx])
            D_BR = avg(d[idx], d[idx + 1])
            J_BL = -D_BL * (c[idx] - interface_compositions[1]) / dB
            J_BR = -D_BR * (c[idx + 1] - c[idx]) / h
            deltaB = 0.5 * (dB + h)
            return idx, float(-(J_BR - J_BL) / deltaB)

        raise ValueError("side must be 'A' or 'B'.")

    def _interface_gradients(self, composition, interface_position, interface_compositions):
        '''
        Computes one-sided interface gradients from three-point quadratic fits
        '''
        z = np.ravel(self.mesh.z)
        geom = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar)
        if geom.p < self.pstar:
            i1, i2 = geom.left_index - 2, geom.left_index - 1
            j1, j2 = geom.right_index, geom.right_index + 1
        else:
            i1, i2 = geom.left_index - 1, geom.left_index
            j1, j2 = geom.right_index + 1, geom.right_index + 2
        i1 = max(0, i1)
        i2 = max(0, i2)
        j1 = min(len(z) - 1, j1)
        j2 = min(len(z) - 1, j2)
        
        if geom.ignored_index in [i1, i2, j1, j2]:
            raise ValueError("Interface gradient evaluation stencils should not include the ignored node.")
        
        x_left = np.array([z[i1], z[i2], interface_position], dtype=np.float64)
        y_left = np.array([composition[i1], composition[i2], interface_compositions[0]], dtype=np.float64)
        x_right = np.array([interface_position, z[j1], z[j2]], dtype=np.float64)
        y_right = np.array([interface_compositions[1], composition[j1], composition[j2]], dtype=np.float64)
        grad_left, _ = quad_fit_derivs(x_left, y_left, interface_position)
        grad_right, _ = quad_fit_derivs(x_right, y_right, interface_position)
        return float(grad_left), float(grad_right)

    def  _max_interface_step_fraction(self, geom, velocity):
        '''
        Returns the largest allowed interface move, in units of ``dz``, for the current regime
        ''' ##XXX: I think this might be incorrect as it may never allow the step to cross the node
        if velocity >= 0:
            node_limit = 1.0 - geom.p
            regime_limit = (self.pstar - geom.p) if geom.p < self.pstar else node_limit
        else:
            node_limit = geom.p
            regime_limit = (geom.p - self.pstar) if geom.p > self.pstar else node_limit
        limit = min(node_limit, regime_limit)
        return max(0.0, float(limit))

    def _compute_fluxes(self, composition, diffusivity_nodes, interface_position, interface_compositions, interface_diffusivities):
        '''
        Computes bulk face fluxes and replaces the interface face with one-sided interface fluxes
        '''
        comp = np.asarray(composition, dtype=np.float64).reshape(-1)
        pairs = [DiffusionPair(diffusivity=diffusivity_nodes[:, np.newaxis], response=comp[:, np.newaxis], averageFunction=arithmeticMean)]
        fluxes = self.mesh.computeFluxes(pairs)[:, 0]
        geom = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar)
        left_flux = -interface_diffusivities[0] * (interface_compositions[0] - comp[geom.left_index]) / geom.left_distance
        right_flux = -interface_diffusivities[1] * (comp[geom.right_index] - interface_compositions[1]) / geom.right_distance
        fluxes[geom.right_index] = 0.5 * (left_flux + right_flux)
        return fluxes, left_flux, right_flux

    def _computeState(self, t, xCurr):
        '''
        Computes composition rates and interface velocity for one explicit FDM step

        This method contains the shared stepping core for both interface update
        modes. It performs the selected bulk-node diffusion update away from
        the interface, reconstructs the ignored node, evaluates one-sided
        interface gradients, and then applies either the basic or
        Lee/Oh-corrected interface motion update. In ``flux_form`` mode, the
        near-interface nodes use conservative cut-cell balances. In
        ``legacy`` mode, they retain the quadratic ``D_i * c_xx`` stencil.
        '''
        c_old = np.asarray(xCurr[0], dtype=np.float64).reshape(-1)
        s_old = self._clipInterfacePosition(float(xCurr[1]))
        geom, _, c_left_int, c_right_int, D_left_int, D_right_int = self._getInterfaceState(t, c_old, s_old)
        diffusivity_nodes = self._bulk_diffusivity_nodes(c_old, t, geom)
        max_diff = float(np.max(np.abs(np.concatenate((diffusivity_nodes, [D_left_int, D_right_int])))))
        min_length = self.mesh.dz * ((1.0 - geom.p) if geom.p < self.pstar else geom.p)
        dt_diff = self.constraints.vonNeumannThreshold * (min_length**2) / max_diff if max_diff > 0 else np.inf

        initial_stage = np.asarray(
            interpolate_previous_ignored_composition(
                self.mesh.z,
                c_old,
                s_old,
                geom.p,
                s_old,
                self.pstar,
                (c_left_int, c_right_int),
            ),
            dtype=np.float64,
        )
        grad_left_pred, grad_right_pred = self._interface_gradients(initial_stage, s_old, (c_left_int, c_right_int))
        denom_basic = c_right_int - c_left_int
        s_dot_pred = (D_left_int * grad_left_pred - D_right_int * grad_right_pred) / denom_basic if abs(denom_basic) > 1e-14 else 0.0
        move_fraction = min(
            self.constraints.movingBoundaryThreshold, np.inf
            # 0.95 * self._max_interface_step_fraction(geom, s_dot_pred),
        )
        dt_move = move_fraction * self.mesh.dz / abs(s_dot_pred) if abs(s_dot_pred) > 0 else np.inf
        allowed_dt = getattr(self, "deltaTime", np.inf)
        self._currdt = min(dt_diff, dt_move, allowed_dt)
        if dt_diff>dt_move:
            raise ValueError("Not Expecting dt_move to control the time step at this point")

        ignored = geom.ignored_index
        if geom.p < self.pstar:
            a_last = geom.left_index - 1
            b_first = geom.right_index
        else:
            a_last = geom.left_index
            b_first = geom.right_index + 1

        bulk_dcdt = self._bulk_dcdt(c_old, diffusivity_nodes)
        bulk_mask = np.zeros(len(c_old), dtype=bool)
        if a_last >= 0:
            bulk_mask[: a_last + 1] = True
        if b_first < len(c_old):
            bulk_mask[b_first:] = True
        bulk_mask[ignored] = False

        left_near_active = geom.left_near_index >= 1 and geom.left_near_index <= a_last
        right_near_active = geom.right_near_index + 1 < len(c_old) and geom.right_near_index >= b_first
        if left_near_active:
            bulk_mask[geom.left_near_index] = False
        if right_near_active:
            bulk_mask[geom.right_near_index] = False

        c_new = c_old.copy()
        c_new[bulk_mask] = c_old[bulk_mask] + self._currdt * bulk_dcdt[bulk_mask]
        if left_near_active:
            if self.bulkUpdateScheme == "legacy":
                self._update_near_interface_node(
                    c_old,
                    c_new,
                    geom.left_near_index,
                    s_old,
                    "A",
                    (c_left_int, c_right_int),
                    diffusivity_nodes[geom.left_near_index],
                )
            else:
                idx, dcdt_left = self._near_interface_dcdt_flux_form(
                    c_old,
                    diffusivity_nodes,
                    geom,
                    "A",
                    (c_left_int, c_right_int),
                    (D_left_int, D_right_int),
                )
                c_new[idx] = c_old[idx] + self._currdt * dcdt_left
        if right_near_active:
            if self.bulkUpdateScheme == "legacy":
                self._update_near_interface_node(
                    c_old,
                    c_new,
                    geom.right_near_index,
                    s_old,
                    "B",
                    (c_left_int, c_right_int),
                    diffusivity_nodes[geom.right_near_index],
                )
            else:
                idx, dcdt_right = self._near_interface_dcdt_flux_form(
                    c_old,
                    diffusivity_nodes,
                    geom,
                    "B",
                    (c_left_int, c_right_int),
                    (D_left_int, D_right_int),
                )
                c_new[idx] = c_old[idx] + self._currdt * dcdt_right

        c_stage = np.asarray(
            interpolate_previous_ignored_composition(
                self.mesh.z,
                c_new,
                s_old,
                geom.p,
                s_old,
                self.pstar,
                (c_left_int, c_right_int),
            ),
            dtype=np.float64,
        )
        grad_left, grad_right = self._interface_gradients(c_stage, s_old, (c_left_int, c_right_int))
        if self.fluxGradientMode == "pre_diffusion":
            flux_left = -D_left_int * grad_left_pred
            flux_right = -D_right_int * grad_right_pred
        else:
            flux_left = -D_left_int * grad_left
            flux_right = -D_right_int * grad_right

        if self.interfaceUpdate == "basic":
            velocity = (flux_right - flux_left) / denom_basic if abs(denom_basic) > 1e-14 else 0.0
            ds = self._currdt * velocity

            current_mass = integrate_binary_fd_profile(self.mesh.z, c_stage, s_old=s_old, p_old=geom.p, 
                s_new=s_old+ds,
                pstar=self.pstar, interface_compositions=(c_left_int, c_right_int), integration_mode=self.integrationMode, s_for_interp="new")
        else:
            denom = c_right_int + c_stage[geom.right_index] - c_left_int - c_stage[geom.left_index]
            ds_intermediate = self._currdt * (2.0 / denom) * (flux_right - flux_left)
            s_intermediate = s_old + ds_intermediate
            current_mass = integrate_binary_fd_profile(
                self.mesh.z,
                c_stage,
                s_old=s_old,
                p_old=geom.p,
                s_new=s_intermediate,
                pstar=self.pstar,
                interface_compositions=(c_left_int, c_right_int),
                integration_mode=self.integrationMode,
                s_for_interp="new",
            )
            delta_mass = current_mass - self._initialInventory
            ds = ds_intermediate + (2.0 * delta_mass) / denom
            velocity = ds / self._currdt if self._currdt > 0 else 0.0

        max_fraction = self.constraints.movingBoundaryThreshold #self._max_interface_step_fraction(geom, velocity)
        requested_fraction = abs(ds) / self.mesh.dz
        if not np.isfinite(requested_fraction):
            raise ValueError("MovingBoundaryFD1DModel produced a non-finite interface increment.")
        max_fraction = max(max_fraction, 1e-12)
        if requested_fraction > max_fraction:
            print(f"(max_fraction, requested_fraction): {(max_fraction, requested_fraction)}")
            raise ValueError("MovingBoundaryFD1DModel requested_fraction is greater than max_fraction") #ds = np.sign(ds) * 0.95 * max_fraction * self.mesh.dz
            velocity = ds / self._currdt if self._currdt > 0 else 0.0

        s_new = self._clipInterfacePosition(s_old + ds, strict=True)
        c_final = np.asarray(
            interpolate_previous_ignored_composition(
                self.mesh.z,
                c_new,
                s_old,
                geom.p,
                s_new,
                self.pstar,
                (c_left_int, c_right_int),
            ),
            dtype=np.float64,
        )

        dcdt = (c_final - c_old) / self._currdt
        fluxes, left_flux, right_flux = self._compute_fluxes(c_final, diffusivity_nodes, s_old, (c_left_int, c_right_int), (D_left_int, D_right_int))
        self._lastFluxes = fluxes
        self._lastInterfaceFluxes = (float(left_flux), float(right_flux))
        self._lastInterfaceVelocity = float(velocity)
        return dcdt[:, np.newaxis], float(velocity)

    def getdXdt(self, t, xCurr):
        '''
        Returns time derivatives for the composition field and interface position
        '''
        dcdt, velocity = self._computeState(t, xCurr)
        return [dcdt, velocity]

    def getFluxes(self, t, xCurr):
        '''
        Returns the last computed face fluxes in a plot-compatible ``(N+1, 1)`` shape
        '''
        self._computeState(t, xCurr)
        return self._lastFluxes[:, np.newaxis]

    def getDt(self, dXdt):
        '''
        Returns the explicit time step estimated from diffusion and interface-motion limits
        '''
        if np.isfinite(self._currdt) and self._currdt > 0:
            return self._currdt
        return getattr(self, "deltaTime", np.inf)

    def correctdXdt(self, dt, x, dXdt):
        '''
        Limits interface motion once the actual explicit time step is known
        '''
        if dt <= 0:
            return
        interface_position = self._clipInterfacePosition(float(x[1]), strict=True)
        velocity = float(dXdt[1])
        if not np.isfinite(velocity):
            dXdt[1] = 0.0
            return

        geom = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar)
        max_fraction = self.constraints.movingBoundaryThreshold #max(1e-12, 0.95 * self._max_interface_step_fraction(geom, velocity))
        max_ds = max_fraction * self.mesh.dz
        z = np.ravel(self.mesh.z)
        eps = max(float(self.mesh.dz) * 1e-8, 1e-14)
        if velocity >= 0:
            domain_ds = max(eps, float(z[-1] - eps - interface_position))
        else:
            domain_ds = max(eps, float(interface_position - (z[0] + eps)))
        allowed_ds = min(max_ds, domain_ds)
        requested_ds = abs(velocity) * dt
        if requested_ds > allowed_ds:
            dXdt[1] = np.sign(velocity) * allowed_ds / dt

    def _isClosedSystem(self):
        '''
        Returns whether the current boundary conditions correspond to a closed system
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

    def _checkMassCorrection(self, composition, interface_position):
        '''
        Checks the current mass residual against the configured moving-boundary tolerance
        '''
        if not self._isClosedSystem() or self._initialInventory is None:
            return
        tolerance = self.constraints.movingBoundaryMassTolerance
        if tolerance is None or not np.isfinite(tolerance):
            return
        geom = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar)
        residual = abs(
            self._initialInventory
            - integrate_binary_fd_profile(
                self.mesh.z,
                composition,
                s_old=interface_position,
                p_old=geom.p,
                s_new=interface_position,
                pstar=self.pstar,
                interface_compositions=self._getInterfaceState(self.currentTime, composition, interface_position)[2:4],
                integration_mode=self.integrationMode,
                s_for_interp="old",
            )
        )
        if residual <= tolerance:
            return
        action = str(self.constraints.movingBoundaryMassAction).lower()
        message = (
            f"MovingBoundaryFD1DModel mass correction residual {residual:.3e} exceeded "
            f"tolerance {tolerance:.3e} at t = {self.currentTime:.3e}."
        )
        if action == "ignore":
            return
        if action == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return
        if action == "raise":
            raise ValueError(message)
        raise ValueError("movingBoundaryMassAction must be one of ['ignore', 'warn', 'raise'].")

    def postProcess(self, time, x):
        '''
        Clips composition values, validates mass behavior, and records the new state
        '''
        GenericModel.postProcess(self, time, x)
        composition = np.asarray(x[0], dtype=np.float64)
        composition[:, 0] = np.clip(composition[:, 0], self.constraints.minComposition, 1 - self.constraints.minComposition)
        interface_position = self._clipInterfacePosition(float(x[1]))
        self._checkMassCorrection(composition[:, 0], interface_position)
        self.data.record(time, composition)
        self.interfaceData.record(time, interface_position)
        self.updateCoupledModels()
        return [composition, interface_position], False

    def solve(self, simTime, iterator=explicitEulerIterator, verbose=False, vIt=10, minDtFrac=1e-8, maxDtFrac=1):
        '''
        Solves the model with explicit Euler only
        '''
        if iterator is not explicitEulerIterator:
            raise ValueError("MovingBoundaryFD1DModel is explicit-only and must be solved with explicitEulerIterator.")
        return super().solve(simTime, iterator=iterator, verbose=verbose, vIt=vIt, minDtFrac=minDtFrac, maxDtFrac=maxDtFrac)

    def postSolve(self):
        self.data.finalize()
        self.interfaceData.finalize()

    def getInterfacePosition(self, time = None):
        '''
        Returns the interface position at a requested time
        '''
        return self.interfaceData.y(time)

    def getTotalMass(self, time = None):
        '''
        Returns the integrated binary composition inventory for the current profile
        '''
        composition = np.asarray(self.data.y(time), dtype=np.float64).reshape(-1)
        interface_position = self.getInterfacePosition(time)
        geom = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar)
        interface_compositions = self._getInterfaceState(self.currentTime if time is None else time, composition, interface_position)[2:4]
        return integrate_binary_fd_profile(
            self.mesh.z,
            composition,
            s_old=interface_position,
            p_old=geom.p,
            s_new=interface_position,
            pstar=self.pstar,
            interface_compositions=interface_compositions,
            integration_mode=self.integrationMode,
            s_for_interp="old",
        )

    def describeMeshState(self, time = None, window: int = 2, precision: int = 9, distance_multiplier: float = 1.0):
        '''
        Returns a compact text summary of the current FDM moving-boundary mesh state.

        This is mainly intended for debugger use.
        '''
        composition = np.asarray(self.data.y(time), dtype=np.float64).reshape(-1)
        interface_position = self.getInterfacePosition(time)
        summary = summarize_moving_boundary_fd_state(
            self.mesh,
            composition,
            interface_position,
            self.pstar,
            window=window,
            precision=precision,
            distance_multiplier=distance_multiplier,
        )
        return f"time = {self.currentTime:.6g}\n{summary}" if time is None else f"time = {time:.6g}\n{summary}"

    def plotMeshState(self, time = None, ax = None, **kwargs):
        '''
        Plots the current FDM moving-boundary mesh state.

        This is intended as a convenience wrapper so the state can be visualized
        directly from a debugger or notebook.
        '''
        from kawin.diffusion.Plot import plotMovingBoundaryState

        return plotMovingBoundaryState(self, time=time, ax=ax, **kwargs)

    def debugMeshState(self, time = None, ax = None, show: bool = True, **kwargs):
        '''
        Prints and plots the current FDM moving-boundary mesh state.

        This is intended as the most direct debugger convenience entry point.
        '''
        composition = np.asarray(self.data.y(time), dtype=np.float64).reshape(-1)
        interface_position = self.getInterfacePosition(time)
        summary = summarize_moving_boundary_fd_state(
            self.mesh,
            composition,
            interface_position,
            self.pstar,
            window=kwargs.get('window', 2),
            precision=kwargs.get('precision', 9),
            distance_multiplier=kwargs.get('distance_multiplier', 1.0),
        )
        print_summary = kwargs.pop('print_summary', True)
        if print_summary:
            time_label = self.currentTime if time is None else time
            print(f"time = {time_label:.6g}\n{summary}")

        from kawin.diffusion.Plot import plotMovingBoundaryState

        ax = plotMovingBoundaryState(self, time=time, ax=ax, **kwargs)
        if show:
            import matplotlib.pyplot as plt

            plt.show()
        return summary, ax
