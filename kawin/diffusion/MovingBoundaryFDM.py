import warnings

import numpy as np
from scipy import optimize

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

def debugInPlace():
    try:
        import debugpy
        # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        print('break on this line')
    except:
        pass

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
    basic Stefan update, the corrected ``lee_oh_corrected`` update, or the
    binary-only ``my_corrected`` update that solves for the interface position
    producing zero net mass change after the explicit diffusion stage. Bulk
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
    interfaceUpdate : {"basic", "lee_oh_corrected", "my_corrected"}
        Interface motion update used after the explicit diffusion stage.
        This argument must be specified explicitly so the chosen mass-balance
        strategy is always visible at the call site.
    integrationMode : {"ignore", "noIgnore", "weighted"}
        Interface-aware inventory integration rule used by mass accounting and
        corrected interface updates. This argument must be specified
        explicitly.
    fluxGradientMode : {"pre_diffusion", "post_diffusion"}
        Selects which interface gradients are used when computing the
        interfacial fluxes for the Stefan update. ``"pre_diffusion"`` uses
        gradients reconstructed from the profile before the bulk diffusion
        update, while ``"post_diffusion"`` uses gradients from the profile
        after that explicit diffusion stage. This argument must be specified
        explicitly.
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
        interfaceUpdate: str | None = None,
        pstar: float = 0.5,
        integrationMode: str | None = None,
        fluxGradientMode: str | None = None,
        balanceElement: str | None = None,
    ):
        self.initialInterfacePosition = float(interfacePosition)
        self.interfaceData = _ScalarHistory(record)
        self.interfaceUpdate = None if interfaceUpdate is None else str(interfaceUpdate)
        self.pstar = float(pstar)
        self.integrationMode = None if integrationMode is None else str(integrationMode)
        self.fluxGradientMode = None if fluxGradientMode is None else str(fluxGradientMode)
        self.bulkUpdateScheme = str(bulkUpdateScheme)
        self.balanceElement = None if balanceElement is None else str(balanceElement)
        self._balanceElementIndex = None
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
        self._initialInventory = self._getStoredInventory()

    def _validateMovingBoundaryModel(self):
        '''
        Validates mesh, thermodynamic, and algorithm assumptions for the model
        '''
        if not isinstance(self.mesh, CartesianFD1D):
            raise TypeError("MovingBoundaryFD1DModel requires a CartesianFD1D mesh.")
        if any(e in interstitials for e in self.allElements):
            raise ValueError("MovingBoundaryFD1DModel currently supports only substitutional systems.")
        if len(self.phases) != 2:
            raise ValueError("MovingBoundaryFD1DModel requires exactly two explicit phases.")
        if not hasattr(self.therm, "getInterfacialComposition") or not hasattr(self.therm, "getInterdiffusivity"):
            raise TypeError("Thermodynamics object must implement interface composition and interdiffusivity methods.")
        if isinstance(getattr(self.mesh, "boundaryConditions", None), PeriodicBoundary1D):
            raise ValueError("Periodic boundary conditions are not supported for MovingBoundaryFD1DModel.")
        if self.interfaceUpdate is None:
            raise ValueError("interfaceUpdate must be specified explicitly.")
        if self.interfaceUpdate not in {"basic", "lee_oh_corrected", "my_corrected"}:
            raise ValueError("interfaceUpdate must be one of ['basic', 'lee_oh_corrected', 'my_corrected'].")
        if self.integrationMode is None:
            raise ValueError("integrationMode must be specified explicitly.")
        if self.integrationMode not in {"ignore", "noIgnore", "weighted"}:
            raise ValueError("integrationMode must be one of ['ignore', 'noIgnore', 'weighted'].")
        if self.fluxGradientMode is None:
            raise ValueError("fluxGradientMode must be specified explicitly.")
        if self.fluxGradientMode not in {"pre_diffusion", "post_diffusion"}:
            raise ValueError("fluxGradientMode must be one of ['pre_diffusion', 'post_diffusion'].")
        if self.bulkUpdateScheme not in {"legacy", "flux_form"}:
            raise ValueError("bulkUpdateScheme must be one of ['legacy', 'flux_form'].")
        if not (0 < self.pstar < 1):
            raise ValueError("pstar must lie strictly between 0 and 1.")
        if self.constraints.movingBoundaryThreshold>=min(self.pstar, 1-self.pstar):
            raise ValueError("movingBoundaryThreshold must be less than the minimum of pstar and 1-pstar.")
        if self._isBinarySystem():
            if self.mesh.numResponses != 1:
                raise ValueError("MovingBoundaryFD1DModel requires one response variable for binary systems.")
            self._balanceElementIndex = 0
        elif self._isTernarySystem():
            if self.mesh.numResponses != 2:
                raise ValueError("MovingBoundaryFD1DModel requires two response variables for ternary systems.")
            if self.bulkUpdateScheme != "flux_form":
                raise ValueError("Ternary MovingBoundaryFD1DModel currently requires bulkUpdateScheme='flux_form'.")
            if self.balanceElement is not None and self.balanceElement not in self.elements:
                raise ValueError(f"balanceElement must be one of {self.elements}.")
            if self.interfaceUpdate == "my_corrected":
                raise ValueError("Ternary MovingBoundaryFD1DModel does not support interfaceUpdate='my_corrected'.")
            if self.interfaceUpdate == "lee_oh_corrected":
                if self.balanceElement is None:
                    raise ValueError("Ternary MovingBoundaryFD1DModel requires balanceElement for interfaceUpdate='lee_oh_corrected'.")
                self._balanceElementIndex = self.elements.index(self.balanceElement)
            else:
                self._balanceElementIndex = None if self.balanceElement is None else self.elements.index(self.balanceElement)
        else:
            raise ValueError("MovingBoundaryFD1DModel currently supports only binary or ternary systems.")
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
            self._validateMovingBoundaryModel()
            self._initialInventory = self._getStoredInventory()

    def toDict(self):
        data = super().toDict()
        data.update(
            {
                "interface_position": self.interfaceData._y,
                "interface_time": self.interfaceData._time,
                "interface_interval": self.interfaceData.recordInterval,
                "interface_index": self.interfaceData.N,
                "interface_update": self.interfaceUpdate,
                "flux_gradient_mode": self.fluxGradientMode,
                "bulk_update_scheme": self.bulkUpdateScheme,
                "balance_element": "" if self.balanceElement is None else self.balanceElement,
            }
        )
        return data

    def fromDict(self, data):
        super().fromDict(data)
        interface_update = data.get("interface_update", self.interfaceUpdate)
        if isinstance(interface_update, np.ndarray):
            interface_update = interface_update.item()
        self.interfaceUpdate = str(interface_update)
        self.fluxGradientMode = str(data.get("flux_gradient_mode", "post_diffusion"))
        self.bulkUpdateScheme = str(data["bulk_update_scheme"])
        balance_element = data.get("balance_element", "")
        if isinstance(balance_element, np.ndarray):
            balance_element = balance_element.item()
        self.balanceElement = None if balance_element in {"", None} else str(balance_element)
        self.interfaceData.recordInterval = int(data["interface_interval"])
        self.interfaceData.N = int(data["interface_index"])
        self.interfaceData._y = np.array(data["interface_position"], dtype=np.float64)
        self.interfaceData._time = np.array(data["interface_time"], dtype=np.float64)
        self.interfaceData.currentY = float(self.interfaceData._y[-1])
        self.interfaceData.currentTime = float(self.interfaceData._time[-1])
        self.interfaceData.currentIndex = self.interfaceData.N
        self._validateMovingBoundaryModel()
        self._initialInventory = self._getStoredInventory(0)

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
        z = np.ravel(self.mesh.z)
        assert z[0] < interface_position < z[-1], "Interface position is outside the domain."
        return interface_position ##NOTE: Bypassing this for now as it is unlikely that the interface will need to be nudged and this function is slow (mostly due to calling np.isclose() on entire array)
        eps = max(float(self.mesh.dz) * 1e-8, 1e-14)
        lower = float(z[0] + eps)
        upper = float(z[-1] - eps)
        if strict and not (lower < interface_position < upper):
            raise ValueError("Interface position must lie strictly inside the FDM node domain.")
        clipped = float(np.clip(interface_position, lower, upper))
        if np.any(np.isclose(z, clipped, atol=eps, rtol=0.0)):
            clipped = float(np.clip(clipped + eps, lower, upper))
        return clipped

    def _isBinarySystem(self) -> bool:
        return len(self.allElements) == 2

    def _isTernarySystem(self) -> bool:
        return len(self.allElements) == 3

    def _clipIndependentCompositionVector(self, composition):
        '''
        Clips an independent-composition vector while preserving a valid reference component.
        '''
        comp = np.asarray(composition, dtype=np.float64).reshape(-1).copy()
        comp = np.clip(comp, self.constraints.minComposition, 1 - self.constraints.minComposition)
        max_sum = 1.0 - self.constraints.minComposition
        total = float(np.sum(comp))
        if total > max_sum:
            comp *= max_sum / total
        return comp

    def _normalizeThermoIndependentComposition(self, composition):
        '''
        Converts thermo interface compositions to the solver's independent-component convention.
        '''
        comp = np.asarray(composition, dtype=np.float64).reshape(-1)
        if np.any(comp < 0):
            raise ValueError("Interface local equilibrium could not be determined for the requested ternary phase pair.")
        if comp.size == len(self.elements):
            return self._clipIndependentCompositionVector(comp)
        if comp.size == len(self.allElements):
            return self._clipIndependentCompositionVector(comp[1:])
        raise ValueError(
            "Multicomponent interface compositions must contain either the independent components "
            "or the full set of elements."
        )

    def _clipCompositionField(self, composition):
        '''
        Clips a node-wise composition field to the admissible substitutional simplex.
        '''
        comp = np.asarray(composition, dtype=np.float64).copy()
        if comp.ndim == 1:
            if self._isBinarySystem():
                return np.clip(comp, self.constraints.minComposition, 1 - self.constraints.minComposition)
            return self._clipIndependentCompositionVector(comp)
        if comp.shape[1] == 1:
            comp[:, 0] = np.clip(comp[:, 0], self.constraints.minComposition, 1 - self.constraints.minComposition)
            return comp
        comp = np.clip(comp, self.constraints.minComposition, 1 - self.constraints.minComposition)
        max_sum = 1.0 - self.constraints.minComposition
        totals = np.sum(comp, axis=1)
        mask = totals > max_sum
        if np.any(mask):
            comp[mask] *= (max_sum / totals[mask])[:, np.newaxis]
        return comp

    def _getStoredInventory(self, time = None):
        '''
        Returns the scalar or vector inventory appropriate for the current system.
        '''
        if self._isBinarySystem():
            return self.getTotalMass(time)
        return self.getTotalInventory(time)

    def _integrateComponentInventory(self, composition, interface_position, interface_compositions, component_index, s_for_interp, s_old=None, p_old=None, s_new=None):
        '''
        Integrates a single independent component using the binary sharp-interface helper.
        '''
        if s_old is None:
            s_old = interface_position
        if s_new is None:
            s_new = interface_position
        if p_old is None:
            p_old = get_moving_boundary_fd_geometry(self.mesh, s_old, self.pstar).p
        comp = np.asarray(composition, dtype=np.float64)
        if comp.ndim == 1:
            comp_field = comp.reshape(-1)
        else:
            comp_field = comp[:, component_index].reshape(-1)
        if np.ndim(interface_compositions[0]) == 0:
            interface_pair = (float(interface_compositions[0]), float(interface_compositions[1]))
        else:
            interface_pair = (
                float(np.asarray(interface_compositions[0], dtype=np.float64).reshape(-1)[component_index]),
                float(np.asarray(interface_compositions[1], dtype=np.float64).reshape(-1)[component_index]),
            )
        return integrate_binary_fd_profile(
            self.mesh.z,
            comp_field,
            s_old=s_old,
            p_old=p_old,
            s_new=s_new,
            pstar=self.pstar,
            interface_compositions=interface_pair,
            integration_mode=self.integrationMode,
            s_for_interp=s_for_interp,
        )

    def _integrateInventory(self, composition, interface_position, interface_compositions, s_for_interp):
        '''
        Integrates either the scalar binary inventory or the vector ternary inventories.
        '''
        comp = np.asarray(composition, dtype=np.float64)
        if comp.ndim == 1 or comp.shape[1] == 1:
            return self._integrateComponentInventory(comp, interface_position, interface_compositions, 0, s_for_interp=s_for_interp)
        return np.array(
            [
                self._integrateComponentInventory(comp, interface_position, interface_compositions, i, s_for_interp=s_for_interp)
                for i in range(comp.shape[1])
            ],
            dtype=np.float64,
        )

    def _reconstructIgnoredComposition(self, composition, s_old, p_old, s_new, interface_compositions):
        '''
        Reconstructs the ignored interface-adjacent node for scalar or vector compositions.
        '''
        comp = np.asarray(composition, dtype=np.float64)
        if comp.ndim == 1:
            return np.asarray(
                interpolate_previous_ignored_composition(
                    self.mesh.z,
                    comp,
                    s_old,
                    p_old,
                    s_new,
                    self.pstar,
                    interface_compositions,
                ),
                dtype=np.float64,
            )

        reconstructed = np.zeros_like(comp, dtype=np.float64)
        left_int = np.asarray(interface_compositions[0], dtype=np.float64).reshape(-1)
        right_int = np.asarray(interface_compositions[1], dtype=np.float64).reshape(-1)
        for i in range(comp.shape[1]):
            reconstructed[:, i] = np.asarray(
                interpolate_previous_ignored_composition(
                    self.mesh.z,
                    comp[:, i],
                    s_old,
                    p_old,
                    s_new,
                    self.pstar,
                    (float(left_int[i]), float(right_int[i])),
                ),
                dtype=np.float64,
            )
        return reconstructed

    def _composeInterfaceProbe(self, composition, geometry, lam: float):
        '''
        Builds the multicomponent local-equilibrium probe state from the bracketing node compositions.
        '''
        comp = np.asarray(composition, dtype=np.float64)
        left = np.asarray(comp[geometry.left_index], dtype=np.float64).reshape(-1)
        right = np.asarray(comp[geometry.right_index], dtype=np.float64).reshape(-1)
        return self._clipIndependentCompositionVector((1.0 - lam) * left + lam * right)

    def _multicomponentResidualTolerance(self, velocities):
        '''
        Returns an absolute tolerance for matching the per-component interface velocities.
        '''
        scale = max(1.0, float(np.max(np.abs(np.asarray(velocities, dtype=np.float64)))))
        return max(1e-10, 1e-6 * scale)

    def _evaluateMulticomponentInterfaceState(self, t, composition, interface_position, lam, geometry=None, temperature=None):
        '''
        Evaluates one candidate ternary interface state and the corresponding equal-velocity residual.
        '''
        if geometry is None:
            geometry = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar)
        if temperature is None:
            temperature = float(self.temperatureParameters(np.array([[interface_position]]), t)[0])

        x_probe = self._composeInterfaceProbe(composition, geometry, lam)
        c_left_int, c_right_int = self.therm.getInterfacialComposition(x_probe, temperature, 0, precPhase=self.phases[1])
        c_left_int = self._normalizeThermoIndependentComposition(c_left_int)
        c_right_int = self._normalizeThermoIndependentComposition(c_right_int)

        D_left_int = np.asarray(self.therm.getInterdiffusivity(c_left_int, temperature, phase=self.phases[0]), dtype=np.float64).reshape(len(self.elements), len(self.elements))
        D_right_int = np.asarray(self.therm.getInterdiffusivity(c_right_int, temperature, phase=self.phases[1]), dtype=np.float64).reshape(len(self.elements), len(self.elements))
        grad_left, grad_right = self._interface_gradients(composition, interface_position, (c_left_int, c_right_int))
        flux_left = -np.matmul(D_left_int, grad_left)
        flux_right = -np.matmul(D_right_int, grad_right)

        comp = np.asarray(composition, dtype=np.float64)
        left_node = np.asarray(comp[geometry.left_index], dtype=np.float64).reshape(-1)
        right_node = np.asarray(comp[geometry.right_index], dtype=np.float64).reshape(-1)
        denom = c_right_int + right_node - c_left_int - left_node
        if np.any(np.abs(denom) <= 1e-14):
            raise ValueError("Ternary MovingBoundaryFD1DModel encountered a near-zero Eq. (22) denominator.")
        velocities = 2.0 * (flux_right - flux_left) / denom
        residual = float(velocities[0] - velocities[1])
        return {
            "lambda": float(lam),
            "geometry": geometry,
            "temperature": float(temperature),
            "probe": x_probe,
            "interface_compositions": (c_left_int, c_right_int),
            "interface_diffusivities": (D_left_int, D_right_int),
            "gradients": (grad_left, grad_right),
            "fluxes": (flux_left, flux_right),
            "denominators": denom,
            "component_velocities": velocities,
            "velocity": float(np.mean(velocities)),
            "residual": residual,
            "tolerance": self._multicomponentResidualTolerance(velocities),
        }

    def _solveMulticomponentInterfaceState(self, t, composition, interface_position):
        '''
        Solves the ternary equal-velocity interface condition by a bracketed 1D search.
        '''
        geometry = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar)
        temperature = float(self.temperatureParameters(np.array([[interface_position]]), t)[0])
        trial_lambdas = np.linspace(0.0, 1.0, 33, dtype=np.float64)
        trial_states = []
        for lam in trial_lambdas:
            try:
                state = self._evaluateMulticomponentInterfaceState(t, composition, interface_position, lam, geometry=geometry, temperature=temperature)
            except Exception:
                continue
            if not np.all(np.isfinite(state["component_velocities"])) or not np.isfinite(state["residual"]):
                continue
            if abs(state["residual"]) <= state["tolerance"]:
                return state
            trial_states.append(state)
        if len(trial_states) == 0:
            raise ValueError("Ternary MovingBoundaryFD1DModel could not evaluate any valid interface states.")

        for left_state, right_state in zip(trial_states[:-1], trial_states[1:]):
            if left_state["residual"] == 0.0:
                return left_state
            if np.sign(left_state["residual"]) == np.sign(right_state["residual"]):
                continue
            lam_lo = left_state["lambda"]
            lam_hi = right_state["lambda"]
            lo_state = left_state
            hi_state = right_state
            for _ in range(50):
                lam_mid = 0.5 * (lam_lo + lam_hi)
                try:
                    mid_state = self._evaluateMulticomponentInterfaceState(
                        t,
                        composition,
                        interface_position,
                        lam_mid,
                        geometry=geometry,
                        temperature=temperature,
                    )
                except Exception:
                    lam_mid = np.nextafter(lam_mid, lam_hi)
                    mid_state = self._evaluateMulticomponentInterfaceState(
                        t,
                        composition,
                        interface_position,
                        lam_mid,
                        geometry=geometry,
                        temperature=temperature,
                    )
                if abs(mid_state["residual"]) <= mid_state["tolerance"]:
                    return mid_state
                if np.sign(mid_state["residual"]) == np.sign(lo_state["residual"]):
                    lam_lo = lam_mid
                    lo_state = mid_state
                else:
                    lam_hi = lam_mid
                    hi_state = mid_state
            best_state = lo_state if abs(lo_state["residual"]) < abs(hi_state["residual"]) else hi_state
            if abs(best_state["residual"]) <= best_state["tolerance"]:
                return best_state

        best_state = min(trial_states, key=lambda s: abs(s["residual"]))
        if abs(best_state["residual"]) <= best_state["tolerance"]:
            return best_state
        raise ValueError(
            "Ternary MovingBoundaryFD1DModel could not match the Eq. (22) interface velocities "
            f"for {self.elements}; smallest residual was {best_state['residual']:.3e}."
        )

    def _getInterfaceState(self, t, composition, interface_position):
        '''
        Returns geometry, interface compositions, and interfacial diffusivities
        '''
        geometry = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar)
        T_interface = float(self.temperatureParameters(np.array([[interface_position]]), t)[0])
        if self._isBinarySystem():
            c_left_int, c_right_int = self.therm.getInterfacialComposition(T_interface, 0, precPhase=self.phases[1])
            c_left_int = float(np.clip(np.squeeze(c_left_int), self.constraints.minComposition, 1 - self.constraints.minComposition))
            c_right_int = float(np.clip(np.squeeze(c_right_int), self.constraints.minComposition, 1 - self.constraints.minComposition))
            D_left_int = float(np.squeeze(self.therm.getInterdiffusivity(c_left_int, T_interface, phase=self.phases[0])))
            D_right_int = float(np.squeeze(self.therm.getInterdiffusivity(c_right_int, T_interface, phase=self.phases[1])))
            return geometry, T_interface, c_left_int, c_right_int, D_left_int, D_right_int

        state = self._solveMulticomponentInterfaceState(t, np.asarray(composition, dtype=np.float64), interface_position)
        c_left_int, c_right_int = state["interface_compositions"]
        D_left_int, D_right_int = state["interface_diffusivities"]
        return geometry, T_interface, c_left_int, c_right_int, D_left_int, D_right_int

    def _bulk_diffusivity_nodes(self, composition, t, geometry):
        '''
        Evaluates phase-appropriate node diffusivities on each side of the interface
        '''
        comp = np.asarray(composition, dtype=np.float64)
        temperatures = self.temperatureParameters(self.mesh.z, t)
        if comp.ndim == 1:
            comp = comp.reshape(-1)
            D = np.zeros_like(comp, dtype=np.float64)
        else:
            D = np.zeros((comp.shape[0], comp.shape[1], comp.shape[1]), dtype=np.float64)
        left_slice = slice(0, geometry.right_index)
        right_slice = slice(geometry.right_index, comp.shape[0])
        if geometry.right_index > 0:
            left_diff = np.asarray(
                self.therm.getInterdiffusivity(comp[left_slice], temperatures[left_slice], phase=self.phases[0]),
                dtype=np.float64,
            )
            if comp.ndim == 1:
                D[left_slice] = left_diff.reshape(-1)
            else:
                if left_diff.ndim == 2:
                    left_diff = left_diff[np.newaxis, :, :]
                D[left_slice] = left_diff.reshape(geometry.right_index, comp.shape[1], comp.shape[1])
        if geometry.right_index < comp.shape[0]:
            right_diff = np.asarray(
                self.therm.getInterdiffusivity(comp[right_slice], temperatures[right_slice], phase=self.phases[1]),
                dtype=np.float64,
            )
            if comp.ndim == 1:
                D[right_slice] = right_diff.reshape(-1)
            else:
                if right_diff.ndim == 2:
                    right_diff = right_diff[np.newaxis, :, :]
                D[right_slice] = right_diff.reshape(comp.shape[0] - geometry.right_index, comp.shape[1], comp.shape[1])
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
        c = np.asarray(composition, dtype=np.float64)
        d = np.asarray(diffusivity_nodes, dtype=np.float64)
        if c.ndim == 1:
            pairs = [
                DiffusionPair(
                    diffusivity=d.reshape(-1, 1),
                    response=c.reshape(-1, 1),
                    averageFunction=arithmeticMean,
                )
            ]
            return self.mesh.computedXdt(pairs)[:, 0]

        num_components = c.shape[1]
        pairs = []
        for i in range(num_components):
            pairs.append(
                DiffusionPair(
                    diffusivity=d[:, :, i],
                    response=np.tile(c[:, i][:, np.newaxis], (1, num_components)),
                    averageFunction=arithmeticMean,
                )
            )
        return self.mesh.computedXdt(pairs)

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
        if np.asarray(c_old).ndim != 1:
            raise ValueError("Legacy near-interface updates are only supported for binary systems.")
        z = np.ravel(self.mesh.z)
        if side == "A":
            xq = np.array([z[idx - 1], z[idx], s], dtype=np.float64)
            yq = np.array([c_old[idx - 1], c_old[idx], interface_compositions[0]], dtype=np.float64)
        else:
            xq = np.array([s, z[idx], z[idx + 1]], dtype=np.float64)
            yq = np.array([interface_compositions[1], c_old[idx], c_old[idx + 1]], dtype=np.float64)
        _, d2 = quad_fit_derivs(xq, yq, z[idx])
        c_new[idx] = c_old[idx] + self._currdt * (diffusivity * d2)

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
        c = np.asarray(composition, dtype=np.float64)
        d = np.asarray(diffusivity_nodes, dtype=np.float64)
        z = np.ravel(self.mesh.z)
        h = float(self.mesh.dz)
        avg = lambda a, b: np.asarray(arithmeticMean([a, b]), dtype=np.float64)

        is_scalar = c.ndim == 1
        if is_scalar:
            c_view = c.reshape(-1)
            d_view = d.reshape(-1)
        else:
            c_view = c
            d_view = d

        if side == "A":
            idx = int(geometry.left_near_index)
            if idx < 1:
                raise ValueError("Left near-interface flux-form update requires a left neighbor.")
            dA = float(geometry.interface_position - z[idx])
            if dA <= 0:
                raise ValueError("Left near-interface distance must be positive.")
            D_AL = avg(d_view[idx - 1], d_view[idx])
            D_AR = avg(d_view[idx], interface_diffusivities[0])
            if is_scalar:
                J_AL = -float(D_AL) * (c_view[idx] - c_view[idx - 1]) / h
                J_AR = -float(D_AR) * (interface_compositions[0] - c_view[idx]) / dA
            else:
                J_AL = -np.matmul(D_AL, (c_view[idx] - c_view[idx - 1]) / h)
                J_AR = -np.matmul(D_AR, (np.asarray(interface_compositions[0], dtype=np.float64) - c_view[idx]) / dA)
            deltaA = 0.5 * (h + dA)
            return idx, -(J_AR - J_AL) / deltaA

        if side == "B":
            idx = int(geometry.right_near_index)
            if idx + 1 >= c_view.shape[0]:
                raise ValueError("Right near-interface flux-form update requires a right neighbor.")
            dB = float(z[idx] - geometry.interface_position)
            if dB <= 0:
                raise ValueError("Right near-interface distance must be positive.")
            D_BL = avg(interface_diffusivities[1], d_view[idx])
            D_BR = avg(d_view[idx], d_view[idx + 1])
            if is_scalar:
                J_BL = -float(D_BL) * (c_view[idx] - interface_compositions[1]) / dB
                J_BR = -float(D_BR) * (c_view[idx + 1] - c_view[idx]) / h
            else:
                J_BL = -np.matmul(D_BL, (c_view[idx] - np.asarray(interface_compositions[1], dtype=np.float64)) / dB)
                J_BR = -np.matmul(D_BR, (c_view[idx + 1] - c_view[idx]) / h)
            deltaB = 0.5 * (dB + h)
            return idx, -(J_BR - J_BL) / deltaB

        raise ValueError("side must be 'A' or 'B'.")

    def _interface_gradients(self, composition, interface_position, interface_compositions):
        '''
        Computes one-sided interface gradients from three-point quadratic fits
        '''
        comp = np.asarray(composition, dtype=np.float64)
        z = np.ravel(self.mesh.z)
        geom = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar)
        if geom.p < self.pstar:
            i1, i2 = geom.left_index - 2, geom.left_index - 1
            j1, j2 = geom.right_index, geom.right_index + 1
        else:
            i1, i2 = geom.left_index - 1, geom.left_index
            j1, j2 = geom.right_index + 1, geom.right_index + 2
        
        if (i1<0) or (i2<0) or (j1>(len(z)-1)) or (j2>(len(z)-1)):
            try:
                import debugpy
                # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
                debugpy.listen(5678)
                print("Waiting for debugger attach")
                debugpy.wait_for_client()
                debugpy.breakpoint()
                print('break on this line')
            except:
                pass

        # i1 = max(0, i1)
        # i2 = max(0, i2)
        # j1 = min(len(z) - 1, j1)
        # j2 = min(len(z) - 1, j2)
        
        if geom.ignored_index in [i1, i2, j1, j2]:
            raise ValueError("Interface gradient evaluation stencils should not include the ignored node.")

        x_left = np.array([z[i1], z[i2], interface_position], dtype=np.float64)
        x_right = np.array([interface_position, z[j1], z[j2]], dtype=np.float64)
        if comp.ndim == 1:
            y_left = np.array([comp[i1], comp[i2], interface_compositions[0]], dtype=np.float64)
            y_right = np.array([interface_compositions[1], comp[j1], comp[j2]], dtype=np.float64)
            grad_left, _ = quad_fit_derivs(x_left, y_left, interface_position)
            grad_right, _ = quad_fit_derivs(x_right, y_right, interface_position)
            return float(grad_left), float(grad_right)

        left_int = np.asarray(interface_compositions[0], dtype=np.float64).reshape(-1)
        right_int = np.asarray(interface_compositions[1], dtype=np.float64).reshape(-1)
        grad_left = np.zeros(comp.shape[1], dtype=np.float64)
        grad_right = np.zeros(comp.shape[1], dtype=np.float64)
        for i in range(comp.shape[1]):
            y_left = np.array([comp[i1, i], comp[i2, i], left_int[i]], dtype=np.float64)
            y_right = np.array([right_int[i], comp[j1, i], comp[j2, i]], dtype=np.float64)
            grad_left[i], _ = quad_fit_derivs(x_left, y_left, interface_position)
            grad_right[i], _ = quad_fit_derivs(x_right, y_right, interface_position)
        return grad_left, grad_right

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
        comp = np.asarray(composition, dtype=np.float64)
        if comp.ndim == 1:
            pairs = [DiffusionPair(diffusivity=np.asarray(diffusivity_nodes, dtype=np.float64)[:, np.newaxis], response=comp[:, np.newaxis], averageFunction=arithmeticMean)]
            fluxes = self.mesh.computeFluxes(pairs)[:, 0]
        else:
            num_components = comp.shape[1]
            pairs = []
            for i in range(num_components):
                pairs.append(
                    DiffusionPair(
                        diffusivity=np.asarray(diffusivity_nodes, dtype=np.float64)[:, :, i],
                        response=np.tile(comp[:, i][:, np.newaxis], (1, num_components)),
                        averageFunction=arithmeticMean,
                    )
                )
            fluxes = self.mesh.computeFluxes(pairs)
        geom = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar)
        if comp.ndim == 1:
            left_flux = -interface_diffusivities[0] * (interface_compositions[0] - comp[geom.left_index]) / geom.left_distance
            right_flux = -interface_diffusivities[1] * (comp[geom.right_index] - interface_compositions[1]) / geom.right_distance
            fluxes[geom.right_index] = 0.5 * (left_flux + right_flux)
            return fluxes, left_flux, right_flux

        left_gradient = (np.asarray(interface_compositions[0], dtype=np.float64) - comp[geom.left_index]) / geom.left_distance
        right_gradient = (comp[geom.right_index] - np.asarray(interface_compositions[1], dtype=np.float64)) / geom.right_distance
        left_flux = -np.matmul(np.asarray(interface_diffusivities[0], dtype=np.float64), left_gradient)
        right_flux = -np.matmul(np.asarray(interface_diffusivities[1], dtype=np.float64), right_gradient)
        fluxes[geom.right_index] = 0.5 * (left_flux + right_flux)
        return fluxes, left_flux, right_flux

    def _computeStateBinary(self, t, xCurr):
        '''
        Computes composition rates and interface velocity for one binary explicit FDM step.
        '''
        c_old = np.asarray(xCurr[0], dtype=np.float64).reshape(-1)
        # if 94341.92<t<94341.94:
        #     try:
        #         import debugpy
        #         # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        #         debugpy.listen(5678)
        #         print("Waiting for debugger attach")
        #         debugpy.wait_for_client()
        #         debugpy.breakpoint()
        #         print('break on this line')
        #     except:
        #         pass
        s_old = self._clipInterfacePosition(float(xCurr[1]))
        geom, _, c_left_int, c_right_int, D_left_int, D_right_int = self._getInterfaceState(t, c_old, s_old)
        diffusivity_nodes = self._bulk_diffusivity_nodes(c_old, t, geom)
        max_diff = float(np.max(np.abs(np.concatenate((diffusivity_nodes, [D_left_int, D_right_int])))))
        min_length = self.mesh.dz * ((1.0 - geom.p) if geom.p < self.pstar else geom.p)
        dt_diff = self.constraints.vonNeumannThreshold * (min_length**2) / max_diff if max_diff > 0 else np.inf

        initial_stage = self._reconstructIgnoredComposition(c_old, s_old, geom.p, s_old, (c_left_int, c_right_int))
        grad_left_pred, grad_right_pred = self._interface_gradients(initial_stage, s_old, (c_left_int, c_right_int))
        denom_basic = c_right_int - c_left_int
        s_dot_pred = (D_left_int * grad_left_pred - D_right_int * grad_right_pred) / denom_basic if abs(denom_basic) > 1e-14 else 0.0
        move_fraction = min(self.constraints.movingBoundaryThreshold, np.inf) # , 0.95 * self._max_interface_step_fraction(geom, s_dot_pred))
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
        if (left_near_active!=True) or (right_near_active!=True):
            try:
                import debugpy
                # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
                debugpy.listen(5678)
                print("Waiting for debugger attach")
                debugpy.wait_for_client()
                debugpy.breakpoint()
                print('break on this line')
            except:
                pass

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

        c_stage = self._reconstructIgnoredComposition(c_new, s_old, geom.p, s_old, (c_left_int, c_right_int))
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
        elif self.interfaceUpdate == "lee_oh_corrected":
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
            
            # delta_mass = current_mass - ((374.5 * 0.291) + (190.5 * 0.394))
            ds = ds_intermediate + (2.0 * delta_mass) / denom
            velocity = ds / self._currdt if self._currdt > 0 else 0.0

            # doubleCheckMass=True
            # if doubleCheckMass:
            #     doubleCheck_mass = integrate_binary_fd_profile(
            #         self.mesh.z,
            #         c_stage,
            #         s_old=s_old,
            #         p_old=geom.p,
            #         s_new=s_old+ds,
            #         pstar=self.pstar,
            #         interface_compositions=(c_left_int, c_right_int),
            #         integration_mode=self.integrationMode,
            #         s_for_interp="new",
            #     )
            #     doubleCheck_massDiff = doubleCheck_mass - self._initialInventory
        elif self.interfaceUpdate == "my_corrected":
            try:
                mass_func = lambda s: integrate_binary_fd_profile(self.mesh.z, c_stage, s_old=s_old, p_old=geom.p, s_new=s, pstar=self.pstar, interface_compositions=(c_left_int, c_right_int), integration_mode=self.integrationMode, s_for_interp="new")
                massDiff_func = lambda s: mass_func(s)-self._initialInventory
                
                # massDiff_func = lambda s: mass_func(s)-((374.5 * 0.291) + (190.5 * 0.394))
                bracket_halfWidth = self.mesh.dz * self.constraints.movingBoundaryThreshold
                # bracket_halfWidth = self.mesh.dz * 0.5
                bracket = [s_old-bracket_halfWidth, s_old+bracket_halfWidth]
                bracket = np.clip(bracket, 1.5 * self.mesh.dz, self.mesh.zlim[0][-1] - (1.5 * self.mesh.dz)).tolist()
                sol = optimize.root_scalar(
                    massDiff_func,
                    # bracket=[np.ravel(self.mesh.z)[geom.left_index-1], np.ravel(self.mesh.z)[geom.right_index+1]],
                    bracket=bracket,
                    method='brentq',
                    rtol=1e-14,
                    xtol=1e-14,
                )
                # debugInPlace()
                ds = sol.root-s_old
                velocity = ds / self._currdt if self._currdt > 0 else 0.0
            except:
                debugInPlace()


        else:
            raise ValueError("Should be one of above options")




        max_fraction = self.constraints.movingBoundaryThreshold #self._max_interface_step_fraction(geom, velocity) # this was removed since _max_interface_step_fraction() will prevent interface from ever crossing node
        if (t==0) and (self.interfaceUpdate == "my_corrected"):
            max_fraction=0.5
        requested_fraction = abs(ds) / self.mesh.dz
        if not np.isfinite(requested_fraction):
            raise ValueError("MovingBoundaryFD1DModel produced a non-finite interface increment.")
        max_fraction = max(max_fraction, 1e-12)
        if requested_fraction > max_fraction:
            print(f"(max_fraction, requested_fraction): {(max_fraction, requested_fraction)}")
            debugInPlace()
            raise ValueError("MovingBoundaryFD1DModel requested_fraction is greater than max_fraction")

        s_new = self._clipInterfacePosition(s_old + ds, strict=True)
        c_final = self._reconstructIgnoredComposition(c_new, s_old, geom.p, s_new, (c_left_int, c_right_int))
        dcdt = (c_final - c_old) / self._currdt
        # fluxes, left_flux, right_flux = self._compute_fluxes(c_final, diffusivity_nodes, s_old, (c_left_int, c_right_int), (D_left_int, D_right_int)) ## This compute_fluxes seems unnecessary and possibly even wrong?
        self._lastFluxes = None # fluxes
        self._lastInterfaceFluxes = None # (float(left_flux), float(right_flux))
        self._lastInterfaceVelocity = None # float(velocity)
        return dcdt[:, np.newaxis], float(velocity)

    def _computeStateTernary(self, t, xCurr):
        '''
        Computes composition rates and interface velocity for one ternary explicit FDM step.
        '''
        c_old = np.asarray(xCurr[0], dtype=np.float64)
        s_old = self._clipInterfacePosition(float(xCurr[1]))
        geom = get_moving_boundary_fd_geometry(self.mesh, s_old, self.pstar)
        pre_state = self._solveMulticomponentInterfaceState(t, c_old, s_old)
        c_left_int_pre, c_right_int_pre = pre_state["interface_compositions"]
        D_left_int_pre, D_right_int_pre = pre_state["interface_diffusivities"]
        diffusivity_nodes = self._bulk_diffusivity_nodes(c_old, t, geom)
        max_diff = float(np.max(np.abs(np.concatenate((diffusivity_nodes.reshape(-1), D_left_int_pre.reshape(-1), D_right_int_pre.reshape(-1))))))
        min_length = self.mesh.dz * ((1.0 - geom.p) if geom.p < self.pstar else geom.p)
        dt_diff = self.constraints.vonNeumannThreshold * (min_length**2) / max_diff if max_diff > 0 else np.inf

        move_fraction = min(self.constraints.movingBoundaryThreshold, np.inf)
        dt_move = move_fraction * self.mesh.dz / abs(pre_state["velocity"]) if abs(pre_state["velocity"]) > 0 else np.inf
        allowed_dt = getattr(self, "deltaTime", np.inf)
        self._currdt = min(dt_diff, dt_move, allowed_dt)

        ignored = geom.ignored_index
        if geom.p < self.pstar:
            a_last = geom.left_index - 1
            b_first = geom.right_index
        else:
            a_last = geom.left_index
            b_first = geom.right_index + 1

        bulk_dcdt = np.asarray(self._bulk_dcdt(c_old, diffusivity_nodes), dtype=np.float64)
        bulk_mask = np.zeros(c_old.shape[0], dtype=bool)
        if a_last >= 0:
            bulk_mask[: a_last + 1] = True
        if b_first < c_old.shape[0]:
            bulk_mask[b_first:] = True
        bulk_mask[ignored] = False

        left_near_active = geom.left_near_index >= 1 and geom.left_near_index <= a_last
        right_near_active = geom.right_near_index + 1 < c_old.shape[0] and geom.right_near_index >= b_first
        if left_near_active:
            bulk_mask[geom.left_near_index] = False
        if right_near_active:
            bulk_mask[geom.right_near_index] = False

        c_new = c_old.copy()
        c_new[bulk_mask] = c_old[bulk_mask] + self._currdt * bulk_dcdt[bulk_mask]
        if left_near_active:
            idx, dcdt_left = self._near_interface_dcdt_flux_form(
                c_old,
                diffusivity_nodes,
                geom,
                "A",
                (c_left_int_pre, c_right_int_pre),
                (D_left_int_pre, D_right_int_pre),
            )
            c_new[idx] = c_old[idx] + self._currdt * dcdt_left
        if right_near_active:
            idx, dcdt_right = self._near_interface_dcdt_flux_form(
                c_old,
                diffusivity_nodes,
                geom,
                "B",
                (c_left_int_pre, c_right_int_pre),
                (D_left_int_pre, D_right_int_pre),
            )
            c_new[idx] = c_old[idx] + self._currdt * dcdt_right

        c_stage = self._reconstructIgnoredComposition(c_new, s_old, geom.p, s_old, (c_left_int_pre, c_right_int_pre))
        actual_state = pre_state if self.fluxGradientMode == "pre_diffusion" else self._solveMulticomponentInterfaceState(t, c_stage, s_old)
        interface_compositions = actual_state["interface_compositions"]
        interface_diffusivities = actual_state["interface_diffusivities"]
        ds_intermediate = self._currdt * actual_state["velocity"]

        if self.interfaceUpdate == "basic":
            ds = ds_intermediate
            velocity = actual_state["velocity"]
        else:
            balance_index = self._balanceElementIndex
            balance_denom = float(actual_state["denominators"][balance_index])
            s_intermediate = s_old + ds_intermediate
            intermediate_inventory = self._integrateComponentInventory(
                c_stage,
                s_intermediate,
                interface_compositions,
                balance_index,
                s_for_interp="new",
                s_old=s_old,
                p_old=geom.p,
                s_new=s_intermediate,
            )
            delta_inventory = intermediate_inventory - self._initialInventory[balance_index]
            ds = ds_intermediate + (2.0 * delta_inventory) / balance_denom
            velocity = ds / self._currdt if self._currdt > 0 else 0.0

        max_fraction = self.constraints.movingBoundaryThreshold
        requested_fraction = abs(ds) / self.mesh.dz
        if not np.isfinite(requested_fraction):
            raise ValueError("MovingBoundaryFD1DModel produced a non-finite interface increment.")
        max_fraction = max(max_fraction, 1e-12)
        if requested_fraction > max_fraction:
            raise ValueError("MovingBoundaryFD1DModel requested_fraction is greater than max_fraction")

        s_new = self._clipInterfacePosition(s_old + ds, strict=True)
        c_final = self._reconstructIgnoredComposition(c_new, s_old, geom.p, s_new, interface_compositions)
        dcdt = (c_final - c_old) / self._currdt
        # fluxes, left_flux, right_flux = self._compute_fluxes(c_final, diffusivity_nodes, s_old, interface_compositions, interface_diffusivities) ## This compute_fluxes seems unnecessary and possibly even wrong?
        self._lastFluxes = None # fluxes
        self._lastInterfaceFluxes = None # (np.asarray(left_flux, dtype=np.float64), np.asarray(right_flux, dtype=np.float64))
        self._lastInterfaceVelocity = None # float(velocity)
        return dcdt, float(velocity)

    def _computeState(self, t, xCurr):
        '''
        Computes composition rates and interface velocity for one explicit FDM step.

        This method contains the shared stepping core for all interface update
        modes. It performs the selected bulk-node diffusion update away from
        the interface, reconstructs the ignored node, evaluates one-sided
        interface gradients, and then applies the selected interface motion
        update. In ``flux_form`` mode, the near-interface nodes use
        conservative cut-cell balances. In ``legacy`` mode, they retain the
        quadratic ``D_i * c_xx`` stencil.
        '''
        if self._isBinarySystem():
            return self._computeStateBinary(t, xCurr)
        return self._computeStateTernary(t, xCurr)

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
        if np.asarray(self._lastFluxes).ndim == 1:
            return self._lastFluxes[:, np.newaxis]
        return self._lastFluxes

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
        max_fraction = self.constraints.movingBoundaryThreshold #max(1e-12, 0.95 * self._max_interface_step_fraction(geom, velocity)) # this was removed since _max_interface_step_fraction() will prevent interface from ever crossing node
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
        interface_compositions = self._getInterfaceState(self.currentTime, composition, interface_position)[2:4]
        residual = np.abs(self._initialInventory - self._integrateInventory(composition, interface_position, interface_compositions, s_for_interp="old"))
        if np.all(residual <= tolerance):
            return
        action = str(self.constraints.movingBoundaryMassAction).lower()
        if np.ndim(residual) == 0:
            residual_text = f"{float(residual):.3e}"
        else:
            residual_text = np.array2string(np.asarray(residual, dtype=np.float64), precision=3, separator=", ")
        message = (
            f"MovingBoundaryFD1DModel mass correction residual {residual_text} exceeded "
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
        composition = self._clipCompositionField(np.asarray(x[0], dtype=np.float64))
        interface_position = self._clipInterfacePosition(float(x[1]))
        mass_check_composition = composition[:, 0] if composition.shape[1] == 1 else composition
        self._checkMassCorrection(mass_check_composition, interface_position)
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
        if not self._isBinarySystem():
            raise ValueError("getTotalMass is only defined for binary MovingBoundaryFD1DModel systems. Use getTotalInventory instead.")
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

    def  getTotalInventory(self, time = None):
        '''
        Returns the integrated inventory of each independent component for ternary systems.
        '''
        composition = np.asarray(self.data.y(time), dtype=np.float64)
        interface_position = self.getInterfacePosition(time)
        interface_compositions = self._getInterfaceState(self.currentTime if time is None else time, composition, interface_position)[2:4]
        return self._integrateInventory(composition, interface_position, interface_compositions, s_for_interp="old")

    def getInterfaceCompositions(self, time = None):
        '''
        Returns the interface compositions at the requested time.
        '''
        composition = np.asarray(self.data.y(time), dtype=np.float64)
        interface_position = self.getInterfacePosition(time)
        interface_state = self._getInterfaceState(self.currentTime if time is None else time, composition, interface_position)
        return interface_state[2], interface_state[3]

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
