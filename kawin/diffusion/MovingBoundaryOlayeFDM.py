import warnings

import numpy as np

from kawin.GenericModel import GenericModel
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.diffusion.mesh import CartesianFD1D, MixedBoundary1D, PeriodicBoundary1D
from kawin.diffusion.mesh.MovingBoundaryOlayeFD1D import (
    build_piecewise_diffusivity_nodes,
    get_olaye_fd_geometry,
    integrate_binary_olaye_fd_profile,
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
def loge_arange(start, stop, log_step):
    logs = np.arange(np.log(start),
                    np.log(stop),
                    log_step)
    return np.exp(logs)

class _ScalarHistory:
    def __init__(self, record: bool | int = False):
        if isinstance(record, bool):
            self.recordInterval = 1 if record else -1
        else:
            self.recordInterval = int(record)
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
            uind = np.argmax(self._time > time)
            lind = uind - 1
            uy, utime = self._y[uind], self._time[uind]
            ly, ltime = self._y[lind], self._time[lind]
            return float((uy - ly) * (time - ltime) / (utime - ltime) + ly)
        return float(self._y[0])

class _VectorHistory:
    def __init__(self, n_components: int, record: bool | int = False):
        if isinstance(record, bool):
            self.recordInterval = 1 if record else -1
        else:
            self.recordInterval = int(record)
        self.n_components = int(n_components)
        self.batchSize = 1000
        self.reset()

    def reset(self):
        '''
        Resets arrays for a vector-valued history.
        '''
        self._y = np.zeros((self.batchSize, self.n_components), dtype=np.float64)
        self._time = np.zeros(self.batchSize, dtype=np.float64)
        self.currentIndex = 0
        self.currentY = np.zeros(self.n_components, dtype=np.float64)
        self.currentTime = 0.0
        self.N = 0

    def record(self, time, y, force: bool = False):
        '''
        Stores current state of time and vector variable.
        '''
        values = np.asarray(y, dtype=np.float64).reshape(-1)
        if values.size != self.n_components:
            raise ValueError(f"Expected {self.n_components} components, got {values.size}.")
        if self.recordInterval > 0:
            if self.currentIndex % self.recordInterval == 0 or force:
                self.N = int(self.currentIndex / self.recordInterval)
                if self.N >= self._time.shape[0]:
                    self._y = np.pad(self._y, ((0, self.batchSize), (0, 0)))
                    self._time = np.pad(self._time, (0, self.batchSize))
                self._y[self.N] = values
                self._time[self.N] = time
            self.currentIndex += 1
        else:
            self._y[self.N] = values
            self._time[self.N] = time
        self.currentY = values.copy()
        self.currentTime = float(time)

    def finalize(self):
        '''
        Removes extra padding.
        '''
        if self.recordInterval > 0 and self.N >= 0 and np.isclose(self._time[self.N], self.currentTime, rtol=0.0, atol=1e-14):
            self._y = self._y[: self.N + 1]
            self._time = self._time[: self.N + 1]
            return
        self.record(self.currentTime, self.currentY, force=True)
        self._y = self._y[: self.N + 1]
        self._time = self._time[: self.N + 1]

    def y(self, time=None):
        '''
        Returns vector value at an exact recorded time.
        '''
        if time is None:
            return self._y[self.N].copy()
        recorded_time = self._time[: self.N + 1]
        matches = np.where(np.isclose(recorded_time, float(time), atol=1e-14, rtol=0.0))[0]
        if len(matches) == 0:
            raise ValueError(
                f"Requested time {float(time):.6g} was not found in exact recorded interface composition times."
            )
        return self._y[matches[-1]].copy()

class MovingBoundaryOlayeFD1DModel(DiffusionModel):
    """
    Binary planar moving-boundary model following Olaye & Ojo (2020).

    The implementation uses the Landau-transformed phase fields described in the
    paper, computes the interface position first from the solute-balance
    relation, uses a small-step classical explicit bootstrap for ``k=1``, and
    then advances both phases with a Leapfrog/Dufort-Frankel style explicit
    update for ``k>1``.

    Notes
    -----
    Runtime support is currently limited to planar geometry. Cylindrical and
    spherical cases remain documented but intentionally raise
    ``NotImplementedError``.
    """

    def __init__(
        self,
        mesh,
        elements,
        phases,
        thermodynamics,
        temperature,
        interfacePosition,
        interface_compositions: tuple[float, float],
        first_step_mode: str,
        main_step_mode: str,
        dt_mode: str,
        geometry: str = "planar",
        phase_a_nodes: int | None = None,
        phase_b_nodes: int | None = None,
        semiLog_dt: float | None = None,
        semiLogT0: float | None = None,
        constraints=None,
        record=False,
    ):
        self.initialInterfacePosition = float(interfacePosition)
        self.interfaceData = _ScalarHistory(record)
        self.concData = _ScalarHistory(record)
        self.concData_alt = _ScalarHistory(record)
        self.pData = None
        self.qData = None
        self.interfaceCompositions = tuple(float(v) for v in interface_compositions)
        self.firstStepMode = str(first_step_mode)
        self.mainStepMode = str(main_step_mode)
        self.dtMode = str(dt_mode)
        self.geometry = str(geometry)
        self.phaseANodes = None if phase_a_nodes is None else int(phase_a_nodes)
        self.phaseBNodes = None if phase_b_nodes is None else int(phase_b_nodes)
        self.semiLog_dt = None if semiLog_dt is None else float(semiLog_dt)
        self.semiLogT0 = None if semiLogT0 is None else float(semiLogT0)

        self._currdt = np.inf
        self._pendingDtDiff = np.inf
        self._pendingDtMove = np.inf
        self._pendingDtSemi = np.inf
        self._lastStepScheme = None
        self._lastFluxes = None
        self._lastInterfaceFluxes = (0.0, 0.0)
        self._lastInterfaceVelocity = 0.0
        self._lastInterfaceCoefficients = (np.nan, np.nan)
        self._lastWindingAB = (1.0, 0.0)
        self._currentPaperDt = np.inf
        self._stepIndex = 0

        self._t_prev = None
        self._p_prev = None
        self._p_curr = None
        self._q_prev = None
        self._q_curr = None
        self._s_prev = None
        self._s_curr = None
        self._semiLogTimes = None
        self._semiLogNextIndex = 0
        self._initialInventory = None

        self._z = None
        self._R = None
        self._u_grid = None
        self._v_grid = None
        self._du = None
        self._dv = None
        self._phase_counts = None

        super().__init__(
            mesh=mesh,
            elements=elements,
            phases=phases,
            thermodynamics=thermodynamics,
            temperature=temperature,
            constraints=constraints,
            record=record,
        )
        self._validateModelConfiguration()
        self.interfaceData.currentY = self.initialInterfacePosition
        self.interfaceData._y[0] = self.initialInterfacePosition

    def _validateModelConfiguration(self):
        if not isinstance(self.mesh, CartesianFD1D):
            raise TypeError("MovingBoundaryOlayeFD1DModel requires a CartesianFD1D mesh.")
        if len(self.allElements) != 2 or self.mesh.numResponses != 1:
            raise ValueError("MovingBoundaryOlayeFD1DModel currently supports only binary systems.")
        if any(e in interstitials for e in self.allElements):
            raise ValueError("MovingBoundaryOlayeFD1DModel supports only substitutional systems.")
        if len(self.phases) != 2:
            raise ValueError("MovingBoundaryOlayeFD1DModel requires exactly two explicit phases.")
        if isinstance(getattr(self.mesh, "boundaryConditions", None), PeriodicBoundary1D):
            raise ValueError("Periodic boundary conditions are not supported.")
        if self.firstStepMode != "classical_explicit":
            raise ValueError("first_step_mode must be 'classical_explicit'.")
        if self.mainStepMode not in {"leapfrog_dufort_frankel", "classical_explicit"}:
            raise ValueError("main_step_mode must be 'leapfrog_dufort_frankel' or 'classical_explicit'.")
        if self.dtMode not in {"cfl", "semi_log_optional"}:
            raise ValueError("dt_mode must be 'cfl' or 'semi_log_optional'.")
        if self.dtMode == "semi_log_optional" and ((self.semiLog_dt is None) or (self.semiLogT0 is None)):
            raise ValueError("semiLog_dt and semiLogT0 must be specified when dt_mode is 'semi_log_optional'.")
        if self.geometry not in {"planar", "cylindrical", "spherical"}:
            raise ValueError("geometry must be one of ['planar', 'cylindrical', 'spherical'].")
        if self.geometry != "planar":
            raise NotImplementedError(
                f"geometry='{self.geometry}' is documented but not implemented yet for "
                "MovingBoundaryOlayeFD1DModel."
            )
        if not (0 < self.semiLogT0):
            raise ValueError("semiLogT0 must be positive.")
        if self.phaseANodes is not None and self.phaseANodes < 3:
            raise ValueError("phase_a_nodes must be at least 3 when specified.")
        if self.phaseBNodes is not None and self.phaseBNodes < 3:
            raise ValueError("phase_b_nodes must be at least 3 when specified.")
        self.initialInterfacePosition = self._clipInterfacePosition(self.initialInterfacePosition, strict=True)

    def _clipInterfacePosition(self, interface_position: float, strict: bool = True) -> float:
        z = np.ravel(self.mesh.z)
        eps = max(float(self.mesh.dz) * 1e-8, 1e-14)
        lower = float(z[0] + eps)
        upper = float(z[-1] - eps)
        if strict and not (lower < interface_position < upper):
            # debugInPlace()
            raise ValueError("Interface position must lie strictly inside the FD domain.")
        clipped = float(np.clip(interface_position, lower, upper))
        matches = np.where(np.isclose(z, clipped, atol=eps, rtol=0.0))[0]
        if len(matches) > 0:
            idx = int(matches[0])
            if idx >= len(z) - 1:
                clipped = float(max(lower, clipped - eps))
            else:
                clipped = float(min(upper, clipped + eps))
        return clipped

    def setTimeInfo(self, currTime, simTime):
        super().setTimeInfo(currTime, simTime)
        if self.dtMode != "semi_log_optional" or simTime <= 0:
            self._semiLogTimes = None
            self._semiLogNextIndex = 0
            return
        t0_rel = max(self.semiLogT0, 1e-15)
        # rel_times = np.geomspace(t0_rel, simTime, self.semiLogPoints)
        rel_times = loge_arange(t0_rel, simTime, self.semiLog_dt)
        if len(rel_times) < 3:
            raise ValueError("len(rel_times) must be at least 3.")
        if rel_times[-1] < simTime:
            rel_times = np.append(rel_times, simTime)
        if rel_times[-1] != simTime:
            raise ValueError(f"rel_times[-1] = {rel_times[-1]} does not match simTime = {simTime}")
            
        self._semiLogTimes = currTime + rel_times
        self._semiLogNextIndex = 0

    def reset(self):
        super().reset()
        self.interfaceData.reset()
        self.interfaceData.record(0, self.initialInterfacePosition)
        self.concData.reset()
        self.concData_alt.reset()
        self.pData = None
        self.qData = None
        self._currdt = np.inf
        self._pendingDtDiff = np.inf
        self._pendingDtMove = np.inf
        self._pendingDtSemi = np.inf
        self._lastStepScheme = None
        self._lastFluxes = None
        self._lastInterfaceFluxes = (0.0, 0.0)
        self._lastInterfaceVelocity = 0.0
        self._lastInterfaceCoefficients = (np.nan, np.nan)
        self._lastWindingAB = (1.0, 0.0)
        self._currentPaperDt = np.inf
        self._stepIndex = 0
        self._semiLogTimes = None
        self._semiLogNextIndex = 0
        self._p_prev = None
        self._p_curr = None
        self._q_prev = None
        self._q_curr = None
        self._s_prev = None
        self._s_curr = None
        self._z = None
        self._R = None
        self._u_grid = None
        self._v_grid = None
        self._du = None
        self._dv = None
        self._phase_counts = None
        if hasattr(self, "mesh") and self.mesh is not None:
            self._validateModelConfiguration()

    def setup(self):
        super().setup()
        self._validateModelConfiguration()
        self._z = np.ravel(self.mesh.z).astype(np.float64)
        self._R = float(self._z[-1] - self._z[0])
        if not np.isclose(self._z[0], 0.0):
            raise ValueError("MovingBoundaryOlayeFD1DModel currently expects a 1D domain starting at 0.")

        c0 = np.asarray(self.data.currentY, dtype=np.float64).reshape(-1)
        s0 = float(self.interfaceData.currentY)
        geom = get_olaye_fd_geometry(self.mesh, s0)

        n_left = self.phaseANodes if self.phaseANodes is not None else geom.left_index + 1
        n_right = self.phaseBNodes if self.phaseBNodes is not None else len(c0) - geom.right_index
        if n_left < 3 or n_right < 3:
            raise ValueError("Each phase must retain at least three transformed grid points.")

        self._u_grid = np.linspace(0.0, 1.0, n_left, dtype=np.float64)
        self._v_grid = np.linspace(0.0, 1.0, n_right, dtype=np.float64)
        self._du = float(self._u_grid[1] - self._u_grid[0])
        self._dv = float(self._v_grid[1] - self._v_grid[0])
        self._phase_counts = (n_left, n_right)
        self.pData = _VectorHistory(n_left, self.interfaceData.recordInterval)
        self.qData = _VectorHistory(n_right, self.interfaceData.recordInterval)

        self._p_curr, self._q_curr = self._initialize_transformed_state(c0, s0)
        self._p_prev = self._p_curr.copy()
        self._q_prev = self._q_curr.copy()
        self._s_curr = s0
        self._s_prev = s0
        self.pData.record(0, self._p_curr)
        self.qData.record(0, self._q_curr)

        self.data.currentY = self._reconstruct_physical_profile(self._p_curr, self._q_curr, s0)[:, np.newaxis]
        self._initialInventory = self.getTotalInventory(time=0)
        averageConc = self.checkMassIntegral(p=self._p_curr.copy(), q=self._q_curr.copy(), s=self._s_curr)
        self.concData.record(0, averageConc)
        averageConc_alt = self.getTotalInventory(time=0) / self._R
        self.concData_alt.record(0, averageConc_alt)
        


    def solve(self, simTime, iterator=explicitEulerIterator, verbose=False, vIt=10, minDtFrac=1e-8, maxDtFrac=1):
        """
        Solves the paper-style recurrence using a single-step explicit Euler wrapper.

        The Olaye/Ojo update is itself a complete explicit recurrence. Multi-stage
        iterators such as RK4 would evaluate inconsistent intermediate states, so
        this model accepts only ``explicitEulerIterator``.
        """
        if iterator is not explicitEulerIterator:
            raise ValueError("MovingBoundaryOlayeFD1DModel supports only explicitEulerIterator.")
        return super().solve(simTime, iterator=iterator, verbose=verbose, vIt=vIt, minDtFrac=minDtFrac, maxDtFrac=maxDtFrac)

    def _initialize_transformed_state(self, composition, interface_position):
        """
        Maps the physical profile onto the transformed phase grids.

        The physical profile is discontinuous at the sharp interface, so the left
        and right phases are interpolated independently. This avoids smearing the
        interface jump across the transformed grids, which would otherwise flip
        the initial near-interface gradient and corrupt the Eq. (12) root.
        """
        c = np.asarray(composition, dtype=np.float64).reshape(-1)
        
        z_left = np.clip(interface_position * self._u_grid, self._z[0], interface_position)
        right_span = max(self._R - interface_position, 1e-15)
        z_right = interface_position + right_span * self._v_grid

        left_mask = self._z <= interface_position
        right_mask = self._z >= interface_position

        z_left_source = self._z[left_mask].copy() ## old way np.concatenate((self._z[left_mask], [interface_position]))
        c_left_source = c[left_mask].copy() ## old way np.concatenate((c[left_mask], [c_ab]))
        z_right_source = self._z[right_mask].copy()## old way np.concatenate(([interface_position], self._z[right_mask]))
        c_right_source = c[right_mask].copy() ## old way np.concatenate(([c_ba], c[right_mask]))

        if len(np.unique(c_left_source))!=1 or len(np.unique(c_right_source))!=1:
             raise ValueError("Initial composition profile has more than one unique composition on at least one side of the interface. The transformed grid initialization will interpolate these values which may cause unexpected results")

        p = np.interp(z_left[:-1], z_left_source, c_left_source)
        q = np.interp(z_right[1:], z_right_source, c_right_source)

        if len(np.unique(p))!=1 or len(np.unique(q))!=1:
             raise ValueError("Transformed composition profile has more than one unique composition on at least one side of the interface. The values were interpolatedwhich may cause unexpected results")

        c_ab, c_ba = self.interfaceCompositions
        p_full = np.concatenate((p, [c_ab]))
        q_full = np.concatenate(([c_ba], q))
        return self._apply_boundary_conditions(p_full, q_full)

    def getCurrentX(self):
        return [self._p_curr.copy(), self._q_curr.copy(), float(self._s_curr)]

    def flattenX(self, X):
        return np.concatenate((np.asarray(X[0], dtype=np.float64), np.asarray(X[1], dtype=np.float64), [float(X[2])]))

    def unflattenX(self, X_flat, X_ref):
        n_p = len(np.asarray(X_ref[0], dtype=np.float64))
        n_q = len(np.asarray(X_ref[1], dtype=np.float64))
        p = np.asarray(X_flat[:n_p], dtype=np.float64)
        q = np.asarray(X_flat[n_p : n_p + n_q], dtype=np.float64)
        s = float(X_flat[n_p + n_q])
        return [p, q, s]

    def _getBoundaryConditions(self):
        bc = getattr(self.mesh, "boundaryConditions", None)
        if bc is None:
            bc = MixedBoundary1D(self.mesh.responses)
            self.mesh.boundaryConditions = bc
        return bc

    def _updateSemiLogIndex(self, t):
        if self._semiLogTimes is None:
            return
        while self._semiLogNextIndex < len(self._semiLogTimes):
            if self._semiLogTimes[self._semiLogNextIndex] > t + 1e-15:
                break
            self._semiLogNextIndex += 1

    def _computeSemiLogDt(self, t):
        # debugInPlace()
        if self.dtMode != "semi_log_optional" or self._semiLogTimes is None:
            return np.inf
        self._updateSemiLogIndex(t)
        if self._semiLogNextIndex >= len(self._semiLogTimes):
            return np.inf
        return max(1e-15, float(self._semiLogTimes[self._semiLogNextIndex] - t))

    def _apply_boundary_conditions(self, p, q):
        p_bc = np.asarray(p, dtype=np.float64).copy()
        q_bc = np.asarray(q, dtype=np.float64).copy()
        c_ab, c_ba = self.interfaceCompositions
        p_bc[-1] = c_ab
        q_bc[0] = c_ba
        p_bc[0] = p_bc[1]
        q_bc[-1] = q_bc[-2]
        return p_bc, q_bc

    def _reconstruct_physical_profile(self, p, q, s):
        p_bc, q_bc = self._apply_boundary_conditions(p, q)
        c = np.empty_like(self._z, dtype=np.float64)

        left_mask = self._z <= s
        right_mask = ~left_mask

        if np.any(left_mask):
            u = np.clip(self._z[left_mask] / max(s, 1e-15), 0.0, 1.0)
            c[left_mask] = np.interp(u, self._u_grid, p_bc)
        if np.any(right_mask):
            denom = max(self._R - s, 1e-15)
            v = np.clip((self._z[right_mask] - s) / denom, 0.0, 1.0)
            c[right_mask] = np.interp(v, self._v_grid, q_bc)

        return np.clip(c, self.constraints.minComposition, 1 - self.constraints.minComposition)

    def _phase_temperatures(self, t, s):
        z_left = s * self._u_grid
        z_right = s + (self._R - s) * self._v_grid
        T_left = np.asarray(self.temperatureParameters(z_left[:, np.newaxis], t), dtype=np.float64).reshape(-1)
        T_right = np.asarray(self.temperatureParameters(z_right[:, np.newaxis], t), dtype=np.float64).reshape(-1)
        return T_left, T_right

    def _phase_diffusivities(self, t, p, q, s):
        T_left, T_right = self._phase_temperatures(t, s)
        D_p = np.asarray(
            self.therm.getInterdiffusivity(np.asarray(p, dtype=np.float64), T_left, phase=self.phases[0]),
            dtype=np.float64,
        ).reshape(-1)
        D_q = np.asarray(
            self.therm.getInterdiffusivity(np.asarray(q, dtype=np.float64), T_right, phase=self.phases[1]),
            dtype=np.float64,
        ).reshape(-1)
        D_p[-1] = float(np.squeeze(self.therm.getInterdiffusivity(self.interfaceCompositions[0], T_left[-1], phase=self.phases[0])))
        D_q[0] = float(np.squeeze(self.therm.getInterdiffusivity(self.interfaceCompositions[1], T_right[0], phase=self.phases[1])))
        return D_p, D_q

    def _interface_velocity(self, p, q, s, D_p, D_q):
        c_ab, c_ba = self.interfaceCompositions
        grad_left = (c_ab - p[-2]) / max(s * self._du, 1e-15)
        grad_right = (q[1] - c_ba) / max((self._R - s) * self._dv, 1e-15)
        denom = c_ba - c_ab
        if abs(denom) <= 1e-15:
            velocity = 0.0
        else:
            velocity = (D_p[-1] * grad_left - D_q[0] * grad_right) / denom
        return float(grad_left), float(grad_right), float(velocity)

    def _planar_interface_root_coefficients(self, p, q, s_curr, dt, told, grad_left, grad_right, D_p, D_q):
        """
        Builds the planar Eq. (38) coefficients from the paper's Eq. (12).

        For planar geometry (``lambda = 1``), the radius-curvature terms in Eq. (12)
        collapse to unity and the future interface position satisfies the linear
        root form ``a_1 s^(k+1) + b_1 = 0`` from Eq. (38). Matching terms gives:

        ``a_1 = p_(N+1/2)^k u_(N+1/2) + q_(1+1/2)^k (1 - v_(1+1/2)) + C_AB - C_BA``

        ``b_1 = dt * ( D_A * dp/dx + D_B * dq/dx ) - a_1 * s^k``

        where the half-node concentrations are arithmetic averages between the
        interface Dirichlet values and the adjacent interior nodes. The gradients
        supplied here are physical gradients, so the ``1/s`` and ``1/(R-s)``
        factors from Eq. (12) are already included.
        """
        # debugInPlace()
        c_ab, c_ba = self.interfaceCompositions
        # p_half = 0.5 * (float(p[-2]) + c_ab)
        # q_half = 0.5 * (c_ba + float(q[1]))
        # u_half = 0.5 * (float(self._u_grid[-2]) + float(self._u_grid[-1]))
        # v_half = 0.5 * (float(self._v_grid[0]) + float(self._v_grid[1]))

        # coeff_a = p_half * u_half + q_half * (1.0 - v_half) + c_ab - c_ba
        # flux_term = D_p[-1] * grad_left + D_q[0] * grad_right
        # coeff_b = dt * flux_term - coeff_a * s_curr

        def coeffsFromMathematicaSolve(
                cA, cB, 
                pN, pNPlus1, qPlus1, qPlus2, 
                uN, uNPlus1, vPlus1, vPlus2, 
                DANPlushalf, DBOnePlushalf,
                told, tnew,
                sold,
                R,
                windOrHalfNode, alpha=None, beta=None
        ):
            
            if windOrHalfNode=="halfNode":
                pNPlushalf = (pN + pNPlus1)/2
                qOnePlushalf = (qPlus1 + qPlus2)/2
            elif windOrHalfNode=="wind":
                pNPlushalf = alpha*pN + beta*pNPlus1
                qOnePlushalf = alpha*qPlus1 + beta*qPlus2
                if not ((alpha==1 and beta==0) or (alpha==0 and beta==1)):
                    raise ValueError("alpha and beta must be 0/1 when windOrHalfNode=='wind' ")
            else:
                raise ValueError("windOrHalfNode must be either 'halfNode' or 'wind'")
            uNPlushalf = (uN + uNPlus1)/2
            vOnePlushalf = (vPlus1 + vPlus2)/2

            # a = cA + qOnePlushalf - (cA * uNPlushalf) + (pNPlushalf * uNPlushalf) - (cB * vOnePlushalf) - (qOnePlushalf * vOnePlushalf)
            a = cA - qOnePlushalf - (cA * uNPlushalf) + (pNPlushalf * uNPlushalf) - (cB * vOnePlushalf) + (qOnePlushalf * vOnePlushalf)
            b = ((DANPlushalf*pNPlus1*tnew)/(sold * (-uN+uNPlus1))) + \
            ((DANPlushalf*pN*told)/(sold *(-uN+uNPlus1))) + \
            ((DANPlushalf*pN*tnew)/(sold*uN - sold*uNPlus1)) + \
            ((DANPlushalf*pNPlus1*told)/(sold*uN - sold*uNPlus1)) + \
            (cA*sold * (-1+uNPlushalf)) + \
            -((pNPlushalf*sold*tnew*uNPlushalf)/(tnew-told)) + \
            ((pNPlushalf*sold*told*uNPlushalf)/(tnew-told)) + \
            (-qOnePlushalf*sold * (-1+vOnePlushalf)) + \
            (cB*sold*vOnePlushalf) + \
            -((DBOnePlushalf*qPlus1*tnew)/((R-sold)*(vPlus1-vPlus2))) + \
            -((DBOnePlushalf*qPlus2*told)/((R-sold)*(vPlus1-vPlus2))) + \
            -((DBOnePlushalf*qPlus2*tnew)/((R-sold)*(-vPlus1+vPlus2))) + \
            -((DBOnePlushalf*qPlus1*told)/((R-sold)*(-vPlus1+vPlus2)))

            return a, b, -b/a
        
        ## Original behavior
        # vals = {
        #     "cA": c_ab,
        #     "cB": c_ba,
        #     "pN": p[-2],
        #     "pNPlus1": p[-1],
        #     "qPlus1": q[0],
        #     "qPlus2": q[1],
        #     "uN": self._u_grid[-2],
        #     "uNPlus1": self._u_grid[-1],
        #     "vPlus1": self._v_grid[0],
        #     "vPlus2": self._v_grid[1],
        #     "DANPlushalf": D_p[-1],
        #     "DBOnePlushalf": D_q[0],
        #     # "told": 0.0010539198280561037,
        #     # "tnew": 0.0010539198280561037 + dt,
        #     "told": told,
        #     "tnew": told + dt,
        #     "sold": s_curr,
        #     "R": self._R,
        #     "windOrHalfNode":'halfNode'
        # }

        # a, b, _ = coeffsFromMathematicaSolve(**vals)
        # return float(a), float(b)

        ## HACK: Use the winding coefficients to compute the half-node concentrations instead of the arithmetic average. Want to see if this fixes the mass drift issues
        vals1 = {
            "cA": c_ab,
            "cB": c_ba,
            "pN": p[-2],
            "pNPlus1": p[-1],
            "qPlus1": q[0],
            "qPlus2": q[1],
            "uN": self._u_grid[-2],
            "uNPlus1": self._u_grid[-1],
            "vPlus1": self._v_grid[0],
            "vPlus2": self._v_grid[1],
            "DANPlushalf": D_p[-1],
            "DBOnePlushalf": D_q[0],
            # "told": 0.0010539198280561037,
            # "tnew": 0.0010539198280561037 + dt,
            "told": told,
            "tnew": told + dt,
            "sold": s_curr,
            "R": self._R,
            "windOrHalfNode":'halfNode',
        }

        a1, b1, _ = coeffsFromMathematicaSolve(**vals1)
        
        alpha, beta = self._winding_coefficients_from_interface_step(-b1/a1, s_curr)
        
        vals2 = {
            "cA": c_ab,
            "cB": c_ba,
            "pN": p[-2],
            "pNPlus1": p[-1],
            "qPlus1": q[0],
            "qPlus2": q[1],
            "uN": self._u_grid[-2],
            "uNPlus1": self._u_grid[-1],
            "vPlus1": self._v_grid[0],
            "vPlus2": self._v_grid[1],
            "DANPlushalf": D_p[-1],
            "DBOnePlushalf": D_q[0],
            # "told": 0.0010539198280561037,
            # "tnew": 0.0010539198280561037 + dt,
            "told": told,
            "tnew": told + dt,
            "sold": s_curr,
            "R": self._R,
            "windOrHalfNode":'wind',
            'alpha': alpha,
            'beta': beta,
        }

        a2, b2, _ = coeffsFromMathematicaSolve(**vals2)
        if np.sign((-b2/a2)-s_curr) != np.sign((-b1/a1)-s_curr):
            debugInPlace()
            raise
        return a2, b2
        
        

   

    def _compute_next_interface_position(self, p, q, s_curr, dt, told, grad_left, grad_right, D_p, D_q):
        """
        Returns the next planar interface position using Eq. (12) -> Eq. (38).

        The Olaye/Ojo planar update is a root of the linear polynomial
        ``a_1 s^(k+1) + b_1 = 0``. This is not the generic Euler interface step
        ``s^(k+1) = s^k + dt * v``; the half-cell concentration terms from Eq. (12)
        contribute directly to ``a_1``.
        """
        coeff_a, coeff_b = self._planar_interface_root_coefficients(
            p, q, s_curr, dt, told, grad_left, grad_right, D_p, D_q
        )
        if abs(coeff_a) <= 1e-15:
            raise ValueError("abs(coeff_a) <= 1e-15")
            # s_new = s_curr
            # coeff_a = 1.0
            # coeff_b = -s_curr
        else:
            s_new = -coeff_b / coeff_a
        self._lastInterfaceCoefficients = (float(coeff_a), float(coeff_b))
        return self._clipInterfacePosition(float(s_new), strict=True)

    def _computeDt(self, t, s, D_p, D_q, velocity):
        if self._stepIndex == 0 or self.mainStepMode == "classical_explicit":
            left_len = max(s, 1e-15)
            right_len = max(self._R - s, 1e-15)
            left_dx = left_len * self._du
            right_dx = right_len * self._dv

            diff_terms = []
            if np.any(np.abs(D_p) > 0):
                diff_terms.append(self.constraints.vonNeumannThreshold * left_dx * left_dx / float(np.max(np.abs(D_p))))
            if np.any(np.abs(D_q) > 0):
                diff_terms.append(self.constraints.vonNeumannThreshold * right_dx * right_dx / float(np.max(np.abs(D_q))))
            dt_diff = min(diff_terms) if diff_terms else np.inf

            adv_terms = []
            left_adv = abs(velocity) / left_len
            right_adv = abs(velocity) / right_len
            if left_adv > 0:
                adv_terms.append(self.constraints.movingBoundaryThreshold / left_adv)
            if right_adv > 0:
                adv_terms.append(self.constraints.movingBoundaryThreshold / right_adv)
            dt_move = min(adv_terms) if adv_terms else np.inf
            dt_semi = self._computeSemiLogDt(t)

            
        else:
            dt_diff = -np.inf
            dt_move = -np.inf
            dt_semi = self._computeSemiLogDt(t)

        self._pendingDtDiff = float(dt_diff)
        self._pendingDtMove = float(dt_move)
        self._pendingDtSemi = float(dt_semi)

        
        if self.dtMode == "semi_log_optional":
            if self._stepIndex == 0 or self.mainStepMode == "classical_explicit":
                print(f"dt_semi: {dt_semi}")
                print(f"dt_diff: {dt_diff}")
                print(f"dt_move: {dt_move}")
                if dt_semi >= min(dt_diff, dt_move):
                    debugInPlace()
                    raise ValueError("First time step (self.semi_log_dt) should be smaller than the computed CFL time step for stability.")
                dt_raw = dt_semi
            else:
                dt_raw = dt_semi
        else:
            raise ValueError("I don't think this should be used. I would to need to verify it is done correctly")
            dt_raw = min(dt_diff, dt_move, getattr(self, "deltaTime", np.inf))
        
        
        
        # dt_floor = max(1e-15, float(getattr(self, "deltaTime", 1.0)) * 1e-12)
        if dt_raw < 1e-15:
            debugInPlace()
            raise ValueError("Computed time step is too small: {}".format(dt_raw))
        dt = dt_raw
        self._currentPaperDt = float(dt)
        return dt

    def _winding_coefficients_from_interface_step(self, s_new, s_curr):
        """
        Returns the paper-style winding coefficients for the advective term.

        The paper defines ``a`` and ``b`` as stepwise constants that are either
        1 or 0 depending on the winding direction. We determine that direction
        from the interface motion implied by Eq. (12), i.e. from the sign of
        ``s^{k+1} - s^k``.
        """
        raise RuntimeError("Base winding convention is legacy; use the rework override.")
        ds = float(s_new - s_curr)
        if ds >= 0:
            return 1.0, 0.0
        return 0.0, 1.0

    def _upwind_gradient(self, values, i, step, a, b):
        face_plus = a * values[i] + b * values[i + 1]
        face_minus = a * values[i - 1] + b * values[i]
        return (face_plus - face_minus) / step

    def _advective_term_p(self, p, sdot, s, a, b):
        out = np.zeros_like(p, dtype=np.float64)
        p_bc, _ = self._apply_boundary_conditions(p, self._q_curr if self._q_curr is not None else self._q_prev)
        for i in range(1, len(p_bc) - 1):
            a_i = (self._u_grid[i] * sdot) / max(s, 1e-15)
            grad = self._upwind_gradient(p_bc, i, self._du, a, b)
            out[i] = a_i * grad
        return out

    def _advective_term_q(self, q, sdot, s, a, b):
        out = np.zeros_like(q, dtype=np.float64)
        _, q_bc = self._apply_boundary_conditions(self._p_curr if self._p_curr is not None else self._p_prev, q)
        scale = max(self._R - s, 1e-15)
        for i in range(1, len(q_bc) - 1):
            a_i = ((1.0 - self._v_grid[i]) * sdot) / scale
            grad = self._upwind_gradient(q_bc, i, self._dv, a, b)
            out[i] = a_i * grad
        return out

    def _diffusion_term(self, c, D, spacing_scale, grid_step):
        out = np.zeros_like(c, dtype=np.float64)
        c_bc = np.asarray(c, dtype=np.float64)
        scale2 = max(spacing_scale * spacing_scale, 1e-15)
        h2 = grid_step * grid_step
        for i in range(1, len(c_bc) - 1):
            d_w = 0.5 * (D[i - 1] + D[i])
            d_e = 0.5 * (D[i] + D[i + 1])
            out[i] = (d_e * (c_bc[i + 1] - c_bc[i]) - d_w * (c_bc[i] - c_bc[i - 1])) / (scale2 * h2)
        return out

    def _classical_explicit_phase_update(self, p, q, s, dt, sdot, D_p, D_q, a, b):
        p_bc, q_bc = self._apply_boundary_conditions(p, q)
        p_rhs = self._advective_term_p(p_bc, sdot, s, a, b) + self._diffusion_term(p_bc, D_p, s, self._du)
        q_rhs = self._advective_term_q(q_bc, sdot, s, a, b) + self._diffusion_term(q_bc, D_q, self._R - s, self._dv)

        p_new = p_bc + dt * p_rhs
        q_new = q_bc + dt * q_rhs
        return self._apply_boundary_conditions(p_new, q_new)

    def _lfdf_phase_update(self, p_prev, p_curr, q_prev, q_curr, s, dt, sdot, D_p, D_q, a, b):
        p_old, q_old = self._apply_boundary_conditions(p_curr, q_curr)
        p_new = p_old.copy()
        q_new = q_old.copy()

        adv_p = self._advective_term_p(p_old, sdot, s, a, b)
        adv_q = self._advective_term_q(q_old, sdot, s, a, b)

        left_scale = max(s, 1e-15)
        right_scale = max(self._R - s, 1e-15)
        left_h2 = self._du * self._du
        right_h2 = self._dv * self._dv

        for i in range(1, len(p_old) - 1):
            d_w = 0.5 * (D_p[i - 1] + D_p[i])
            d_e = 0.5 * (D_p[i] + D_p[i + 1])
            sigma = dt * (d_w + d_e) / max(left_scale * left_scale * left_h2, 1e-15)
            numer = (
                (1.0 - sigma) * p_prev[i]
                + 2.0 * dt * adv_p[i]
                + 2.0 * dt * (d_w * p_old[i - 1] + d_e * p_old[i + 1]) / max(left_scale * left_scale * left_h2, 1e-15)
            )
            p_new[i] = numer / (1.0 + sigma)

        for i in range(1, len(q_old) - 1):
            d_w = 0.5 * (D_q[i - 1] + D_q[i])
            d_e = 0.5 * (D_q[i] + D_q[i + 1])
            sigma = dt * (d_w + d_e) / max(right_scale * right_scale * right_h2, 1e-15)
            numer = (
                (1.0 - sigma) * q_prev[i]
                + 2.0 * dt * adv_q[i]
                + 2.0 * dt * (d_w * q_old[i - 1] + d_e * q_old[i + 1]) / max(right_scale * right_scale * right_h2, 1e-15)
            )
            q_new[i] = numer / (1.0 + sigma)

        return self._apply_boundary_conditions(p_new, q_new)

    def _build_physical_fluxes(self, t, p, q, s):
        c = self._reconstruct_physical_profile(p, q, s)
        geom = get_olaye_fd_geometry(self.mesh, s)
        T_nodes = self.temperatureParameters(self.mesh.z, t)
        D_nodes = build_piecewise_diffusivity_nodes(
            therm=self.therm,
            phases=(self.phases[0], self.phases[1]),
            composition=c,
            temperature_nodes=T_nodes,
            geometry=geom,
        )

        flux = np.zeros(len(c) + 1, dtype=np.float64)
        face_diffusivity = 0.5 * (D_nodes[:-1] + D_nodes[1:])
        face_flux = -face_diffusivity * np.diff(c) / float(self.mesh.dz)
        flux[1 : geom.left_index + 1] = face_flux[: geom.left_index]
        flux[geom.right_index + 1 : len(c)] = face_flux[geom.right_index :]

        # flux2 = np.zeros(len(c) + 1, dtype=np.float64)
        # for face in range(1, geom.left_index + 1):
        #     d_face = 0.5 * (D_nodes[face - 1] + D_nodes[face])
        #     flux2[face] = -d_face * (c[face] - c[face - 1]) / float(self.mesh.dz)
        # for face in range(geom.right_index + 1, len(c)):
        #     d_face = 0.5 * (D_nodes[face - 1] + D_nodes[face])
        #     flux2[face] = -d_face * (c[face] - c[face - 1]) / float(self.mesh.dz)
        # flux2[0] = 0.0
        # flux2[-1] = 0.0
        # assert np.abs(flux-flux2).max() < 1e-10

        return flux, c, D_nodes

    def _computeState(self, t, xCurr):
        p_curr = np.asarray(xCurr[0], dtype=np.float64).reshape(-1)
        q_curr = np.asarray(xCurr[1], dtype=np.float64).reshape(-1)
        # if (p_curr[:-1]<0.12).any():
        # if (p_curr[:-1]<0.10223).any():
        #     debugInPlace()
        s_curr = self._clipInterfacePosition(float(xCurr[2]))
        p_curr, q_curr = self._apply_boundary_conditions(p_curr, q_curr)

        D_p, D_q = self._phase_diffusivities(t, p_curr, q_curr, s_curr)
        grad_left, grad_right, velocity = self._interface_velocity(p_curr, q_curr, s_curr, D_p, D_q)
        dt = self._computeDt(t, s_curr, D_p, D_q, velocity)
        self._currdt = float(dt)

        
        s_new = self._compute_next_interface_position(p_curr, q_curr, s_curr, dt, t, grad_left, grad_right, D_p, D_q)
        sdot = (s_new - s_curr) / dt
        a_wind, b_wind = self._winding_coefficients_from_interface_step(s_new, s_curr)
        self._lastWindingAB = (float(a_wind), float(b_wind))

        if self._stepIndex == 0 or self.mainStepMode == "classical_explicit":
            p_new, q_new = self._classical_explicit_phase_update(p_curr, q_curr, s_curr, dt, sdot, D_p, D_q, a_wind, b_wind)
            self._lastStepScheme = "bootstrap_classical_explicit" if self._stepIndex == 0 else "classical_explicit"
        else:
            p_new, q_new = self._lfdf_phase_update(self._p_prev, p_curr, self._q_prev, q_curr, s_curr, dt, sdot, D_p, D_q, a_wind, b_wind)
            self._lastStepScheme = "leapfrog_dufort_frankel"

        p_new = np.clip(p_new, self.constraints.minComposition, 1 - self.constraints.minComposition)
        q_new = np.clip(q_new, self.constraints.minComposition, 1 - self.constraints.minComposition)
        p_new, q_new = self._apply_boundary_conditions(p_new, q_new)

        dpdt = (p_new - p_curr) / dt
        dqdt = (q_new - q_curr) / dt

        fluxes, _, _ = self._build_physical_fluxes(t, p_curr, q_curr, s_curr)
        self._lastFluxes = fluxes
        self._lastInterfaceFluxes = (-D_p[-1] * grad_left, -D_q[0] * grad_right)
        self._lastInterfaceVelocity = float(sdot)
        self._t_prev = t
        return dpdt, dqdt, float(sdot)

    def getdXdt(self, t, xCurr):
        dpdt, dqdt, velocity = self._computeState(t, xCurr)
        return [dpdt, dqdt, velocity]

    def getDt(self, dXdt):
        if np.isfinite(self._currdt) and self._currdt > 0:
            return self._currdt
        return getattr(self, "deltaTime", np.inf)

    def getTransformedState(self, time=None):
        """
        Returns the recorded transformed left/right phase state at an exact time.
        """
        return self.getTransformedStateLeft(time), self.getTransformedStateRight(time)

    def getTransformedStateLeft(self, time=None):
        """
        Returns the recorded left transformed composition vector ``p``.
        """
        if self.pData is None:
            raise ValueError("Transformed left-state history is not initialized.")
        return self.pData.y(time)

    def getTransformedStateRight(self, time=None):
        """
        Returns the recorded right transformed composition vector ``q``.
        """
        if self.qData is None:
            raise ValueError("Transformed right-state history is not initialized.")
        return self.qData.y(time)

    def postProcess(self, time, x):
        GenericModel.postProcess(self, time, x)

        p = np.asarray(x[0], dtype=np.float64).reshape(-1)
        q = np.asarray(x[1], dtype=np.float64).reshape(-1)
        s = self._clipInterfacePosition(float(x[2]))
        p, q = self._apply_boundary_conditions(p, q)
        physical = self._reconstruct_physical_profile(p, q, s)[:, np.newaxis]

        
        self.data.record(time, physical)
        self.interfaceData.record(time, s)
        self.pData.record(time, p)
        self.qData.record(time, q)
        
        averageConc = self.checkMassIntegral(p=p, q=q, s=s)
        self.concData.record(time, averageConc)
        averageConc_alt = self.getTotalInventory() / self._R
        self.concData_alt.record(time, averageConc_alt)

        self._p_prev = self._p_curr.copy()
        self._q_prev = self._q_curr.copy()
        self._p_curr = p.copy()
        self._q_curr = q.copy()
        self._s_prev = float(self._s_curr)
        self._s_curr = float(s)
        self._stepIndex += 1

        self.updateCoupledModels()
        return [self._p_curr.copy(), self._q_curr.copy(), self._s_curr], False

    def postSolve(self):
        self.data.finalize()
        self.interfaceData.finalize()
        self.concData.finalize()
        self.concData_alt.finalize()
        if self.pData is not None:
            self.pData.finalize()
        if self.qData is not None:
            self.qData.finalize()

    def getInterfacePosition(self, time=None):
        return self.interfaceData.y(time)

    def getTotalInventory(self, time=None):
        composition = np.asarray(self.data.y(time), dtype=np.float64).reshape(-1)
        interface_position = self.getInterfacePosition(time)
        return integrate_binary_olaye_fd_profile(
            z=self.mesh.z,
            composition=composition,
            interface_position=interface_position,
            interface_compositions=self.interfaceCompositions,
        )

    def getTotalMass(self, time=None):
        return self.getTotalInventory(time=time)

    def getFluxes(self, t, xCurr):
        self._computeState(t, xCurr)
        return self._lastFluxes[:, np.newaxis]

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

    def checkConservation(self, tolerance: float, time=None):
        """
        Checks absolute inventory drift against the provided tolerance.

        Returns
        -------
        drift : float
            Absolute inventory difference from stored initial inventory.
        """
        if self._initialInventory is None:
            raise ValueError("Model must be setup before conservation checks.")
        drift = abs(self.getTotalInventory(time=time) - self._initialInventory)
        if drift > float(tolerance):
            warnings.warn(
                f"Olaye inventory drift {drift:.3e} exceeded tolerance {float(tolerance):.3e}.",
                RuntimeWarning,
                stacklevel=2,
            )
        return drift
    
    def checkMassIntegral(self, p, q, s):
        # debugInPlace()
        assert abs((((len(p)-2) * self._du) + self._du/2 + self._du/2)-1)<1e-10
        assert abs((((len(q)-2) * self._dv) + self._dv/2 + self._dv/2)-1)<1e-10

        left_mass = s * ( (self._du/2)*p[0] + (self._du*p[1:-1]).sum() + (self._du/2)*p[-1] )
        right_mass = (self._R - s) * ( (self._dv/2)*q[0] + (self._dv*q[1:-1]).sum() + (self._dv/2)*q[-1] )
        total_mass = left_mass + right_mass
        total_conc = total_mass/self._R
        return total_conc
