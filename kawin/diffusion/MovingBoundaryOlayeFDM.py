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
from kawin.thermo.Mobility import interstitials


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
        semi_log_points: int = 200,
        semi_log_t0_fraction: float = 1e-6,
        constraints=None,
        record=False,
    ):
        self.initialInterfacePosition = float(interfacePosition)
        self.interfaceData = _ScalarHistory(record)
        self.interfaceCompositions = tuple(float(v) for v in interface_compositions)
        self.firstStepMode = str(first_step_mode)
        self.mainStepMode = str(main_step_mode)
        self.dtMode = str(dt_mode)
        self.geometry = str(geometry)
        self.semiLogPoints = int(semi_log_points)
        self.semiLogT0Fraction = float(semi_log_t0_fraction)

        self._currdt = np.inf
        self._pendingDtDiff = np.inf
        self._pendingDtMove = np.inf
        self._pendingDtSemi = np.inf
        self._lastStepScheme = None
        self._lastFluxes = None
        self._lastInterfaceFluxes = (0.0, 0.0)
        self._lastInterfaceVelocity = 0.0
        self._stepIndex = 0

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
        if self.geometry not in {"planar", "cylindrical", "spherical"}:
            raise ValueError("geometry must be one of ['planar', 'cylindrical', 'spherical'].")
        if self.geometry != "planar":
            raise NotImplementedError(
                f"geometry='{self.geometry}' is documented but not implemented yet for "
                "MovingBoundaryOlayeFD1DModel."
            )
        if self.semiLogPoints < 3:
            raise ValueError("semi_log_points must be at least 3.")
        if not (0 < self.semiLogT0Fraction < 1):
            raise ValueError("semi_log_t0_fraction must be between 0 and 1.")
        c_ab, c_ba = self.interfaceCompositions
        if c_ba <= c_ab:
            raise ValueError("Expected interface_compositions=(C_AB, C_BA) with C_BA > C_AB.")
        self.initialInterfacePosition = self._clipInterfacePosition(self.initialInterfacePosition, strict=True)

    def _clipInterfacePosition(self, interface_position: float, strict: bool = True) -> float:
        z = np.ravel(self.mesh.z)
        eps = max(float(self.mesh.dz) * 1e-8, 1e-14)
        lower = float(z[0] + eps)
        upper = float(z[-1] - eps)
        if strict and not (lower < interface_position < upper):
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
        t0_rel = max(self.semiLogT0Fraction * simTime, 1e-15)
        rel_times = np.geomspace(t0_rel, simTime, self.semiLogPoints)
        self._semiLogTimes = currTime + rel_times
        self._semiLogNextIndex = 0

    def reset(self):
        super().reset()
        self.interfaceData.reset()
        self.interfaceData.record(0, self.initialInterfacePosition)
        self._currdt = np.inf
        self._pendingDtDiff = np.inf
        self._pendingDtMove = np.inf
        self._pendingDtSemi = np.inf
        self._lastStepScheme = None
        self._lastFluxes = None
        self._lastInterfaceFluxes = (0.0, 0.0)
        self._lastInterfaceVelocity = 0.0
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

        n_left = geom.left_index + 2
        n_right = len(c0) - geom.right_index + 1
        if n_left < 3 or n_right < 3:
            raise ValueError("Each phase must retain at least three transformed grid points.")

        self._u_grid = np.linspace(0.0, 1.0, n_left, dtype=np.float64)
        self._v_grid = np.linspace(0.0, 1.0, n_right, dtype=np.float64)
        self._du = float(self._u_grid[1] - self._u_grid[0])
        self._dv = float(self._v_grid[1] - self._v_grid[0])
        self._phase_counts = (n_left, n_right)

        self._p_curr, self._q_curr = self._initialize_transformed_state(c0, s0)
        self._p_prev = self._p_curr.copy()
        self._q_prev = self._q_curr.copy()
        self._s_curr = s0
        self._s_prev = s0

        self.data.currentY = self._reconstruct_physical_profile(self._p_curr, self._q_curr, s0)[:, np.newaxis]
        self._initialInventory = self.getTotalInventory(time=0)

    def _initialize_transformed_state(self, composition, interface_position):
        c = np.asarray(composition, dtype=np.float64).reshape(-1)
        c_ab, c_ba = self.interfaceCompositions

        z_left = np.clip(interface_position * self._u_grid, self._z[0], interface_position)
        right_span = max(self._R - interface_position, 1e-15)
        z_right = interface_position + right_span * self._v_grid

        p = np.interp(z_left[:-1], self._z, c)
        q = np.interp(z_right[1:], self._z, c)

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

    def _computeDt(self, t, s, D_p, D_q, velocity):
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
        self._pendingDtDiff = float(dt_diff)
        self._pendingDtMove = float(dt_move)
        self._pendingDtSemi = float(dt_semi)

        if self.dtMode == "semi_log_optional":
            dt_raw = min(dt_diff, dt_move, dt_semi, getattr(self, "deltaTime", np.inf))
        else:
            dt_raw = min(dt_diff, dt_move, getattr(self, "deltaTime", np.inf))

        dt_floor = max(1e-15, float(getattr(self, "deltaTime", 1.0)) * 1e-8)
        return max(float(dt_raw), dt_floor)

    def _advective_term_p(self, p, sdot, s):
        out = np.zeros_like(p, dtype=np.float64)
        p_bc, _ = self._apply_boundary_conditions(p, self._q_curr if self._q_curr is not None else self._q_prev)
        for i in range(1, len(p_bc) - 1):
            a_i = (self._u_grid[i] * sdot) / max(s, 1e-15)
            if a_i >= 0:
                grad = (p_bc[i] - p_bc[i - 1]) / self._du
            else:
                grad = (p_bc[i + 1] - p_bc[i]) / self._du
            out[i] = a_i * grad
        return out

    def _advective_term_q(self, q, sdot, s):
        out = np.zeros_like(q, dtype=np.float64)
        _, q_bc = self._apply_boundary_conditions(self._p_curr if self._p_curr is not None else self._p_prev, q)
        scale = max(self._R - s, 1e-15)
        for i in range(1, len(q_bc) - 1):
            a_i = -((1.0 - self._v_grid[i]) * sdot) / scale
            if a_i >= 0:
                grad = (q_bc[i] - q_bc[i - 1]) / self._dv
            else:
                grad = (q_bc[i + 1] - q_bc[i]) / self._dv
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

    def _classical_explicit_phase_update(self, p, q, s, dt, sdot, D_p, D_q):
        p_bc, q_bc = self._apply_boundary_conditions(p, q)
        p_rhs = self._advective_term_p(p_bc, sdot, s) + self._diffusion_term(p_bc, D_p, s, self._du)
        q_rhs = self._advective_term_q(q_bc, sdot, s) + self._diffusion_term(q_bc, D_q, self._R - s, self._dv)

        p_new = p_bc + dt * p_rhs
        q_new = q_bc + dt * q_rhs
        return self._apply_boundary_conditions(p_new, q_new)

    def _lfdf_phase_update(self, p_prev, p_curr, q_prev, q_curr, s, dt, sdot, D_p, D_q):
        p_old, q_old = self._apply_boundary_conditions(p_curr, q_curr)
        p_new = p_old.copy()
        q_new = q_old.copy()

        adv_p = self._advective_term_p(p_old, sdot, s)
        adv_q = self._advective_term_q(q_old, sdot, s)

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
        for face in range(1, geom.left_index + 1):
            d_face = 0.5 * (D_nodes[face - 1] + D_nodes[face])
            flux[face] = -d_face * (c[face] - c[face - 1]) / float(self.mesh.dz)
        for face in range(geom.right_index + 1, len(c)):
            d_face = 0.5 * (D_nodes[face - 1] + D_nodes[face])
            flux[face] = -d_face * (c[face] - c[face - 1]) / float(self.mesh.dz)
        flux[0] = 0.0
        flux[-1] = 0.0
        return flux, c, D_nodes

    def _computeState(self, t, xCurr):
        p_curr = np.asarray(xCurr[0], dtype=np.float64).reshape(-1)
        q_curr = np.asarray(xCurr[1], dtype=np.float64).reshape(-1)
        s_curr = self._clipInterfacePosition(float(xCurr[2]))
        p_curr, q_curr = self._apply_boundary_conditions(p_curr, q_curr)

        D_p, D_q = self._phase_diffusivities(t, p_curr, q_curr, s_curr)
        grad_left, grad_right, velocity = self._interface_velocity(p_curr, q_curr, s_curr, D_p, D_q)
        dt = self._computeDt(t, s_curr, D_p, D_q, velocity)
        self._currdt = float(dt)

        s_new = self._clipInterfacePosition(s_curr + dt * velocity, strict=True)
        sdot = (s_new - s_curr) / dt

        if self._stepIndex == 0 or self.mainStepMode == "classical_explicit":
            p_new, q_new = self._classical_explicit_phase_update(p_curr, q_curr, s_curr, dt, sdot, D_p, D_q)
            self._lastStepScheme = "classical_explicit"
        else:
            p_new, q_new = self._lfdf_phase_update(self._p_prev, p_curr, self._q_prev, q_curr, s_curr, dt, sdot, D_p, D_q)
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
        return dpdt, dqdt, float(sdot)

    def getdXdt(self, t, xCurr):
        dpdt, dqdt, velocity = self._computeState(t, xCurr)
        return [dpdt, dqdt, velocity]

    def getDt(self, dXdt):
        if np.isfinite(self._currdt) and self._currdt > 0:
            return self._currdt
        return getattr(self, "deltaTime", np.inf)

    def postProcess(self, time, x):
        GenericModel.postProcess(self, time, x)

        p = np.asarray(x[0], dtype=np.float64).reshape(-1)
        q = np.asarray(x[1], dtype=np.float64).reshape(-1)
        s = self._clipInterfacePosition(float(x[2]))
        p, q = self._apply_boundary_conditions(p, q)
        physical = self._reconstruct_physical_profile(p, q, s)[:, np.newaxis]

        self.data.record(time, physical)
        self.interfaceData.record(time, s)

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
