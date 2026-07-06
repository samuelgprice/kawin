import warnings

import numpy as np

from kawin.GenericModel import GenericModel
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.diffusion.mesh import CartesianFD1D, MixedBoundary1D, PeriodicBoundary1D
from kawin.diffusion.mesh.MovingBoundaryIllingworthFD1D import (
    flatten_1d_coordinates,
    integrate_planar_transformed_profile,
    reconstruct_planar_transformed_profile,
    solve_illingworth_tridiagonal,
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
        if self.recordInterval > 0 and self.N >= 0 and np.isclose(self._time[self.N], self.currentTime, rtol=0.0, atol=1e-14):
            self._y = self._y[: self.N + 1]
            self._time = self._time[: self.N + 1]
            return
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

class MovingBoundaryIllingworthFD1DModel(DiffusionModel):
    """
    Binary planar moving-boundary model following Illingworth and Golosnoy.

    This v1 implementation ports the planar, constant-diffusivity implicit
    front-fixing algorithm from the authors' MAP C++ code. Each step solves a
    fixed-point problem for the future interface position and the two future
    transformed concentration profiles; the phase solves are tri-diagonal and
    the interface equation uses the paper's conservative planar balance.

    Notes
    -----
    Only planar geometry is supported. Cylindrical and spherical equations are
    intentionally excluded until their coefficient paths are separately tested.
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
        time_step: float,
        geometry: str = "planar",
        phase_a_nodes: int | None = None,
        phase_b_nodes: int | None = None,
        tolerance: float = 1e-8,
        max_iterations: int = 100,
        constraints=None,
        record=False,
    ):
        self.initialInterfacePosition = float(interfacePosition)
        self.interfaceCompositions = tuple(float(v) for v in interface_compositions)
        self.timeStep = float(time_step)
        self.geometry = str(geometry)
        self.phaseANodes = None if phase_a_nodes is None else int(phase_a_nodes)
        self.phaseBNodes = None if phase_b_nodes is None else int(phase_b_nodes)
        self.tolerance = float(tolerance)
        self.maxIterations = int(max_iterations)

        self.interfaceData = _ScalarHistory(record)
        self.concData = _ScalarHistory(record)
        self.pData = None
        self.qData = None
        self._currdt = np.inf
        self._lastImplicitIterations = 0
        self._lastImplicitError = np.nan
        self._nearFinalNoop = False
        self._initialInventory = None

        self._z = None
        self._R = None
        self._u_grid = None
        self._v_grid = None
        self._p_curr = None
        self._q_curr = None
        self._s_curr = None
        self._s_old = None
        self._D_left = None
        self._D_right = None

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
            raise TypeError("MovingBoundaryIllingworthFD1DModel requires a CartesianFD1D mesh.")
        if len(self.allElements) != 2 or self.mesh.numResponses != 1:
            raise ValueError("MovingBoundaryIllingworthFD1DModel currently supports only binary systems.")
        if any(e in interstitials for e in self.allElements):
            raise ValueError("MovingBoundaryIllingworthFD1DModel supports only substitutional systems.")
        if len(self.phases) != 2:
            raise ValueError("MovingBoundaryIllingworthFD1DModel requires exactly two explicit phases.")
        if isinstance(getattr(self.mesh, "boundaryConditions", None), PeriodicBoundary1D):
            raise ValueError("Periodic boundary conditions are not supported.")
        if self.geometry != "planar":
            raise NotImplementedError("MovingBoundaryIllingworthFD1DModel currently implements only planar geometry.")
        if not np.isfinite(self.timeStep) or self.timeStep <= 0:
            raise ValueError("time_step must be a positive finite value.")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be a positive finite value.")
        if self.maxIterations < 2:
            raise ValueError("max_iterations must be at least 2.")
        if self.phaseANodes is not None and self.phaseANodes < 3:
            raise ValueError("phase_a_nodes must be at least 3 when specified.")
        if self.phaseBNodes is not None and self.phaseBNodes < 3:
            raise ValueError("phase_b_nodes must be at least 3 when specified.")
        self.initialInterfacePosition = self._clipInterfacePosition(self.initialInterfacePosition, strict=True)

    def _clipInterfacePosition(self, interface_position: float, strict: bool = True) -> float:
        z = flatten_1d_coordinates(self.mesh.z)
        eps = max(float(z[-1] - z[0]) * 1e-14, 1e-14)
        lower = float(z[0] + eps)
        upper = float(z[-1] - eps)
        if strict and not (lower < interface_position < upper):
            raise ValueError("Interface position must lie strictly inside the FD domain.")
        return float(np.clip(interface_position, lower, upper))

    def _getBoundaryConditions(self):
        bc = getattr(self.mesh, "boundaryConditions", None)
        if bc is None:
            bc = MixedBoundary1D(self.mesh.responses)
            self.mesh.boundaryConditions = bc
        return bc

    def reset(self):
        super().reset()
        self.interfaceData.reset()
        self.interfaceData.record(0, self.initialInterfacePosition)
        self.concData.reset()
        self.pData = None
        self.qData = None
        self._currdt = np.inf
        self._lastImplicitIterations = 0
        self._lastImplicitError = np.nan
        self._nearFinalNoop = False
        self._initialInventory = None
        self._z = None
        self._R = None
        self._u_grid = None
        self._v_grid = None
        self._p_curr = None
        self._q_curr = None
        self._s_curr = None
        self._s_old = None
        self._D_left = None
        self._D_right = None
        if hasattr(self, "mesh") and self.mesh is not None:
            self._validateModelConfiguration()

    def setup(self):
        super().setup()
        self._validateModelConfiguration()
        self._z = flatten_1d_coordinates(self.mesh.z).astype(np.float64)
        if not np.isclose(self._z[0], 0.0):
            raise ValueError("MovingBoundaryIllingworthFD1DModel expects a 1D domain starting at 0.")
        self._R = float(self._z[-1] - self._z[0])

        c0 = np.asarray(self.data.currentY, dtype=np.float64).reshape(-1)
        s0 = float(self.interfaceData.currentY)
        n_left = self.phaseANodes if self.phaseANodes is not None else max(3, int(np.searchsorted(self._z, s0, side="right")))
        n_right = self.phaseBNodes if self.phaseBNodes is not None else max(3, len(self._z) - int(np.searchsorted(self._z, s0, side="left")))

        self._u_grid = np.linspace(0.0, 1.0, int(n_left), dtype=np.float64)
        self._v_grid = np.linspace(0.0, 1.0, int(n_right), dtype=np.float64)
        self.pData = _VectorHistory(int(n_left), self.interfaceData.recordInterval)
        self.qData = _VectorHistory(int(n_right), self.interfaceData.recordInterval)
        self._p_curr, self._q_curr = self._initialize_transformed_state(c0, s0)
        self._s_curr = s0
        self._s_old = s0
        self._D_left, self._D_right = self._constant_phase_diffusivities()
        self.pData.record(0, self._p_curr)
        self.qData.record(0, self._q_curr)

        physical = self._reconstruct_physical_profile(self._p_curr, self._q_curr, s0)[:, np.newaxis]
        self.data.currentY = physical
        self._initialInventory = self.getTotalInventoryFromState(self._p_curr, self._q_curr, self._s_curr)
        self.concData.record(0, self.checkMassIntegral(self._p_curr, self._q_curr, self._s_curr))

    def _initialize_transformed_state(self, composition, interface_position):
        c = np.asarray(composition, dtype=np.float64).reshape(-1)
        c_left_int, c_right_int = self.interfaceCompositions

        z_left = interface_position * self._u_grid
        z_right = interface_position + (self._R - interface_position) * self._v_grid
        left_mask = self._z <= interface_position
        right_mask = self._z >= interface_position

        if not np.any(left_mask) or not np.any(right_mask):
            raise ValueError("Initial interface leaves an empty phase.")

        p = np.interp(z_left, self._z[left_mask], c[left_mask])
        q = np.interp(z_right, self._z[right_mask], c[right_mask])
        p[-1] = c_left_int
        q[0] = c_right_int
        return p.astype(np.float64), q.astype(np.float64)

    def _constant_phase_diffusivities(self):
        values = []
        for phase, c in zip(self.phases, self.interfaceCompositions):
            x = np.clip(
                np.asarray([c - 0.05, c, c + 0.05], dtype=np.float64),
                self.constraints.minComposition,
                1.0 - self.constraints.minComposition,
            )
            T = np.asarray(self.temperatureParameters(np.zeros((3, 1)), 0), dtype=np.float64).reshape(-1)
            if T.size == 1:
                T = np.full(3, float(T[0]), dtype=np.float64)
            D = np.asarray(self.therm.getInterdiffusivity(x, T, phase=phase), dtype=np.float64)
            if D.ndim == 0:
                D = np.full(3, float(D), dtype=np.float64)
            D = np.ravel(D)
            if not np.all(np.isfinite(D)):
                raise ValueError("Illingworth model requires finite constant diffusivities.")
            if not np.allclose(D, D[0], rtol=1e-12, atol=0.0):
                raise ValueError("Illingworth model currently supports only constant diffusivity per phase.")
            values.append(float(D[0]))
        return values[0], values[1]

    def solve(self, simTime, iterator=explicitEulerIterator, verbose=False, vIt=10, minDtFrac=1e-8, maxDtFrac=1):
        """
        Solves the implicit recurrence using a single-stage Euler wrapper.

        Multi-stage iterators are rejected because the implicit Illingworth step
        is not a continuously valid right-hand side for RK intermediate states.
        """
        if iterator is not explicitEulerIterator:
            raise ValueError("MovingBoundaryIllingworthFD1DModel supports only explicitEulerIterator.")
        return super().solve(simTime, iterator=iterator, verbose=verbose, vIt=vIt, minDtFrac=minDtFrac, maxDtFrac=maxDtFrac)

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

    def _compute_dt(self, t):
        remaining = getattr(self, "finalTime", np.inf) - float(t)
        self._nearFinalNoop = bool(np.isfinite(remaining) and 0 < remaining <= self.timeStep * 1e-10)
        if self._nearFinalNoop:
            self._currdt = self.timeStep
            return self.timeStep
        dt = min(self.timeStep, remaining)
        if not np.isfinite(dt) or dt <= 0:
            dt = self.timeStep
        self._currdt = float(dt)
        return float(dt)

    def _new_interface_planar(self, p_future, q_future, s, old_s, future_s, dt, pass_number):
        c_left_int, c_right_int = self.interfaceCompositions
        if pass_number == 0:
            velocity_sign_probe = s - old_s
        else:
            velocity_sign_probe = future_s - s

        diff_l = (c_left_int - p_future[-2]) / (1.0 - self._u_grid[-2])
        diff_l = diff_l * self._D_left / future_s
        diff_r = (q_future[1] - c_right_int) / self._v_grid[1]
        diff_r = diff_r * self._D_right / (self._R - future_s)
        rhs = (diff_r - diff_l) * dt

        if velocity_sign_probe >= 0:
            lhs = c_left_int
            lhs = lhs - q_future[1] * (1.0 - self._v_grid[1] / 2.0)
            lhs = lhs - c_right_int * self._v_grid[1] / 2.0
        else:
            lhs = p_future[-2] * (0.5 + self._u_grid[-2] / 2.0)
            lhs = lhs + c_left_int * (0.5 - self._u_grid[-2] / 2.0)
            lhs = lhs - c_right_int
        if abs(lhs) <= 1e-300:
            raise ZeroDivisionError("Illingworth interface update encountered a zero denominator.")
        return float(s + rhs / lhs)

    def _new_concentration_left_planar(self, p, s, future_s, dt):
        n = len(p)
        lo = np.zeros(n, dtype=np.float64)
        diag = np.zeros(n, dtype=np.float64)
        up = np.zeros(n, dtype=np.float64)
        rhs = np.zeros(n, dtype=np.float64)
        tmpA = self._D_left * dt / future_s
        tmpB = future_s - s

        if future_s >= s:
            diag[0] = -tmpA / self._u_grid[1] - future_s * self._u_grid[1] / 2.0
            up[0] = tmpA / self._u_grid[1] + tmpB * self._u_grid[1] / 2.0
            rhs[0] = -p[0] * s * self._u_grid[1] / 2.0
            for i in range(1, n - 1):
                left_diff = self._u_grid[i] - self._u_grid[i - 1]
                left_sum = self._u_grid[i] + self._u_grid[i - 1]
                right_diff = self._u_grid[i + 1] - self._u_grid[i]
                right_sum = self._u_grid[i + 1] + self._u_grid[i]
                lo[i] = tmpA / left_diff
                diag[i] = -tmpA * (1.0 / left_diff + 1.0 / right_diff) - tmpB * left_sum / 2.0
                diag[i] = diag[i] - future_s * (right_sum - left_sum) / 2.0
                up[i] = tmpA / right_diff + tmpB * right_sum / 2.0
                rhs[i] = -s * p[i] * (right_sum - left_sum) / 2.0
        else:
            diag[0] = -tmpA / self._u_grid[1] + tmpB * self._u_grid[1] / 2.0 - future_s * self._u_grid[1] / 2.0
            up[0] = tmpA / self._u_grid[1]
            rhs[0] = -p[0] * s * self._u_grid[1] / 2.0
            for i in range(1, n - 1):
                left_diff = self._u_grid[i] - self._u_grid[i - 1]
                left_sum = self._u_grid[i] + self._u_grid[i - 1]
                right_diff = self._u_grid[i + 1] - self._u_grid[i]
                right_sum = self._u_grid[i + 1] + self._u_grid[i]
                lo[i] = tmpA / left_diff - tmpB * left_sum / 2.0
                diag[i] = -tmpA * (1.0 / left_diff + 1.0 / right_diff) + tmpB * right_sum / 2.0
                diag[i] = diag[i] - future_s * (right_sum - left_sum) / 2.0
                up[i] = tmpA / right_diff
                rhs[i] = -s * p[i] * (right_sum - left_sum) / 2.0

        diag[-1] = -1.0
        rhs[-1] = -self.interfaceCompositions[0]
        return solve_illingworth_tridiagonal(lo, diag, up, rhs)

    def _new_concentration_right_planar(self, q, s, future_s, dt):
        n = len(q)
        lo = np.zeros(n, dtype=np.float64)
        diag = np.zeros(n, dtype=np.float64)
        up = np.zeros(n, dtype=np.float64)
        rhs = np.zeros(n, dtype=np.float64)
        tmpA = self._D_right * dt / (self._R - future_s)
        tmpB = future_s - s

        diag[0] = -1.0
        rhs[0] = -self.interfaceCompositions[1]

        if future_s >= s:
            for i in range(1, n - 1):
                left_diff = self._v_grid[i] - self._v_grid[i - 1]
                left_sum = self._v_grid[i] + self._v_grid[i - 1]
                right_diff = self._v_grid[i + 1] - self._v_grid[i]
                right_sum = self._v_grid[i + 1] + self._v_grid[i]
                lo[i] = tmpA / left_diff
                diag[i] = -tmpA * (1.0 / right_diff + 1.0 / left_diff) - tmpB * (1.0 - left_sum / 2.0)
                diag[i] = diag[i] - (self._R - future_s) * (right_sum - left_sum) / 2.0
                up[i] = tmpA / right_diff + tmpB * (1.0 - right_sum / 2.0)
                rhs[i] = -(self._R - s) * q[i] * (right_sum - left_sum) / 2.0

            tmp = self._v_grid[-2]
            lo[-1] = tmpA / (1.0 - tmp)
            diag[-1] = -tmpA / (1.0 - tmp) - tmpB * (1.0 - (1.0 + tmp) / 2.0)
            diag[-1] = diag[-1] - (self._R - future_s) * (1.0 - tmp) / 2.0
            rhs[-1] = -q[-1] * (self._R - s) * (1.0 - tmp) / 2.0
        else:
            for i in range(1, n - 1):
                left_diff = self._v_grid[i] - self._v_grid[i - 1]
                left_sum = self._v_grid[i] + self._v_grid[i - 1]
                right_diff = self._v_grid[i + 1] - self._v_grid[i]
                right_sum = self._v_grid[i + 1] + self._v_grid[i]
                lo[i] = tmpA / left_diff - tmpB * (1.0 - left_sum / 2.0)
                diag[i] = -tmpA * (1.0 / right_diff + 1.0 / left_diff) + tmpB * (1.0 - right_sum / 2.0)
                diag[i] = diag[i] - (self._R - future_s) * (right_sum - left_sum) / 2.0
                up[i] = tmpA / right_diff
                rhs[i] = -(self._R - s) * q[i] * (right_sum - left_sum) / 2.0

            tmp = self._v_grid[-2]
            lo[-1] = tmpA / (1.0 - tmp) - tmpB * (1.0 - (1.0 + tmp) / 2.0)
            diag[-1] = -tmpA / (1.0 - tmp) - (self._R - future_s) * (1.0 - tmp) / 2.0
            rhs[-1] = -q[-1] * (self._R - s) * (1.0 - tmp) / 2.0

        return solve_illingworth_tridiagonal(lo, diag, up, rhs)

    def _take_implicit_step_planar(self, p, q, s, old_s, dt):
        p_future = np.asarray(p, dtype=np.float64).copy()
        q_future = np.asarray(q, dtype=np.float64).copy()
        future_s = float(s)
        error = np.inf

        for count in range(self.maxIterations):
            previous_future_s = future_s
            future_s = self._new_interface_planar(p_future, q_future, s, old_s, future_s, dt, count)
            if not np.isfinite(future_s):
                raise ValueError("Illingworth interface update produced a non-finite position.")
            p_future = self._new_concentration_left_planar(p, s, future_s, dt)
            q_future = self._new_concentration_right_planar(q, s, future_s, dt)
            error = abs(previous_future_s - future_s)
            if error <= self.tolerance and count >= 1:
                self._lastImplicitIterations = count + 1
                self._lastImplicitError = float(error)
                return p_future, q_future, self._clipInterfacePosition(future_s, strict=True)

        raise RuntimeError(
            "Illingworth implicit step failed to converge within "
            f"{self.maxIterations} iterations; final interface error was {error:.3e}."
        )

    def _reconstruct_physical_profile(self, p, q, s):
        return reconstruct_planar_transformed_profile(
            z=self._z,
            p=p,
            q=q,
            s=s,
            domain_length=self._R,
            u=self._u_grid,
            v=self._v_grid,
        )

    def getdXdt(self, t, xCurr):
        p = np.asarray(xCurr[0], dtype=np.float64).reshape(-1)
        q = np.asarray(xCurr[1], dtype=np.float64).reshape(-1)
        s = self._clipInterfacePosition(float(xCurr[2]), strict=True)
        p[-1] = self.interfaceCompositions[0]
        q[0] = self.interfaceCompositions[1]
        dt = self._compute_dt(t)
        if self._nearFinalNoop:
            return [np.zeros_like(p), np.zeros_like(q), 0.0]
        p_new, q_new, s_new = self._take_implicit_step_planar(p, q, s, self._s_old, dt)
        return [(p_new - p) / dt, (q_new - q) / dt, (s_new - s) / dt]

    def getDt(self, dXdt):
        if np.isfinite(self._currdt) and self._currdt > 0:
            return self._currdt
        return self.timeStep

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
        if self._nearFinalNoop:
            self.currentTime = time
            self._nearFinalNoop = False
            return [self._p_curr.copy(), self._q_curr.copy(), self._s_curr], True
        GenericModel.postProcess(self, time, x)
        p = np.asarray(x[0], dtype=np.float64).reshape(-1).copy()
        q = np.asarray(x[1], dtype=np.float64).reshape(-1).copy()
        s = self._clipInterfacePosition(float(x[2]), strict=True)
        p[-1] = self.interfaceCompositions[0]
        q[0] = self.interfaceCompositions[1]

        physical = self._reconstruct_physical_profile(p, q, s)[:, np.newaxis]
        self.data.record(time, physical)
        self.interfaceData.record(time, s)
        self.pData.record(time, p)
        self.qData.record(time, q)
        self.concData.record(time, self.checkMassIntegral(p, q, s))

        self._s_old = float(self._s_curr)
        self._p_curr = p
        self._q_curr = q
        self._s_curr = float(s)
        self.updateCoupledModels()
        return [self._p_curr.copy(), self._q_curr.copy(), self._s_curr], False

    def postSolve(self):
        self.data.finalize()
        self.interfaceData.finalize()
        self.concData.finalize()
        if self.pData is not None:
            self.pData.finalize()
        if self.qData is not None:
            self.qData.finalize()

    def getInterfacePosition(self, time=None):
        return self.interfaceData.y(time)

    def getTotalInventoryFromState(self, p, q, s):
        return integrate_planar_transformed_profile(
            p=p,
            q=q,
            s=s,
            domain_length=self._R,
            u=self._u_grid,
            v=self._v_grid,
        )

    def getTotalInventory(self, time=None):
        if time is None:
            return self.getTotalInventoryFromState(self._p_curr, self._q_curr, self._s_curr)
        if self.concData is not None:
            return self.concData.y(time) * self._R
        composition = np.asarray(self.data.y(time), dtype=np.float64).reshape(-1)
        return float(np.trapezoid(composition, self._z))

    def getTotalMass(self, time=None):
        return self.getTotalInventory(time=time)

    def checkConservation(self, tolerance: float, time=None):
        """
        Checks absolute transformed-inventory drift against the initial value.

        Returns
        -------
        drift : float
            Absolute inventory difference from the stored initial inventory.
        """
        if self._initialInventory is None:
            raise ValueError("Model must be setup before conservation checks.")
        drift = abs(self.getTotalInventory(time=time) - self._initialInventory)
        if drift > float(tolerance):
            warnings.warn(
                f"Illingworth inventory drift {drift:.3e} exceeded tolerance {float(tolerance):.3e}.",
                RuntimeWarning,
                stacklevel=2,
            )
        return drift

    def checkMassIntegral_old(self, p, q, s):
        """
        Returns the conserved average composition of the transformed state.

        Illingworth's planar scheme conserves solute in the Landau coordinates,
        so the numerically meaningful inventory is
        ``s int_0^1 p du + (R-s) int_0^1 q dv``. This helper records the
        corresponding average composition, matching ``concData`` in the Olaye
        model while avoiding any extra interpolation through the physical mesh.
        """
        total_mass = self.getTotalInventoryFromState(p, q, s)
        return float(total_mass / self._R)
    
    def checkMassIntegral(self, p, q, s):
        # debugInPlace()
        [du] = np.unique(np.diff(self._u_grid.copy()).round(15)).tolist()
        [dv] = np.unique(np.diff(self._v_grid.copy()).round(15)).tolist()
        assert abs((((len(p)-2) * du) + du/2 + du/2)-1)<1e-10, ((((len(p)-2) * du) + du/2 + du/2)-1, len(p), du)
        assert abs((((len(q)-2) * dv) + dv/2 + dv/2)-1)<1e-10, ((((len(q)-2) * dv) + dv/2 + dv/2)-1, len(q), dv)

        left_mass = s * ( (du/2)*p[0] + (du*p[1:-1]).sum() + (du/2)*p[-1] )
        right_mass = (self._R - s) * ( (dv/2)*q[0] + (dv*q[1:-1]).sum() + (dv/2)*q[-1] )
        total_mass = left_mass + right_mass
        total_conc = total_mass/self._R
        return total_conc
