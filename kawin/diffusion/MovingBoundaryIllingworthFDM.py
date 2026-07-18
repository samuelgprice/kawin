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

def _loge_arange(start, stop, log_step):
    """Returns exponentially spaced target times with fixed natural-log spacing."""
    logs = np.arange(np.log(start), np.log(stop), log_step)
    return np.exp(logs)

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

    def preallocate(self, capacity):
        """Preallocates scalar history storage without changing recorded values."""
        capacity = int(capacity)
        if capacity <= self._time.shape[0]:
            return
        y_new = np.zeros(capacity, dtype=self._y.dtype)
        time_new = np.zeros(capacity, dtype=self._time.dtype)
        y_new[: self._y.shape[0]] = self._y
        time_new[: self._time.shape[0]] = self._time
        self._y = y_new
        self._time = time_new

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

    def preallocate(self, capacity):
        """Preallocates vector history storage without changing recorded values."""
        capacity = int(capacity)
        if capacity <= self._time.shape[0]:
            return
        y_new = np.zeros((capacity, self.n_components), dtype=self._y.dtype)
        time_new = np.zeros(capacity, dtype=self._time.dtype)
        y_new[: self._y.shape[0]] = self._y
        time_new[: self._time.shape[0]] = self._time
        self._y = y_new
        self._time = time_new

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
    The transformed profile histories ``pData`` and ``qData`` are recorded by
    default, but may be disabled with ``record_pq_data=False`` to reduce
    memory use in long runs. Recording arrays can also be preallocated once the
    timestep schedule is known; this avoids repeated padding but may allocate
    large full-profile histories up front. Custom transformed grids may be
    supplied with ``transformed_u_grid`` and ``transformed_v_grid`` for planar
    validation cases that use nonuniform Landau-coordinate spacing; grids must
    be finite, strictly increasing, and span exactly from 0 to 1.

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
        dt_mode: str = "fixed",
        semiLog_dt: float | None = None,
        semiLogT0: float | None = None,
        geometry: str = "planar",
        phase_a_nodes: int | None = None,
        phase_b_nodes: int | None = None,
        tolerance: float = 1e-8,
        max_iterations: int = 100,
        constraints=None,
        record=False,
        record_pq_data: bool = True,
        preallocate_recordings: bool = False,
        transformed_u_grid=None,
        transformed_v_grid=None,
    ):
        self.initialInterfacePosition = float(interfacePosition)
        self.interfaceCompositions = tuple(float(v) for v in interface_compositions)
        self.timeStep = float(time_step)
        self.dtMode = str(dt_mode)
        self.semiLog_dt = None if semiLog_dt is None else float(semiLog_dt)
        self.semiLogT0 = None if semiLogT0 is None else float(semiLogT0)
        self.geometry = str(geometry)
        self.phaseANodes = None if phase_a_nodes is None else int(phase_a_nodes)
        self.phaseBNodes = None if phase_b_nodes is None else int(phase_b_nodes)
        self.tolerance = float(tolerance)
        self.maxIterations = int(max_iterations)
        self.recordPqData = bool(record_pq_data)
        self.preallocateRecordings = bool(preallocate_recordings)
        self._inputUGrid = self._validate_transformed_grid(transformed_u_grid, "transformed_u_grid")
        self._inputVGrid = self._validate_transformed_grid(transformed_v_grid, "transformed_v_grid")
        if self._inputUGrid is not None:
            if self.phaseANodes is not None and self.phaseANodes != len(self._inputUGrid):
                raise ValueError("phase_a_nodes must match the length of transformed_u_grid.")
            self.phaseANodes = len(self._inputUGrid)
        if self._inputVGrid is not None:
            if self.phaseBNodes is not None and self.phaseBNodes != len(self._inputVGrid):
                raise ValueError("phase_b_nodes must match the length of transformed_v_grid.")
            self.phaseBNodes = len(self._inputVGrid)

        self.interfaceData = _ScalarHistory(record)
        self.concData = _ScalarHistory(record)
        self.pData = None
        self.qData = None
        self._currdt = np.inf
        self._semiLogTimes = None
        self._semiLogNextIndex = 0
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

    def _validate_transformed_grid(self, grid, name):
        """Validates an optional planar Landau-coordinate grid."""
        if grid is None:
            return None
        values = np.asarray(grid, dtype=np.float64).reshape(-1)
        if len(values) < 3:
            raise ValueError(f"{name} must contain at least three nodes.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values.")
        if not np.isclose(values[0], 0.0, rtol=0.0, atol=1e-14) or not np.isclose(values[-1], 1.0, rtol=0.0, atol=1e-14):
            raise ValueError(f"{name} must start at 0 and end at 1.")
        if not np.all(np.diff(values) > 0.0):
            raise ValueError(f"{name} must be strictly increasing.")
        values[0] = 0.0
        values[-1] = 1.0
        return values

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
        if self.dtMode not in {"fixed", "semi_log"}:
            raise ValueError("dt_mode must be 'fixed' or 'semi_log'.")
        if self.dtMode == "semi_log" and ((self.semiLog_dt is None) or (self.semiLogT0 is None)):
            raise ValueError("semiLog_dt and semiLogT0 must be specified when dt_mode is 'semi_log'.")
        if self.semiLog_dt is not None and (not np.isfinite(self.semiLog_dt) or self.semiLog_dt <= 0):
            raise ValueError("semiLog_dt must be a positive finite value.")
        if self.semiLogT0 is not None and (not np.isfinite(self.semiLogT0) or self.semiLogT0 <= 0):
            raise ValueError("semiLogT0 must be a positive finite value.")
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
        self._semiLogTimes = None
        self._semiLogNextIndex = 0
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

        self._u_grid = (
            self._inputUGrid.copy()
            if self._inputUGrid is not None
            else np.linspace(0.0, 1.0, int(n_left), dtype=np.float64)
        )
        self._v_grid = (
            self._inputVGrid.copy()
            if self._inputVGrid is not None
            else np.linspace(0.0, 1.0, int(n_right), dtype=np.float64)
        )
        n_left = len(self._u_grid)
        n_right = len(self._v_grid)
        if self.recordPqData:
            self.pData = _VectorHistory(int(n_left), self.interfaceData.recordInterval)
            self.qData = _VectorHistory(int(n_right), self.interfaceData.recordInterval)
        self._p_curr, self._q_curr = self._initialize_transformed_state(c0, s0)
        self._s_curr = s0
        self._s_old = s0
        self._D_left, self._D_right = self._constant_phase_diffusivities()
        if self.recordPqData:
            self.pData.record(0, self._p_curr)
            self.qData.record(0, self._q_curr)

        physical = self._reconstruct_physical_profile(self._p_curr, self._q_curr, s0)[:, np.newaxis]
        self.data.currentY = physical
        self._initialInventory = self.getTotalInventoryFromState(self._p_curr, self._q_curr, self._s_curr)
        self.concData.record(0, self.checkMassIntegral(self._p_curr, self._q_curr, self._s_curr))

    def setTimeInfo(self, currTime, simTime):
        """
        Stores solve-time bounds and prepares optional semi-log target times.

        In ``dt_mode='semi_log'``, the implicit Illingworth recurrence still
        advances one complete step at a time; only the target output/step times
        are nonuniform. ``semiLogT0`` is the first relative target time and
        ``semiLog_dt`` is the spacing in natural-log time.
        """
        super().setTimeInfo(currTime, simTime)
        self._currdt = np.inf
        self._nearFinalNoop = False
        if self.dtMode != "semi_log" or simTime <= 0:
            self._semiLogTimes = None
            self._semiLogNextIndex = 0
            self._preallocateRecordingHistories(simTime)
            return

        t0_rel = max(float(self.semiLogT0), 1e-15)
        sim_time = float(simTime)
        if sim_time <= t0_rel:
            rel_times = np.asarray([sim_time], dtype=np.float64)
        else:
            rel_times = _loge_arange(t0_rel, sim_time, float(self.semiLog_dt))
            rel_times = rel_times[(rel_times > 0.0) & (rel_times < sim_time)]
            rel_times = np.append(rel_times, sim_time)
        self._semiLogTimes = float(currTime) + np.asarray(rel_times, dtype=np.float64)
        self._semiLogNextIndex = 0
        self._preallocateRecordingHistories(simTime)

    def _estimateStepCount(self, simTime):
        """Returns the number of implicit steps expected for the current schedule."""
        sim_time = float(simTime)
        if sim_time <= 0:
            return 0
        if self.dtMode == "semi_log" and self._semiLogTimes is not None:
            return int(len(self._semiLogTimes))
        return int(np.ceil(sim_time / self.timeStep))

    def _recordCapacityForSteps(self, record_interval, step_count):
        """
        Estimates rows needed by the history record/finalize convention.

        Histories already contain the initial state when this is called. The
        extra two rows cover the final forced record path and roundoff-driven
        near-final no-op step.
        """
        if record_interval <= 0:
            return 1
        return int(np.ceil((int(step_count) + 2) / int(record_interval))) + 2

    def _preallocateRecordingHistories(self, simTime):
        """Preallocates enabled history arrays when the run length is predictable."""
        if not self.preallocateRecordings:
            return
        step_count = self._estimateStepCount(simTime)
        histories = [self.data, self.interfaceData, self.concData]
        if self.recordPqData:
            histories.extend([self.pData, self.qData])
        for history in histories:
            if history is None or not hasattr(history, "preallocate"):
                continue
            capacity = self._recordCapacityForSteps(history.recordInterval, step_count)
            history.preallocate(capacity)

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
        step_scale = self.timeStep
        if self.dtMode == "semi_log":
            scheduled_dt = self._computeSemiLogDt(t)
            if np.isfinite(scheduled_dt) and scheduled_dt > 0:
                step_scale = scheduled_dt
            dt = min(scheduled_dt, remaining)
        else:
            dt = min(self.timeStep, remaining)
        self._nearFinalNoop = bool(np.isfinite(remaining) and 0 < remaining <= max(step_scale, 1e-15) * 1e-10)
        if self._nearFinalNoop:
            self._currdt = max(step_scale, 1e-15)
            return self._currdt
        if not np.isfinite(dt) or dt <= 0:
            dt = self.timeStep
        self._currdt = float(dt)
        return float(dt)

    def _updateSemiLogIndex(self, t):
        if self._semiLogTimes is None:
            return
        while self._semiLogNextIndex < len(self._semiLogTimes):
            if self._semiLogTimes[self._semiLogNextIndex] > float(t) + 1e-15:
                break
            self._semiLogNextIndex += 1

    def _computeSemiLogDt(self, t):
        """Returns the step needed to reach the next semi-log target time."""
        if self.dtMode != "semi_log" or self._semiLogTimes is None:
            return np.inf
        self._updateSemiLogIndex(t)
        if self._semiLogNextIndex >= len(self._semiLogTimes):
            return np.inf
        return max(1e-15, float(self._semiLogTimes[self._semiLogNextIndex] - float(t)))

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
        """
        Solves the left transformed concentration profile for one implicit step.

        Coefficients are the planar Illingworth finite-volume coefficients from
        the MAP implementation. Interior rows are assembled with vectorized
        slices, but the boundary rows retain the original one-sided forms.
        """
        n = len(p)
        lo = np.zeros(n, dtype=np.float64)
        diag = np.zeros(n, dtype=np.float64)
        up = np.zeros(n, dtype=np.float64)
        rhs = np.zeros(n, dtype=np.float64)
        tmpA = self._D_left * dt / future_s
        tmpB = future_s - s
        u = self._u_grid
        left_diff = u[1:-1] - u[:-2]
        left_sum = u[1:-1] + u[:-2]
        right_diff = u[2:] - u[1:-1]
        right_sum = u[2:] + u[1:-1]
        cell_width = right_sum - left_sum

        if future_s >= s:
            diag[0] = -tmpA / u[1] - future_s * u[1] / 2.0
            up[0] = tmpA / u[1] + tmpB * u[1] / 2.0
            rhs[0] = -p[0] * s * u[1] / 2.0
            lo[1:-1] = tmpA / left_diff
            diag[1:-1] = -tmpA * (1.0 / left_diff + 1.0 / right_diff) - tmpB * left_sum / 2.0
            diag[1:-1] = diag[1:-1] - future_s * cell_width / 2.0
            up[1:-1] = tmpA / right_diff + tmpB * right_sum / 2.0
            rhs[1:-1] = -s * p[1:-1] * cell_width / 2.0
        else:
            diag[0] = -tmpA / u[1] + tmpB * u[1] / 2.0 - future_s * u[1] / 2.0
            up[0] = tmpA / u[1]
            rhs[0] = -p[0] * s * u[1] / 2.0
            lo[1:-1] = tmpA / left_diff - tmpB * left_sum / 2.0
            diag[1:-1] = -tmpA * (1.0 / left_diff + 1.0 / right_diff) + tmpB * right_sum / 2.0
            diag[1:-1] = diag[1:-1] - future_s * cell_width / 2.0
            up[1:-1] = tmpA / right_diff
            rhs[1:-1] = -s * p[1:-1] * cell_width / 2.0

        diag[-1] = -1.0
        rhs[-1] = -self.interfaceCompositions[0]
        return solve_illingworth_tridiagonal(lo, diag, up, rhs)

    def _new_concentration_right_planar(self, q, s, future_s, dt):
        """
        Solves the right transformed concentration profile for one implicit step.

        This is the large phase for the Figure-3 setup, so interior coefficient
        assembly is vectorized while preserving the MAP code's boundary rows and
        tridiagonal sign convention.
        """
        n = len(q)
        lo = np.zeros(n, dtype=np.float64)
        diag = np.zeros(n, dtype=np.float64)
        up = np.zeros(n, dtype=np.float64)
        rhs = np.zeros(n, dtype=np.float64)
        tmpA = self._D_right * dt / (self._R - future_s)
        tmpB = future_s - s
        span = self._R - future_s
        v = self._v_grid
        left_diff = v[1:-1] - v[:-2]
        left_sum = v[1:-1] + v[:-2]
        right_diff = v[2:] - v[1:-1]
        right_sum = v[2:] + v[1:-1]
        cell_width = right_sum - left_sum

        diag[0] = -1.0
        rhs[0] = -self.interfaceCompositions[1]

        if future_s >= s:
            lo[1:-1] = tmpA / left_diff
            diag[1:-1] = -tmpA * (1.0 / right_diff + 1.0 / left_diff) - tmpB * (1.0 - left_sum / 2.0)
            diag[1:-1] = diag[1:-1] - span * cell_width / 2.0
            up[1:-1] = tmpA / right_diff + tmpB * (1.0 - right_sum / 2.0)
            rhs[1:-1] = -(self._R - s) * q[1:-1] * cell_width / 2.0

            tmp = v[-2]
            lo[-1] = tmpA / (1.0 - tmp)
            diag[-1] = -tmpA / (1.0 - tmp) - tmpB * (1.0 - (1.0 + tmp) / 2.0)
            diag[-1] = diag[-1] - span * (1.0 - tmp) / 2.0
            rhs[-1] = -q[-1] * (self._R - s) * (1.0 - tmp) / 2.0
        else:
            lo[1:-1] = tmpA / left_diff - tmpB * (1.0 - left_sum / 2.0)
            diag[1:-1] = -tmpA * (1.0 / right_diff + 1.0 / left_diff) + tmpB * (1.0 - right_sum / 2.0)
            diag[1:-1] = diag[1:-1] - span * cell_width / 2.0
            up[1:-1] = tmpA / right_diff
            rhs[1:-1] = -(self._R - s) * q[1:-1] * cell_width / 2.0

            tmp = v[-2]
            lo[-1] = tmpA / (1.0 - tmp) - tmpB * (1.0 - (1.0 + tmp) / 2.0)
            diag[-1] = -tmpA / (1.0 - tmp) - span * (1.0 - tmp) / 2.0
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

        print(f"s: {s}")
        print(f"old_s: {old_s}")
        print(f"self.currentTime: {self.currentTime}")
        print(f"dt: {dt}")
        debugInPlace()
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
            raise ValueError("Transformed left-state history is not available; set record_pq_data=True.")
        return self.pData.y(time)

    def getTransformedStateRight(self, time=None):
        """
        Returns the recorded right transformed composition vector ``q``.
        """
        if self.qData is None:
            raise ValueError("Transformed right-state history is not available; set record_pq_data=True.")
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
        if self.recordPqData:
            self.pData.record(time, p)
            self.qData.record(time, q)
        if time>1e3:
            debugInPlace()
            self.checkMassIntegral(p, q, s)
            self.getTotalInventoryFromState(p, q, s)

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
        """
        Returns the transformed-coordinate average composition.

        The trapezoidal integration supports both the standard uniform Landau
        grids and custom nonuniform grids used for validation against the MAP
        implementation.
        """
        return float(self.getTotalInventoryFromState(p, q, s) / self._R)
