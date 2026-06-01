import warnings

from datetime import date
TODAY = date.today()

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

def getMaxEigVal(diffusivities_input):
    diff_eigvals = np.linalg.eigvals(diffusivities_input)
    diff_minEigval, diff_maxEigVal = np.min(diff_eigvals), np.max(diff_eigvals)
    if diff_minEigval<0:
        debugInPlace()
        raise ValueError("I think it might be unstable if any eigvals are less than zero")
    return diff_maxEigVal

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
    interfaceUpdate : {"basic", "lee_oh_corrected", "my_corrected", "1999_lee_allSolute_corrected"}
        Interface motion update used after the explicit diffusion stage.
        This argument must be specified explicitly so the chosen mass-balance
        strategy is always visible at the call site.
    integrationMode : {"ignore", "noIgnore", "weighted"}
        Interface-aware inventory integration rule used by mass accounting and
        corrected interface updates. This argument must be specified
        explicitly.
    ignoredNodeReconstructionMode : {"lagrange", "linear"}
        Rule used to reconstruct a previously ignored interface-adjacent node
        when the ignored-node side changes during an explicit step.
        ``"lagrange"`` uses the existing quadratic interpolation behavior;
        ``"linear"`` uses a two-point line between the interface composition
        and the nearest retained node on that side. This argument must be
        specified explicitly.
    ignoredNodeRule : {"legacy_two_region", "lee_oh_1996_three_region"}
        Rule used to determine when an interface-adjacent node is ignored and
        reconstructed. ``"legacy_two_region"`` preserves the existing behavior
        (ignore left when ``p < pstar``, otherwise ignore right).
        ``"lee_oh_1996_three_region"`` ignores left for ``p < pstar``, right
        for ``p > 1-pstar``, and ignores neither node in the middle band.
    denom_type : {"eqn22", "eqn11"}
    fluxGradientMode : {"pre_diffusion", "post_diffusion"}
        Selects which interface gradients are used when computing the
        interfacial fluxes for the Stefan update. ``"pre_diffusion"`` uses
        gradients reconstructed from the profile before the bulk diffusion
        update, while ``"post_diffusion"`` uses gradients from the profile
        after that explicit diffusion stage. This argument must be specified
        explicitly.
    initialInventoryMode : {"integrated", "phase_length_idealized"}
        Rule used to compute the stored initial inventory ``_initialInventory``.
        ``"integrated"`` uses the interface-aware integration currently used by
        ``getTotalMass`` and ``getTotalInventory``. ``"phase_length_idealized"``
        uses constant composition in each phase multiplied by each phase length
        and raises an error if either phase is not constant. This argument must
        be specified explicitly and affects only the stored initial inventory.
    multicomponentInterfaceStateUpdate : {"pre_diffusion_only", "pre_and_post_diffusion"}
        Ternary-only policy controlling when the multicomponent equal-velocity
        interface solve is performed within one explicit step. The default
        ``"pre_diffusion_only"`` matches the Lee/Oh sequencing most closely by
        solving once at the start of the step and reusing that interface state
        through the rest of the step. ``"pre_and_post_diffusion"`` performs one
        additional solve after the explicit diffusion update and uses that
        second state for the interface-motion update.
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
        ignoredNodeReconstructionMode: str | None = None,
        ignoredNodeRule: str | None = None,
        denom_type: str | None = None,
        fluxGradientMode: str | None = None,
        initialInventoryMode: str | None = None,
        balanceElement: str | None = None,
        multicomponentInterfaceStateUpdate: str = "pre_diffusion_only",
        pstarSchedule: dict | None = None,
    ):
        self.initialInterfacePosition = float(interfacePosition)
        self.interfaceData = _ScalarHistory(record)
        self.interfaceUpdate = None if interfaceUpdate is None else str(interfaceUpdate)
        self.pstar = float(pstar)
        self.integrationMode = None if integrationMode is None else str(integrationMode)
        self.ignoredNodeReconstructionMode = None if ignoredNodeReconstructionMode is None else str(ignoredNodeReconstructionMode)
        self.ignoredNodeRule = None if ignoredNodeRule is None else str(ignoredNodeRule)
        self.denom_type = None if denom_type is None else str(denom_type)
        self.fluxGradientMode = None if fluxGradientMode is None else str(fluxGradientMode)
        self.initialInventoryMode = None if initialInventoryMode is None else str(initialInventoryMode)
        self.bulkUpdateScheme = str(bulkUpdateScheme)
        self.balanceElement = None if balanceElement is None else str(balanceElement)
        self.multicomponentInterfaceStateUpdate = str(multicomponentInterfaceStateUpdate)
        self._balanceElementIndex = None
        self._initialInventory = None
        self._currdt = np.inf
        self._lastFluxes = None
        self._lastInterfaceFluxes = (0.0, 0.0)
        self._lastInterfaceVelocity = 0.0
        self.dtDiffData = _ScalarHistory(record)
        self.dtMoveData = _ScalarHistory(record)
        self._pendingDtDiff = np.inf
        self._pendingDtMove = np.inf
        self._cachedMulticomponentInterfaceState = None
        self._interfaceCompositionHistory = None
        self._pendingInterfaceCompositionRecord = None
        self._hasInterfaceCompositionHistoryData = False
        self._pstarScheduleTimes = np.array([], dtype=np.float64)
        self._pstarScheduleValues = np.array([], dtype=np.float64)
        self._nextPstarScheduleIndex = 0
        self._lastAppliedPstarScheduleIndex = -1
        self._pstarScheduleUpdateLog = []
        self._pstarStrictAudit = True
        self._pstarChangedSinceLastPreProcess = False
        super().__init__(
            mesh=mesh,
            elements=elements,
            phases=phases,
            thermodynamics=thermodynamics,
            temperature=temperature,
            constraints=constraints,
            record=record,
        )
        if self.interfaceUpdate=="1999_lee_allSolute_corrected":
            self.previousMassDelta = np.zeros_like(self.elements, dtype=np.float64)
        self._validateMovingBoundaryModel()
        self.interfaceData.currentY = self.initialInterfacePosition
        self.interfaceData._y[0] = self.initialInterfacePosition
        # self._initialInventory = self._getStoredInventory()
        raise ValueError("This is just to let you know that the initialInventory is being hard-coded here")
        self._initialInventory = np.array([0.229945, 0.09035])*self.mesh.zlim[0][-1]
        self._cachedMulticomponentInterfaceState = None
        self._initializeInterfaceCompositionHistory()
        if pstarSchedule is not None:
            if not isinstance(pstarSchedule, dict):
                raise ValueError("pstarSchedule must be a dict with 'times' and 'values'.")
            self.setPstarSchedule(
                schedule_times=pstarSchedule.get("times", None),
                schedule_values=pstarSchedule.get("values", None),
            )

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
        if self.interfaceUpdate not in {"basic", "lee_oh_corrected", "my_corrected", "1999_lee_allSolute_corrected"}:
            raise ValueError("interfaceUpdate must be one of ['basic', 'lee_oh_corrected', 'my_corrected', '1999_lee_allSolute_corrected'].")
        if self.integrationMode is None:
            raise ValueError("integrationMode must be specified explicitly.")
        if self.integrationMode not in {"ignore", "noIgnore", "weighted"}:
            raise ValueError("integrationMode must be one of ['ignore', 'noIgnore', 'weighted'].")
        if self.ignoredNodeReconstructionMode is None:
            raise ValueError("ignoredNodeReconstructionMode must be specified explicitly.")
        if self.ignoredNodeReconstructionMode not in {"lagrange", "linear"}:
            raise ValueError("ignoredNodeReconstructionMode must be one of ['lagrange', 'linear'].")
        if self.ignoredNodeRule is None:
            raise ValueError("ignoredNodeRule must be specified explicitly.")
        if self.ignoredNodeRule not in {"legacy_two_region", "lee_oh_1996_three_region"}:
            raise ValueError("ignoredNodeRule must be one of ['legacy_two_region', 'lee_oh_1996_three_region'].")
        if self.denom_type is None:
            raise ValueError("denom_type must be specified explicitly.")
        if self.denom_type not in {"eqn22", "eqn11"}:
            raise ValueError("denom_type must be one of ['eqn22', 'eqn11'].")
        if self.fluxGradientMode is None:
            raise ValueError("fluxGradientMode must be specified explicitly.")
        if self.fluxGradientMode not in {"pre_diffusion", "post_diffusion"}:
            raise ValueError("fluxGradientMode must be one of ['pre_diffusion', 'post_diffusion'].")
        if self.initialInventoryMode is None:
            raise ValueError("initialInventoryMode must be specified explicitly.")
        if self.initialInventoryMode not in {"integrated", "phase_length_idealized"}:
            raise ValueError("initialInventoryMode must be one of ['integrated', 'phase_length_idealized'].")
        if self.bulkUpdateScheme not in {"legacy", "flux_form"}:
            raise ValueError("bulkUpdateScheme must be one of ['legacy', 'flux_form'].")
        if self.multicomponentInterfaceStateUpdate not in {"pre_diffusion_only", "pre_and_post_diffusion"}:
            raise ValueError(
                "multicomponentInterfaceStateUpdate must be one of "
                "['pre_diffusion_only', 'pre_and_post_diffusion']."
            )
        if not (0 < self.pstar < 1):
            raise ValueError("pstar must lie strictly between 0 and 1.")
        if self.ignoredNodeRule == "lee_oh_1996_three_region":
            if (self.pstar>0.5) or (self.pstar<=0):
                raise ValueError(f"pstar should be between 0 and 0.5 for ignoredNodeRule='lee_oh_1996_three_region', got {self.pstar}")
        if self.constraints.movingBoundaryThreshold>=min(self.pstar, 1-self.pstar):
            raise ValueError("movingBoundaryThreshold must be less than the minimum of pstar and 1-pstar.")
        if self._isBinarySystem():
            if self.interfaceUpdate not in {"basic", "lee_oh_corrected", "my_corrected"}:
                raise ValueError("Binary MovingBoundaryFD1DModel does not support interfaceUpdate='1999_lee_allSolute_corrected'.")
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
            if self.interfaceUpdate in {"lee_oh_corrected", "my_corrected"}:
                if self.balanceElement is None:
                    raise ValueError(
                        "Ternary MovingBoundaryFD1DModel requires balanceElement for "
                        f"interfaceUpdate='{self.interfaceUpdate}'."
                    )
                self._balanceElementIndex = self.elements.index(self.balanceElement)
            else:
                self._balanceElementIndex = None if self.balanceElement is None else self.elements.index(self.balanceElement)
        else:
            raise ValueError("MovingBoundaryFD1DModel currently supports only binary or ternary systems.")
        self._clipInterfacePosition(self.initialInterfacePosition, strict=True, initial=True)

    def reset(self):
        super().reset()
        self.interfaceData.reset()
        self.interfaceData.record(0, self.initialInterfacePosition)
        self.dtDiffData.reset()
        self.dtMoveData.reset()
        self.dtDiffData.record(0, np.inf)
        self.dtMoveData.record(0, np.inf)
        self._pendingDtDiff = np.inf
        self._pendingDtMove = np.inf
        self._lastFluxes = None
        self._lastInterfaceFluxes = None
        self._lastInterfaceVelocity = None
        if self.interfaceUpdate=="1999_lee_allSolute_corrected":
            self.previousMassDelta = np.zeros_like(self.elements, dtype=np.float64)
        self._currdt = np.inf
        self._cachedMulticomponentInterfaceState = None
        self._pstarChangedSinceLastPreProcess = False
        self._pendingInterfaceCompositionRecord = None
        self._hasInterfaceCompositionHistoryData = False
        if hasattr(self, "mesh") and self.mesh is not None:
            self._validateMovingBoundaryModel()
            self._initialInventory = self._getStoredInventory()
            self._cachedMulticomponentInterfaceState = None
            self._initializeInterfaceCompositionHistory()

    def _validatePstarValue(self, pstar_value: float):
        '''
        Validates one ``pstar`` value against runtime moving-boundary constraints.
        '''
        pstar_value = float(pstar_value)
        if not (0 < pstar_value < 1):
            raise ValueError("pstar must lie strictly between 0 and 1.")
        if self.ignoredNodeRule == "lee_oh_1996_three_region" and pstar_value > 0.5:
            raise ValueError(
                "pstar should be between 0 and 0.5 for ignoredNodeRule='lee_oh_1996_three_region'."
            )
        if self.constraints.movingBoundaryThreshold >= min(pstar_value, 1 - pstar_value):
            raise ValueError("movingBoundaryThreshold must be less than the minimum of pstar and 1-pstar.")

    def setPstarSchedule(self, schedule_times, schedule_values):
        '''
        Configures a piecewise-constant runtime ``pstar`` schedule.

        The schedule is checked and then applied at iteration boundaries during
        ``preProcess``.
        '''
        if schedule_times is None or schedule_values is None:
            raise ValueError("Both schedule_times and schedule_values must be provided.")
        times = np.asarray(schedule_times, dtype=np.float64).reshape(-1)
        values = np.asarray(schedule_values, dtype=np.float64).reshape(-1)
        if len(times) != len(values):
            raise ValueError("schedule_times and schedule_values must have the same length.")
        if len(times) == 0:
            raise ValueError("pstar schedule must include at least one entry.")
        if not np.all(np.isfinite(times)):
            raise ValueError("schedule_times must all be finite.")
        if not np.all(np.diff(times) > 0):
            raise ValueError("schedule_times must be strictly increasing.")
        for pstar_value in values:
            self._validatePstarValue(float(pstar_value))

        self._pstarScheduleTimes = times.copy()
        self._pstarScheduleValues = values.copy()
        self._nextPstarScheduleIndex = 0
        self._lastAppliedPstarScheduleIndex = -1
        self._pstarScheduleUpdateLog = []

    def clearPstarSchedule(self):
        '''
        Disables runtime ``pstar`` scheduling and clears schedule state.
        '''
        self._pstarScheduleTimes = np.array([], dtype=np.float64)
        self._pstarScheduleValues = np.array([], dtype=np.float64)
        self._nextPstarScheduleIndex = 0
        self._lastAppliedPstarScheduleIndex = -1
        self._pstarScheduleUpdateLog = []

    def getPstarScheduleStatus(self):
        '''
        Returns schedule state and runtime update history.
        '''
        next_time = None
        next_value = None
        if self._nextPstarScheduleIndex < len(self._pstarScheduleTimes):
            next_time = float(self._pstarScheduleTimes[self._nextPstarScheduleIndex])
            next_value = float(self._pstarScheduleValues[self._nextPstarScheduleIndex])
        return {
            "enabled": len(self._pstarScheduleTimes) > 0,
            "current_pstar": float(self.pstar),
            "next_index": int(self._nextPstarScheduleIndex),
            "last_applied_index": int(self._lastAppliedPstarScheduleIndex),
            "next_time": next_time,
            "next_value": next_value,
            "history": list(self._pstarScheduleUpdateLog),
        }

    def _recordPstarScheduleEvent(self, **event):
        '''
        Appends one schedule/update event to the runtime history.
        '''
        event_copy = {k: v for k, v in event.items()}
        event_copy["time"] = float(event_copy.get("time", self.currentTime))
        event_copy["pstar_before"] = float(event_copy.get("pstar_before", self.pstar))
        if "pstar_after" in event_copy and event_copy["pstar_after"] is not None:
            event_copy["pstar_after"] = float(event_copy["pstar_after"])
        print(event_copy)
        self._pstarScheduleUpdateLog.append(event_copy)

    def updatePstar(self, new_pstar, *, reason="manual", time=None, pAtUpdate=None):
        '''
        Applies a new ``pstar`` value and clears stale runtime state.
        '''
        t_eval = self.currentTime if time is None else float(time)
        pstar_old = float(self.pstar)
        new_pstar = float(new_pstar)
        self._validatePstarValue(new_pstar)
        if np.isclose(new_pstar, pstar_old, rtol=0.0, atol=1e-14):
            self._recordPstarScheduleEvent(
                event="no_change",
                reason=str(reason),
                time=t_eval,
                pstar_before=pstar_old,
                pstar_after=pstar_old,
                pAtUpdate=pAtUpdate,
            )
            return pstar_old

        self.pstar = new_pstar
        self._cachedMulticomponentInterfaceState = None
        self._lastFluxes = None
        self._lastInterfaceFluxes = None
        self._lastInterfaceVelocity = None
        self._pstarChangedSinceLastPreProcess = True

        interface_now = float(self.interfaceData.currentY)
        geom_new = get_moving_boundary_fd_geometry(self.mesh, interface_now, self.pstar, self.ignoredNodeRule)
        if self._pstarStrictAudit and self._cachedMulticomponentInterfaceState is not None:
            raise ValueError("Stale ternary interface cache detected immediately after pstar update.")
        if self._pstarStrictAudit and geom_new.ignore_mode not in {"ignore_left", "ignore_right", "ignore_none"}:
            raise ValueError("Unexpected ignored-node regime after pstar update.")
        self._recordPstarScheduleEvent(
            event="applied",
            reason=str(reason),
            time=t_eval,
            pstar_before=pstar_old,
            pstar_after=new_pstar,
            ignore_mode_after=geom_new.ignore_mode,
            interface_position=float(interface_now),
            pAtUpdate=pAtUpdate,
        )
        return new_pstar

    def preProcess(self):
        '''
        Applies due scheduled ``pstar`` updates at iteration boundaries.
        '''
        self._pstarChangedSinceLastPreProcess = False
        if len(self._pstarScheduleTimes) == 0:
            return

        eps_t = 1e-14
        while self._nextPstarScheduleIndex < len(self._pstarScheduleTimes):
            schedule_time = float(self._pstarScheduleTimes[self._nextPstarScheduleIndex])
            if self.currentTime + eps_t < schedule_time:
                break

            target_pstar = float(self._pstarScheduleValues[self._nextPstarScheduleIndex])
            interface_now = float(self.interfaceData.currentY)
            geom_old = get_moving_boundary_fd_geometry(self.mesh, interface_now, self.pstar, self.ignoredNodeRule)
            geom_new = get_moving_boundary_fd_geometry(self.mesh, interface_now, target_pstar, self.ignoredNodeRule)
            if geom_old.ignore_mode != geom_new.ignore_mode:
                self._recordPstarScheduleEvent(
                    event="deferred",
                    reason="regime_flip",
                    time=self.currentTime,
                    schedule_index=int(self._nextPstarScheduleIndex),
                    pstar_before=float(self.pstar),
                    pstar_after=float(target_pstar),
                    ignore_mode_before=geom_old.ignore_mode,
                    ignore_mode_after=geom_new.ignore_mode,
                    p=float(geom_old.p),
                )
                break

            self.updatePstar(
                target_pstar,
                reason=f"schedule[{self._nextPstarScheduleIndex}]",
                time=self.currentTime,
                pAtUpdate=geom_old.p,
            )
            self._lastAppliedPstarScheduleIndex = int(self._nextPstarScheduleIndex)
            self._nextPstarScheduleIndex += 1

    def toDict(self):
        '''
        Serializes solved state, including interface and timestep-limit histories.
        '''
        data = super().toDict()
        data.update(
            {
                "interface_position": self.interfaceData._y,
                "interface_time": self.interfaceData._time,
                "interface_interval": self.interfaceData.recordInterval,
                "interface_index": self.interfaceData.N,
                "interface_update": self.interfaceUpdate,
                "ignored_node_reconstruction_mode": self.ignoredNodeReconstructionMode,
                "ignored_node_rule": self.ignoredNodeRule,
                "denom_type": self.denom_type,
                "flux_gradient_mode": self.fluxGradientMode,
                "initial_inventory_mode": self.initialInventoryMode,
                "bulk_update_scheme": self.bulkUpdateScheme,
                "balance_element": "" if self.balanceElement is None else self.balanceElement,
                "multicomponent_interface_state_update": self.multicomponentInterfaceStateUpdate,
                "dt_diff": self.dtDiffData._y,
                "dt_diff_time": self.dtDiffData._time,
                "dt_diff_interval": self.dtDiffData.recordInterval,
                "dt_diff_index": self.dtDiffData.N,
                "dt_move": self.dtMoveData._y,
                "dt_move_time": self.dtMoveData._time,
                "dt_move_interval": self.dtMoveData.recordInterval,
                "dt_move_index": self.dtMoveData.N,
            }
        )
        if self._isTernarySystem() and self._interfaceCompositionHistory is not None and self._hasInterfaceCompositionHistoryData:
            h = self._interfaceCompositionHistory
            data.update(
                {
                    "interface_comp_time": h["time"]._time,
                    "interface_comp_interval": h["time"].recordInterval,
                    "interface_comp_index": h["time"].N,
                    "interface_comp_pre_left": h["pre_left"]._y,
                    "interface_comp_pre_right": h["pre_right"]._y,
                    "interface_comp_post_left": h["post_left"]._y,
                    "interface_comp_post_right": h["post_right"]._y,
                    "interface_comp_used_stage": h["used_stage"]._y,
                }
            )
        return data

    def fromDict(self, data):
        '''
        Restores solved state, including interface and timestep-limit histories.
        '''
        super().fromDict(data)
        interface_update = data.get("interface_update", self.interfaceUpdate)
        if isinstance(interface_update, np.ndarray):
            interface_update = interface_update.item()
        self.interfaceUpdate = str(interface_update)
        if "ignored_node_reconstruction_mode" not in data:
            raise ValueError(
                "Saved MovingBoundaryFD1DModel data does not include 'ignored_node_reconstruction_mode'. "
                "Please set ignoredNodeReconstructionMode explicitly before loading legacy data."
            )
        ignored_node_reconstruction_mode = data["ignored_node_reconstruction_mode"]
        if isinstance(ignored_node_reconstruction_mode, np.ndarray):
            ignored_node_reconstruction_mode = ignored_node_reconstruction_mode.item()
        self.ignoredNodeReconstructionMode = str(ignored_node_reconstruction_mode)
        if "ignored_node_rule" not in data:
            raise ValueError(
                "Saved MovingBoundaryFD1DModel data does not include 'ignored_node_rule'. "
                "Please set ignoredNodeRule explicitly before loading legacy data."
            )
        ignored_node_rule = data["ignored_node_rule"]
        if "denom_type" not in data:
            raise ValueError(
                "Saved MovingBoundaryFD1DModel data does not include 'denom_type'. "
                "Please set denom_type explicitly before loading legacy data."
            )
        denom_type = data["denom_type"]
        if isinstance(ignored_node_rule, np.ndarray):
            ignored_node_rule = ignored_node_rule.item()

        self.ignoredNodeRule = str(ignored_node_rule)
        self.denom_type = str(denom_type)
        self.fluxGradientMode = str(data.get("flux_gradient_mode", None))
        if "initial_inventory_mode" not in data:
            raise ValueError(
                "Saved MovingBoundaryFD1DModel data does not include 'initial_inventory_mode'. "
                "Please set initialInventoryMode explicitly before loading legacy data."
            )
        self.initialInventoryMode = str(data["initial_inventory_mode"])
        self.bulkUpdateScheme = str(data["bulk_update_scheme"])
        self.multicomponentInterfaceStateUpdate = str(data.get("multicomponent_interface_state_update", "pre_diffusion_only"))
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
        self.dtDiffData.recordInterval = int(data.get("dt_diff_interval", self.interfaceData.recordInterval))
        self.dtDiffData.N = int(data.get("dt_diff_index", self.interfaceData.N))
        self.dtDiffData._y = np.array(data.get("dt_diff", np.full_like(self.interfaceData._y, np.inf)), dtype=np.float64)
        self.dtDiffData._time = np.array(data.get("dt_diff_time", self.interfaceData._time), dtype=np.float64)
        self.dtDiffData.currentY = float(self.dtDiffData._y[self.dtDiffData.N])
        self.dtDiffData.currentTime = float(self.dtDiffData._time[self.dtDiffData.N])
        self.dtDiffData.currentIndex = self.dtDiffData.N
        self.dtMoveData.recordInterval = int(data.get("dt_move_interval", self.interfaceData.recordInterval))
        self.dtMoveData.N = int(data.get("dt_move_index", self.interfaceData.N))
        self.dtMoveData._y = np.array(data.get("dt_move", np.full_like(self.interfaceData._y, np.inf)), dtype=np.float64)
        self.dtMoveData._time = np.array(data.get("dt_move_time", self.interfaceData._time), dtype=np.float64)
        self.dtMoveData.currentY = float(self.dtMoveData._y[self.dtMoveData.N])
        self.dtMoveData.currentTime = float(self.dtMoveData._time[self.dtMoveData.N])
        self.dtMoveData.currentIndex = self.dtMoveData.N
        self._validateMovingBoundaryModel()
        self._initialInventory = self._getStoredInventory(0)
        self._cachedMulticomponentInterfaceState = None
        self._initializeInterfaceCompositionHistory()
        self._hasInterfaceCompositionHistoryData = False
        if self._isTernarySystem() and "interface_comp_time" in data:
            h = self._interfaceCompositionHistory
            if h is not None:
                h["time"].recordInterval = int(data["interface_comp_interval"])
                h["time"].N = int(data["interface_comp_index"])
                h["time"]._time = np.array(data["interface_comp_time"], dtype=np.float64)
                h["time"]._y = np.array(data["interface_comp_time"], dtype=np.float64)
                h["time"].currentTime = float(h["time"]._time[h["time"].N])
                h["time"].currentY = float(h["time"]._y[h["time"].N])
                h["time"].currentIndex = h["time"].N

                for key, arr_key in [
                    ("pre_left", "interface_comp_pre_left"),
                    ("pre_right", "interface_comp_pre_right"),
                    ("post_left", "interface_comp_post_left"),
                    ("post_right", "interface_comp_post_right"),
                ]:
                    h[key].recordInterval = h["time"].recordInterval
                    h[key].N = h["time"].N
                    h[key]._time = h["time"]._time.copy()
                    h[key]._y = np.array(data[arr_key], dtype=np.float64)
                    h[key].currentY = h[key]._y[h[key].N].copy()
                    h[key].currentTime = float(h[key]._time[h[key].N])
                    h[key].currentIndex = h[key].N

                h["used_stage"].recordInterval = h["time"].recordInterval
                h["used_stage"].N = h["time"].N
                h["used_stage"]._time = h["time"]._time.copy()
                h["used_stage"]._y = np.array(data["interface_comp_used_stage"], dtype=np.float64)
                h["used_stage"].currentY = float(h["used_stage"]._y[h["used_stage"].N])
                h["used_stage"].currentTime = float(h["used_stage"]._time[h["used_stage"].N])
                h["used_stage"].currentIndex = h["used_stage"].N
                self._hasInterfaceCompositionHistoryData = h["time"].N >= 0

    def setup(self):
        super().setup()
        self._validateMovingBoundaryModel()

    def _initializeInterfaceCompositionHistory(self):
        '''
        Initializes ternary-only interface composition histories.
        '''
        self._interfaceCompositionHistory = None
        if not self._isTernarySystem():
            return
        self._hasInterfaceCompositionHistoryData = False
        n_components = len(self.elements)
        self._interfaceCompositionHistory = {
            "time": _ScalarHistory(self.interfaceData.recordInterval),
            "pre_left": _VectorHistory(n_components, self.interfaceData.recordInterval),
            "pre_right": _VectorHistory(n_components, self.interfaceData.recordInterval),
            "post_left": _VectorHistory(n_components, self.interfaceData.recordInterval),
            "post_right": _VectorHistory(n_components, self.interfaceData.recordInterval),
            "used_stage": _ScalarHistory(self.interfaceData.recordInterval),
        }

    def _recordInterfaceCompositionHistory(self, time, pre_comp, post_comp, used_stage, force=False):
        '''
        Records one ternary interface composition history sample.
        '''
        if self._interfaceCompositionHistory is None:
            return
        stage_value = 1.0 if str(used_stage) == "post" else 0.0
        h = self._interfaceCompositionHistory
        h["time"].record(time, float(time), force=force)
        h["pre_left"].record(time, np.asarray(pre_comp[0], dtype=np.float64), force=force)
        h["pre_right"].record(time, np.asarray(pre_comp[1], dtype=np.float64), force=force)
        h["post_left"].record(time, np.asarray(post_comp[0], dtype=np.float64), force=force)
        h["post_right"].record(time, np.asarray(post_comp[1], dtype=np.float64), force=force)
        h["used_stage"].record(time, stage_value, force=force)
        self._hasInterfaceCompositionHistoryData = True

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

    def _clipInterfacePosition(self, interface_position: float, strict: bool = True, initial: bool = False) -> float:
        '''
        Clips the interface to the open domain and nudges it off exact node locations
        '''
        z = np.ravel(self.mesh.z)
        assert z[0] < interface_position < z[-1], "Interface position is outside the domain."
        if not initial:
            return interface_position ##NOTE: Bypassing this for now as it is unlikely that the interface will need to be nudged and this function is slow (mostly due to calling np.isclose() on entire array)
        eps = max(float(self.mesh.dz) * 1e-8, 1e-14)
        lower = float(z[0] + eps)
        upper = float(z[-1] - eps)
        if strict and not (lower < interface_position < upper):
            debugInPlace()
            raise ValueError("Interface position must lie strictly inside the FDM node domain.")
        clipped = float(np.clip(interface_position, lower, upper))
        if np.any(np.isclose(z, clipped, atol=eps, rtol=0.0)):
            if strict:
                # debugInPlace()
                raise ValueError("Interface position is too close to a node location (after clipping). Consider increasing the mesh spacing or perturbing the interface position.")
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
        Returns stored initial inventory using the configured initialization mode.

        This helper is used only to seed ``self._initialInventory`` at
        initialization/reset/load. Runtime mass and inventory checks still use
        the standard interface-aware integration paths.
        '''
        if self.initialInventoryMode == "integrated":
            if self._isBinarySystem():
                return self.getTotalMass(time)
            return self.getTotalInventory(time)
        if self.initialInventoryMode == "phase_length_idealized":
            initInventory = self._getPhaseLengthIdealizedInitialInventory(time)
            print(f"Initial mass: {initInventory}")
            return initInventory
        raise ValueError("initialInventoryMode must be one of ['integrated', 'phase_length_idealized'].")

    def _getPhaseLengthIdealizedInitialInventory(self, time=None):
        '''
        Computes initial inventory from phase lengths times constant phase composition.

        The left phase is defined over nodes ``[:geometry.right_index]`` and the
        right phase over ``[geometry.right_index:]``. Each phase must be
        compositionally constant (within ``atol=self.constraints.minComposition``).
        '''
        composition = np.asarray(self.data.y(time), dtype=np.float64)
        interface_position = self.getInterfacePosition(time)
        geometry = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar, self.ignoredNodeRule)
        z = np.asarray(self.mesh.z, dtype=np.float64).reshape(-1)
        left_slice = slice(0, geometry.right_index)
        right_slice = slice(geometry.right_index, composition.shape[0])

        if self._isBinarySystem() or (composition.ndim > 1 and composition.shape[1] == 1):
            composition = composition.reshape(-1)
            left_phase = composition[left_slice]
            right_phase = composition[right_slice]
            left_value = self._requireConstantPhaseComposition(left_phase, "left", None)
            right_value = self._requireConstantPhaseComposition(right_phase, "right", None)
            left_length = float(interface_position - z[0])
            right_length = float(z[-1] - interface_position)
            return float(left_value * left_length + right_value * right_length)

        left_phase = composition[left_slice, :]
        right_phase = composition[right_slice, :]
        left_values = np.zeros(composition.shape[1], dtype=np.float64)
        right_values = np.zeros(composition.shape[1], dtype=np.float64)
        for i in range(composition.shape[1]):
            left_values[i] = self._requireConstantPhaseComposition(left_phase[:, i], "left", i)
            right_values[i] = self._requireConstantPhaseComposition(right_phase[:, i], "right", i)
        left_length = float(interface_position - z[0])
        right_length = float(z[-1] - interface_position)
        return left_values * left_length + right_values * right_length

    def _requireConstantPhaseComposition(self, phase_values, phase_name, component_index):
        '''
        Validates that a 1D phase composition array is constant and returns its value.
        '''
        values = np.asarray(phase_values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError(f"Cannot evaluate {phase_name} phase constancy because it contains no nodes.")
        reference = float(values[0])
        if not np.allclose(values, reference, rtol=0.0, atol=self.constraints.minComposition):
            if component_index is None:
                component_label = self.elements[0] if len(self.elements) > 0 else "independent component"
            else:
                component_label = self.elements[component_index]
            debugInPlace()
            raise ValueError(
                f"phase_length_idealized initial inventory requires constant composition in each phase; "
                f"{phase_name} phase is not constant for component '{component_label}'."
            )
        return reference

    def _integrateComponentInventory(self, composition, interface_position, interface_compositions, component_index, s_for_interp, s_old=None, p_old=None, s_new=None):
        '''
        Integrates a single independent component using the binary sharp-interface helper.
        '''
        # s_for_interp="old"
        if s_old is None:
            s_old = interface_position
        if s_new is None:
            s_new = interface_position
        if p_old is None:
            p_old = get_moving_boundary_fd_geometry(self.mesh, s_old, self.pstar, self.ignoredNodeRule).p
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
            ignored_node_rule=self.ignoredNodeRule,
            ignored_node_reconstruction_mode=self.ignoredNodeReconstructionMode,
            integration_mode=self.integrationMode,
            s_for_interp=s_for_interp,
        )

    def _integrateInventory(self, composition, interface_position, interface_compositions, s_for_interp, s_old=None, p_old=None, s_new=None):
        '''
        Integrates either the scalar binary inventory or the vector ternary inventories.
        '''
        comp = np.asarray(composition, dtype=np.float64)
        if comp.ndim == 1 or comp.shape[1] == 1:
            return self._integrateComponentInventory(comp, interface_position, interface_compositions, 0, s_for_interp=s_for_interp, s_old=s_old, p_old=p_old, s_new=s_new)
        return np.array(
            [
                self._integrateComponentInventory(comp, interface_position, interface_compositions, i, s_for_interp=s_for_interp, s_old=s_old, p_old=p_old, s_new=s_new)
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
                    ignored_node_rule=self.ignoredNodeRule,
                    ignored_node_reconstruction_mode=self.ignoredNodeReconstructionMode,
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
                    ignored_node_rule=self.ignoredNodeRule,
                    ignored_node_reconstruction_mode=self.ignoredNodeReconstructionMode,
                ),
                dtype=np.float64,
            )
        return reconstructed

    def _composeInterfaceProbe(self, composition, geometry, lam: float):
        '''
        Builds the multicomponent local-equilibrium probe state from the bracketing node compositions.
        '''
        comp = np.asarray(composition, dtype=np.float64)
        # left = np.asarray(comp[geometry.left_index], dtype=np.float64).reshape(-1) ##XXX: I don't think these are the appropriate compositions for bracketing. They should correspond to the extreme tielines which these don't necessarily do.
        # right = np.asarray(comp[geometry.right_index], dtype=np.float64).reshape(-1) ##XXX: I don't think these are the appropriate compositions for bracketing. They should correspond to the extreme tielines which these don't necessarily do.
        left = np.asarray(self.therm.left_probe, dtype=np.float64).reshape(-1)
        right = np.asarray(self.therm.right_probe, dtype=np.float64).reshape(-1)
        return self._clipIndependentCompositionVector((1.0 - lam) * left + lam * right)

    def _multicomponentResidualTolerance(self, velocities):
        '''
        Returns an absolute tolerance for matching the per-component interface velocities.
        '''
        scale = max(1.0, float(np.max(np.abs(np.asarray(velocities, dtype=np.float64)))))
        return max(1e-10, 1e-6 * scale)

    def _assembleMulticomponentInterfaceState(
        self,
        t,
        composition,
        interface_position,
        c_left_int,
        c_right_int,
        D_left_int,
        D_right_int,
        lam=None,
        geometry=None,
        temperature=None,
        probe=None,
        dt=None,
        denom_type="eqn22",
        reconstructPreviouslyIgnoredComposition=False,
        reconstructPreviouslyIgnoredCompositionFunc=None,
        
    ):
        '''
        Assembles fluxes and interface velocities for a known ternary interface tie-line.
        '''
        if reconstructPreviouslyIgnoredComposition:
            composition = reconstructPreviouslyIgnoredCompositionFunc((c_left_int, c_right_int))
        if geometry is None:
            geometry = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar, self.ignoredNodeRule)
        if temperature is None:
            temperature = float(self.temperatureParameters(np.array([[interface_position]]), t)[0])

        grad_left, grad_right = self._interface_gradients(composition, interface_position, (c_left_int, c_right_int))
        flux_left = -np.matmul(D_left_int, grad_left)
        flux_right = -np.matmul(D_right_int, grad_right)

        comp = np.asarray(composition, dtype=np.float64)
        left_node = np.asarray(comp[geometry.left_index], dtype=np.float64).reshape(-1)
        right_node = np.asarray(comp[geometry.right_index], dtype=np.float64).reshape(-1)
        if denom_type=="eqn22":
            denom = c_right_int + right_node - c_left_int - left_node
        elif denom_type=="eqn11":
            denom = c_right_int - c_left_int
        else:
            debugInPlace()
            raise ValueError("denom_type should be one of the above")
        if np.any(np.abs(denom) <= 1e-14):
            raise ValueError("Ternary MovingBoundaryFD1DModel encountered a near-zero Eq. (22) denominator.")
        
        if self.interfaceUpdate=="1999_lee_allSolute_corrected":
        # if True==False:
            # raise ValueError("Interface update method '1999_lee_allSolute_corrected' needs to be reexamined")
            if denom_type!="eqn22":
                raise ValueError(f"denom_type must be 'eqn22' for interface update method '1999_lee_allSolute_corrected'. Got denom_type='{denom_type}'")
            if dt is None:
                debugInPlace()
                raise ValueError("dt should not be None when self.interfaceUpdate=='1999_lee_allSolute_corrected'")
            if dt=="0": ## This is used by getTotalInventory and so should be at the start when self.previousMassDelta is all zeros
                if (self.previousMassDelta==0).all()!=True:
                    debugInPlace()
                    raise ValueError("dt should not be 0 when self.interfaceUpdate=='1999_lee_allSolute_corrected'")
                velocities = (2.0 * (flux_right - flux_left)) / denom
            else:
                velocities = (2.0 * (flux_right - flux_left) + (2.0 * self.previousMassDelta/dt)) / denom
        else: 
            if denom_type=="eqn22":
                velocities = (2.0 * (flux_right - flux_left)) / denom
            elif denom_type=="eqn11":
                velocities = (flux_right - flux_left) / denom
            else:
                raise ValueError("denom_type should be one of the above")
            
            
        residual = float(velocities[0] - velocities[1])
        return {
            "lambda": None if lam is None else float(lam),
            "geometry": geometry,
            "temperature": float(temperature),
            "probe": probe,
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

    def _evaluateMulticomponentInterfaceState(self, t, composition, interface_position, lam, geometry=None, temperature=None, dt=None, denom_type=None, reconstructPreviouslyIgnoredComposition=False, reconstructPreviouslyIgnoredCompositionFunc=None):
        '''
        Evaluates one candidate ternary interface state and the corresponding equal-velocity residual.
        '''
        
        if geometry is None:
            geometry = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar, self.ignoredNodeRule)
        if temperature is None:
            temperature = float(self.temperatureParameters(np.array([[interface_position]]), t)[0])

        # x_probe = self._composeInterfaceProbe(composition, geometry, lam)
        try:
            c_a_int, c_b_int, meta = self.therm.getInterfacialComposition(
                # x_probe,
                lam,
                temperature,
                0,
                precPhase=self.phases[1],
                returnMeta=True,
                xIsLambda=True,
            )
        except TypeError as exc:
            raise ValueError(
                "Ternary MovingBoundaryFD1DModel requires getInterfacialComposition(..., returnMeta=True) "
                "with endpoint phase metadata."
            ) from exc
        c_left_int, c_right_int = self._orderMulticomponentInterfaceByPhase(
            c_a_int,
            c_b_int,
            meta,
            t,
            interface_position,
        )
        c_left_int = self._normalizeThermoIndependentComposition(c_left_int)
        c_right_int = self._normalizeThermoIndependentComposition(c_right_int)

        D_left_int = np.asarray(self.therm.getInterdiffusivity(c_left_int, temperature, phase=self.phases[0], query_context="interface"), dtype=np.float64).reshape(len(self.elements), len(self.elements))
        D_right_int = np.asarray(self.therm.getInterdiffusivity(c_right_int, temperature, phase=self.phases[1], query_context="interface"), dtype=np.float64).reshape(len(self.elements), len(self.elements))
        return self._assembleMulticomponentInterfaceState(
            t,
            composition,
            interface_position,
            c_left_int,
            c_right_int,
            D_left_int,
            D_right_int,
            lam=lam,
            geometry=geometry,
            temperature=temperature,
            # probe=x_probe,
            probe=None,
            dt=dt,
            denom_type=denom_type,
            reconstructPreviouslyIgnoredComposition=reconstructPreviouslyIgnoredComposition,
            reconstructPreviouslyIgnoredCompositionFunc=reconstructPreviouslyIgnoredCompositionFunc,
        )

    def _orderMulticomponentInterfaceByPhase(self, c_a_int, c_b_int, meta, t, interface_position):
        '''
        Reorders ternary interface endpoints so left/right compositions match the
        left/right phase convention of the moving-boundary model.
        '''
        if meta is None:
            raise ValueError(
                f"Missing interface endpoint phase metadata at t={t:.6g}, s={interface_position:.6g}."
            )

        expected_left = self.phases[0]
        expected_right = self.phases[1]
        if not isinstance(meta, dict) or "endpoints" not in meta:
            raise ValueError(
                f"Invalid interface metadata format at t={t:.6g}, s={interface_position:.6g}: {meta!r}"
            )
        endpoints = meta["endpoints"]
        if endpoints is None or len(endpoints) != 2:
            raise ValueError(
                f"Invalid interface metadata endpoints at t={t:.6g}, s={interface_position:.6g}: {meta!r}"
            )

        phase_a = endpoints[0].get("phase", None) if isinstance(endpoints[0], dict) else None
        phase_b = endpoints[1].get("phase", None) if isinstance(endpoints[1], dict) else None
        if phase_a is None or phase_b is None:
            raise ValueError(
                f"Missing endpoint phases at t={t:.6g}, s={interface_position:.6g}: "
                f"phase_a={phase_a}, phase_b={phase_b}, expected=({expected_left}, {expected_right})."
            )
        if phase_a == phase_b:
            raise ValueError(
                f"Ambiguous interface endpoint phases at t={t:.6g}, s={interface_position:.6g}: "
                f"both endpoints reported phase '{phase_a}'."
            )

        if phase_a == expected_left and phase_b == expected_right:
            return c_a_int, c_b_int
        if phase_a == expected_right and phase_b == expected_left:
            return c_b_int, c_a_int

        raise ValueError(
            f"Interface endpoint phase mismatch at t={t:.6g}, s={interface_position:.6g}: "
            f"returned=({phase_a}, {phase_b}), expected=({expected_left}, {expected_right})."
        )

    def _solveMulticomponentInterfaceState(self, t, composition, interface_position, dt=None, denom_type=None, reconstructPreviouslyIgnoredComposition=False, reconstructPreviouslyIgnoredCompositionFunc=None):
        '''
        Solves the ternary equal-velocity interface condition by a bracketed 1D search.
        '''
        return self._solveMulticomponentInterfaceState_alt(t, composition, interface_position, dt=dt, denom_type=denom_type, reconstructPreviouslyIgnoredComposition=reconstructPreviouslyIgnoredComposition, reconstructPreviouslyIgnoredCompositionFunc=reconstructPreviouslyIgnoredCompositionFunc)
        geometry = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar, self.ignoredNodeRule)
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

    def _solveMulticomponentInterfaceState_alt(self, t, composition, interface_position, root_scalar_method="brentq", dt=None, denom_type=None, reconstructPreviouslyIgnoredComposition=False, reconstructPreviouslyIgnoredCompositionFunc=None):
        '''
        Alternate ternary equal-velocity solver using ``scipy.optimize.root_scalar``.

        This variant keeps the same physical residual as
        ``_solveMulticomponentInterfaceState`` but delegates the 1D root solve to
        SciPy (``brentq`` on a bracket with a sign change).
        '''
        geometry = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar, self.ignoredNodeRule)
        temperature = float(self.temperatureParameters(np.array([[interface_position]]), t)[0])
        # left_state = self._evaluateMulticomponentInterfaceState(t, composition, interface_position, lam=0, geometry=geometry,temperature=temperature, dt=dt, denom_type=denom_type, reconstructPreviouslyIgnoredComposition=reconstructPreviouslyIgnoredComposition, reconstructPreviouslyIgnoredCompositionFunc=reconstructPreviouslyIgnoredCompositionFunc)
        # right_state = self._evaluateMulticomponentInterfaceState(t, composition, interface_position, lam=1, geometry=geometry,temperature=temperature, dt=dt, denom_type=denom_type, reconstructPreviouslyIgnoredComposition=reconstructPreviouslyIgnoredComposition, reconstructPreviouslyIgnoredCompositionFunc=reconstructPreviouslyIgnoredCompositionFunc)
        # if np.sign(left_state["residual"]) != np.sign(right_state["residual"]):
        #     bracket_pair = (left_state, right_state)
        # else:
        #     debugInPlace()
        #     raise ValueError("Extreme bracket (lambda=0 and lambda=1) is not valid as both have same sign")

        # if bracket_pair is not None:
        #     left_state, right_state = bracket_pair
        def residual_func(lam):
            try:
                state = self._evaluateMulticomponentInterfaceState(
                    t,
                    composition,
                    interface_position,
                    float(lam),
                    geometry=geometry,
                    temperature=temperature,
                    dt=dt,
                    denom_type=denom_type,
                    reconstructPreviouslyIgnoredComposition=reconstructPreviouslyIgnoredComposition,
                    reconstructPreviouslyIgnoredCompositionFunc=reconstructPreviouslyIgnoredCompositionFunc,
                )
                if not np.isfinite(state["residual"]):
                    return np.nan
                return float(state["residual"])
            except Exception as e:
                print(lam)
                raise e
                return np.nan
        try:
            sol = optimize.root_scalar(
                residual_func,
                # bracket=[0, 1],
                bracket=[0.25, 0.5],
                method=root_scalar_method,
                # rtol=1e-14,
                xtol=1e-10,
                maxiter=100,
            )
        except Exception as e:
            debugInPlace()
            raise e
        if sol.converged:
            root_state = self._evaluateMulticomponentInterfaceState(
                t,
                composition,
                interface_position,
                float(sol.root),
                geometry=geometry,
                temperature=temperature,
                dt=dt,
                denom_type=denom_type,
                reconstructPreviouslyIgnoredComposition=reconstructPreviouslyIgnoredComposition,
                reconstructPreviouslyIgnoredCompositionFunc=reconstructPreviouslyIgnoredCompositionFunc,
            )
            if abs(root_state["residual"]) <= root_state["tolerance"]:
                return root_state
            else:
                debugInPlace()
        debugInPlace()
        raise ValueError(
            "Ternary MovingBoundaryFD1DModel (alt solver) could not match the Eq. (22) interface velocities "
            f"for {self.elements}; smallest residual was {root_state['residual']}."
        )

    def _cacheMulticomponentInterfaceState(self, t, composition, interface_position, solve_stage, state):
        '''
        Stores the most recent ternary interface state for later retrieval.
        '''
        self._cachedMulticomponentInterfaceState = {
            "time": float(t),
            "interface_position": float(interface_position),
            "solve_stage": str(solve_stage),
            "composition": np.array(composition, dtype=np.float64, copy=True),
            "state": state,
        }
        return state

    def _getCachedMulticomponentInterfaceState(self, t=None, composition=None, interface_position=None, solve_stage=None):
        '''
        Returns the cached ternary interface state when the requested state matches.
        '''
        cached = self._cachedMulticomponentInterfaceState
        if cached is None:
            return None
        if solve_stage is not None and cached["solve_stage"] != str(solve_stage):
            return None
        if t is not None and not np.isclose(float(cached["time"]), float(t), rtol=0.0, atol=1e-14):
            return None
        if interface_position is not None and not np.isclose(
            float(cached["interface_position"]),
            float(interface_position),
            rtol=0.0,
            atol=max(1e-14, float(self.mesh.dz) * 1e-10),
        ):
            return None
        ##XXX: This check will likely be slow and should eventually be changed to avoid it
        if composition is not None and not np.allclose(
            np.asarray(cached["composition"], dtype=np.float64),
            np.asarray(composition, dtype=np.float64),
            rtol=0.0,
            atol=1e-14,
        ):
            return None
        return cached["state"]

    def _getInterfaceState(self, t, composition, interface_position, allow_solve=False, dt=None):
        '''
        Returns geometry, interface compositions, and interfacial diffusivities.

        For ternary systems, this method prefers a cached interface state and
        only launches a new multicomponent solve when ``allow_solve`` is set
        explicitly to ``True``.
        '''
        geometry = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar, self.ignoredNodeRule)
        T_interface = float(self.temperatureParameters(np.array([[interface_position]]), t)[0])
        if self._isBinarySystem():
            c_left_int, c_right_int = self.therm.getInterfacialComposition(T_interface, 0, precPhase=self.phases[1])
            c_left_int = float(np.clip(np.squeeze(c_left_int), self.constraints.minComposition, 1 - self.constraints.minComposition))
            c_right_int = float(np.clip(np.squeeze(c_right_int), self.constraints.minComposition, 1 - self.constraints.minComposition))
            D_left_int = float(np.squeeze(self.therm.getInterdiffusivity(c_left_int, T_interface, phase=self.phases[0], query_context="interface")))
            D_right_int = float(np.squeeze(self.therm.getInterdiffusivity(c_right_int, T_interface, phase=self.phases[1], query_context="interface")))
            return geometry, T_interface, c_left_int, c_right_int, D_left_int, D_right_int

        state = self._getCachedMulticomponentInterfaceState(t=t, composition=composition, interface_position=interface_position)
        if state is None:
            if not allow_solve:
                raise ValueError(
                    "No cached ternary interface state is available for the requested time/profile. "
                    "Call the ternary step driver first or request allow_solve=True."
                )
            state = self._solveMulticomponentInterfaceState(t, np.asarray(composition, dtype=np.float64), interface_position, dt=dt, denom_type=self.denom_type)
            self._cacheMulticomponentInterfaceState(t, composition, interface_position, "explicit_request", state)
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
                self.therm.getInterdiffusivity(comp[left_slice], temperatures[left_slice], phase=self.phases[0], query_context="general"),
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
                self.therm.getInterdiffusivity(comp[right_slice], temperatures[right_slice], phase=self.phases[1], query_context="general"),
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
        geom = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar, self.ignoredNodeRule)
        if geom.ignore_mode == "ignore_left":
            i1, i2 = geom.left_index - 2, geom.left_index - 1
            j1, j2 = geom.right_index, geom.right_index + 1
        elif geom.ignore_mode == "ignore_right":
            i1, i2 = geom.left_index - 1, geom.left_index
            j1, j2 = geom.right_index + 1, geom.right_index + 2
        else:
            i1, i2 = geom.left_index - 1, geom.left_index
            j1, j2 = geom.right_index, geom.right_index + 1
        
        if (i1<0) or (i2<0) or (j1>(len(z)-1)) or (j2>(len(z)-1)):
            debugInPlace()

        # i1 = max(0, i1)
        # i2 = max(0, i2)
        # j1 = min(len(z) - 1, j1)
        # j2 = min(len(z) - 1, j2)
        
        if geom.ignored_index is not None and geom.ignored_index in [i1, i2, j1, j2]:
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
        raise ValueError("This function is currently disabled as it may be overly restrictive and needs review.")
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
        raise ValueError("I don't think this is currently implemented correctly and it shouldn't be needed right now anyways so I have disabled it")
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
        geom = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar, self.ignoredNodeRule)
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
            # debugInPlace()
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
        self._pendingDtDiff = float(dt_diff)
        self._pendingDtMove = float(dt_move)
        self._currdt = min(dt_diff, dt_move, allowed_dt)
        if dt_diff>dt_move:
            raise ValueError("Not Expecting dt_move to control the time step at this point")

        ignored = geom.ignored_index
        if geom.ignore_mode == "ignore_left":
            a_last = geom.left_index - 1
            b_first = geom.right_index
        elif geom.ignore_mode == "ignore_right":
            a_last = geom.left_index
            b_first = geom.right_index + 1
        else:
            a_last = geom.left_index
            b_first = geom.right_index

        bulk_dcdt = self._bulk_dcdt(c_old, diffusivity_nodes)
        bulk_mask = np.zeros(len(c_old), dtype=bool)
        if a_last >= 0:
            bulk_mask[: a_last + 1] = True
        if b_first < len(c_old):
            bulk_mask[b_first:] = True
        if ignored is not None:
            bulk_mask[ignored] = False

        left_near_active = geom.left_near_index >= 1 and geom.left_near_index <= a_last
        right_near_active = geom.right_near_index + 1 < len(c_old) and geom.right_near_index >= b_first
        if left_near_active:
            bulk_mask[geom.left_near_index] = False
        if right_near_active:
            bulk_mask[geom.right_near_index] = False
        if (left_near_active!=True) or (right_near_active!=True):
            debugInPlace()

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
                ignored_node_rule=self.ignoredNodeRule,
                ignored_node_reconstruction_mode=self.ignoredNodeReconstructionMode,
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
                mass_func = lambda s: integrate_binary_fd_profile(self.mesh.z, c_stage, s_old=s_old, p_old=geom.p, s_new=s, pstar=self.pstar, interface_compositions=(c_left_int, c_right_int), ignored_node_rule=self.ignoredNodeRule, ignored_node_reconstruction_mode=self.ignoredNodeReconstructionMode, integration_mode=self.integrationMode, s_for_interp="new")
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
        elif self.interfaceUpdate == "1999_lee_allSolute_corrected":
            debugInPlace()


        else:
            raise ValueError("Should be one of above options")




        max_fraction = self.constraints.movingBoundaryThreshold #self._max_interface_step_fraction(geom, velocity) # this was removed since _max_interface_step_fraction() will prevent interface from ever crossing node
        # if (t==0) and (self.interfaceUpdate == "my_corrected"):
        #     max_fraction=0.5
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
        # fluxes, left_flux, right_flux = self._compute_fluxes(c_final, diffusivity_nodes, s_old, interface_compositions, interface_diffusivities) ## This compute_fluxes seems unnecessary and possibly even wrong?
        self._lastFluxes = None # fluxes
        self._lastInterfaceFluxes = None # (np.asarray(left_flux, dtype=np.float64), np.asarray(right_flux, dtype=np.float64))
        self._lastInterfaceVelocity = None # float(velocity)
        return dcdt[:, np.newaxis], float(velocity)

    def _computeStateTernary(self, t, xCurr):
        '''
        Computes composition rates and interface velocity for one ternary explicit FDM step.
        '''
        # debugInPlace()    
        c_old = np.asarray(xCurr[0], dtype=np.float64)
        s_old = self._clipInterfacePosition(float(xCurr[1]))
        geom = get_moving_boundary_fd_geometry(self.mesh, s_old, self.pstar, self.ignoredNodeRule)

        last_s_old = getattr(self, "_last_s_old", None)
        if last_s_old is not None:
            last_interface_compositions = self._last_interface_compositions
            last_c_final = self._last_c_final
            c_reconstruct = self._reconstructIgnoredComposition(c_old, last_s_old, geom.p, s_old, last_interface_compositions)
            if (np.max(np.abs(c_reconstruct-c_old))>1e-16) or (np.max(np.abs(c_reconstruct-c_old))>1e-16):
                debugInPlace()
                raise ValueError("Should be the same")
            
        diffusivity_nodes = self._bulk_diffusivity_nodes(c_old, t, geom)
        
        # if True==False:
        if self.interfaceUpdate in ["1999_lee_allSolute_corrected"]:
            # max_diff_onlyBulk = float(np.max(np.abs(diffusivity_nodes.reshape(-1))))
            max_diff_onlyBulk = getMaxEigVal(diffusivity_nodes)

            if self.ignoredNodeRule == "legacy_two_region":
                min_length = self.mesh.dz * ((1.0 - geom.p) if geom.p < self.pstar else geom.p)
            else:
                if self.ignoredNodeRule != "lee_oh_1996_three_region":
                    raise ValueError("ignoredNodeRule must be either 'legacy_two_region' or 'lee_oh_1996_three_region'")
                if geom.p<self.pstar:
                    min_length = 1-geom.p
                elif geom.p>(1-self.pstar):
                    min_length = geom.p
                else:
                    min_length = min(geom.p, 1-geom.p)
                min_length = self.mesh.dz * min_length
            
            trial_factor = 0.9 ## used to make sure dt_trial is less than dt_diff
            dt_trial = trial_factor * self.constraints.vonNeumannThreshold * (min_length**2) / max_diff_onlyBulk if max_diff_onlyBulk > 0 else np.inf

            dt_overall = -np.inf
            
            trial_count = 0
            maxNumTrials=5
            
            for trial in range(maxNumTrials):
                trial_count += 1
                
                pre_state = self._solveMulticomponentInterfaceState(t, c_old, s_old, dt=dt_trial, denom_type=self.denom_type)
                # self._cacheMulticomponentInterfaceState(t, c_old, s_old, "pre", pre_state)
                c_left_int_pre, c_right_int_pre = pre_state["interface_compositions"]
                # sideOfProbe = self.therm._check_side_of_probe([c_old[0], c_old[-1], c_left_int_pre, c_right_int_pre])
                # assert(sideOfProbe[0]==sideOfProbe[2] and sideOfProbe[1]==sideOfProbe[3])
                D_left_int_pre, D_right_int_pre = pre_state["interface_diffusivities"]
                
                # max_diff = float(np.max(np.abs(np.concatenate((diffusivity_nodes.reshape(-1), D_left_int_pre.reshape(-1), D_right_int_pre.reshape(-1))))))
                D_ints = np.stack((D_left_int_pre, D_right_int_pre))
                max_diff = getMaxEigVal(np.concatenate((diffusivity_nodes, D_ints)))

                min_length = self.mesh.dz * ((1.0 - geom.p) if geom.p < self.pstar else geom.p)
                dt_diff = self.constraints.vonNeumannThreshold * (min_length**2) / max_diff if max_diff > 0 else np.inf

                move_fraction = min(self.constraints.movingBoundaryThreshold, np.inf)
                dt_move = move_fraction * self.mesh.dz / abs(pre_state["velocity"]) if abs(pre_state["velocity"]) > 0 else np.inf
                allowed_dt = getattr(self, "deltaTime", np.inf)
                dt_overall = min(dt_diff, dt_move, allowed_dt)

                if dt_overall < dt_trial:
                    if (trial_count)>=maxNumTrials:
                        debugInPlace()
                        raise ValueError("Max number of trials exceeded")
                    # debugInPlace()
                    print(f"Trial dt {dt_trial} larger than min(dt_diff, dt_move, allowed_dt)=min({dt_diff}, {dt_move}, {allowed_dt})={min(dt_diff, dt_move, allowed_dt)}, reducing and retrying.")
                    # dt_trial *= 1/2
                    dt_trial = dt_overall * 0.95
                    # numRetrials = getattr(self, "numRetrials", 0)
                    # self.numRetrials = numRetrials
                    # self.numRetrials += 1
                    
                else:
                    self._pendingDtDiff = float(dt_diff)
                    self._pendingDtMove = float(dt_move)
                    self._currdt = dt_trial
                    break
            

        else:
            pre_state = self._solveMulticomponentInterfaceState(t, c_old, s_old, denom_type=self.denom_type)
            # self._cacheMulticomponentInterfaceState(t, c_old, s_old, "pre", pre_state)
            c_left_int_pre, c_right_int_pre = pre_state["interface_compositions"]
            # sideOfProbe = self.therm._check_side_of_probe([c_old[0], c_old[-1], c_left_int_pre, c_right_int_pre])
            # assert(sideOfProbe[0]==sideOfProbe[2] and sideOfProbe[1]==sideOfProbe[3])
            D_left_int_pre, D_right_int_pre = pre_state["interface_diffusivities"]
            
            # max_diff = float(np.max(np.abs(np.concatenate((diffusivity_nodes.reshape(-1), D_left_int_pre.reshape(-1), D_right_int_pre.reshape(-1))))))
            D_ints = np.stack((D_left_int_pre, D_right_int_pre))
            max_diff = getMaxEigVal(np.concatenate((diffusivity_nodes, D_ints)))
            # debugInPlace()
            # min_length = self.mesh.dz * ((1.0 - geom.p) if geom.p < self.pstar else geom.p)
            if self.ignoredNodeRule == "legacy_two_region":
                min_length = self.mesh.dz * ((1.0 - geom.p) if geom.p < self.pstar else geom.p)
            else:
                if self.ignoredNodeRule != "lee_oh_1996_three_region":
                    raise ValueError("ignoredNodeRule must be either 'legacy_two_region' or 'lee_oh_1996_three_region'")
                if geom.p<self.pstar:
                    min_length = 1-geom.p
                elif geom.p>(1-self.pstar):
                    min_length = geom.p
                else:
                    min_length = min(geom.p, 1-geom.p)
                min_length = self.mesh.dz * min_length
            
            dt_diff = self.constraints.vonNeumannThreshold * (min_length**2) / max_diff if max_diff > 0 else np.inf

            move_fraction = min(self.constraints.movingBoundaryThreshold, np.inf)
            dt_move = move_fraction * self.mesh.dz / abs(pre_state["velocity"]) if abs(pre_state["velocity"]) > 0 else np.inf
            allowed_dt = getattr(self, "deltaTime", np.inf)
            self._pendingDtDiff = float(dt_diff)
            self._pendingDtMove = float(dt_move)
            self._currdt = min(dt_diff, dt_move, allowed_dt)

        ignored = geom.ignored_index
        if geom.ignore_mode == "ignore_left":
            a_last = geom.left_index - 1
            b_first = geom.right_index
        elif geom.ignore_mode == "ignore_right":
            a_last = geom.left_index
            b_first = geom.right_index + 1
        else:
            a_last = geom.left_index
            b_first = geom.right_index

        bulk_dcdt = np.asarray(self._bulk_dcdt(c_old, diffusivity_nodes), dtype=np.float64)
        bulk_mask = np.zeros(c_old.shape[0], dtype=bool)
        if a_last >= 0:
            bulk_mask[: a_last + 1] = True
        if b_first < c_old.shape[0]:
            bulk_mask[b_first:] = True
        if ignored is not None:
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
        # raise ValueError("LOOK AT AND THINK ABOUT THE LINES BELOW!")
        if self.multicomponentInterfaceStateUpdate == "pre_and_post_diffusion":
            raise ValueError("I DON'T THINK I SHOULD BE USING THIS OPTION (at least not as written) 5-16-26")
            actual_state = self._solveMulticomponentInterfaceState(t, c_stage, s_old, dt=self._currdt)
            self._cacheMulticomponentInterfaceState(t, c_stage, s_old, "post", actual_state)
            chosen_stage = "post"
            post_compositions = actual_state["interface_compositions"]
        else:
            if self.fluxGradientMode == "post_diffusion":
                actual_state = self._assembleMulticomponentInterfaceState(
                    t,
                    c_stage,
                    s_old,
                    c_left_int_pre,
                    c_right_int_pre,
                    D_left_int_pre,
                    D_right_int_pre,
                    lam=pre_state["lambda"],
                    geometry=geom,
                    temperature=pre_state["temperature"],
                    probe=pre_state["probe"],
                    dt=None,
                    denom_type=None,
                ) 
                ## I think the        
                ## the INTERFACE GRADIENTS will change due to the change in the compositions around the interface. This will in turn led to different INTERFACIAL FLUXES.
                ## The different compositions around the interface lead to different composition values for LEFT and RIGHT nodes which then leads to different DENOM and thus a different VELOCITY
                ## Seems like it should not be used with self.interfaceUpdate=="1999_lee_allSolute_corrected" so I set dt=None so it will throw an error if it is used
                debugInPlace()
                actual_state, pre_state
            else:
                actual_state = pre_state
            chosen_stage = "pre"
            post_compositions = pre_state["interface_compositions"]
        pre_compositions = pre_state["interface_compositions"]
        interface_compositions = actual_state["interface_compositions"]
        # interface_diffusivities = actual_state["interface_diffusivities"]
        
        useOGfluxesButDiffusedComps=False
        if (self.multicomponentInterfaceStateUpdate == "pre_diffusion_only") and (self.fluxGradientMode == "pre_diffusion") and (useOGfluxesButDiffusedComps):
            if TODAY>date(2026, 5, 16):
                raise ValueError("Consider if this should still be done")
            left_node = np.asarray(c_stage[geom.left_index], dtype=np.float64).reshape(-1)
            right_node = np.asarray(c_stage[geom.right_index], dtype=np.float64).reshape(-1)
            if self.denom_type=="eqn22":
                denom = interface_compositions[1] + right_node - interface_compositions[0] - left_node
            elif self.denom_type=="eqn11":
                denom = interface_compositions[1] - interface_compositions[0]
            else:
                raise ValueError()
            if np.any(np.abs(denom) <= 1e-14):
                raise ValueError("Ternary MovingBoundaryFD1DModel encountered a near-zero Eq. (22) denominator.")
            if self.denom_type=="eqn22":
                velocities = (2.0 * (actual_state["fluxes"][1] - actual_state["fluxes"][0])) / denom
            elif self.denom_type=="eqn11":
                velocities = (actual_state["fluxes"][1] - actual_state["fluxes"][0]) / denom
            else:
                raise ValueError()
            
            velocity = float(np.mean(velocities))
            # debugInPlace()
            # velocities, actual_state["component_velocities"]
            # velocity, actual_state["velocity"]
            ds_intermediate = self._currdt * velocity
            
            actual_state["denominators"] = denom
        else:
            ds_intermediate = self._currdt * actual_state["velocity"]

        if self.interfaceUpdate == "basic":
            ds = ds_intermediate
            velocity = actual_state["velocity"]
        elif self.interfaceUpdate == "lee_oh_corrected":
            balance_index = self._balanceElementIndex
            balance_denom = float(actual_state["denominators"][balance_index])
            s_intermediate = s_old + ds_intermediate

            # debugInPlace()
            # post_state = self._solveMulticomponentInterfaceState(t, c_stage, s_intermediate, dt=None, denom_type=self.denom_type)
            # balance_denom = float(post_state["denominators"][balance_index])
            # interface_compositions = post_state["interface_compositions"]

            intermediate_inventory = self._integrateComponentInventory(
                c_stage, ##XXX: This might be something that should be altered
                s_intermediate,
                interface_compositions,
                balance_index,
                s_for_interp="new",
                s_old=s_old,
                p_old=geom.p,
                s_new=s_intermediate,
            )
            # debugInPlace()
            delta_inventory = intermediate_inventory - self._initialInventory[balance_index]
            ds = ds_intermediate + (2.0 * delta_inventory) / balance_denom
            velocity = ds / self._currdt if self._currdt > 0 else 0.0

            # interface_compositions = actual_state["interface_compositions"]
        elif self.interfaceUpdate == "my_corrected":
            balance_index = self._balanceElementIndex
            if balance_index is None:
                raise ValueError(
                    "Ternary my_corrected update requires balanceElement to choose the inventory-closure component."
                )

            ## Trying out a version where the interface compositions are solved each time and previously ignored points are reconstructed
            
            

            def component_inventory_residual(s, return_comps=False):

                
                def reconstructPreviouslyIgnoredCompositionFunc(interface_compositions):
                    return self._reconstructIgnoredComposition(c_stage, s_old, geom.p, s, interface_compositions)
                
                state_forMyCorrected = self._solveMulticomponentInterfaceState(t, c_stage, s_old, dt=self._currdt, denom_type=self.denom_type, reconstructPreviouslyIgnoredComposition=True, reconstructPreviouslyIgnoredCompositionFunc=reconstructPreviouslyIgnoredCompositionFunc)
                # # debugInPlace()
                # state_forMyCorrected
                # interface_compositions
                # interface_compositions_forMyCorrected = interface_compositions
                interface_compositions_forMyCorrected = state_forMyCorrected['interface_compositions']
                c_forMyCorrected = reconstructPreviouslyIgnoredCompositionFunc(interface_compositions_forMyCorrected)
                # c_forMyCorrected = c_stage
                
                
                # c_stage[np.where(np.linalg.norm(c_stage - c_forMyCorrected, axis=1)!=0)[0]]
                # c_forMyCorrected[np.where(np.linalg.norm(c_stage - c_forMyCorrected, axis=1)!=0)[0]]
                inventory = self._integrateComponentInventory(
                    c_forMyCorrected,
                    float(s),
                    interface_compositions_forMyCorrected,
                    balance_index,
                    s_for_interp="new",
                    s_old=s_old,
                    p_old=geom.p,
                    s_new=float(s),
                )
                if return_comps:
                    return float(inventory - self._initialInventory[balance_index]), interface_compositions_forMyCorrected, c_forMyCorrected
                return float(inventory - self._initialInventory[balance_index])
            
            # def component_inventory_residual(s):
                # inventory = self._integrateComponentInventory(
                #     c_stage,
                #     float(s),
                #     interface_compositions,
                #     balance_index,
                #     s_for_interp="new",
                #     s_old=s_old,
                #     p_old=geom.p,
                #     s_new=float(s),
                # )
            #     return float(inventory - self._initialInventory[balance_index])

            bracket_half_width = self.mesh.dz * self.constraints.movingBoundaryThreshold
            bracket = [s_old - bracket_half_width, s_old + bracket_half_width]
            bracket = np.clip(bracket, 1.5 * self.mesh.dz, self.mesh.zlim[0][-1] - (1.5 * self.mesh.dz)).tolist()
            try:
                sol = optimize.root_scalar(
                    component_inventory_residual,
                    bracket=bracket,
                    method='brentq',
                    rtol=1e-14,
                    xtol=1e-14,
                )
            except ValueError as exc:
                raise ValueError(
                    "Ternary my_corrected could not bracket a root for balance-element inventory closure "
                    f"({self.elements[balance_index]})."
                ) from exc
            if not sol.converged:
                raise ValueError(
                    "Ternary my_corrected root solve did not converge for balance-element inventory closure "
                    f"({self.elements[balance_index]})."
                )
            ds = float(sol.root - s_old)
            velocity = ds / self._currdt if self._currdt > 0 else 0.0

            _, interface_compositions_forMyCorrected, c_forMyCorrected = component_inventory_residual(sol.root, return_comps=True)
            interface_compositions = interface_compositions_forMyCorrected 
            
        elif self.interfaceUpdate == "1999_lee_allSolute_corrected":
            # debugInPlace()

            ds = ((2.0 * self._currdt * (actual_state['fluxes'][1]-actual_state['fluxes'][0]).sum()) + (2*self.previousMassDelta.sum())) /  (actual_state['denominators'].sum())
            intermediate_inventory = self._integrateInventory(
                c_stage, ##XXX: This might be something that should be altered
                s_old + ds,
                interface_compositions,
                s_for_interp="new",
                s_old=s_old,
                p_old=geom.p,
                s_new=s_old + ds,
                
            )
            delta_inventory = intermediate_inventory - self._initialInventory
            self.previousMassDelta = delta_inventory

            # s_intermediate = s_old + ds_intermediate
            
            # intermediate_inventory = self._integrateInventory(
            #     c_stage, ##XXX: This might be something that should be altered
            #     s_intermediate,
            #     interface_compositions,
            #     s_for_interp="new",
            # )
            # delta_inventory = intermediate_inventory - self._initialInventory
            # self.previousMassDelta = delta_inventory
            # ds = ds_intermediate

            velocity = ds / self._currdt if self._currdt > 0 else 0.0
        else:
            raise ValueError("interfaceUpdate should be one of ['basic', 'lee_oh_corrected', 'my_corrected', '1999_lee_allSolute_corrected'].")

        max_fraction = self.constraints.movingBoundaryThreshold
        requested_fraction = abs(ds) / self.mesh.dz
        if not np.isfinite(requested_fraction):
            raise ValueError("MovingBoundaryFD1DModel produced a non-finite interface increment.")
        max_fraction = max(max_fraction, 1e-12)
        if requested_fraction > max_fraction:
            requested_fraction_signed = ds / self.mesh.dz
            import math
            def binCurrentAndFuturep(current_p, requested_fraction_input):
                current_p_plus10 = current_p + 10
                current_p_plus10_modf = math.modf(current_p_plus10)
                future_p_modf = math.modf(current_p_plus10 + requested_fraction_input)
                # get_modf_relativeTo_pstar = lambda modf_input: (0.25, modf_input[1]) if modf_input[0] < self.pstar else (0.75, modf_input[1])
                def get_modf_relativeTo_pstar(modf_input): 
                    if modf_input[0] < self.pstar:
                        return (0.25, modf_input[1])
                    elif modf_input[0] > (1-self.pstar):
                        return (0.75, modf_input[1])
                    elif self.pstar < modf_input[0] < (1-self.pstar):
                        return (0.5, modf_input[1])
                    raise ValueError("Should be one of the above options!")
                return get_modf_relativeTo_pstar(current_p_plus10_modf), get_modf_relativeTo_pstar(future_p_modf)
            currentPBin, futurePBin = binCurrentAndFuturep(geom.p, requested_fraction_signed)
            Pbin_threshold = 0.5 if self.pstar==0.5 else 0.25
            # if abs(sum(currentPBin)-sum(futurePBin))>Pbin_threshold:
            if False==True:
                print(f"(max_fraction, requested_fraction_signed, t): {(max_fraction, requested_fraction_signed, t)}")
                print(f"(currentPBin, futurePBin): {(currentPBin, futurePBin)}")
                debugInPlace()
                raise ValueError("MovingBoundaryFD1DModel requested_fraction is greater than max_fraction")

        s_new = self._clipInterfacePosition(s_old + ds, strict=True)
        c_final = self._reconstructIgnoredComposition(c_new, s_old, geom.p, s_new, interface_compositions)
        # if (c_forMyCorrected==c_final).all()!=True:
        #     debugInPlace()
        # self._cacheMulticomponentInterfaceState(t + self._currdt, c_final, s_new, chosen_stage, actual_state)
        self._pendingInterfaceCompositionRecord = {
            "pre": (
                np.asarray(pre_compositions[0], dtype=np.float64).copy(),
                np.asarray(pre_compositions[1], dtype=np.float64).copy(),
            ),
            "post": (
                np.asarray(post_compositions[0], dtype=np.float64).copy(),
                np.asarray(post_compositions[1], dtype=np.float64).copy(),
            ),
            "used_stage": str(chosen_stage),
        }
        dcdt = (c_final - c_old) / self._currdt
        # fluxes, left_flux, right_flux = self._compute_fluxes(c_final, diffusivity_nodes, s_old, interface_compositions, interface_diffusivities) ## This compute_fluxes seems unnecessary and possibly even wrong?
        self._lastFluxes = None # fluxes
        self._lastInterfaceFluxes = None # (np.asarray(left_flux, dtype=np.float64), np.asarray(right_flux, dtype=np.float64))
        self._lastInterfaceVelocity = None # float(velocity)
        self._last_s_old = s_old
        self._last_interface_compositions = interface_compositions
        self._last_c_final = c_final
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
        if self._pstarStrictAudit and self._pstarChangedSinceLastPreProcess and self._cachedMulticomponentInterfaceState is not None:
            raise ValueError("Ternary interface cache should be cleared after a pstar update before getdXdt.")
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
        if TODAY>date(2026, 5, 30):
            raise ValueError("Consider if this should still be skipped")
        return ## I'm skipping this while I'm letting binCurrentAndFuturep() override max_fraction limits
        if dt <= 0:
            return
        interface_position = self._clipInterfacePosition(float(x[1]), strict=True)
        velocity = float(dXdt[1])
        if not np.isfinite(velocity):
            raise ValueError("velocity is infinite and I don't want this being zeroed silently")
            dXdt[1] = 0.0
            return

        geom = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar, self.ignoredNodeRule)
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
            raise ValueError("requested_ds > allowed_ds and I don't want this clipping velocity silently")
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
        if self._isBinarySystem():
            interface_compositions = self._getInterfaceState(self.currentTime, composition, interface_position)[2:4]
        else:
            cached = self._getCachedMulticomponentInterfaceState(t=self.currentTime, composition=composition, interface_position=interface_position)
            if cached is None:
                interface_compositions = self._getInterfaceState(self.currentTime, composition, interface_position, allow_solve=True)[2:4]
            else:
                interface_compositions = cached["interface_compositions"]
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
        Clips composition values, validates mass behavior, and records accepted-step state.
        '''
        GenericModel.postProcess(self, time, x)
        composition = self._clipCompositionField(np.asarray(x[0], dtype=np.float64))
        interface_position = self._clipInterfacePosition(float(x[1]))
        mass_check_composition = composition[:, 0] if composition.shape[1] == 1 else composition
        self._checkMassCorrection(mass_check_composition, interface_position)
        self.data.record(time, composition)
        self.interfaceData.record(time, interface_position)
        self.dtDiffData.record(time, self._pendingDtDiff)
        self.dtMoveData.record(time, self._pendingDtMove)
        if self._isTernarySystem() and self._interfaceCompositionHistory is not None:
            pending = self._pendingInterfaceCompositionRecord
            if pending is None:
                cached = self._getCachedMulticomponentInterfaceState(
                    t=time,
                    composition=composition,
                    interface_position=interface_position,
                )
                if cached is None:
                    raise ValueError("No ternary interface composition record is available for this accepted step.")
                left, right = cached["interface_compositions"]
                pending = {
                    "pre": (np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)),
                    "post": (np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)),
                    "used_stage": "pre",
                }
            self._recordInterfaceCompositionHistory(
                time,
                pending["pre"],
                pending["post"],
                pending["used_stage"],
            )
            self._pendingInterfaceCompositionRecord = None
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
        '''
        Finalizes recorded histories after solve completion.
        '''
        self.data.finalize()
        self.interfaceData.finalize()
        self.dtDiffData.finalize()
        self.dtMoveData.finalize()
        if self._isTernarySystem() and self._interfaceCompositionHistory is not None and self._hasInterfaceCompositionHistoryData:
            pending = self._pendingInterfaceCompositionRecord
            if pending is not None:
                self._recordInterfaceCompositionHistory(
                    self.interfaceData.currentTime,
                    pending["pre"],
                    pending["post"],
                    pending["used_stage"],
                    force=True,
                )
                self._pendingInterfaceCompositionRecord = None
            for key in ["time", "pre_left", "pre_right", "post_left", "post_right", "used_stage"]:
                self._interfaceCompositionHistory[key].finalize()

    def getInterfacePosition(self, time = None):
        '''
        Returns the interface position at a requested time
        '''
        return self.interfaceData.y(time)

    def getDtDiff(self, time = None):
        '''
        Returns the diffusion-limited timestep estimate at a requested time.
        '''
        return self.dtDiffData.y(time)

    def getDtMove(self, time = None):
        '''
        Returns the interface-motion-limited timestep estimate at a requested time.
        '''
        return self.dtMoveData.y(time)

    def getTotalMass(self, time = None):
        '''
        Returns the integrated binary composition inventory for the current profile
        '''
        if not self._isBinarySystem():
            raise ValueError("getTotalMass is only defined for binary MovingBoundaryFD1DModel systems. Use getTotalInventory instead.")
        composition = np.asarray(self.data.y(time), dtype=np.float64).reshape(-1)
        interface_position = self.getInterfacePosition(time)
        geom = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar, self.ignoredNodeRule)
        interface_compositions = self._getInterfaceState(self.currentTime if time is None else time, composition, interface_position)[2:4]
        return integrate_binary_fd_profile(
            self.mesh.z,
            composition,
            s_old=interface_position,
            p_old=geom.p,
            s_new=interface_position,
            pstar=self.pstar,
            interface_compositions=interface_compositions,
            ignored_node_rule=self.ignoredNodeRule,
            ignored_node_reconstruction_mode=self.ignoredNodeReconstructionMode,
            integration_mode=self.integrationMode,
            s_for_interp="old",
        )

    def  getTotalInventory(self, time = None):
        '''
        Returns the integrated inventory of each independent component for ternary systems.
        '''
        composition = np.asarray(self.data.y(time), dtype=np.float64)
        interface_position = self.getInterfacePosition(time)
        if self._isBinarySystem():
            interface_compositions = self._getInterfaceState(self.currentTime if time is None else time, composition, interface_position)[2:4]
        else:
            cached = self._getCachedMulticomponentInterfaceState(t=time, composition=composition, interface_position=interface_position)
            if cached is None:
                cached = self._getCachedMulticomponentInterfaceState(
                    t=self.currentTime if time is None else time,
                    composition=composition if time is None else None,
                    interface_position=interface_position if time is None else None,
                )
            if cached is not None:
                interface_compositions = cached["interface_compositions"]
            else:
                interface_compositions = self._getInterfaceState(
                    self.currentTime if time is None else time,
                    composition,
                    interface_position,
                    allow_solve=True,
                    dt="0",
                )[2:4]
        return self._integrateInventory(composition, interface_position, interface_compositions, s_for_interp="old")

    def getInterfaceCompositions(self, time = None, stage: str = "used"):
        '''
        Returns interface compositions at an exact recorded time for ternary systems.
        '''
        if self._isBinarySystem():
            if stage != "used":
                raise ValueError("Binary MovingBoundaryFD1DModel does not record ternary interface composition stages.")
            composition = np.asarray(self.data.y(time), dtype=np.float64)
            interface_position = self.getInterfacePosition(time)
            interface_state = self._getInterfaceState(
                self.currentTime if time is None else time,
                composition,
                interface_position,
                allow_solve=False,
            )
            return interface_state[2], interface_state[3]

        if stage not in {"used", "pre", "post"}:
            raise ValueError("stage must be one of ['used', 'pre', 'post'].")
        if self._interfaceCompositionHistory is None:
            raise ValueError("Ternary interface composition history is not initialized.")
        if not self._hasInterfaceCompositionHistoryData:
            raise ValueError("No ternary interface composition history is available yet.")

        h = self._interfaceCompositionHistory

        if time is None:
            if stage == "pre":
                return h["pre_left"].y(None), h["pre_right"].y(None)
            if stage == "post":
                return h["post_left"].y(None), h["post_right"].y(None)
            used_stage = "post" if float(h["used_stage"].y(None)) >= 0.5 else "pre"
            return (
                h["post_left"].y(None),
                h["post_right"].y(None),
            ) if used_stage == "post" else (
                h["pre_left"].y(None),
                h["pre_right"].y(None),
            )

        # exact-time only retrieval
        if stage == "pre":
            return h["pre_left"].y(time), h["pre_right"].y(time)
        if stage == "post":
            return h["post_left"].y(time), h["post_right"].y(time)
        used_stage = "post" if float(h["used_stage"].y(time)) >= 0.5 else "pre"
        return (
            h["post_left"].y(time),
            h["post_right"].y(time),
        ) if used_stage == "post" else (
            h["pre_left"].y(time),
            h["pre_right"].y(time),
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
            ignored_node_rule=self.ignoredNodeRule,
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

    def plotTernaryState(
        self,
        composition=None,
        interface_compositions=None,
        time=None,
        stage="used",
        ax=None,
        show_background=True,
        show_scatter=True,
        show_path=True,
        show_interface=True,
        highlight_near_interface=True,
        cond_step=0.01,
        scatter_kwargs=None,
        path_kwargs=None,
        interface_kwargs=None,
        background_kwargs=None,
        axis_lims={'x': (0, 1), 'y': (0, 1)},
        OG_interface_comps=None,
    ):
        '''
        Plots a ternary moving-boundary state on top of a ternary phase diagram.

        This debugger-oriented helper overlays the current (or user-supplied)
        composition trajectory and interface compositions onto a ``pycalphad``
        ternary diagram. Mesh points to the left and right of the interface are
        rendered with different colors for easier side-by-side debugging.
        '''
        if not self._isTernarySystem():
            raise ValueError("plotTernaryState is only available for ternary MovingBoundaryFD1DModel systems.")

        if composition is None:
            composition = np.asarray(self.data.y(time), dtype=np.float64)
        else:
            composition = np.asarray(composition, dtype=np.float64)
        if composition.ndim != 2 or composition.shape[1] != 2:
            raise ValueError("composition must have shape (n_nodes, 2) for ternary independent components.")

        if interface_compositions is None:
            interface_compositions = self.getInterfaceCompositions(time=time, stage=stage)
        left_int = np.asarray(interface_compositions[0], dtype=np.float64).reshape(-1)
        right_int = np.asarray(interface_compositions[1], dtype=np.float64).reshape(-1)
        if left_int.size != 2 or right_int.size != 2:
            raise ValueError("interface_compositions must contain two length-2 independent-component vectors.")

        interface_position = self.getInterfacePosition(time)
        t_eval = self.currentTime if time is None else float(time)
        T_plot = float(self.temperatureParameters(np.array([[interface_position]]), t_eval)[0])

        # Build full ternary compositions [X_ref, X_1, X_2] in model allElements order.
        ref = 1.0 - np.sum(composition, axis=1)
        full_comp = np.column_stack((ref, composition))
        full_comp = np.clip(full_comp, 0.0, 1.0)
        left_full = np.clip(np.array([1.0 - np.sum(left_int), left_int[0], left_int[1]], dtype=np.float64), 0.0, 1.0)
        right_full = np.clip(np.array([1.0 - np.sum(right_int), right_int[0], right_int[1]], dtype=np.float64), 0.0, 1.0)
        geom = get_moving_boundary_fd_geometry(self.mesh, interface_position, self.pstar, self.ignoredNodeRule)
        left_slice = slice(0, geom.left_index + 1)
        right_slice = slice(geom.right_index, full_comp.shape[0])

        import matplotlib.pyplot as plt
        from pycalphad import ternplot
        from pycalphad import variables as v

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
        else:
            fig = ax.figure

        if show_background:
            try:
                dbf = getattr(self.therm, "db", None)
                if dbf is None:
                    dbf = getattr(self.therm, "db_forPlotting", None)
                    if dbf is None:
                        raise ValueError("thermodynamics object does not expose a pycalphad database via '.db'.")
                bg_kwargs = {} if background_kwargs is None else dict(background_kwargs)
                conds = {
                    v.T: T_plot,
                    v.P: 101325,
                    v.X(self.elements[0]): (0, 1, cond_step),
                    v.X(self.elements[1]): (0, 1, cond_step),
                }
                ternplot(
                    dbf,
                    self.allElements + ["VA"],
                    self.phases,
                    conds,
                    x=v.X(self.elements[0]),
                    y=v.X(self.elements[1]),
                    ax=ax,
                    **bg_kwargs,
                )
            except Exception as exc:
                raise ValueError(
                    "plotTernaryState could not draw ternary background with ternplot. "
                    "Check thermodynamics database/phases/components compatibility."
                ) from exc

        if show_scatter:
            skw = {"s": 20, "alpha": 0.8}
            if scatter_kwargs is not None:
                skw.update(scatter_kwargs)
            if "color" in skw or "c" in skw:
                ax.scatter(full_comp[:, 1], full_comp[:, 2], **skw)
            else:
                ax.scatter(full_comp[left_slice, 1], full_comp[left_slice, 2], color="C0", label="Left Side", **skw)
                ax.scatter(full_comp[right_slice, 1], full_comp[right_slice, 2], color="C1", label="Right Side", **skw)

        if show_path:
            pkw = {"linewidth": 1.2, "alpha": 0.9}
            if path_kwargs is not None:
                pkw.update(path_kwargs)
            if "color" in pkw or "c" in pkw:
                ax.plot(full_comp[:, 1], full_comp[:, 2], **pkw)
            else:
                ax.plot(full_comp[left_slice, 1], full_comp[left_slice, 2], color="C0", label="Left Side", **pkw)
                ax.plot(full_comp[right_slice, 1], full_comp[right_slice, 2], color="C1", label="Right Side", **pkw)

        if show_interface:
            ikw = {"s": 90, "alpha": 1.0}
            if interface_kwargs is not None:
                ikw.update(interface_kwargs)
            ax.scatter([left_full[1]], [left_full[2]], marker="D", color="C2", label="Interface Left", **ikw)
            ax.scatter([right_full[1]], [right_full[2]], marker="P", color="C3", label="Interface Right", **ikw)

        if highlight_near_interface:
            near_indices = [geom.left_index, geom.right_index]
            near_indices = [i for i in near_indices if 0 <= i < full_comp.shape[0]]
            if len(near_indices) > 0:
                ax.scatter(
                    full_comp[near_indices, 1],
                    full_comp[near_indices, 2],
                    s=120,
                    marker="x",
                    color="k",
                    linewidths=1.5,
                    label="Near Interface Nodes",
                )
        if OG_interface_comps is not None:
            ax.scatter([OG_interface_comps[0][0],  OG_interface_comps[1][0]], [OG_interface_comps[0][1], OG_interface_comps[1][1]], marker="o", color="k", label="OG Interface")
            


        ax.set_xlabel(f"X({self.elements[0]})")
        ax.set_ylabel(f"X({self.elements[1]})")
        ax.set_title(f"Ternary State at t={t_eval:.6g}, T={T_plot:.2f} K")
        ax.legend()

        ax.set_xlim(axis_lims['x'])
        ax.set_ylim(axis_lims['y'])
        return fig, ax

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
            ignored_node_rule=self.ignoredNodeRule,
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
