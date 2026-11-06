"""Fork step policy on the unmodified public-kawin integration loop.

Policy source: samuelgprice/kawin 06dc7b4, solver/Solver.py and GenericModel.py.
The host owns iteration, callbacks, flattening, correction and stopping.
"""
import math
from kawin.solver import DESolver as _HostSolver, rk4Iterator
from kawin.GenericModel import GenericModel as _HostModel


class DESolver(_HostSolver):
    """Reject undersized steps except the exact final remainder.

    The inherited loop is unchanged. Remaining time is computed at each iterator
    callback; Illingworth solvers support explicit Euler only.
    """
    def solve(self, t0, X0, tf, verbose=False, vIt=10):
        """Use the host loop with the requested final time available to step policy."""
        self._compatFinalTime = tf
        return super().solve(t0, X0, tf, verbose, vIt)

    def _getdXdt(self, t, x, getDt = False):
        '''
        Wrapper around getdXdt which will handle the following:
            Handle flattening/unfalttening the x and dx/dt arrays
            Calculate dt if not supplied

        The API for the iterator will be that all arrays are 1D np.arrays where operators will be trivial

        Parameters
        ----------
        t : float
            Time
        x : 1D np.array
            Model values
        getDt : bool
            Will calculate dt if True
        '''
        self._remainingTime = self._compatFinalTime - t
        unflatX = self._unflattenX(x, self._X0)
        dXdt = self._f(t, unflatX)
        if getDt:
            dt = self._getDt(dXdt)
            final_remainder = bool(
                self._remainingTime < self._dtmin
                and math.isclose(
                    float(dt),
                    self._remainingTime,
                    rel_tol=2e-13,
                    abs_tol=64.0 * math.ulp(1.0) * max(1.0, abs(self._remainingTime)),
                )
            )
            if dt<self._dtmin and not final_remainder:
                raise ValueError(f"Calculated dt {dt} is smaller than minimum allowed dt {self._dtmin}.")
            dt = dt if dt > self._dtmin or final_remainder else self._dtmin
            dt = dt if dt < self._dtmax else self._dtmax
            return self._flattenX(dXdt), dt
        else:
            return self._flattenX(dXdt)


class GenericModel(_HostModel):
    """Use public model services with the package's timestep policy adapter."""
    def solve(self, simTime, iterator = rk4Iterator, verbose=False, vIt=10, minDtFrac = 1e-8, maxDtFrac = 1):
        '''
        Solves model using the DESolver

        Steps:
            1. Call setup
            2. Create DESolver object and set necessary functions
            3. Get current values of t and X
            4. Solve from current t to t+simTime

        Parameters
        ----------
        simTime : float
            Simulation time (as a delta from current time)
        iterator : Iterator function (defaults to rk4Iterator)
            Defines what iteration scheme to use
        verbose : bool (defaults to False)
            Outputs status if true
        vIt : integer (defaults to 10)
            Number of iterations before printing status
        minDtFrac : float (defaults to 1e-8)
            Minimum dt as fraction of simulation time. The exact final
            remainder may be smaller than this limit.
        maxDtFrac : float (defaults to 1)
            Maximum dt as fraction of simulation time
        '''
        self.setup()

        solver = DESolver(iterator, minDtFrac = minDtFrac, maxDtFrac = maxDtFrac)
        solver.setFunctions(preProcess=self.preProcess, postProcess=self.postProcess, printHeader=self.printHeader, printStatus=self.printStatus)
        solver.setdXdtFunctions(self.getdXdt, self.correctdXdt, self.getDt, self.flattenX, self.unflattenX)

        X0 = self.getCurrentX()
        self.setTimeInfo(self.currentTime, simTime)
        solver.solve(self.initialTime, X0, self.finalTime, verbose, vIt)

        self.postSolve()
