"""Small recording policies on public kawin MeshData.

Source: 06dc7b4 kawin/diffusion/mesh/MeshBase.py; recording/reset remain inherited.
"""
import numpy as np
from kawin.diffusion.mesh import MeshData as _HostData


class MeshData(_HostData):
    """Allow preallocation and exact-time-only snapshots when interpolation is unsafe."""
    def __init__(self, mesh, record=False):
        """Initialize host storage and the optional exact-time lookup policy."""
        super().__init__(mesh, record)
        self._timeInterpolationEnabled = True
        self._timeInterpolationError = None

    def preallocate(self, capacity):
        '''
        Preallocates recording storage for at least ``capacity`` rows.

        This is intended for solvers that can estimate their recording count
        before time integration starts. The normal ``record`` growth path is
        unchanged, so models that cannot estimate their size still expand in
        batches with ``np.pad``.
        '''
        capacity = int(capacity)
        if capacity <= self._time.shape[0]:
            return
        y_new = np.zeros((capacity, *self.yShape), dtype=self._y.dtype)
        time_new = np.zeros(capacity, dtype=self._time.dtype)
        y_new[: self._y.shape[0]] = self._y
        time_new[: self._time.shape[0]] = self._time
        self._y = y_new
        self._time = time_new

    def y(self, time = None):
        '''
        Returns reponse variable at time

        If recording is disabled, then this will return the current state.
        Models with topology-changing fixed-mesh snapshots may disable time
        interpolation and require exact recorded times.
        '''
        # If time is not supplied, then return current state of response
        if time is None:
            return self._y[self.N]

        # If time is supplied, then interpolate if recording, else return current state
        if self.recordInterval > 0:
            recorded_time = self._time[: self.N + 1]
            if not self._timeInterpolationEnabled:
                matches = np.flatnonzero(np.isclose(recorded_time, float(time), rtol=0.0, atol=1e-14))
                if len(matches) == 0:
                    message = self._timeInterpolationError or "Time interpolation is disabled; request an exact recorded time."
                    raise ValueError(message)
                return self._y[matches[-1]].copy()
            if time <= recorded_time[0]:
                if time < recorded_time[0]:
                    print('Input time is lower than smallest recorded time, returning data at t = {:.3e}'.format(self._time[0]))
                return self._y[0]
            elif time >= recorded_time[-1]:
                if time > recorded_time[-1]:
                    print('Input time is larger than longest recorded time, return data at t = {:.3e}'.format(recorded_time[-1]))
                return self._y[self.N]
            else:
                uind = int(np.searchsorted(recorded_time, float(time), side="right"))
                lind = uind - 1

                ux, utime = self._y[uind], self._time[uind]
                lx, ltime = self._y[lind], self._time[lind]

                flatY = (ux - lx) * (time - ltime) / (utime - ltime) + lx
                return flatY
        else:
            print('Recording is disabled. Returning current state.')
            return self._y[0]
