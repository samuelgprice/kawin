"""Small recording policies on public kawin MeshData.

Source: 06dc7b4 kawin/diffusion/mesh/MeshBase.py. Storage, padding and timed
lookup follow the host; the row-selection rule (LEDGERS R09) is package-owned
and shared with the Illingworth history classes through the helpers below.
"""
import numpy as np
from kawin.diffusion.mesh import MeshData as _HostData


def _history_row_target(history, force=False):
    """
    Returns the row a ``record`` call writes, or ``None`` when it stores no row.

    Shared row rule for every Illingworth history (LEDGERS R09):

    - Enabled recording (``recordInterval > 0``): rows are append-only. A
      regular call stores a row when the cadence counter ``currentIndex`` is a
      multiple of ``recordInterval``; a forced call always stores one. A stored
      row goes to ``_rowCount``, the number of rows already stored, so the
      initial row and earlier regular rows are never overwritten.
    - Disabled recording: the single row ``N`` (0) is overwritten every call.

    The history must carry ``_rowCount`` and ``_lastRecordStored`` (set by its
    ``reset``); ``N`` remains the index of the last stored row.
    """
    interval = history.recordInterval
    if interval > 0:
        if force or history.currentIndex % interval == 0:
            return int(history._rowCount)
        return None
    return int(history.N)


def _advance_history_rows(history, target, force=False):
    """
    Updates row bookkeeping after a ``record`` call wrote ``target`` (or nothing).

    Only regular calls advance the cadence counter ``currentIndex``, so a forced
    row never consumes a cadence slot and the regular sampling counted from
    ``reset`` continues unchanged across solves. ``_lastRecordStored`` records
    whether this call stored a row; ``_finalize_history_rows`` uses it instead of
    comparing times. Disabled histories keep ``N == 0`` and no counters.
    """
    if history.recordInterval <= 0:
        return
    if target is not None:
        history.N = int(target)
        history._rowCount = int(target) + 1
    history._lastRecordStored = target is not None
    if not force:
        history.currentIndex += 1


def _finalize_history_rows(history):
    """
    Adds the forced final row if needed, then trims unused capacity.

    With enabled recording the current state is appended unless the most recent
    ``record`` call itself stored a row (flag-based; times are never compared, so
    a distinct snapshot a few ulp after the last stored row is kept). Repeated
    calls add nothing. With recording disabled the single row is rewritten with
    the current state. Afterwards ``_y``/``_time`` hold exactly ``N + 1`` rows.
    """
    if history.recordInterval <= 0 or not history._lastRecordStored:
        history.record(history.currentTime, history.currentY, force=True)
    history._y = history._y[: history.N + 1]
    history._time = history._time[: history.N + 1]


class MeshData(_HostData):
    """
    Host ``MeshData`` with the package row rule, preallocation and exact-time lookup.

    ``record``/``finalize`` follow ``_history_row_target``: the initial row and
    regular rows are never overwritten and a forced final row is appended
    without consuming a cadence slot. ``y()`` without a time returns the current
    state (``currentY``) even when the latest accepted step was not stored.
    ``finalize`` uses the shared flag rule at every interval (LEDGERS R09c), so
    ``data`` stays synchronized with the other Illingworth histories: no
    duplicate final row, and a repeated ``finalize`` adds nothing. (Before R09c
    the host behavior of always appending a row was kept at interval 1.)
    """
    def __init__(self, mesh, record=False):
        """Initialize host storage and the optional exact-time lookup policy."""
        super().__init__(mesh, record)
        self._timeInterpolationEnabled = True
        self._timeInterpolationError = None

    def reset(self):
        """Clears storage like the host and starts the row counter and stored-row flag."""
        super().reset()
        self._rowCount = 0
        self._lastRecordStored = False

    def _syncRowBookkeeping(self):
        """
        Derives ``_rowCount``/``_lastRecordStored`` after adopting host-written rows.

        ``DiffusionModel.__init__`` copies the state of the host ``MeshData`` that
        the host constructor recorded into with regular calls only, so the stored
        rows are ``0..N`` and the last call's cadence slot decides the flag.
        """
        if self.recordInterval > 0 and self.currentIndex > 0:
            self._rowCount = int(self.N) + 1
            self._lastRecordStored = (self.currentIndex - 1) % self.recordInterval == 0
        else:
            self._rowCount = 0
            self._lastRecordStored = False

    def record(self, time, y, force=False):
        """
        Stores time and flattened response under the shared row rule.

        Rows are appended at ``_rowCount`` (regular calls on the cadence, forced
        calls always) and never overwrite an earlier row; disabled recording
        overwrites the single row. ``currentY``/``currentTime`` always hold this
        call's values, as in the host.
        """
        target = _history_row_target(self, force)
        if target is not None:
            if target >= self._time.shape[0]:
                self._y = np.pad(self._y, ((0, self.batchSize), *[(0, 0)] * (self._y.ndim - 1)))
                self._time = np.pad(self._time, (0, self.batchSize))
            self._y[target] = y
            self._time[target] = time
        _advance_history_rows(self, target, force)
        self.currentY = y
        self.currentTime = time

    def finalize(self):
        """
        Appends the forced final row if needed and trims unused rows (``_finalize_history_rows``).

        The current state is appended only when the latest ``record`` stored no
        row; times are not compared. At interval 1 every record stores a row, so
        nothing is appended (the host always appended a duplicate; LEDGERS R09c).
        """
        _finalize_history_rows(self)

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

        Without a time this returns a copy of the current state (``currentY``),
        at any recording interval, including steps that stored no row. A
        requested time is always looked up in the stored rows: linear
        interpolation, clamped to the first/last stored row (asking for the
        current, unstored time therefore returns the last stored row). If
        recording is disabled, the single row holds the current state.
        Models with topology-changing fixed-mesh snapshots may disable time
        interpolation and require exact recorded times.
        '''
        # If time is not supplied, then return current state of response
        if time is None:
            return np.array(self.currentY, dtype=self._y.dtype)

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
