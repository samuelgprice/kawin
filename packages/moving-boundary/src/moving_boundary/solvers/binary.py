"""Bounded binary bulk extraction with the research host retained.

Source: samuelgprice/kawin, ae9fb2971a9fd49132f71ddc883a5bd5406644b6.
The fork identifies the binary scheme as a port of the authors' MAP C++ code.
See NOTICE and docs/BINARY_SLICE.md; redistribution clearance remains unresolved.

Original: kawin/diffusion/MovingBoundaryIllingworthFDM.py,
MovingBoundaryIllingworthFD1DModel._new_concentration_{left,right}_planar.
The two method bodies and their numerical docstrings are unchanged.
"""

import numpy as np

from kawin.diffusion.MovingBoundaryIllingworthFDM import (
    MovingBoundaryIllingworthFD1DModel as _ForkModel,
)
from moving_boundary.numerics.tridiagonal import solve_illingworth_tridiagonal


class MovingBoundaryIllingworthFD1DModel(_ForkModel):
    """Binary Illingworth model with package-owned planar bulk solves.

    Constructor arguments, interface iteration, retry and timestep policies,
    transformed grids, composition conversion, and results are inherited from
    the pinned research-fork model. See its constructor and class documentation
    for the full input contract. Only the two concentration solves and their
    Thomas kernel are extracted here; this is not an independent solver.

    The bulk discretization retains zero-flux outer boundaries, fixed interface
    concentrations, scalar or face diffusivities, and the original operation
    order. Supported geometry is planar. The inherited host validates inputs.
    Temperature arrays retain the host's hours convention.
    """

    def _new_concentration_left_planar(self, p, s, future_s, dt, D_left):
        """
        Solves the left transformed concentration profile for one implicit step.

        Coefficients are the planar Illingworth finite-volume coefficients from
        the MAP implementation. Interior rows are assembled with vectorized
        slices, but the boundary rows retain the original one-sided forms.

        ``D_left`` is either a scalar phase-uniform diffusivity, assembled with
        the original MAP expressions so constant-D results are bitwise
        unchanged, or a face array of shape ``(len(p) - 1,)`` where face ``j``
        lies between nodes ``j`` and ``j + 1`` and the interface face is
        ``D_left[-1]``. The upwind and cell-volume terms do not depend on D.
        """
        n = len(p)
        lo = np.zeros(n, dtype=np.float64)
        diag = np.zeros(n, dtype=np.float64)
        up = np.zeros(n, dtype=np.float64)
        rhs = np.zeros(n, dtype=np.float64)
        D_values = np.asarray(D_left, dtype=np.float64)
        phase_uniform = D_values.ndim == 0
        if not phase_uniform and D_values.shape != (n - 1,):
            raise ValueError(f"Left face diffusivities must have shape ({n - 1},); received {D_values.shape}.")
        tmpB = future_s - s
        u = self._u_grid
        left_diff = u[1:-1] - u[:-2]
        left_sum = u[1:-1] + u[:-2]
        right_diff = u[2:] - u[1:-1]
        right_sum = u[2:] + u[1:-1]
        cell_width = right_sum - left_sum
        if phase_uniform:
            tmpA = float(D_values) * dt / future_s
            A_first = tmpA
            A_lo = tmpA / left_diff
            A_up = tmpA / right_diff
            A_diag = -tmpA * (1.0 / left_diff + 1.0 / right_diff)
        else:
            A_faces = D_values * dt / future_s
            A_first = A_faces[0]
            A_lo = A_faces[:-1] / left_diff
            A_up = A_faces[1:] / right_diff
            A_diag = -(A_lo + A_up)

        if future_s >= s:
            diag[0] = -A_first / u[1] - future_s * u[1] / 2.0
            up[0] = A_first / u[1] + tmpB * u[1] / 2.0
            rhs[0] = -p[0] * s * u[1] / 2.0
            lo[1:-1] = A_lo
            diag[1:-1] = A_diag - tmpB * left_sum / 2.0
            diag[1:-1] = diag[1:-1] - future_s * cell_width / 2.0
            up[1:-1] = A_up + tmpB * right_sum / 2.0
            rhs[1:-1] = -s * p[1:-1] * cell_width / 2.0
        else:
            diag[0] = -A_first / u[1] + tmpB * u[1] / 2.0 - future_s * u[1] / 2.0
            up[0] = A_first / u[1]
            rhs[0] = -p[0] * s * u[1] / 2.0
            lo[1:-1] = A_lo - tmpB * left_sum / 2.0
            diag[1:-1] = A_diag + tmpB * right_sum / 2.0
            diag[1:-1] = diag[1:-1] - future_s * cell_width / 2.0
            up[1:-1] = A_up
            rhs[1:-1] = -s * p[1:-1] * cell_width / 2.0

        diag[-1] = -1.0
        rhs[-1] = -self.interfaceCompositions[0]
        return solve_illingworth_tridiagonal(lo, diag, up, rhs)

    def _new_concentration_right_planar(self, q, s, future_s, dt, D_right):
        """
        Solves the right transformed concentration profile for one implicit step.

        This is the large phase for the Figure-3 setup, so interior coefficient
        assembly is vectorized while preserving the MAP code's boundary rows and
        tridiagonal sign convention.

        ``D_right`` is either a scalar phase-uniform diffusivity (original MAP
        expressions, bitwise unchanged) or a face array of shape
        ``(len(q) - 1,)`` where the interface face is ``D_right[0]`` and the
        zero-flux far-boundary row uses ``D_right[-1]``.
        """
        n = len(q)
        lo = np.zeros(n, dtype=np.float64)
        diag = np.zeros(n, dtype=np.float64)
        up = np.zeros(n, dtype=np.float64)
        rhs = np.zeros(n, dtype=np.float64)
        D_values = np.asarray(D_right, dtype=np.float64)
        phase_uniform = D_values.ndim == 0
        if not phase_uniform and D_values.shape != (n - 1,):
            raise ValueError(f"Right face diffusivities must have shape ({n - 1},); received {D_values.shape}.")
        tmpB = future_s - s
        span = self._R - future_s
        v = self._v_grid
        left_diff = v[1:-1] - v[:-2]
        left_sum = v[1:-1] + v[:-2]
        right_diff = v[2:] - v[1:-1]
        right_sum = v[2:] + v[1:-1]
        cell_width = right_sum - left_sum
        if phase_uniform:
            tmpA = float(D_values) * dt / (self._R - future_s)
            A_last = tmpA
            A_lo = tmpA / left_diff
            A_up = tmpA / right_diff
            A_diag = -tmpA * (1.0 / right_diff + 1.0 / left_diff)
        else:
            A_faces = D_values * dt / (self._R - future_s)
            A_last = A_faces[-1]
            A_lo = A_faces[:-1] / left_diff
            A_up = A_faces[1:] / right_diff
            A_diag = -(A_up + A_lo)

        diag[0] = -1.0
        rhs[0] = -self.interfaceCompositions[1]

        if future_s >= s:
            lo[1:-1] = A_lo
            diag[1:-1] = A_diag - tmpB * (1.0 - left_sum / 2.0)
            diag[1:-1] = diag[1:-1] - span * cell_width / 2.0
            up[1:-1] = A_up + tmpB * (1.0 - right_sum / 2.0)
            rhs[1:-1] = -(self._R - s) * q[1:-1] * cell_width / 2.0

            tmp = v[-2]
            lo[-1] = A_last / (1.0 - tmp)
            diag[-1] = -A_last / (1.0 - tmp) - tmpB * (1.0 - (1.0 + tmp) / 2.0)
            diag[-1] = diag[-1] - span * (1.0 - tmp) / 2.0
            rhs[-1] = -q[-1] * (self._R - s) * (1.0 - tmp) / 2.0
        else:
            lo[1:-1] = A_lo - tmpB * (1.0 - left_sum / 2.0)
            diag[1:-1] = A_diag + tmpB * (1.0 - right_sum / 2.0)
            diag[1:-1] = diag[1:-1] - span * cell_width / 2.0
            up[1:-1] = A_up
            rhs[1:-1] = -(self._R - s) * q[1:-1] * cell_width / 2.0

            tmp = v[-2]
            lo[-1] = A_last / (1.0 - tmp) - tmpB * (1.0 - (1.0 + tmp) / 2.0)
            diag[-1] = -A_last / (1.0 - tmp) - span * (1.0 - tmp) / 2.0
            rhs[-1] = -q[-1] * (self._R - s) * (1.0 - tmp) / 2.0

        return solve_illingworth_tridiagonal(lo, diag, up, rhs)
