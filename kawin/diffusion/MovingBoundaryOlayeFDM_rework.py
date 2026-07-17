import numpy as np

from .MovingBoundaryOlayeFDM import debugInPlace
from .MovingBoundaryOlayeFDM import MovingBoundaryOlayeFD1DModel


class MovingBoundaryOlayeFD1DReworkModel(MovingBoundaryOlayeFD1DModel):
    """
    Paper-closer rework of the Olaye binary planar moving-boundary model.

    This class intentionally reuses the current planar interface root solve and
    timestep generation from ``MovingBoundaryOlayeFD1DModel``. The rework is
    focused on the phase-field recurrence so it more closely follows the paper's
    winding, bootstrap, Leapfrog/Dufort-Frankel, and fixed-boundary treatment.

    Notes
    -----
    - Runtime support remains planar-only.
    - The rework currently targets step initial profiles.
    - Equation-number comments are included near the main update blocks for
      traceability while comparing against Olaye & Ojo (2020).
    """

    def _apply_boundary_conditions(self, p, q):
        """
        Applies only the interface Dirichlet values from the paper.

        The fixed-end zero-flux boundary nodes are updated explicitly inside the
        bootstrap and LF/DF recurrences rather than being mirrored here.
        """
        p_bc = np.asarray(p, dtype=np.float64).copy()
        q_bc = np.asarray(q, dtype=np.float64).copy()
        c_ab, c_ba = self.interfaceCompositions

        # Eq. (22): p_N^{k+1} = C_AB
        p_bc[-1] = c_ab
        # Eq. (23): q_1^{k+1} = C_BA
        q_bc[0] = c_ba
        return p_bc, q_bc

    def _winding_coefficients_from_interface_step(self, s_new, s_curr):
        """
        Returns the paper's winding coefficients for the advective term.

        Positive interface motion uses ``alpha = 0, beta = 1`` and negative
        interface motion uses ``alpha = 1, beta = 0`` as described in the
        implementation section of the paper.
        """
        ds = float(s_new - s_curr)
        if ds >= 0:
            return 0.0, 1.0
            # return 1.0, 0.0
        return 1.0, 0.0
        # return 0.0, 1.0

    def _wound_face_values(self, values, a, b, pq_str):
        """
        Builds the wound face concentrations used in Eq. (17).
        """
        values = np.asarray(values, dtype=np.float64)
        # debugInPlace()
        ## For actual wound face values use this
        self.forceHalfNodeValues_HACK=False
        return a * values[:-1] + b * values[1:]
        ## HACK: for testing half-node values instead of wound face values
        raise ValueError("THIS HACK DOES NOT ACTUALLY CHANGE THE LFDF PHASE UPDATE!!")
        # self.forceHalfNodeValues_HACK=True
        # # return (values[:-1] + values[1:])/2
        # valsToReturn = a * values[:-1] + b * values[1:]
        # if pq_str=='p':
        #     valsToReturn[-1]=(values[-2]+values[-1])/2
        # elif pq_str=='q':
        #     valsToReturn[0]=(values[0]+values[1])/2
        # else:
        #     raise ValueError("Invalid pq_str. Must be 'p' or 'q'.")
        # return valsToReturn


    def _half_node_diffusivities(self, values, temperatures, phase):
        """
        Evaluates half-node diffusivities using Eq. (37).
        """
        values = np.asarray(values, dtype=np.float64)
        temperatures = np.asarray(temperatures, dtype=np.float64)
        t_half = 0.5 * (temperatures[:-1] + temperatures[1:])
        c_half = 0.5 * (values[:-1] + values[1:])
        return np.asarray(
            self.therm.getInterdiffusivity(c_half, t_half, phase=phase),
            dtype=np.float64,
        ).reshape(-1)

    def _bootstrap_phase_a_boundary(self, p_curr, s, dt, D_half):
        """
        Left fixed-boundary bootstrap update using the zero-flux condition.
        """
        scale = max(s * s * self._du * self._du, 1e-15)
        # Eq. (24): zero flux at the fixed boundary, applied in the Eq. (10) step.
        return p_curr[0] + 2.0 * dt * D_half[0] * (p_curr[1] - p_curr[0]) / scale

    def _bootstrap_phase_b_boundary(self, q_curr, right_scale, dt, D_half):
        """
        Right fixed-boundary bootstrap update using the zero-flux condition.
        """
        scale = max(right_scale * right_scale * self._dv * self._dv, 1e-15)
        # Eq. (25): zero flux at the fixed boundary, applied in the Eq. (11) step.
        return q_curr[-1] + 2.0 * dt * D_half[-1] * (q_curr[-2] - q_curr[-1]) / scale

    def _lfdf_phase_a_boundary(self, p_prev, p_curr, s, dt, D_half):
        """
        Left fixed-boundary LF/DF update corresponding to Eq. (26).
        """
        sigma = dt * D_half[0] / max(s * s * self._du * self._du, 1e-15)
        return ((1.0 - 2.0 * sigma) * p_prev[0] + 4.0 * sigma * p_curr[1]) / (1.0 + 2.0 * sigma)

    def _lfdf_phase_b_boundary(self, q_prev, q_curr, right_scale, dt, D_half):
        """
        Right fixed-boundary LF/DF update corresponding to Eq. (27).
        """
        sigma = dt * D_half[-1] / max(right_scale * right_scale * self._dv * self._dv, 1e-15)
        return ((1.0 - 2.0 * sigma) * q_prev[-1] + 4.0 * sigma * q_curr[-2]) / (1.0 + 2.0 * sigma)

    def _classical_explicit_phase_update(self, t_curr, p_curr, q_curr, s_curr, s_next, dt, sdot, a, b):
        """
        Bootstrap update matching the paper's classical explicit form.
        """
        p_curr, q_curr = self._apply_boundary_conditions(p_curr, q_curr)
        

        ''' Old way '''
        # p_new = p_curr.copy()
        # q_new = q_curr.copy()
        # t=t_curr
        # s=s_curr
        # T_left, T_right = self._phase_temperatures(t, s)
        # Dp_half = self._half_node_diffusivities(p_curr, T_left, self.phases[0])
        # Dq_half = self._half_node_diffusivities(q_curr, T_right, self.phases[1])
        # p_faces = self._wound_face_values(p_curr, a, b)
        # q_faces = self._wound_face_values(q_curr, a, b)

        # left_scale = max(s, 1e-15)
        # right_scale = max(self._R - s, 1e-15)
        # left_den = max(left_scale * left_scale * self._du * self._du, 1e-15)
        # right_den = max(right_scale * right_scale * self._dv * self._dv, 1e-15)

        # # Eq. (10): classical explicit update for phase A at k = 1.
        # for i in range(1, len(p_curr) - 1):
        #     adv_i = (self._u_grid[i] * sdot / left_scale) * (p_faces[i] - p_faces[i - 1]) / self._du
        #     diff_i = (
        #         Dp_half[i] * (p_curr[i + 1] - p_curr[i])
        #         - Dp_half[i - 1] * (p_curr[i] - p_curr[i - 1])
        #     ) / left_den
        #     p_new[i] = p_curr[i] + dt * (adv_i + diff_i)

        # # Eq. (11): classical explicit update for phase B at k = 1.
        # for i in range(1, len(q_curr) - 1):
        #     adv_i = ((1.0 - self._v_grid[i]) * sdot / right_scale) * (q_faces[i] - q_faces[i - 1]) / self._dv
        #     diff_i = (
        #         Dq_half[i] * (q_curr[i + 1] - q_curr[i])
        #         - Dq_half[i - 1] * (q_curr[i] - q_curr[i - 1])
        #     ) / right_den
        #     q_new[i] = q_curr[i] + dt * (adv_i + diff_i)

        # p_new[0] = self._bootstrap_phase_a_boundary(p_curr, s, dt, Dp_half)
        # q_new[-1] = self._bootstrap_phase_b_boundary(q_curr, right_scale, dt, Dq_half)
        
        ''' New way '''
        # debugInPlace()
        T_left, T_right = self._phase_temperatures(t_curr, s_curr)
        Dp_iPlusHalf = self._half_node_diffusivities(p_curr, T_left, self.phases[0])
        Dq_iPlusHalf = self._half_node_diffusivities(q_curr, T_right, self.phases[1])
        

        Dp_iPlusHalf = np.append(Dp_iPlusHalf, np.nan)
        Dp_iMinusHalf = np.roll(Dp_iPlusHalf, 1)
        Dq_iPlusHalf = np.append(Dq_iPlusHalf, np.nan)
        Dq_iMinusHalf = np.roll(Dq_iPlusHalf, 1)


        uShape_nan_arr = np.full(self._u_grid.shape, np.nan)
        u_iPlusHalf = uShape_nan_arr.copy()
        u_iPlus1 = uShape_nan_arr.copy()
        u_iMinus1 = uShape_nan_arr.copy()
        p_k_iPlusHalf = uShape_nan_arr.copy()
        p_k_iPlus1 = uShape_nan_arr.copy()
        p_k_iMinus1 = uShape_nan_arr.copy()
        r_kPlus1_iPlusHalf = uShape_nan_arr.copy()
        r_k_iPlusHalf = uShape_nan_arr.copy()
        V_p_kPlus1_i = uShape_nan_arr.copy()
        V_p_k_i = uShape_nan_arr.copy()

        u_i = self._u_grid.copy()
        u_iPlus1[:-1] = u_i[1:].copy()
        u_iMinus1[1:] = u_i[:-1].copy()
        p_k_i = p_curr.copy()
        p_k_iPlus1[:-1] = p_k_i[1:].copy()
        p_k_iMinus1[1:] = p_k_i[:-1].copy()
        p_k_iPlusHalf[:-1] = self._wound_face_values(p_k_i, a, b, 'p')
        # p_k_iPlusHalf[:-1] = (p_k_i[:-1] + p_k_i[1:])/2
        p_k_iMinusHalf = np.roll(p_k_iPlusHalf, 1)
        if not self.forceHalfNodeValues_HACK:
            if s_next>s_curr:
                assert all([p_k_iPlusHalf[i] == p_curr[i+1] for i in range(len(p_curr)-1)]) == True
            else:
                assert all([p_k_iPlusHalf[i] == p_curr[i] for i in range(len(p_curr)-1)]) == True
        
        u_iPlusHalf[:-1] = (self._u_grid[:-1] + self._u_grid[1:])/2
        u_iMinusHalf = np.roll(u_iPlusHalf, 1)
        r_kPlus1_iPlusHalf = s_next * u_iPlusHalf
        r_kPlus1_iMinusHalf = s_next * u_iMinusHalf
        r_k_iPlusHalf = s_curr * u_iPlusHalf
        r_k_iMinusHalf = s_curr * u_iMinusHalf

        V_p_kPlus1_i = r_kPlus1_iPlusHalf - r_kPlus1_iMinusHalf
        V_p_k_i = r_k_iPlusHalf - r_k_iMinusHalf

        assert all([u_iPlusHalf[i]==((self._u_grid[i] + self._u_grid[i+1])/2) for i in range(len(self._u_grid)-1)]) == True
        assert all([u_iMinusHalf[i]==((self._u_grid[i-1] + self._u_grid[i])/2) for i in range(1, len(self._u_grid))]) == True
        assert all([u_iPlus1[i]==self._u_grid[i+1] for i in range(len(self._u_grid)-1)]) == True
        assert all([u_iMinus1[i]==self._u_grid[i-1] for i in range(1, len(self._u_grid))]) == True
        assert all([p_k_iPlus1[i]==p_k_i[i+1] for i in range(len(p_k_i)-1)]) == True
        assert all([p_k_iMinus1[i]==p_k_i[i-1] for i in range(1, len(p_k_i))]) == True

        p_new_i = (1/V_p_kPlus1_i) * ( dt * ( (sdot*p_k_iPlusHalf*u_iPlusHalf + (Dp_iPlusHalf/s_curr)*((p_k_iPlus1-p_k_i)/(u_iPlus1-u_i))) - \
                                            (sdot*p_k_iMinusHalf*u_iMinusHalf + (Dp_iMinusHalf/s_curr)*((p_k_i-p_k_iMinus1)/(u_i-u_iMinus1))) ) + \
                                        (p_k_i * V_p_k_i) )



        vShape_nan_arr = np.full(self._v_grid.shape, np.nan)
        v_iPlusHalf = vShape_nan_arr.copy()
        v_iPlus1 = vShape_nan_arr.copy()
        v_iMinus1 = vShape_nan_arr.copy()
        q_k_iPlusHalf = vShape_nan_arr.copy()
        q_k_iPlus1 = vShape_nan_arr.copy()
        q_k_iMinus1 = vShape_nan_arr.copy()
        l_kPlus1_iPlusHalf = vShape_nan_arr.copy()
        l_k_iPlusHalf = vShape_nan_arr.copy()
        V_q_kPlus1_i = vShape_nan_arr.copy()
        V_q_k_i = vShape_nan_arr.copy()

        v_i = self._v_grid.copy()
        v_iPlus1[:-1] = v_i[1:].copy()
        v_iMinus1[1:] = v_i[:-1].copy()
        q_k_i = q_curr.copy()
        q_k_iPlus1[:-1] = q_k_i[1:].copy()
        q_k_iMinus1[1:] = q_k_i[:-1].copy()
        q_k_iPlusHalf[:-1] = self._wound_face_values(q_k_i, a, b, 'q')
        # q_k_iPlusHalf[:-1] = (q_k_i[:-1] + q_k_i[1:])/2
        q_k_iMinusHalf = np.roll(q_k_iPlusHalf, 1)
        if not self.forceHalfNodeValues_HACK:
            if s_next>s_curr:
                assert all([q_k_iPlusHalf[i] == q_curr[i+1] for i in range(len(q_curr)-1)]) == True
            else:
                assert all([q_k_iPlusHalf[i] == q_curr[i] for i in range(len(q_curr)-1)]) == True
        
        v_iPlusHalf[:-1] = (self._v_grid[:-1] + self._v_grid[1:])/2
        v_iMinusHalf = np.roll(v_iPlusHalf, 1)
        l_kPlus1_iPlusHalf = s_next + (self._R-s_next) * v_iPlusHalf
        l_kPlus1_iMinusHalf = s_next + (self._R-s_next) * v_iMinusHalf
        l_k_iPlusHalf = s_curr + (self._R-s_curr) * v_iPlusHalf
        l_k_iMinusHalf = s_curr + (self._R-s_curr) * v_iMinusHalf

        V_q_kPlus1_i = l_kPlus1_iPlusHalf - l_kPlus1_iMinusHalf
        V_q_k_i = l_k_iPlusHalf - l_k_iMinusHalf

        assert all([v_iPlusHalf[i]==((self._v_grid[i] + self._v_grid[i+1])/2) for i in range(len(self._v_grid)-1)]) == True
        assert all([v_iMinusHalf[i]==((self._v_grid[i-1] + self._v_grid[i])/2) for i in range(1, len(self._v_grid))]) == True
        assert all([v_iPlus1[i]==self._v_grid[i+1] for i in range(len(self._v_grid)-1)]) == True
        assert all([v_iMinus1[i]==self._v_grid[i-1] for i in range(1, len(self._v_grid))]) == True
        assert all([q_k_iPlus1[i]==q_k_i[i+1] for i in range(len(q_k_i)-1)]) == True
        assert all([q_k_iMinus1[i]==q_k_i[i-1] for i in range(1, len(q_k_i))]) == True

        q_new_i = (1/V_q_kPlus1_i) * ( dt * ( (sdot*q_k_iPlusHalf*(1-v_iPlusHalf) + (Dq_iPlusHalf/(self._R-s_curr))*((q_k_iPlus1-q_k_i)/(v_iPlus1-v_i))) - \
                                            (sdot*q_k_iMinusHalf*(1-v_iMinusHalf) + (Dq_iMinusHalf/(self._R-s_curr))*((q_k_i-q_k_iMinus1)/(v_i-v_iMinus1))) ) + \
                                        (q_k_i * V_q_k_i) )


        ## Outside boundary conditions:
        u_1PlusHalf = u_iPlusHalf[0]
        u_1Plus1 = u_iPlus1[0]
        u_1 = u_i[0]
        p_k_1PlusHalf = p_k_iPlusHalf[0]
        p_k_1Plus1 = p_k_iPlus1[0]
        p_k_1 = p_k_i[0]
        Dp_1PlusHalf = Dp_iPlusHalf[0]
        V_p_kPlus1_1 = r_kPlus1_iPlusHalf[0] # - 0 
        V_p_k_1 = r_k_iPlusHalf[0] # - 0 

        p_new_i[0] = (1/V_p_kPlus1_1) * ( dt * ( (sdot*p_k_1PlusHalf*u_1PlusHalf + (Dp_1PlusHalf/s_curr)*((p_k_1Plus1-p_k_1)/(u_1Plus1-u_1))) ) + \
                                        (p_k_1 * V_p_k_1) )


        ## tecnically using the olaye notation this should be M+1 and M+1/2 (instead of M and M-1/2)
        v_MMinusHalf = v_iMinusHalf[-1]
        v_MMinus1 = v_iMinus1[-1]
        v_M = v_i[-1]
        q_k_MMinusHalf = q_k_iMinusHalf[-1]
        q_k_MMinus1 = q_k_iMinus1[-1]
        q_k_M = q_k_i[-1]
        Dq_MMinusHalf = Dq_iMinusHalf[-1]
        V_q_kPlus1_M = self._R - l_kPlus1_iMinusHalf[-1]
        V_q_k_M = self._R - l_k_iMinusHalf[-1]
        

        q_new_i[-1] = (1/V_q_kPlus1_M) * ( dt * ( -(sdot*q_k_MMinusHalf*(1-v_MMinusHalf) + (Dq_MMinusHalf/(self._R-s_curr))*((q_k_M-q_k_MMinus1)/(v_M-v_MMinus1))) ) + \
                                        (q_k_M * V_q_k_M) )


        p_new_i, q_new_i = self._apply_boundary_conditions(p_new_i, q_new_i)
        return p_new_i, q_new_i

    def _lfdf_phase_update(self, t_prev, t_curr, t_next, p_prev, p_curr, q_prev, q_curr, s_prev, s_curr, s_next, dt, D_p, D_q, a, b):
        """
        Leapfrog/Dufort-Frankel update arranged around the paper's recurrence.

        New equations (including factors to account for the two timesteps not being equal) can be found in "Branch · Ternary system extension search" chat on account
        """
        p_curr, q_curr = self._apply_boundary_conditions(p_curr, q_curr)
    
        sdot = (s_next - s_prev) / (t_next - t_prev)

        ''' Old way '''
        # p_new = p_curr.copy()
        # q_new = q_curr.copy()
        # T_left, T_right = self._phase_temperatures(t_curr, s_curr)
        # Dp_iPlusHalf = self._half_node_diffusivities(p_curr, T_left, self.phases[0])
        # Dq_iPlusHalf = self._half_node_diffusivities(q_curr, T_right, self.phases[1])
        
        # p_faces = self._wound_face_values(p_curr, a, b)
        # q_faces = self._wound_face_values(q_curr, a, b)

        # left_scale = max(s, 1e-15)
        # right_scale = max(self._R - s, 1e-15)
        # left_den = max(left_scale * left_scale * self._du * self._du, 1e-15)
        # right_den = max(right_scale * right_scale * self._dv * self._dv, 1e-15)

        
        # # Eq. (17): wound intermediate concentrations at i +/- 1/2.
        # # Eq. (18): Leapfrog / Dufort-Frankel center substitution.
        # # Eq. (20): collected explicit recurrence for phase A.
        # Dp_half = self._half_node_diffusivities(p_curr, T_left, self.phases[0])
        # left_scale = max(s_curr, 1e-15)
        # left_den = max(left_scale * left_scale * self._du * self._du, 1e-15)
        # p_faces = self._wound_face_values(p_curr, a, b)
        # for i in range(1, len(p_curr) - 1):
        #     sigma_w = dt * Dp_half[i - 1] / left_den
        #     sigma_e = dt * Dp_half[i] / left_den
        #     adv_i = dt * (self._u_grid[i] * sdot / left_scale) * (p_faces[i] - p_faces[i - 1]) / self._du
        #     numer = (
        #         (1.0 - sigma_w - sigma_e) * p_prev[i]
        #         + 2.0 * sigma_w * p_curr[i - 1]
        #         + 2.0 * sigma_e * p_curr[i + 1]
        #         + 2.0 * adv_i
        #     )
        #     p_new[i] = numer / (1.0 + sigma_w + sigma_e)

        # # Eq. (17): wound intermediate concentrations at i +/- 1/2.
        # # Eq. (18): Leapfrog / Dufort-Frankel center substitution.
        # # Eq. (21): collected explicit recurrence for phase B.
        # Dq_half = self._half_node_diffusivities(q_curr, T_right, self.phases[1])
        # right_scale = max(self._R - s_curr, 1e-15)
        # right_den = max(right_scale * right_scale * self._dv * self._dv, 1e-15)
        # q_faces = self._wound_face_values(q_curr, a, b)
        # for i in range(1, len(q_curr) - 1):
        #     sigma_w = dt * Dq_half[i - 1] / right_den
        #     sigma_e = dt * Dq_half[i] / right_den
        #     adv_i = dt * ((1.0 - self._v_grid[i]) * sdot / right_scale) * (q_faces[i] - q_faces[i - 1]) / self._dv
        #     numer = (
        #         (1.0 - sigma_w - sigma_e) * q_prev[i]
        #         + 2.0 * sigma_w * q_curr[i - 1]
        #         + 2.0 * sigma_e * q_curr[i + 1]
        #         + 2.0 * adv_i
        #     )
        #     q_new[i] = numer / (1.0 + sigma_w + sigma_e)

        # # Eq. (24)-(27): zero-flux fixed-boundary recurrence updates.
        # p_new[0] = self._lfdf_phase_a_boundary(np.asarray(p_prev, dtype=np.float64), p_old, s, dt, Dp_half)
        # q_new[-1] = self._lfdf_phase_b_boundary(np.asarray(q_prev, dtype=np.float64), q_old, right_scale, dt, Dq_half)
        # return self._apply_boundary_conditions(p_new, q_new)

        ''' New way '''
         # debugInPlace()         
        T_left, T_right = self._phase_temperatures(t_curr, s_curr)
        Dp_iPlusHalf = self._half_node_diffusivities(p_curr, T_left, self.phases[0])
        Dq_iPlusHalf = self._half_node_diffusivities(q_curr, T_right, self.phases[1])
        

        Dp_iPlusHalf = np.append(Dp_iPlusHalf, np.nan)
        Dp_iMinusHalf = np.roll(Dp_iPlusHalf, 1)
        Dq_iPlusHalf = np.append(Dq_iPlusHalf, np.nan)
        Dq_iMinusHalf = np.roll(Dq_iPlusHalf, 1)


        t_kPlus1 = t_next
        t_k = t_curr 
        t_kMinus1 = t_prev
        delta_t0 = t_kPlus1 - t_kMinus1
        delta_tPlus = t_kPlus1 - t_k
        delta_tMinus = t_k - t_kMinus1
        ## Non-uniform timesteps
        t_plus_deltaRatio = delta_tPlus / delta_t0
        t_minus_deltaRatio = delta_tMinus / delta_t0
        ## HACK: Pretend timesteps are uniform
        # raise
        # t_plus_deltaRatio = 1/2
        # t_minus_deltaRatio = 1/2


        uShape_nan_arr = np.full(self._u_grid.shape, np.nan)
        u_iPlusHalf = uShape_nan_arr.copy()
        r_kPlus1_iPlusHalf = uShape_nan_arr.copy()
        r_kMinus1_iPlusHalf = uShape_nan_arr.copy()
        V_p_kPlus1_i = uShape_nan_arr.copy()
        V_p_kMinus1_i = uShape_nan_arr.copy()
        
        
        u_iPlusHalf[:-1] = (self._u_grid[:-1] + self._u_grid[1:])/2
        u_iMinusHalf = np.roll(u_iPlusHalf, 1)
        r_kPlus1_iPlusHalf = s_next * u_iPlusHalf
        r_kPlus1_iMinusHalf = s_next * u_iMinusHalf
        r_kMinus1_iPlusHalf = s_prev * u_iPlusHalf
        r_kMinus1_iMinusHalf = s_prev * u_iMinusHalf
        
        # assert all([u_iPlusHalf[i]==((self._u_grid[i] + self._u_grid[i+1])/2) for i in range(len(self._u_grid)-1)]) == True
        # assert all([u_iMinusHalf[i]==((self._u_grid[i-1] + self._u_grid[i])/2) for i in range(1, len(self._u_grid))]) == True

        V_p_kPlus1_i = r_kPlus1_iPlusHalf - r_kPlus1_iMinusHalf
        V_p_kMinus1_i = r_kMinus1_iPlusHalf - r_kMinus1_iMinusHalf

        
        F_p_k_iPlusHalf = sdot * u_iPlusHalf  ## Adjective term
        F_p_k_iMinusHalf = sdot * u_iMinusHalf  ## Adjective term
        G_p_k_iPlusHalf = Dp_iPlusHalf / (s_curr * self._du)  ## Diffusive term
        G_p_k_iMinusHalf = Dp_iMinusHalf / (s_curr * self._du)  ## Diffusive term

        p_k_iPlus1 = np.append(p_curr[1:], np.nan).copy()
        p_k_iMinus1 = np.insert(p_curr[:-1], 0, np.nan).copy()
        # p_k_iPlusHalf = uShape_nan_arr.copy()
        # p_k_iPlusHalf[:-1] = self._wound_face_values(p_curr, a, b, 'p')
        # if not self.forceHalfNodeValues_HACK:
        #     if s_next>s_curr:
        #         assert all([p_k_iPlusHalf[i] == p_curr[i+1] for i in range(len(p_curr)-1)]) == True
        #     else:
        #         assert all([p_k_iPlusHalf[i] == p_curr[i] for i in range(len(p_curr)-1)]) == True

        
        H_p_k_i = (a*F_p_k_iPlusHalf) - (b*F_p_k_iMinusHalf) - G_p_k_iPlusHalf - G_p_k_iMinusHalf
        # print(f"H_p_k_i.shape: {H_p_k_i.shape}")

        delta_p1 = V_p_kPlus1_i - (t_minus_deltaRatio*delta_t0 * H_p_k_i)
        delta_p2 = V_p_kMinus1_i + (t_plus_deltaRatio*delta_t0 * H_p_k_i)
        delta_p3 = delta_t0 * (b*F_p_k_iPlusHalf + G_p_k_iPlusHalf)
        delta_p4 = delta_t0 * (G_p_k_iMinusHalf - a*F_p_k_iMinusHalf)

        p_new_i = (((delta_p2*p_prev) + (delta_p3*p_k_iPlus1) + (delta_p4*p_k_iMinus1)) / delta_p1).copy()

        

        vShape_nan_arr = np.full(self._v_grid.shape, np.nan)
        v_iPlusHalf = vShape_nan_arr.copy()
        l_kPlus1_iPlusHalf = vShape_nan_arr.copy()
        l_kMinus1_iPlusHalf = vShape_nan_arr.copy()
        V_q_kPlus1_i = vShape_nan_arr.copy()
        V_q_kMinus1_i = vShape_nan_arr.copy()


        v_iPlusHalf[:-1] = (self._v_grid[:-1] + self._v_grid[1:])/2
        v_iMinusHalf = np.roll(v_iPlusHalf, 1)
        l_kPlus1_iPlusHalf = s_next + (self._R-s_next) * v_iPlusHalf
        l_kPlus1_iMinusHalf = s_next + (self._R-s_next) * v_iMinusHalf
        l_kMinus1_iPlusHalf = s_prev + (self._R-s_prev) * v_iPlusHalf
        l_kMinus1_iMinusHalf = s_prev + (self._R-s_prev) * v_iMinusHalf

        # assert all([v_iPlusHalf[i]==((self._v_grid[i] + self._v_grid[i+1])/2) for i in range(len(self._v_grid)-1)]) == True
        # assert all([v_iMinusHalf[i]==((self._v_grid[i-1] + self._v_grid[i])/2) for i in range(1, len(self._v_grid))]) == True

        V_q_kPlus1_i = l_kPlus1_iPlusHalf - l_kPlus1_iMinusHalf
        V_q_kMinus1_i = l_kMinus1_iPlusHalf - l_kMinus1_iMinusHalf

        F_q_k_iPlusHalf = sdot * (1-v_iPlusHalf)  ## Adjective term
        F_q_k_iMinusHalf = sdot * (1-v_iMinusHalf)  ## Adjective term
        G_q_k_iPlusHalf = Dq_iPlusHalf / ((self._R-s_curr) * self._dv)  ## Diffusive term
        G_q_k_iMinusHalf = Dq_iMinusHalf / ((self._R-s_curr) * self._dv)  ## Diffusive term

        q_k_iPlus1 = np.append(q_curr[1:], np.nan).copy()
        q_k_iMinus1 = np.insert(q_curr[:-1], 0, np.nan).copy()
        # q_k_iPlusHalf = vShape_nan_arr.copy()
        # q_k_iPlusHalf[:-1] = self._wound_face_values(q_curr, a, b, 'q')
        # if not self.forceHalfNodeValues_HACK:
        #     if s_next>s_curr:
        #         assert all([q_k_iPlusHalf[i] == q_curr[i+1] for i in range(len(q_curr)-1)]) == True
        #     else:
        #         assert all([q_k_iPlusHalf[i] == q_curr[i] for i in range(len(q_curr)-1)]) == True


        H_q_k_i = (a*F_q_k_iPlusHalf) - (b*F_q_k_iMinusHalf) - G_q_k_iPlusHalf - G_q_k_iMinusHalf

        delta_q1 = V_q_kPlus1_i - (t_minus_deltaRatio*delta_t0 * H_q_k_i)
        delta_q2 = V_q_kMinus1_i + (t_plus_deltaRatio*delta_t0 * H_q_k_i)
        delta_q3 = delta_t0 * (b*F_q_k_iPlusHalf + G_q_k_iPlusHalf)
        delta_q4 = delta_t0 * (G_q_k_iMinusHalf - a*F_q_k_iMinusHalf)

        q_new_i = (((delta_q2*q_prev) + (delta_q3*q_k_iPlus1) + (delta_q4*q_k_iMinus1)) / delta_q1).copy()




        
        V_p_kPlus1_1 = r_kPlus1_iPlusHalf[0] - 0
        V_p_kMinus1_1 = r_kMinus1_iPlusHalf[0] - 0

        F_p_k_1PlusHalf = F_p_k_iPlusHalf[0]  ## Adjective term
        G_p_k_1PlusHalf = G_p_k_iPlusHalf[0]  ## Diffusive term

        H_p_k_1 = (a*F_p_k_1PlusHalf) - G_p_k_1PlusHalf

        delta_p5 = V_p_kPlus1_1 - (t_minus_deltaRatio*delta_t0 * H_p_k_1)
        delta_p6 = V_p_kMinus1_1 + (t_plus_deltaRatio*delta_t0 * H_p_k_1)
        delta_p7 = delta_t0 * (b*F_p_k_1PlusHalf + G_p_k_1PlusHalf)

        p_new_i[0] = (((delta_p6*p_prev[0]) + (delta_p7*p_k_iPlus1[0])) / delta_p5).copy()



        ## tecnically using the olaye notation this should be M+1 and M+1/2 (instead of M and M-1/2)
        V_q_kPlus1_M = self._R - l_kPlus1_iMinusHalf[-1]
        V_q_kMinus1_M = self._R - l_kMinus1_iMinusHalf[-1]

        F_q_k_MMinusHalf = F_q_k_iMinusHalf[-1]  ## Adjective term
        G_q_k_MMinusHalf = G_q_k_iMinusHalf[-1]  ## Diffusive term

        H_q_k_M = - (b*F_q_k_MMinusHalf) - G_q_k_MMinusHalf

        delta_q5 = V_q_kPlus1_M - (t_minus_deltaRatio*delta_t0 * H_q_k_M)
        delta_q6 = V_q_kMinus1_M + (t_plus_deltaRatio*delta_t0 * H_q_k_M)
        delta_q7 = delta_t0 * (G_q_k_MMinusHalf - a*F_q_k_MMinusHalf)

        q_new_i[-1] = (((delta_q6*q_prev[-1]) + (delta_q7*q_k_iMinus1[-1])) / delta_q5).copy()

        
        p_new_i, q_new_i = self._apply_boundary_conditions(p_new_i, q_new_i)
        return p_new_i, q_new_i

    def _computeState(self, t, xCurr):
        """
        Advances one explicit paper-style recurrence using the reworked updates.
        """
        # debugInPlace()
        p_curr = np.asarray(xCurr[0], dtype=np.float64).reshape(-1)
        q_curr = np.asarray(xCurr[1], dtype=np.float64).reshape(-1)
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
            p_new, q_new = self._classical_explicit_phase_update(
                t_curr=t,
                p_curr=p_curr,
                q_curr=q_curr,
                s_curr=s_curr,
                s_next=s_new,
                dt=dt,
                sdot=sdot,
                a=a_wind,
                b=b_wind,
                # t, p_curr, q_curr, s_curr, dt, sdot, D_p, D_q, a_wind, b_wind
            )
            self._lastStepScheme = "bootstrap_classical_explicit" if self._stepIndex == 0 else "classical_explicit"
        else:
            p_new, q_new = self._lfdf_phase_update(
                t_prev=self._t_prev, 
                t_curr=t,
                t_next=t+dt,
                p_prev=self._p_prev,
                p_curr=p_curr,
                q_prev=self._q_prev,
                q_curr=q_curr,
                s_prev=self._s_prev,
                s_curr=s_curr,
                s_next=s_new,
                dt=dt,
                D_p=D_p,
                D_q=D_q,
                a=a_wind,
                b=b_wind
            )
            self._lastStepScheme = "leapfrog_dufort_frankel_rework"

        p_new = np.clip(p_new, self.constraints.minComposition, 1 - self.constraints.minComposition)
        q_new = np.clip(q_new, self.constraints.minComposition, 1 - self.constraints.minComposition)
        p_new, q_new = self._apply_boundary_conditions(p_new, q_new)

        # self.checkMassIntegral(p=self._p_prev.copy(), q=self._q_prev.copy(), s=self._s_prev)
        # self.averageConc = self.checkMassIntegral(p=p_curr, q=q_curr, s=s_curr)
        # self.checkMassIntegral(p=p_new, q=q_new, s=s_new)
        # self.getTotalInventory()/self._R


        dpdt = (p_new - p_curr) / dt
        dqdt = (q_new - q_curr) / dt

        # fluxes, _, _ = self._build_physical_fluxes(t, p_curr, q_curr, s_curr)
        # self._lastFluxes = fluxes
        self._lastInterfaceFluxes = (-D_p[-1] * grad_left, -D_q[0] * grad_right)
        self._lastInterfaceVelocity = float(sdot)
        self._t_prev = t
        return dpdt, dqdt, float(sdot)

    