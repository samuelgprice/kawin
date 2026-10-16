/*
 * Materials Algorithms Project (MAP), University of Cambridge:
 *   MAP_KINETICS_MOVING_BOUNDARY (ConservativeIFF)
 *   https://www.phase-trans.msm.cam.ac.uk/map/kinetics/programs/ConservativeIFF.html
 * Code originators: T.C. Illingworth and I.O. Golosnoy, Department of Materials
 *   Science and Metallurgy, University of Cambridge (added to MAP February 2005).
 * Method: T.C. Illingworth, I.O. Golosnoy, "Numerical solutions of diffusion-controlled
 *   moving boundary problems which conserve solute", J. Comput. Phys. 209 (2005) 207-225.
 *
 * Redistributed in kawin under the MAP conditions (see README.md in this folder),
 * not under kawin's MIT licence. Modification made for kawin: this comment block was
 * added; the code below is unchanged from the MAP download (kinetics/tar/illingworth.zip).
 */


/*	Given the concentration profiles and interface position at timestep j and estimates for the values at j+1,
	updates the estimate for interface position at timestep j+1 */
int new_interface_planar(two_phase *tp, double time_step, double pass_nmb);
int new_interface_spherical(two_phase *tp, double time_step, double pass_nmb);

/*	Given the concentration profiles and interface position at timestep j and estimates for the interface at j+1,
	finds concentrations at j+1 in phase to the left of the boundary	*/
void new_concentration_left_planar(two_phase *tp, double time_step);
void new_concentration_left_spherical(two_phase *tp, double time_step);

/*	Given the concentration profiles and interface position at timestep j and estimates for the interface at j+1,
	finds concentrations at j+1 in phase to the right of the boundary	*/
void new_concentration_right_planar(two_phase *tp, double time_step);
void new_concentration_right_spherical(two_phase *tp, double time_step);		


/*	Takes system from time step j to timestep j+1 */
int	take_step_planar(two_phase *tp, double dt, double tolerance);
int	take_step_spherical(two_phase *tp, double dt, double tolerance);
