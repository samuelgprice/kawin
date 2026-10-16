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


/* data structure for a single phase region in a two-phase system */
typedef struct {
		double *u;			 /* co-ordinate of point i (in new co-ordinate system)      */
		double *c;           /* concentration at timestep k (in new co-ordinate system) */
		double *future_c;    /* concentration at timestep k+1                           */
		int n;               /* number of nodes                                         */
		double c_boundary;   /* Concentration at phase boundary                         */
		double d_coeff;      /* diffusion coefficient                                   */
		double a1_diffusion; /* frequency factor 1                                      */
		double q1_diffusion; /* activation energy 1                                     */
		double a2_diffusion; /* frequency factor 2                                      */
		double q2_diffusion; /* activation energy 2                                     */
		} single_phase;

typedef struct{
		single_phase *left;  /* Phase region on left                    */
		single_phase *right; /* Phase region on right                   */
		double old_s;        /* Interface position at timestep k-1      */
		double s;            /* Interface position at timestep k        */
		double future_s;     /* Interface position at timestep k+1      */
		double l;            /* Position of bounday (original co-ords)  */
		} two_phase;


