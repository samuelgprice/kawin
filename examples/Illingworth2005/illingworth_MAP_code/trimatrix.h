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


/* data structure for a tri-diagonal matrix: */
typedef struct {
		double *lo;   /* lower diagonal           */
		double *diag; /* diagonal                 */
		double *up;   /* upper diagonal           */
		double *rhs;  /* right hand side          */
		double *c;    /* solution - concentration */
		int dim;      /* dimension of matrix =n   */
		} trimatrix_system;

//   system looks like:		/diag0 up0    0      0  ..   0    0    \  / c0   \   /  rhs0  \
//							| lo1  diag1  up1    0  ..   0    0     | | c1   |   |  rhs1  |
//							|  0    lo2  diag2  up2 ..   0    0     | | c2   |   |  rhs2  |
//							|            ....                       | |  ..  | = |   ..   |
//							|            ....                       | |  ..  |   |   ..   |
//							\  0     0     0  ..      lon-1 diagn-1 /  \cn-1/    \ rhsn-1/


extern trimatrix_system tms;

/* algorithm for solving system of linear equations with tridiagonal matrix */
void solve_trimatrix_system (trimatrix_system tms);