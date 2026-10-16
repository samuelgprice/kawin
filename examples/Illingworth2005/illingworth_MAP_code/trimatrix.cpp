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

#include <stdlib.h>
#include "trimatrix.h"

/**************************************
 algorithm for solving system of linear equations with tridiagonal matrix
***************************************/
void solve_trimatrix_system (trimatrix_system tms)
{
	double *alf, *bet;
	int n;
	int i;

	n=tms.dim;
	alf = (double *) calloc(n+1, sizeof(double));
	bet = (double *) calloc(n+1, sizeof(double));

	alf[0]=bet[0]=0.0;
	for( i=0; i<=n-1; i++)
	{
		alf[i+1] = tms.up[i]/(-tms.diag[i] - tms.lo[i]*alf[i] );
		bet[i+1]=( tms.lo[i]*bet[i] - tms.rhs[i] )/(-tms.diag[i] - tms.lo[i]*alf[i] );
	}

	tms.c[n-1] = bet[n];
	for( i=n-2; i>=0; i--)
	{
		tms.c[i] = alf[i+1]*tms.c[i+1]+bet[i+1];
	}

	free(alf);
	free(bet);
}
