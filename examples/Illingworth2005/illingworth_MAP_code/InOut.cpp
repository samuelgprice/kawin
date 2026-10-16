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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data_structures.h"
#include "InOut.h"


void out_profile(two_phase *tp, FILE *fp, double time)
{		// Prints:
			//	time
			//	x[0],	c(x[0]),
			//  x[1],	c(x[1]),
			//	....	....
			//  x[n],	c(x[n]),
			//	x[n+1],	c(x[n+1]),
			//  x[x+2],	c(x[n+2]),
			//	...		...
			//	x[n+N],	c(x[n+N]),
			//

	int i;

	fprintf(fp, "%lg,\n", time);
	for(i=0; i<(tp->left->n); i++)
	{
		fprintf(fp, "%lg,\t", tp->s*tp->left->u[i]);
		fprintf(fp, "%lg,\n", tp->left->c[i]);
	}

	for(i=0; i<(tp->right->n); i++)
	{
		fprintf(fp, "%lg,\t", tp->s + (tp->l-tp->s)*tp->right->u[i]);
		fprintf(fp, "%lg,\n", tp->right->c[i]);
	}	
	fprintf(fp, "\n");
}

void out_interface(two_phase *tp, double time, FILE *fp)
{
	fprintf(fp, "%lg\t%lg\n", time, tp->s);
}

