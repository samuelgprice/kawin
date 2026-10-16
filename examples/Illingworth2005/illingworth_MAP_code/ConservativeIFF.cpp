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
#include <math.h>
#include "data_structures.h"
#include "subroutines.h"
#include "InOut.h"

int main( void )
{

/********************************************************************************************
 ****************************** User-defined input parameters *******************************
 ********************************************************************************************/

	int lambda = 1;		// Parameter describing geometry of system (=1 for planar, =3 for spherical)

	double s_0 = 1.;	// Initial interface position
	double R   = 5.;	// Position of far boundary

	int nAlpha = 100;			// Number of points in phase alpha
	double dAlpha = 1E-7;		// Diffusion coefficient of phase alpha
	double initialAlpha = 0.8;	// Initial concentration in phase alpha
	double interAlpha   = 0.6;	// Interfacial concentration in phase alpha


	int nBeta    = 100;
	double dBeta = 1E-5;
	double initialBeta  = 0.4;
	double interBeta   = 0.0;

	double time_step=0.1;
	int n_time_steps=10;	// As written, output written to maximum 100000000 steps

	double tol = 1.E-8;		// Used to determine whether linearisation converges at each timestep

/********************************************************************************************
 ******************************************* ENDS *******************************************
 ********************************************************************************************/



	FILE *fpt;
	int i, tmp;
	two_phase *whole_system;

	whole_system =(two_phase *) calloc (1, sizeof(two_phase));

	whole_system->s = s_0;
	whole_system->l = R;
	whole_system->old_s = whole_system->s;
	whole_system->future_s = whole_system->s;


	// Alpha is on left
	whole_system->left =(single_phase *) calloc (1, sizeof(single_phase));

	whole_system->left->n = nAlpha;
	whole_system->left->d_coeff = dAlpha;
	whole_system->left->c_boundary = interAlpha;

	whole_system->left->u =(double *) calloc (whole_system->left->n, sizeof(double));
	whole_system->left->c =(double *) calloc (whole_system->left->n, sizeof(double));
	whole_system->left->future_c =(double *) calloc (whole_system->left->n, sizeof(double));


	for (i=0; i<whole_system->left->n; i++)
	{
	whole_system->left->u[i] = double(i)/double(whole_system->left->n - 1);
	whole_system->left->c[i] = initialAlpha;
	whole_system->left->future_c[i] = whole_system->left->c[i];
	}
	whole_system->left->c[whole_system->left->n-1] = whole_system->left->c_boundary;
	whole_system->left->future_c[whole_system->left->n-1] = whole_system->left->c_boundary;

	// Beta is on right
	whole_system->right =(single_phase *) calloc (1, sizeof(single_phase));

	whole_system->right->n = nBeta;
	whole_system->right->d_coeff = dBeta;
	whole_system->right->c_boundary = interBeta;
	
	whole_system->right->u =(double *) calloc (whole_system->right->n, sizeof(double));
	whole_system->right->c =(double *) calloc (whole_system->right->n, sizeof(double));
	whole_system->right->future_c =(double *) calloc (whole_system->right->n, sizeof(double));

	for (i=0; i<whole_system->right->n; i++)
	{
	whole_system->right->u[i] = double(i)/double(whole_system->right->n - 1);
	whole_system->right->c[i] = initialBeta;
	whole_system->right->future_c[i] = whole_system->right->c[i];
	}
	whole_system->right->c[0] = whole_system->right->c_boundary;
	whole_system->right->future_c[0] = whole_system->right->c_boundary;



	fpt=fopen("results.txt","w");

	fprintf(fpt, "Time\tInterface Position\n");
	/***** Looping over time (for the given timestep) *****/
	for (i=0; i<n_time_steps+1; i++)
	{
		if(i<100)
			{out_interface(whole_system, double(i)*time_step, fpt);}
		else if((i<1000) && (i%10 == 0))
			{out_interface(whole_system, double(i)*time_step, fpt);}
		else if((i<10000) && (i%100 == 0))
			{out_interface(whole_system, double(i)*time_step, fpt);}
		else if((i<100000) && (i%1000 == 0))
			{out_interface(whole_system, double(i)*time_step, fpt);}
		else if((i<1000000) && (i%10000 == 0))
			{out_interface(whole_system, double(i)*time_step, fpt);}
		else if((i<10000000) && (i%100000 == 0))
			{out_interface(whole_system, double(i)*time_step, fpt);}
		else if((i<100000000) && (i%1000000 == 0))
			{out_interface(whole_system, double(i)*time_step, fpt);}

		if(lambda == 1)
			{tmp = take_step_planar(whole_system, time_step, tol);}
		else if(lambda == 3)
			{tmp = take_step_spherical(whole_system, time_step, tol);}
		else
			{printf("\nProblem with geometry\n\n");}


		if(tmp<0)	// Interface has moved beyond the end of the system
			{printf("Finished at time = %lg", double(i)*time_step); fclose(fpt); return 0;}
		
		if(i%1000 == 0)
			{printf("\nTimestep %d complete\n", i);}
		else if (i%10 == 0)
			{printf(".");}
	}

	free(whole_system->left->u);
	free(whole_system->left->c);
	free(whole_system->left->future_c);
	free(whole_system->right->u);
	free(whole_system->right->c);
	free(whole_system->right->future_c);
	free(whole_system->left);
	free(whole_system->right);

	fclose(fpt);

	printf("\n \n");
	return 1;

}