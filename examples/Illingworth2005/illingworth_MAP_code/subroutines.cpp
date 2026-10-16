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
#include <stdio.h>
#include "trimatrix.h"
#include "data_structures.h"
#include "subroutines.h"


// #########  This file uses 1st order conservative scheme to generate a solution on an irregular mesh ######### //

/*********************************************
	Given the concentration profiles and interface position at timestep j and estimates for the values at j+1,
	updates the estimate for interface position at timestep j+1 
**********************************************/
int new_interface_planar(two_phase *tp, double time_step, double pass_nmb)
{	

	//note: For first (implicit step), we require some first estimates for solution variables
			// 1) the elements in future_c to take the values of 'current_' c
			// 2) the value of future_s to be the same as s

	double v;
	double diff_l, diff_r;
	double rhs, lhs;

	if(pass_nmb==0)				// FIRST PASS - 
	{v=tp->s-tp->old_s;}			// Use old interface position to determine sign of interface velocity
	else
	{v=tp->future_s-tp->s;}		// SECOND (or higher) PASS -
									// Use estimate of future position to determine sign of interface velocity

	diff_l=(tp->left->c_boundary - tp->left->future_c[tp->left->n-2])/(1. - tp->left->u[tp->left->n-2]);
	diff_l= diff_l*tp->left->d_coeff / tp->future_s;
	
	diff_r=(tp->right->future_c[1] - tp->right->c_boundary)/(tp->right->u[1]);
	diff_r= diff_r*tp->right->d_coeff / (tp->l-tp->future_s);

	rhs = (diff_r - diff_l)*time_step;


	if(v>=0)			//POSITIVE VELOCITY - Use points to the right:
	{
		lhs = tp->left->c_boundary;
		lhs = lhs - tp->right->future_c[1]*(1-tp->right->u[1]/2.);
		lhs = lhs -	tp->right->c_boundary*tp->right->u[1]/2.;
	}
	else				//NEGATIVE VELOCITY - Use points to the left:
	{
		lhs = tp->left->future_c[tp->left->n-2] * (0.5+tp->left->u[tp->left->n-2]/2.);
 		lhs = lhs + tp->left->c_boundary * (0.5-tp->left->u[tp->left->n-2]/2.);
		lhs = lhs - tp->right->c_boundary;
	}

	tp->future_s=tp->s + rhs/lhs;

	return 0;

}


int new_interface_spherical(two_phase *tp, double time_step, double pass_nmb)
{	

	//note: For first (implicit step), we require some first estimates for solution variables
			// 1) the elements in future_c to take the values of 'current_' c
			// 2) the value of future_s to be the same as s

	double s_dot;
	double u_minus, v_plus;
	double x_minus_j, x_minus_jp1;
	double x_plus_j, x_plus_jp1;
	double rhs, lhs;
	double tmp, tmp1, tmp2;

	if(pass_nmb==0)				// FIRST PASS - 
	{s_dot=tp->s-tp->old_s;}	// Use old interface position to determine sign of interface velocity
	else
	{s_dot=tp->future_s-tp->s;}		// SECOND (or higher) PASS -
									// Use estimate of future position to determine sign of interface velocity


	u_minus     = (1+tp->left->u[(tp->left->n)-2])/2;
	x_minus_jp1 = u_minus * tp->future_s;				// Requires something sensible in future_s even at first pass
	x_minus_j   = u_minus * tp->s;

	v_plus     = (tp->right->u[1])/2;
	x_plus_jp1 = v_plus * (tp->l-tp->future_s) + tp->future_s;
	x_plus_j   = v_plus * (tp->l-tp->s) + tp->s;


	tmp1 = x_plus_jp1*x_plus_jp1 * tp->right->d_coeff * (tp->right->future_c[1]-tp->right->c_boundary);
	tmp1 = tmp1/((tp->l-tp->future_s) * tp->right->u[1]);

	tmp2 = x_minus_jp1*x_minus_jp1 * tp->left->d_coeff * (tp->left->c_boundary-tp->left->future_c[(tp->left->n)-2]);
	tmp2 = tmp2/(tp->future_s * (1-tp->left->u[(tp->left->n)-2]));

	rhs = tmp1 - tmp2;
	rhs = 3. * time_step * rhs;


	tmp = (tp->future_s*tp->future_s) + (tp->future_s*tp->s) + (tp->s*tp->s);
	
	tmp1 = x_minus_jp1*x_minus_jp1 + x_minus_jp1*x_minus_j + x_minus_j*x_minus_j;
	tmp1 = tmp - tmp1*u_minus;
	tmp1 = tmp1*tp->left->c_boundary;

	tmp2 = x_plus_jp1*x_plus_jp1 + x_plus_jp1*x_plus_j + x_plus_j*x_plus_j;
	tmp2 = tmp - tmp2*(1-v_plus);
	tmp2 = tmp2*tp->right->c_boundary;

	lhs = tmp1 - tmp2;


	if(s_dot>=0)			//POSITIVE VELOCITY - Use points to the right:
		{tmp = (x_minus_jp1*x_minus_jp1*u_minus*tp->left->c_boundary) - (x_plus_jp1*x_plus_jp1*(1-v_plus)*tp->right->future_c[1]);}
	else				//NEGATIVE VELOCITY - Use points to the left:
		{tmp = (x_minus_jp1*x_minus_jp1*u_minus*tp->left->future_c[(tp->left->n)-2]) - (x_plus_jp1*x_plus_jp1*(1-v_plus)*tp->right->c_boundary);}		
	lhs = lhs + (3.*tmp);


	tp->future_s = tp->s + (rhs/lhs);

	return 0;

}






/*********************************************
	Given the concentration profiles and interface position at timestep j and estimates for the interface at j+1,
	finds concentrations at j+1 in phase to the left of the boundary
*********************************************/
void new_concentration_left_planar(two_phase *tp, double time_step)
{
	int i;
	double tmpA, tmpB, left_diff, right_diff, left_sum, right_sum;
	trimatrix_system tms;


	tms.dim = tp->left->n;

// RESERVE MEMORY FOR tms
	tms.lo   = (double *) calloc(tms.dim, sizeof(double));
	tms.diag = (double *) calloc(tms.dim, sizeof(double));
	tms.up   = (double *) calloc(tms.dim, sizeof(double));
	tms.rhs  = (double *) calloc(tms.dim, sizeof(double));
	tms.c    = tp->left->future_c;



// FILL tms
	tmpA=tp->left->d_coeff*time_step / tp->future_s;
	tmpB=(tp->future_s-tp->s);

	if(tp->future_s >= tp->s)				// POSITIVE VELOCITY - use points to the right:
	{
		tms.lo[0]  =0.;
		tms.diag[0]=-tmpA/tp->left->u[1] - tp->future_s*tp->left->u[1]/2.;
		tms.up[0]  = tmpA/tp->left->u[1] + tmpB*tp->left->u[1]/2.;
		tms.rhs[0] =-tp->left->c[0]*tp->s*tp->left->u[1] / 2.;

		for(i=1; i<(tms.dim-1); i++)
		{
			left_diff = tp->left->u[i] - tp->left->u[i-1];
			left_sum  = tp->left->u[i] + tp->left->u[i-1];
			right_diff= tp->left->u[i+1] - tp->left->u[i];
			right_sum = tp->left->u[i+1] + tp->left->u[i];

			tms.lo[i]  = tmpA/left_diff;
			tms.diag[i]=-tmpA*(1/left_diff + 1/right_diff)  - tmpB*left_sum/2.
						- tp->future_s*(right_sum-left_sum)/2.;
			tms.up[i]  = tmpA/right_diff + tmpB*right_sum/2.;
			tms.rhs[i] =-tp->s*tp->left->c[i]*(right_sum-left_sum)/2.;
		}

	}

	else									// NEGATIVE VELOCITY - use points to the left:
	{
		tms.lo[0]  =0.;
		tms.diag[0]=-tmpA/tp->left->u[1] + tmpB*tp->left->u[1]/2. - tp->future_s*tp->left->u[1]/2.;
		tms.up[0]  = tmpA/tp->left->u[1];
		tms.rhs[0] =-tp->left->c[0]*tp->s*tp->left->u[1] / 2.;

		for(i=1; i<(tms.dim-1); i++)
		{
			
			left_diff = tp->left->u[i] - tp->left->u[i-1];
			left_sum  = tp->left->u[i] + tp->left->u[i-1];
			right_diff= tp->left->u[i+1] - tp->left->u[i];
			right_sum = tp->left->u[i+1] + tp->left->u[i];

			tms.lo[i]  = tmpA/left_diff  - tmpB*left_sum/2;
			tms.diag[i]=-tmpA*(1/left_diff + 1/right_diff) + tmpB*right_sum/2.
						- tp->future_s*(right_sum-left_sum)/2.;
			tms.up[i]  = tmpA/right_diff;
			tms.rhs[i] =-tp->s*tp->left->c[i]*(right_sum-left_sum)/2.;
		}
	}

	tms.lo[tms.dim-1]  = 0.;
	tms.diag[tms.dim-1]=-1.;
	tms.up[tms.dim-1]  = 0.;
	tms.rhs[tms.dim-1] =-tp->left->c_boundary;


// SOLVE tms
	solve_trimatrix_system(tms);


// FREE MEMORY RESERVED FOR tms	
	free(tms.lo);
	free(tms.diag);
	free(tms.up);
	free(tms.rhs);
}



void new_concentration_left_spherical(two_phase *tp, double time_step)
{
	int i;
	double r_plus, r_minus;
	double tmp;
	trimatrix_system tms;


	tms.dim = tp->left->n;

// RESERVE MEMORY FOR tms
	tms.lo   = (double *) calloc(tms.dim, sizeof(double));
	tms.diag = (double *) calloc(tms.dim, sizeof(double));
	tms.up   = (double *) calloc(tms.dim, sizeof(double));
	tms.rhs  = (double *) calloc(tms.dim, sizeof(double));
	tms.c    = tp->left->future_c;



// FILL tms		
	if(tp->future_s >= tp->s)				// POSITIVE VELOCITY - use points to the right:
	{
		r_plus  = (tp->left->u[1])/2;
		r_plus  = r_plus * tp->future_s;

		tms.lo[0] = 0;

		tmp =-(r_plus * r_plus)/tp->future_s * 1/(tp->left->u[1]) * time_step * tp->left->d_coeff;
		tmp = tmp - (r_plus*r_plus*r_plus)/3.;
		tms.diag[0] = tmp;

		tmp = (r_plus * r_plus)/tp->future_s * 1/(tp->left->u[1]) * time_step * tp->left->d_coeff;
		tmp = tmp + (tp->future_s - tp->s) * r_plus*r_plus * (tp->left->u[1]/2.);
		tms.up[0]  = tmp;

		r_plus  = (tp->left->u[1])/2;
		r_plus  = r_plus * tp->s;
		tmp = tp->left->c[0] * (r_plus*r_plus*r_plus/3.);
		tms.rhs[0] =-tmp;

		for(i=1; i<(tms.dim-1); i++)
		{
			r_plus  = (tp->left->u[i+1]+tp->left->u[i])/2;
			r_plus  = r_plus * tp->future_s;
			r_minus = (tp->left->u[i]+tp->left->u[i-1])/2;
			r_minus = r_minus * tp->future_s;

			tmp = ((r_minus * r_minus)/tp->future_s) * (1/(tp->left->u[i]-tp->left->u[i-1])) * time_step * tp->left->d_coeff;
			tms.lo[i] = tmp;

			tmp = (r_plus * r_plus)/(tp->left->u[i+1]-tp->left->u[i]);
			tmp = tmp + (r_minus * r_minus)/(tp->left->u[i]-tp->left->u[i-1]);
			tmp =-tmp * time_step * tp->left->d_coeff/tp->future_s;
			tmp = tmp - (tp->future_s - tp->s) * (r_minus*r_minus) * ((tp->left->u[i]+tp->left->u[i-1])/2.);
			tmp = tmp - ((r_plus*r_plus*r_plus)-(r_minus*r_minus*r_minus))/3.;
			tms.diag[i] = tmp;

			tmp = ((r_plus * r_plus)/tp->future_s) * (1/(tp->left->u[i+1]-tp->left->u[i])) * time_step * tp->left->d_coeff;
			tmp = tmp + (tp->future_s - tp->s) * (r_plus*r_plus) * ((tp->left->u[i+1]+tp->left->u[i])/2.);
			tms.up[i]  = tmp;


			r_plus  = (tp->left->u[i+1]+tp->left->u[i])/2;
			r_plus  = r_plus * tp->s;
			r_minus = (tp->left->u[i]+tp->left->u[i-1])/2;
			r_minus = r_minus * tp->s;
			tmp = tp->left->c[i] * ((r_plus*r_plus*r_plus - r_minus*r_minus*r_minus)/3.);
			tms.rhs[i] =-tmp;
		}

	}

	else									// NEGATIVE VELOCITY - use points to the left:
	{
		r_plus  = (tp->left->u[1]+tp->left->u[0])/2;
		r_plus  = r_plus * tp->future_s;

		tms.lo[0] = 0;

		tmp =-(r_plus * r_plus)/tp->future_s * 1/(tp->left->u[1]-tp->left->u[0]) * time_step * tp->left->d_coeff;
		tmp = tmp - (r_plus*r_plus*r_plus)/3.;
		tmp = tmp + (tp->future_s - tp->s) * r_plus*r_plus * ((tp->left->u[1]+tp->left->u[0])/2.);
		tms.diag[0] = tmp;

		tmp = (r_plus * r_plus)/tp->future_s * 1/(tp->left->u[1]-tp->left->u[0]) * time_step * tp->left->d_coeff;
		tms.up[0]  = tmp;


		r_plus  = (tp->left->u[1]+tp->left->u[0])/2;
		r_plus  = r_plus * tp->s;
		tmp = tp->left->c[0] * (r_plus*r_plus*r_plus/3.);
		tms.rhs[0] =-tmp;

		for(i=1; i<(tms.dim-1); i++)
		{
			
			r_plus  = (tp->left->u[i+1]+tp->left->u[i])/2;
			r_plus  = r_plus * tp->future_s;
			r_minus = (tp->left->u[i]+tp->left->u[i-1])/2;
			r_minus = r_minus * tp->future_s;

			tmp = (r_minus * r_minus)/tp->future_s * 1/(tp->left->u[i]-tp->left->u[i-1]) * time_step * tp->left->d_coeff;
			tmp = tmp - (tp->future_s - tp->s) * r_minus*r_minus * ((tp->left->u[i]+tp->left->u[i-1])/2.);
			tms.lo[i] = tmp;

			tmp =-(r_plus * r_plus)/tp->future_s * 1/(tp->left->u[i+1]-tp->left->u[i]);
			tmp = tmp - ((r_minus * r_minus)/tp->future_s * 1/(tp->left->u[i]-tp->left->u[i-1]));
			tmp = tmp * time_step * tp->left->d_coeff;
			tmp = tmp + (tp->future_s - tp->s) * r_plus*r_plus * ((tp->left->u[i+1]+tp->left->u[i])/2.);
			tmp = tmp - ((r_plus*r_plus*r_plus)-(r_minus*r_minus*r_minus))/3.;
			tms.diag[i] = tmp;

			tmp = (r_plus * r_plus)/tp->future_s * 1/(tp->left->u[i+1]-tp->left->u[i]) * time_step * tp->left->d_coeff;
			tms.up[i]  = tmp;


			r_plus  = (tp->left->u[i+1]+tp->left->u[i])/2;
			r_plus  = r_plus * tp->s;
			r_minus = (tp->left->u[i]+tp->left->u[i-1])/2;
			r_minus = r_minus * tp->s;
			tmp = tp->left->c[i] * ((r_plus*r_plus*r_plus - r_minus*r_minus*r_minus)/3.);
			tms.rhs[i] =-tmp;
		}
	}

	tms.lo[tms.dim-1]  = 0.;
	tms.diag[tms.dim-1]= 1.;
	tms.up[tms.dim-1]  = 0.;
	tms.rhs[tms.dim-1] = tp->left->c_boundary;


// SOLVE tms
	solve_trimatrix_system(tms);


// FREE MEMORY RESERVED FOR tms	
	free(tms.lo);
	free(tms.diag);
	free(tms.up);
	free(tms.rhs);
}








/*********************************************
	Given the concentration profiles and interface position at timestep j and estimates for the interface at j+1,
	finds concentrations at j+1 in phase to the right of the boundary
*********************************************/
void new_concentration_right_planar(two_phase *tp, double time_step)
{
	int i;
	double tmp, tmpA, tmpB, left_diff, right_diff, left_sum, right_sum;
	trimatrix_system tms;

	tms.dim = tp->right->n;

// RESERVE MEMORY FOR tms
	tms.lo   = (double *) calloc(tms.dim, sizeof(double));
	tms.diag = (double *) calloc(tms.dim, sizeof(double));
	tms.up   = (double *) calloc(tms.dim, sizeof(double));
	tms.rhs  = (double *) calloc(tms.dim, sizeof(double));
	tms.c    = tp->right->future_c;



// FILL tms
	tmpA=tp->l-tp->future_s;
	tmpA=tp->right->d_coeff*time_step/tmpA;
	tmpB=(tp->future_s-tp->s);


	tms.lo[0]  = 0.;
	tms.diag[0]=-1.;
	tms.up[0]  = 0.;
	tms.rhs[0] =-tp->right->c_boundary;

	if(tp->future_s >= tp->s)				// POSITIVE VELOCITY - use points to the right:
	{
		for(i=1; i<(tms.dim-1); i++)
		{
			left_diff = tp->right->u[i] - tp->right->u[i-1];
			left_sum  = tp->right->u[i] + tp->right->u[i-1];
			right_diff= tp->right->u[i+1] - tp->right->u[i];
			right_sum = tp->right->u[i+1] + tp->right->u[i];

			tms.lo[i]  = tmpA/left_diff;
			tms.diag[i]=-tmpA*(1/right_diff + 1/left_diff) - tmpB*(1.- left_sum/2.)
				        - (tp->l-tp->future_s)*(right_sum-left_sum)/ 2.;
			tms.up[i]  = tmpA/right_diff + tmpB*(1. - right_sum/2.);
			tms.rhs[i] =-(tp->l-tp->s)*tp->right->c[i]*(right_sum-left_sum) / 2.;
		}

		tmp=tp->right->u[tp->right->n-2];

		tms.lo[tms.dim-1]  = tmpA/(1.-tmp);
		tms.diag[tms.dim-1]=-tmpA/(1.-tmp) - tmpB*(1. - (1.+tmp)/2.) - (tp->l-tp->future_s)*(1.-tmp)/2.;
		tms.up[tms.dim-1]  = 0;
		tms.rhs[tms.dim-1] =-tp->right->c[tp->right->n-1]*(tp->l-tp->s)*(1.-tmp) / 2.;
	}

	else								// NEGATIVE VELOCITY - use points to the left:
	{
		for(i=1; i<(tms.dim-1); i++)
		{
			left_diff = tp->right->u[i] - tp->right->u[i-1];
			left_sum  = tp->right->u[i] + tp->right->u[i-1];
			right_diff= tp->right->u[i+1] - tp->right->u[i];
			right_sum = tp->right->u[i+1] + tp->right->u[i];

			tms.lo[i]  = tmpA/left_diff - tmpB*(1.- left_sum/2.);
			tms.diag[i]=-tmpA*(1/right_diff + 1/left_diff) + tmpB*(1. - right_sum/2.)
						- (tp->l-tp->future_s)*(right_sum - left_sum) / 2.;
			tms.up[i]  = tmpA/right_diff;
			tms.rhs[i] =-(tp->l-tp->s)*tp->right->c[i]*(right_sum-left_sum) / 2.;
		}

		tmp=tp->right->u[tp->right->n-2];

		tms.lo[tms.dim-1]  = tmpA/(1.-tmp) - tmpB*(1. - (1.+tmp)/2.);
		tms.diag[tms.dim-1]=-tmpA/(1.-tmp) - (tp->l-tp->future_s)*(1.-tmp)/2.;
		tms.up[tms.dim-1]  =0;
		tms.rhs[tms.dim-1] =-tp->right->c[tp->right->n-1]*(tp->l-tp->s)*(1.-tmp) / 2.;
	}


// SOLVE tms
	solve_trimatrix_system(tms);


// FREE MEMORY RESERVED FOR tms	
	free(tms.lo);
	free(tms.diag);
	free(tms.up);
	free(tms.rhs);
}



void new_concentration_right_spherical(two_phase *tp, double time_step)
{
	int i;
	double r_plus, r_minus;
	double tmp;
	trimatrix_system tms;

	tms.dim = tp->right->n;

// RESERVE MEMORY FOR tms
	tms.lo   = (double *) calloc(tms.dim, sizeof(double));
	tms.diag = (double *) calloc(tms.dim, sizeof(double));
	tms.up   = (double *) calloc(tms.dim, sizeof(double));
	tms.rhs  = (double *) calloc(tms.dim, sizeof(double));
	tms.c    = tp->right->future_c;



// FILL tms

	tms.lo[0]  = 0.;
	tms.diag[0]= 1.0;
	tms.up[0]  = 0.;
	tms.rhs[0] = tp->right->c_boundary;

	if(tp->future_s >= tp->s)				// POSITIVE VELOCITY - use points to the right:
	{
		for(i=1; i<(tms.dim-1); i++)
		{
			r_plus  = (tp->right->u[i+1] + tp->right->u[i])/2.;
			r_plus  = (r_plus * (tp->l-tp->future_s)) + tp->future_s;
			r_minus = (tp->right->u[i] + tp->right->u[i-1])/2.;
			r_minus = (r_minus * (tp->l-tp->future_s)) + tp->future_s;

			tmp = (r_minus * r_minus)/(tp->l-tp->future_s) * 1./(tp->right->u[i]-tp->right->u[i-1]) * time_step * tp->right->d_coeff;
			tms.lo[i] = tmp;

			tmp =-(r_plus * r_plus)/(tp->right->u[i+1]-tp->right->u[i]) - (r_minus * r_minus)/(tp->right->u[i]-tp->right->u[i-1]);
			tmp = tmp * time_step * tp->right->d_coeff/(tp->l-tp->future_s);
			tmp = tmp - (tp->future_s - tp->s) * r_minus*r_minus * (1. - (tp->right->u[i]+tp->right->u[i-1])/2.);
			tmp = tmp - ((r_plus*r_plus*r_plus)-(r_minus*r_minus*r_minus))/3.;
			tms.diag[i] = tmp;

			tmp = (r_plus * r_plus)/(tp->l-tp->future_s) * 1./(tp->right->u[i+1]-tp->right->u[i]) * time_step * tp->right->d_coeff;
			tmp = tmp + (tp->future_s - tp->s) * r_plus*r_plus * (1. - (tp->right->u[i+1]+tp->right->u[i])/2.);
			tms.up[i]  = tmp;


			r_plus  = (tp->right->u[i+1] + tp->right->u[i])/2.;
			r_plus  = (r_plus * (tp->l-tp->s)) + tp->s;
			r_minus = (tp->right->u[i] + tp->right->u[i-1])/2.;
			r_minus = (r_minus * (tp->l-tp->s)) + tp->s;
			tmp = tp->right->c[i] * ((r_plus*r_plus*r_plus - r_minus*r_minus*r_minus)/3.);
			tms.rhs[i] =-tmp;
		}

		r_minus = (1 + tp->right->u[(tp->right->n)-2])/2.;
		r_minus = (r_minus * (tp->l-tp->future_s)) + tp->future_s;

		tmp = (r_minus * r_minus)/(tp->l-tp->future_s) * 1./(1-tp->right->u[(tp->right->n)-2]) * time_step * tp->right->d_coeff;
		tms.lo[(tp->right->n)-1] = tmp;

		tmp =-(r_minus * r_minus)/(tp->l-tp->future_s) * 1./(1-tp->right->u[(tp->right->n)-2]) * time_step * tp->right->d_coeff;
		tmp = tmp - (tp->future_s - tp->s) * r_minus*r_minus * (1. - (1+tp->right->u[(tp->right->n)-2])/2.);
		tmp = tmp - (tp->l*tp->l*tp->l - r_minus*r_minus*r_minus)/3.;
		tms.diag[(tp->right->n)-1] = tmp;

		tms.up[(tp->right->n)-1]  = 0.;


		r_minus = (1 + tp->right->u[(tp->right->n)-2])/2.;
		r_minus = (r_minus * (tp->l-tp->s)) + tp->s;		
		tmp = tp->right->c[(tp->right->n)-1] * (tp->l*tp->l*tp->l - r_minus*r_minus*r_minus)/3.;
		tms.rhs[(tp->right->n)-1] =-tmp;
	}

	else								// NEGATIVE VELOCITY - use points to the left:
	{
		for(i=1; i<(tms.dim-1); i++)
		{
			r_plus  = (tp->right->u[i+1] + tp->right->u[i])/2.;
			r_plus  = r_plus * (tp->l-tp->future_s) + tp->future_s;
			r_minus = (tp->right->u[i] + tp->right->u[i-1])/2.;
			r_minus = (r_minus * (tp->l-tp->future_s)) + tp->future_s;

			tmp = (r_minus * r_minus)/(tp->l-tp->future_s) * 1/(tp->right->u[i]-tp->right->u[i-1]) * time_step * tp->right->d_coeff;
			tmp = tmp - (tp->future_s - tp->s) * r_minus*r_minus * (1 - (tp->right->u[i]+tp->right->u[i-1])/2.);
			tms.lo[i] = tmp;

			tmp =-(r_plus * r_plus)/(tp->l-tp->future_s) * 1/(tp->right->u[i+1]-tp->right->u[i]);
			tmp = tmp - ((r_minus * r_minus)/(tp->l-tp->future_s) * 1/(tp->right->u[i]-tp->right->u[i-1]));
			tmp = tmp * time_step * tp->right->d_coeff;
			tmp = tmp + (tp->future_s - tp->s) * r_plus*r_plus * (1 - (tp->right->u[i+1]+tp->right->u[i])/2.);
			tmp = tmp - ((r_plus*r_plus*r_plus)-(r_minus*r_minus*r_minus))/3.;
			tms.diag[i] = tmp;

			tmp = (r_plus * r_plus)/(tp->l-tp->future_s) * 1/(tp->right->u[i+1]-tp->right->u[i]) * time_step * tp->right->d_coeff;
			tms.up[i]  = tmp;

			r_plus  = (tp->right->u[i+1] + tp->right->u[i])/2.;
			r_plus  = (r_plus * (tp->l-tp->s)) + tp->s;
			r_minus = (tp->right->u[i] + tp->right->u[i-1])/2.;
			r_minus = (r_minus * (tp->l-tp->s)) + tp->s;
			tmp = tp->right->c[i] * ((r_plus*r_plus*r_plus - r_minus*r_minus*r_minus)/3.);
			tms.rhs[i] =-tmp;
		}
		
		r_minus = (tp->right->u[(tp->right->n)-1] + tp->right->u[(tp->right->n)-2])/2.;
		r_minus = (r_minus * (tp->l-tp->future_s)) + tp->future_s;

		tmp = (r_minus * r_minus)/(tp->l-tp->future_s) * 1/(tp->right->u[(tp->right->n)-1]-tp->right->u[(tp->right->n)-2]) * time_step * tp->right->d_coeff;
		tmp = tmp - (tp->future_s - tp->s) * r_minus*r_minus * (1 - (tp->right->u[(tp->right->n)-1]+tp->right->u[(tp->right->n)-2])/2.);
		tms.lo[(tp->right->n)-1] = tmp;

		tmp =-(r_minus * r_minus)/(tp->l-tp->future_s) * 1/(tp->right->u[(tp->right->n)-1]-tp->right->u[(tp->right->n)-2]) * time_step * tp->right->d_coeff;
		tmp = tmp - (tp->l*tp->l*tp->l - r_minus*r_minus*r_minus)/3.;
		tms.diag[(tp->right->n)-1] = tmp;

		tms.up[(tp->right->n)-1]  = 0.;


		r_minus = (tp->right->u[(tp->right->n)-1] + tp->right->u[(tp->right->n)-2])/2.;
		r_minus = (r_minus * (tp->l-tp->s)) + tp->s;
		tmp = tp->right->c[(tp->right->n)-1] * (tp->l*tp->l*tp->l - r_minus*r_minus*r_minus)/3.;
		tms.rhs[(tp->right->n)-1] =-tmp;
	}


// SOLVE tms
	solve_trimatrix_system(tms);

// FREE MEMORY RESERVED FOR tms	
	free(tms.lo);
	free(tms.diag);
	free(tms.up);
	free(tms.rhs);
}








/*********************************************
	Takes system from time step j to timestep j+1
*********************************************/
int	take_step_planar(two_phase *tp, double dt, double tolerance)
{
	int i, count;
	double error;
	double *tmp_dbl_ptr;
	

	/************************** BOOKKEEPING **************************/	
	// First prediction of interface position at j+1 is position at j
	tp->future_s = tp->s;
	for(i=0; i<(tp->left->n); i++)
		{tp->left->future_c[i]  = tp->left->c[i];}
	for(i=0; i<(tp->right->n); i++)
		{tp->right->future_c[i] = tp->right->c[i];}



	/************************** TAKE STEP **************************/
	error=tolerance+1;
	count=0;
	while((error>tolerance) || (count==1)) // need at least one implicit loop
	{

		error=tp->future_s;

		// find new interface position
		new_interface_planar(tp, dt, count);

		// solve for concentration to left of interface:
		new_concentration_left_planar(tp, dt);

		// solve for concentration to right of interface:
		new_concentration_right_planar(tp, dt);

		if(error>tp->future_s)				// (old_estimate > new_estimate)
			{error = error-tp->future_s;}
		else								// (new_estimate <= old_estimate)
			{error = tp->future_s-error;}

		count++;
	}


	/************************** BOOKKEEPING **************************/
	// Update s:
	tp->old_s = tp->s;
	tp->s     = tp->future_s;

	if((tp->s<0.) || (tp->s>tp->l))
	{// Interface has moved beyond the ends of the system
		printf("\n\nInterface has moved beyond the end of the system\n");
		return -1;
	}
	else
	{// Update c:
		tmp_dbl_ptr = tp->left->c;
		tp->left->c        = tp->left->future_c;
		tp->left->future_c = tmp_dbl_ptr;
		tmp_dbl_ptr = tp->right->c;
		tp->right->c        = tp->right->future_c;
		tp->right->future_c = tmp_dbl_ptr;
		return count;
	}
}


int	take_step_spherical(two_phase *tp, double dt, double tolerance)
{
	int i, count;
	double error;
	double *tmp_dbl_ptr;
	

	/************************** BOOKKEEPING **************************/	
	// First prediction of interface position at j+1 is position at j
	tp->future_s = tp->s;
	for(i=0; i<(tp->left->n); i++)
		{tp->left->future_c[i]  = tp->left->c[i];}
	for(i=0; i<(tp->right->n); i++)
		{tp->right->future_c[i] = tp->right->c[i];}



	/************************** TAKE STEP **************************/
	error=tolerance+1;
	count=0;
	while((error>tolerance) || (count==1)) // need at least one implicit loop
	{

		error=tp->future_s;

		// find new interface position
		new_interface_spherical(tp, dt, count);

		// solve for concentration to left of interface:
		new_concentration_left_spherical(tp, dt);

		// solve for concentration to right of interface:
		new_concentration_right_spherical(tp, dt);

		if(error>tp->future_s)				// (old_estimate > new_estimate)
			{error = error-tp->future_s;}
		else								// (new_estimate <= old_estimate)
			{error = tp->future_s-error;}

		count++;
	}


	/************************** BOOKKEEPING **************************/
	// Update s:
	tp->old_s = tp->s;
	tp->s     = tp->future_s;

	if((tp->s<0.) || (tp->s>tp->l))
	{// Interface has moved beyond the ends of the system
		printf("\n\nInterface has moved beyond the end of the system\n");
		return -1;
	}
	else
	{// Update c:
		tmp_dbl_ptr = tp->left->c;
		tp->left->c        = tp->left->future_c;
		tp->left->future_c = tmp_dbl_ptr;
		tmp_dbl_ptr = tp->right->c;
		tp->right->c        = tp->right->future_c;
		tp->right->future_c = tmp_dbl_ptr;
		return count;
	}
}
