// iBayes - Elastic Bayesian Inference Framework with Sparse Grid
// Copyright (C) 2015-today Ao Mo-Hellenbrand
//
// All copyrights remain with the respective authors.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#ifndef SIM_NSSIM_HPP_
#define SIM_NSSIM_HPP_

#include <tools/matrix.hpp>
#include <Eigen/Sparse>
#include <Eigen/Eigen>
#include <cstdio>
#include <cmath>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>


/* Class Definiation */
class NSSim
{
friend class NSSimDist;

private:
	// Simulation Setup
	static constexpr double DOMAIN_SIZE_X = 27.0;	/// Domain size in x-direction
	static constexpr double DOMAIN_SIZE_Y = 15.0;	/// Domain size in y-direction
	static constexpr double INITIAL_VELOCITY_X = 1.0;	/// Initial velocity in x-direction
	static constexpr double INITIAL_VELOCITY_Y = 0.0;	/// Initial velocity in y-direction
	static constexpr double INITIAL_PRESSURE = 0.0;		/// Initial pressure
	static constexpr double INFLOW_VELOCITY_X = 1.0;	/// In-flow velocity in x-direction
	static constexpr double INFLOW_VELOCITY_Y = 0.0;	/// In-flow velocity in y-direction
	static constexpr double EXTERNAL_FORCE_X = 0.0;		/// External force in x-direction
	static constexpr double EXTERNAL_FORCE_Y = 0.0;		/// External force in y-direction
	static constexpr double RE = 100.0;		/// Reynolds number
	static constexpr double TAU = 0.5;		/// Safety factor for time step size computation
	static constexpr double ALPHA = 0.9;	/// Upwind differecing factor
	static constexpr double OMEGA = 1.7;	/// Pressure SOR solver related

	// Boundary values: 1 for inflow; 2 for outflow; 3 for no-slip; 4 for free-slip
	static const int BOUNDARY_LEFT = 1;
	static const int BOUNDARY_RIGHT = 2;
	static const int BOUNDARY_TOP = 3;
	static const int BOUNDARY_BOTTOM = 3;
	// All inner domain boundaries are no-slip
	static const int NUM_OBS = 17;
	static constexpr double OBS_XMIN[] = {0.0, 2.0,  9.0, 11.0, 16.0,  0.0,
			5.0, 10.0, 11.0, 16.0, 14.0, 15.0, 18.0, 24.0, 23.0, 22.0, 21.0};
	static constexpr double OBS_XMAX[] = {9.0, 8.0, 11.0, 16.0, 18.0,  5.0,
		   10.0, 11.0, 16.0, 21.0, 15.0, 18.0, 19.0, 27.0, 24.0, 23.0, 22.0};
	static constexpr double OBS_YMIN[] = {0.0, 5.0,  0.0,  0.0,  0.0,  9.0,
		   10.0, 11.0, 12.0, 13.0,  7.0,  6.0,  8.0,  0.0,  3.0,  4.0,  5.0};
	static constexpr double OBS_YMAX[] = {5.0, 6.0,  4.0,  3.0,  2.0, 15.0,
		   15.0, 15.0, 15.0, 15.0,  9.0, 10.0, 10.0, 12.0, 11.0, 10.0, 10.0};

	//MASK VALUES [CEWNS]
	static const int B_E  = 10111;
	static const int B_W  = 11011;
	static const int B_N  = 11101;
	static const int B_S  = 11110;
	static const int B_NE = 10101;
	static const int B_SE = 10110;
	static const int B_NW = 11001;
	static const int B_SW = 11010;
	static const int B_IN = 11111;

	// Simulation variables
	std::size_t ncx;	/// Number of grid cells in x-direction
	std::size_t ncy;	/// Number of grid cells in y-direction
	double dx;
	double dy;

public:
	/* DEFINE destructor (no ;)*/
	~NSSim() {}

	/* Declare constructor */
	NSSim(std::size_t num_cells_x, std::size_t num_cells_y);

	void run();

private:
	/*****************************************
	 * Internal core functions
	 *****************************************/

	/**
	 * Create a mask array to distinguish fluid and different types of obstacle cells
	 */
	int** create_geometry_mask();

	/**
	 * Create system matrix A
	 * A is N-by-N (no boundary cells)
	 */
	Eigen::SparseMatrix<double> create_system_matrix();

	/**
	 * Compute the right hand side of the pressure equation
	 */
	void compute_rhs(
			double&  dt,
			double**& F,
			double**& G,
			double**& P,
			double**& RHS);

	/**
	 * Update boundaries of U,V: BOTTOM (no-slip), TOP (no-slip), LEFT (in-flow), RIGHT (out-flow)
	 */
	void update_boundaries_uv(
			double**& U,
			double**& V);

	/**
	 * Update boundaries of F,G
	 */
	void update_boundaries_fg(
			double**& U,
			double**& V,
			double**& F,
			double**& G);

	/**
	 * Update boundaries of P: discrete Neumann condition
	 */
	void update_boundaries_p(double** &P);

	/**
	 * Update obstacle cells: Set U,V
	 */
	void update_domain_uv(
			int**&    M,
			double**& U,
			double**& V);

	/**
	 * Update obstacle cells: Set F G
	 */
	void update_domain_fg(
			int**&    M,
			double**& U,
			double**& V,
			double**& F,
			double**& G);

	/**
	 * Update obstacle cells: Set P
	 */
	void update_domain_p(
			int**&    M,
			double**& P);

	/**
	 * Compute time step size dt
	 */
	void compute_dt(
			double**& U,
			double**& V,
			double& dt);

	/**
	 * Determine the values of F and G.
	 */
	void compute_fg(
			double& dt,
			double**& U,
			double**& V,
			double**& F,
			double**& G);

	void solve_for_p_sor(
			double& dt,
			int**& M,
			double**& F,
			double**& G,
			double**& P);

	/**
	 * Solve the pressure equation (using an external solver from Eigen)
	 */
	void solve_for_p_direct(
			Eigen::SparseMatrix<double>& A,
			double**& RHS,
			double**& P);

	/**
	 * Compute new velocity values
	 */
	void compute_uv(
			double&  dt,
			double**& F,
			double**& G,
			double**& P,
			double**& U,
			double**& V);

	/**
	 * Same as write_vtkFile(...)
	 * but with in-domain obstacle cells darkened (artificially set U,V,P values to 0)
	 */
	void write_vtk_file(
			const char *szProblem,
			int timeStepNumber,
			int**&    M,
			double**& U,
			double**& V,
			double**& P);

	/**
	 * Method for writing header information and coordinate points of a VTK file
	 */
	void write_vtk_header_coord(std::ofstream& fout);


	/*****************************************
	 * In-line functions (Finite-Difference)
	 *****************************************/
	/**
	 * Laplacian - second order accurate (spatial) Laplacian
	 */
	inline
	static
	double laplacian (double **m, std::size_t j, std::size_t i, double dx, double dy)
	{
		return (m[j][i+1] - 2.0*m[j][i] + m[j][i-1]) / (dx*dx) +
			   (m[j+1][i] - 2.0*m[j][i] + m[j-1][i]) / (dy*dy);
	}

	/**
	 * FD_x_U2 - first derivative along x of U (velocity along x-direction) squared, with alpha
	 */
	inline
	static
	double FD_x_U2 (double **u, std::size_t j, std::size_t i, double dx, double alpha)
	{
		return (            ( (u[j][i] + u[j][i+1]) * (u[j][i] + u[j][i+1]) -
							  (u[j][i-1] + u[j][i]) * (u[j][i-1] + u[j][i])
						    )
				  + alpha * ( fabs(u[j][i] + u[j][i+1]) * (u[j][i] - u[j][i+1]) -
							  fabs(u[j][i-1] + u[j][i]) * (u[j][i-1] - u[j][i])
						    )
			   ) / (dx * 4.0);
	}

	/**
	 * FD_y_V2 - first derivative along y of V (velocity along y-direction) squared, with alpha
	 */
	inline
	static
	double FD_y_V2 (double **v, std::size_t j, std::size_t i, double dy, double alpha)
	{
		return (            ( (v[j  ][i] + v[j+1][i]) * (v[j  ][i] + v[j+1][i]) -
				              (v[j-1][i] + v[j  ][i]) * (v[j-1][i] + v[j  ][i])
				            )
				  + alpha * ( fabs(v[j  ][i] + v[j+1][i]) * (v[j  ][i] - v[j+1][i]) -
						      fabs(v[j-1][i] + v[j  ][i]) * (v[j-1][i] - v[j  ][i])
						    )
		       ) / (dy * 4.0);
	}

	/**
	 * FD_x_UV - first derivative along x of UV,
	 * evaluated at the respective point of our staggered grid (with alpha)
	 */
	inline
	static
	double FD_x_UV (double **u, double **v, std::size_t j, std::size_t i, double dx, double alpha)
	{
		return (            ( (u[j][i  ] + u[j+1][i  ]) * (v[j][i  ] + v[j][i+1]) -
				              (u[j][i-1] + u[j+1][i-1]) * (v[j][i-1] + v[j][i  ])
		                    )
		          + alpha * ( fabs(u[j][i  ] + u[j+1][i  ]) * (v[j][i  ] - v[j][i+1]) -
				              fabs(u[j][i-1] + u[j+1][i-1]) * (v[j][i-1] - v[j][i  ])
		                    )
		       ) / (dx * 4.0);
	}

	/**
	 * FD_y_UV - first derivative along y of UV,
	 * evaluated at the respective point of our staggered grid (with alpha)
	 */
	inline
	static
	double FD_y_UV (double **u, double **v, std::size_t j, std::size_t i, double dy, double alpha)
	{
		return (            ( (v[j  ][i] + v[j  ][i+1]) * (u[j  ][i] + u[j+1][i]) -
				              (v[j-1][i] + v[j-1][i+1]) * (u[j-1][i] + u[j  ][i])
		                    )
		          + alpha * ( fabs(v[j  ][i] + v[j  ][i+1]) * (u[j  ][i] - u[j+1][i]) -
				              fabs(v[j-1][i] + v[j-1][i+1]) * (u[j-1][i] - u[j  ][i])
		                    )
		       ) / (dy * 4.0);
	}

	/**
	 * Check if a point lies inside any obstacle cell
	 */
	inline
	static
	bool is_point_in_obs_cell(int** M, std::size_t j, std::size_t i)
	{
		return ((M[j][i] > 0) || (M[j+1][i] > 0) || (M[j+1][i+1] > 0) || (M[j][i+1] > 0)) ? true : false;
	}

}; //end of class

#endif /* SIM_NSSIM_HPP_ */
