// This file is part of BayeSIFSG - Bayesian Statistical Inference Framework with Sparse Grid
// Copyright (C) 2015-today Ao Mo-Hellenbrand.
//
// SIPFSG is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// SIPFSG is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License.
// If not, see <http://www.gnu.org/licenses/>.

#ifndef MODEL_NS_HPP_
#define MODEL_NS_HPP_

#include <config.h>
#include <ForwardModel.hpp>
#include <Eigen/Sparse>
#include <Eigen/Eigen>

#include <cstdio>
#include <cmath>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>


/* Class Definiation */
class NS : public FullModel
{
private:
	// Simulation Setup
	static constexpr double DOMAIN_SIZE_X = 10.0;
	static constexpr double DOMAIN_SIZE_Y = 2.0;
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
	static const std::size_t BOUNDARY_LEFT = 1;
	static const std::size_t BOUNDARY_RIGHT = 2;
	static const std::size_t BOUNDARY_TOP = 3;
	static const std::size_t BOUNDARY_BOTTOM = 3;

	// Domain setup
	static const std::size_t NUM_OBS = NS_NUM_OBS;
	static constexpr double OBS_XMIN[] = { 1.0, 3.0, 5.5, 8.2 };
	static constexpr double OBS_XMAX[] = { 1.4, 3.4, 5.9, 8.6 };
	static constexpr double OBS_YMIN[] = { 0.8, 1.5, 0.2, 1.0 };
	static constexpr double OBS_YMAX[] = { 1.2, 1.9, 0.6, 1.4 };

	// Bayesian Inference Setup
	static constexpr std::size_t NUM_SAMPLING_TIMES = 4;
	static constexpr std::size_t NUM_SAMPLING_LOCATIONS = 10;
	static constexpr std::size_t PARAM_SIZE = NS_NUM_OBS * 2;
	static constexpr std::size_t DATA_SIZE = NUM_SAMPLING_TIMES * NUM_SAMPLING_LOCATIONS;
	static constexpr double NOISE_IN_DATA = 0.2;
	static constexpr double SAMPLING_TIMES[] = { 2.5, 5.0, 7.5, 10.0 };
	static constexpr double SAMPLING_LOCATIONS_X[] = {
			1.5, 3.1, 4.7, 6.3, 7.9, 1.5, 3.1, 4.7, 6.3, 7.9 };
	static constexpr double SAMPLING_LOCATIONS_Y[] = {
			0.6, 0.6, 0.6, 0.6, 0.6, 1.3, 1.3, 1.3, 1.3, 1.3 };

#if (NS_NUM_OBS == 1)
	static constexpr double OBSERVED_DATA[] = {
			1.550702, 1.381798, 1.169803, 1.284388, 1.212167, 0.937018, 1.208724, 1.232352, 1.200904, 1.122890,
			1.485995, 1.400837, 1.450246, 1.272755, 1.335284, 0.893536, 1.216029, 1.345854, 1.356359, 1.350194,
			1.575831, 1.355027, 1.447311, 1.317794, 1.345888, 0.844182, 1.316034, 1.246116, 1.341811, 1.214408,
			1.603132, 1.338208, 1.353566, 1.207331, 1.384773, 0.871862, 1.263172, 1.183300, 1.317644, 1.296239
	};
#elif (NS_NUM_OBS == 2)
	static constexpr double OBSERVED_DATA[] = {
			1.439478, 1.427827, 1.677685, 1.137933, 1.087015, 0.941399, 1.685526, 1.436638, 1.121427, 1.270909,
			1.509609, 1.405698, 1.553082, 1.371944, 1.306100, 1.065633, 1.766364, 1.148386, 1.349967, 1.350273,
			1.448787, 1.455805, 1.615799, 1.619908, 1.255048, 0.932650, 1.617136, 1.087480, 1.287901, 1.398660,
			1.622786, 1.518387, 1.651682, 1.579377, 1.259158, 1.050172, 1.730922, 1.016449, 1.348416, 1.333852
	};
#elif (NS_NUM_OBS == 3)
	static constexpr double OBSERVED_DATA[] = {
			1.452448, 1.462823, 1.488205, 1.308171, 1.251706, 1.058352, 1.590764, 1.322284, 1.218652, 0.835348,
			1.590159, 1.435235, 1.756499, 1.458708, 1.368215, 0.994503, 1.715847, 1.206932, 1.352263, 0.868761,
			1.600331, 1.516845, 1.626249, 1.670804, 1.509140, 1.111133, 1.883070, 1.169152, 1.357879, 1.062469,
			1.526908, 1.319665, 1.635458, 1.586238, 1.459840, 1.011157, 1.631830, 1.037050, 1.188381, 0.983786
	};
#elif (NS_NUM_OBS == 4)
	static constexpr double OBSERVED_DATA[] = {
			1.434041, 1.375464, 1.402000, 0.234050, 1.387931, 1.006520, 1.850871, 1.545131, 1.563303, 0.973778,
			1.512808, 1.387468, 1.608557, 0.141381, 1.313631, 0.990608, 1.741001, 1.551365, 1.789867, 1.170761,
			1.597586, 1.509048, 1.549320, 0.135403, 1.191323, 1.015913, 1.682937, 1.592488, 1.743632, 1.296677,
			1.535493, 1.341702, 1.541945, 0.137985, 1.272473, 1.041918, 1.824279, 1.690430, 1.810520, 1.358992
	};
#endif

	//MASK VALUES [C E W N S]
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
	~NS() {}

	/* Declare constructor */
	NS(const std::size_t num_cells_x, const std::size_t num_cells_y);

	/* Declare member functions */
	void run(const double* m, double* d);

	std::size_t get_param_size();

	std::size_t get_data_size();

	void get_param_space(int dim, double& min, double& max);

	double compute_posterior_sigma();

	double compute_posterior(const double sigma, const double* d);

	void get_resolution(std::size_t& nx, std::size_t& ny);

private:

	/*****************************************
	 * Internal core functions
	 *****************************************/

	/**
	 * Create a mask array to distinguish fluid and different types of obstacle cells
	 */
	int** create_geometry_mask(const double* m);

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
	 * Update boundaries of U,V: BOTTOM (no-slip), TOP (no-slip),
	 *                           LEFT (in-flow), RIGHT (out-flow)
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
	double laplacian (double **m, std::size_t j, std::size_t i, double dx, double dy)
	{
		return (m[j][i+1] - 2.0*m[j][i] + m[j][i-1]) / (dx*dx) +
			   (m[j+1][i] - 2.0*m[j][i] + m[j-1][i]) / (dy*dy);
	}

	/**
	 * FD_x_U2 - first derivative along x of U (velocity along x-direction) squared, with alpha
	 */
	inline
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
	bool is_point_in_obs_cell(int** M, std::size_t j, std::size_t i)
	{
		return ((M[j][i] > 0) || (M[j+1][i] > 0) || (M[j+1][i+1] > 0) || (M[j][i+1] > 0)) ? true : false;
	}


	/*****************************************
	 * Matrix related helper methods
	 *****************************************/

	template<typename T>
	T** alloc_matrix(std::size_t nrows, std::size_t ncols, bool is_row_major)
	{
		T** m;
		T* elems = new T[nrows * ncols];

		if (is_row_major) {
			m = new T*[nrows];
			for (std::size_t i=0; i < nrows; i++) {
				m[i] = &elems[i*ncols];
			}
		} else {
			m = new T*[ncols];
			for (std::size_t i=0; i < ncols; i++) {
				m[i] = &elems[i*nrows];
			}
		}
		return m;
	}

	template<typename T>
	void free_matrix(T** m)
	{
		delete[] m[0];
		delete[] m;
		return;
	}

	template<typename T>
	void init_matrix(T** m, std::size_t nrows, std::size_t ncols, bool is_row_major, T value)
	{
		if (is_row_major) {
			for (std::size_t r=0; r < nrows; r++)
				for (std::size_t c=0; c < ncols; c++)
					m[r][c] = value;
		} else {
			for (std::size_t c=0; c < ncols; c++)
				for (std::size_t r=0; r < nrows; r++)
					m[c][r] = value;
		}
		return;
	}

	template<typename T>
	void print_matrix(T** m, std::size_t nrows, std::size_t ncols, bool is_row_major)
	{
		if (is_row_major) {
			for (std::size_t r=0; r < nrows; r++) {
				for (std::size_t c=0; c < ncols; c++) {
					std::cout << m[nrows-1-r][c] << " ";
				}
				std::cout << std::endl;
			}
		} else {
			for (std::size_t r=0; r < nrows; r++) {
				for (std::size_t c=0; c < ncols; c++) {
					std::cout << m[c][nrows-1-r] << " ";
				}
				std::cout << std::endl;
			}
		}
		return;
	}

}; //end of class

#endif /* MODEL_NS_HPP_ */
