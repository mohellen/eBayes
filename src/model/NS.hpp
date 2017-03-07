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

#ifndef MODEL_NS_HPP_
#define MODEL_NS_HPP_

#include "config.h"
#include "model/ForwardModel.hpp"
#include "Eigen/Sparse"
#include "Eigen/Eigen"

#include <cstdio>
#include <cmath>
#include <utility>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <vector>

#define NS_USE_DIRECT_SOLVER 1 // 1: yes, 0: no

/* Class Definiation */
class NS : public ForwardModel
{
private:
	struct Obstacle {
		double locx;
		double locy;
		double sizex;
		double sizey;

		Obstacle(double lx, double ly, double sx, double sy):
			locx(lx), locy(ly), sizex(sx), sizey(sy)
		{}
	};

	//MASK VALUES [CEWNS]
	static const int FLUID = 0;
	static const int B_E   = 10111;
	static const int B_W   = 11011;
	static const int B_N   = 11101;
	static const int B_S   = 11110;
	static const int B_NE  = 10101;
	static const int B_SE  = 10110;
	static const int B_NW  = 11001;
	static const int B_SW  = 11010;
	static const int B_IN  = 11111;

    //Boundary types
    static const int BOUNDARY_TYPE_INLET    = -10;
    static const int BOUNDARY_TYPE_OUTLET   = -20;
    static const int BOUNDARY_TYPE_NOSLIP   = -30;
    static const int BOUNDARY_TYPE_FREESLIP = -40;

    //Simulation variables (From input file)
    double domain_size_x;		/// Domain size in x-direction
    double domain_size_y;	  	/// Domain size in y-direction
    double initial_velocity_x;	/// Initial velocity in x-direction
    double initial_velocity_y;	/// Initial velocity in y-direction
    double initial_pressure;	/// Initial pressure
    double inlet_velocity_x;	/// Inlet velocity in x-direction
    double inlet_velocity_y;	/// Inlet velocity in y-direction
    double external_force_x;	/// External force in x-direction
    double external_force_y;	/// External force in y-direction
    double re;					/// Reynolds number
    double tau;					/// Safety factor for time step size computation
    double alpha;				/// Upwind differecing factor
    double omega;				/// Pressure related
    int boundary_north;	/// North boundary type
    int boundary_south;	/// South boundary type
    int boundary_east;	/// East boundary type
    int boundary_west;	/// West boundary type

    //Simulation domain resolution
	std::size_t ncx;	/// Number of grid cells in x-direction
	std::size_t ncy;	/// Number of grid cells in y-direction
	double dx;			/// Grid cell size in x-direction
	double dy;			/// Grid cell size in y-direction

	//Inverse problem variables: y = f(x) (From input file)
    std::vector<Obstacle> obs;		/// List of obstacles inside domain
    std::vector<double> out_times;	/// List of output sampling time instances
    std::vector< std::pair<double, double> > out_locs;	/// List of output sampling locations
	std::size_t input_size;			/// size of input parameter x
	std::size_t output_size;		/// size of output parameter y


public:
	/* DEFINE destructor (no ;)*/
	~NS();

	/* Declare constructor */
	NS(std::string input_file, int resx, int resy);

	/* Run simulation with VTK output */
	void sim();

	/* Declare member functions */
	double* run(const double * m);

	std::size_t get_input_size();

	std::size_t get_output_size();

	void get_input_space(int dim, double& min, double& max);

	void get_resolution(std::size_t& nx, std::size_t& ny);

	/* Debug only */
	void print_info();
	void print_mask(int **& M);

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

	static
	std::string trim(const std::string& str, const std::string& whitespace=" \t");


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
