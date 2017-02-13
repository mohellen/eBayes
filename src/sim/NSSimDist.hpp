// This file is part of BayeSIFSG - Bayesian Statistical Inference Framework with Sparse Grid
// Copyright (C) 2015-today Ao Mo-Hellenbrand.
//
// SIPFSG is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// BayeSIFSG is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License.
// If not, see <http://www.gnu.org/licenses/>.


#ifndef SIM_NSSIMDIST_HPP_
#define SIM_NSSIMDIST_HPP_

#include <config.h>
#include <mpi/MPIObject.hpp>
#include <sim/NSSim.hpp>
#include <sim/DomainDecomposer2D.hpp>
#include <tools/matrix.hpp>
#include <memory>
#include <iostream>
#include <fstream>


class NSSimDist
{
private:
	MPIObject* mpi;	/// Link to external MPI object
	std::unique_ptr<DomainBlock> block;	/// Local domain block

	std::size_t gncx; /// Global resolution
	std::size_t gncy;
	std::size_t lncx; /// Local resolution
	std::size_t lncy;
	double dx;	/// cell size in x-direction
	double dy;	/// cell size in x-direction

#if (ENABLE_IMPI == YES)
public:
	struct {
		int** M;
		double** U;
		double** V;
		double** P;
		double** F;
		double** G;
		double t;
		double tic;
		int vtk_cnt;
	} CarryOver;
#endif

public:
	NSSimDist(
			std::size_t num_cells_x,
			std::size_t num_cells_y,
			MPIObject* mpi_obj);

	~NSSimDist() {}

	void run();

#if (ENABLE_IMPI == YES)
	void impi_adapt();
#endif

private:
	// convenient inline: convert local index to global index
	inline
	int yl2g(int j, DomainBlock* b)
	{
		return j + b->ymin - 1;
	}

	inline
	int xl2g(int i, DomainBlock* b)
	{
		return i + b->xmin - 1;
	}

	int** create_geometry_mask();

	void update_boundaries_uv(
			double** &U,
			double** &V);

	void update_boundaries_fg(
			double**& U,
			double**& V,
			double**& F,
			double**& G);

	void update_boundaries_p(
			double**& P);

	void update_domain_uv(
			int** &M,
			double** &U,
			double** &V);

	void update_domain_fg(
			int**&    M,
			double**& U,
			double**& V,
			double**& F,
			double**& G);

	void update_domain_p(
			int**&    M,
			double**& P);

	void compute_dt_mpi(
			double**& U,
			double**& V,
			double&  dt);

	void compute_fg(
			double&  dt,
			double**& U,
			double**& V,
			double**& F,
			double**& G);

	void compute_uv(
			double&  dt,
			double**& F,
			double**& G,
			double**& P,
			double**& U,
			double**& V);

	void solve_for_p_sor_mpi(
			double& dt,
			double**& U,
			double**& V,
			double**& F,
			double**& G,
			double**& P);

	void exchange_ghost_layers_mpi(
			double**& m1);

	void exchange_ghost_layers_mpi(
			double**& m1,
			double**& m2);

	void exchange_ghost_layers_mpi(
			double**& m1,
			double**& m2,
			double**& m3,
			double**& m4,
			double**& m5);

	void write_vtk_dist(
			const char *szProblem,
			int timeStepNumber,
			int**&    M,
			double**& U,
			double**& V,
			double**& P);

	/*********************************
	 *******    iMPI Stuff     *******
	 *********************************/

	double** gather_global_matrix_mpi(double**& m);

	int** gather_global_matrix_mpi(int**& m);

	void update_redistribute_domain_mpi(
			MPI_Comm newcomm,
			int**& M,
			double**& U,
			double**& V,
			double**& P,
			double**& F,
			double**& G);

};
#endif /* SIM_NSSIMDIST_HPP_ */
