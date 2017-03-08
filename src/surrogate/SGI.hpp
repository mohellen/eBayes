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

#ifndef SURROGATE_SGI_HPP_
#define SURROGATE_SGI_HPP_

#include "model/ForwardModel.hpp"
#include "model/NS.hpp"
#include "sgpp_base.hpp"

#include <mpi.h>
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

class SGI : public ForwardModel
{
private:
	int mpi_rank;	/// MPI rank
	int mpi_size;	/// Size of MPI_COMM_WORLD
#if (ENABLE_IMPI==1)
	int mpi_status;	/// iMPI adapt status
#endif
	
	std::size_t input_size;
	std::size_t output_size;

	std::unique_ptr<ForwardModel> 			  	fullmodel;
	std::unique_ptr<sgpp::base::Grid> 		  	grid;
	std::unique_ptr<sgpp::base::DataVector[]> 	alphas;
	std::unique_ptr<sgpp::base::OperationEval> 	eval;
	
	std::unique_ptr<double[]> odata;
	double noise;
	double sigma;
	double maxpos;
	std::size_t maxpos_seq;
	
public:
	~SGI(){}
	
	SGI(ForwardModel* fm, const std::string& observed_data_file);

	std::size_t get_input_size();

	std::size_t get_output_size();

	void get_input_space(
			int dim,
			double& min,
			double& max);

	void run(const double* m, double* d);
	
	void initialize(
			std::size_t level,
			bool is_masterworker=false); // MPI scheme default to naive


private:
	sgpp::base::DataVector arr_to_vec(const double *& in, std::size_t size);

	double* vec_to_arr(sgpp::base::DataVector& in);

	double* seg_to_coord_arr(std::size_t seq);

	std::string vec_to_str(sgpp::base::DataVector& v);

	sgpp::base::BoundingBox* create_boundingbox();

	void create_alphas();

	void mpiio_read_full_grid();

	void mpiio_write_full_grid();

	void mpiio_partial_data(
			bool is_read,
			std::size_t seq_min,
			std::size_t seq_max,
			double* buff);

	void mpiio_partial_posterior(
			bool is_read,
			std::size_t seq_min,
			std::size_t seq_max,
			double* buff);

	void mpi_find_global_update_maxpos();

	void compute_grid_points(
			std::size_t gp_offset,
			bool is_masterworker);

	void mpina_get_local_range(
			const std::size_t& gmin,
			const std::size_t& gmax,
			std::size_t& lmin,
			std::size_t& lmax);

	void mpina_compute_range(
			const std::size_t& seq_min,
			const std::size_t& seq_max);



};
#endif /* SURROGATE_SGI_HPP_ */
