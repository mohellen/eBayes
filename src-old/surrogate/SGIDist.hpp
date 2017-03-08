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

#ifndef SURROGATE_SGIDIST_HPP_
#define SURROGATE_SGIDIST_HPP_

#include <config.h>
#include <model/ForwardModel.hpp>
#include <mpi/MPIObject.hpp>
#include <model/NS.hpp>
#include <tools/io.hpp>
#include <sgpp_base.hpp>

#include <mpi.h>
#include <vector>
#include <memory>
#include <cmath>
#include <string>


class SGIDist : public ForwardModel
{
private:
	MPIObject* const mpi;	/// External MPI object
	FullModel* const fullmodel;	/// External full model, is fixed throughout this object's life-time (const pointer)
	const std::size_t param_size;	/// param_size is the same as full model's, and is fixed throughout life-time
	const std::size_t data_size;	/// data_size is the same as full model's, and is fixed throughout life-time

	std::unique_ptr<sgpp::base::Grid> grid;							/// Sparse grid
	std::vector< std::unique_ptr<sgpp::base::DataVector> > alphas;	/// Grid point coefficients
	std::unique_ptr<sgpp::base::OperationEval> op_eval;				/// Eval operation with grid
	std::size_t maxpos_gp_seq;	/// The seq of the grid point at which max posterior occurs
	double maxpos;				/// The max posterior

#if (ENABLE_IMPI == YES)
public:
	struct {
		int phase;
		std::size_t gp_offset;
	} CarryOver;
#endif

public:
	~SGIDist() {}

	SGIDist(
			FullModel* fm_obj,
			MPIObject* mpi_obj);

	std::size_t get_param_size();

	std::size_t get_data_size();

	void get_param_space(
			int dim,
			double& min,
			double& max);

	double compute_posterior_sigma();

	double compute_posterior(const double sigma, const double* d);

	void run(
			const double* m,
			double* d);

	void initialize(
			std::size_t level,
			std::string mpi_scheme="na"); // MPI scheme default to naive

	void refine(
			double portion,
			std::string mpi_scheme="na"); // MPI scheme default to naive

	void get_point_coord(
			std::size_t seq,
			double* m);

	void get_maxpos_point(
			double* m);

#if (ENABLE_IMPI == YES)
	void impi_adapt();
#endif

private:
	sgpp::base::BoundingBox* create_boundingbox();

	void mpi_ior_grid();

	void mpi_iow_grid();

	void mpi_io_data(
			std::string which_type,
			char which_action,
			std::size_t seq_min,
			std::size_t seq_max,
			double* buff);

	void find_maxpos();

	void create_alphas();

	void refine_grid(double portion);

	/*Functions for major compuation*/
	void compute_grid_points(
			std::size_t gp_offset,
			std::string mpi_scheme);

	void compute_range(
			const std::size_t& seq_min,
			const std::size_t& seq_max);

	/*Functions for Naive MPI implementation*/
	void get_local_range(
			const std::size_t& global_min,
			const std::size_t& global_max,
			std::size_t& mymin,
			std::size_t& mymax);

	/*Functions for Master-Minion MPI implementation*/
	void compute_gp_master(
			const std::size_t& gp_offset);

	void compute_gp_minion(
			const std::size_t& gp_offset);

	void jobid_to_range(
			const std::size_t& jobid,
			const std::size_t& seq_offset,
			std::size_t& seq_min,
			std::size_t& seq_max);

	void seed_minions(
			const int& num_jobs,
			int& scnt,
			int* jobs);

	void prepare_minions_for_adapt(
			std::vector<int> & jobs_done,
			int & jobs_per_tic);
};


#endif /* SURROGATE_SGIDIST_HPP_ */
