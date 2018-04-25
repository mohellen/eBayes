// eBayes - Elastic Bayesian Inference Framework with iMPI
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

#include <model/ForwardModel.hpp>
#include <tools/Parallel.hpp>
#include <tools/Config.hpp>
#include <model/NS.hpp>
#include <sgpp_base.hpp>

#include <mpi.h>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <sstream>
//#include <iomanip>

// Local constants
#define MPIMW_TAG_TERMINATE 10
#define MPIMW_TAG_WORK		20
#define MPIMW_TAG_ADAPT		30


class SGI : public ForwardModel
{
public:
	~SGI(){}
	
	SGI(
		Config const& c,
		Parallel & p,
		ForwardModel & m);

	std::vector<double> run(
			std::vector<double> const& m);
	
	void build();

	void duplicate(
			const std::string& gridfile,
			const std::string& datafile,
			const std::string& posfile);

	void impi_adapt();

	std::vector<double> get_nth_maxpos(std::size_t n);

	std::pair<double,double> get_input_space(int dim) const {return fullmodel.get_input_space(dim);}

private:
	//Config const& cfg;			// ForwardModel contains cfg
	Parallel & par;				// Reference to parallel object
	ForwardModel & fullmodel;	// Reference to a FULL forward model

	// Internal sparse grid objects
	// f(x) ~= sum_i ( alpha_i * phi_i (x) )
	std::vector<sgpp::base::DataVector> 		alphas; // list of alphas. vector.size() = output_size, each alpha.size() = num_grid_points)
	// Abstract type cannot be instanciated, must use pointers
	std::unique_ptr<sgpp::base::Grid> 			grid; // Sparse grid, containing grid points (input parameters)
	std::unique_ptr<sgpp::base::OperationEval> 	eval;
	std::unique_ptr<sgpp::base::BoundingBox>	bbox;

	// sorted list of top maxpos grid points (pos + gp_seq), in ascending order (LAST one is the MAX)
	// multimap because the key value (posterior) is not unique
	std::multimap<double, std::size_t> maxpos_list;

#if (IMPI==1)
	std::size_t impi_gpoffset = 0; //MPI_SIZE_T
#endif

private:
	std::vector<double> get_gp_coord(std::size_t seq);

	sgpp::base::BoundingBox* create_boundingbox();

	void compute_hier_alphas(const std::string& outfile="");

	void compute_grid_points(
			std::size_t gp_offset,
			bool is_masterworker);

	void compute_gp_range(
			const std::size_t& seq_min,
			const std::size_t& seq_max);

	bool refine_grid(double portion_to_refine);

	void mpiio_read_grid(
			const std::string& outfile="");

	void mpiio_write_grid(
			const std::string& outfile="");

	void mpiio_readwrite_data(
			bool is_read,
			std::size_t seq_min,
			std::size_t seq_max,
			double* buff,
			const std::string& outfile="");

	void mpiio_readwrite_posterior(
			bool is_read,
			std::size_t seq_min,
			std::size_t seq_max,
			double* buff,
			const std::string& outfile="");

	void mpina_get_local_range(
			const std::size_t& gmin,
			const std::size_t& gmax,
			std::size_t& lmin,
			std::size_t& lmax);

	void mpina_find_global_maxpos();

	void mpimw_master_compute(std::size_t gp_offset);

	void mpimw_worker_compute(std::size_t gp_offset);

	void mpimw_exchange_maxpos(int worker_rank);

	void mpimw_sync_maxpos();	

	void mpimw_get_job_range(
			const std::size_t& jobid,
			const std::size_t& seq_offset,
			std::size_t& seq_min,
			std::size_t& seq_max);

	void mpimw_seed_workers(
			const int& num_jobs,
			int& scnt,
			int* jobs);

	void mpimw_adapt_preparation(
			std::vector<int> & jobs_done,
			int & jobs_per_tic);

};
#endif /* SURROGATE_SGI_HPP_ */
