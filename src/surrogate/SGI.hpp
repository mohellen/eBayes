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

#define JOBTODO		't'
#define JOBDONE		'd'
#define JOBINPROG	'p'

#define RANKACTIVE	'a'
#define RANKIDLE	'i'

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

//	void duplicate(
//			const std::string& gridfile,
//			const std::string& datafile,
//			const std::string& posfile);

	std::vector<double> get_maxpos();

	// Need to implement this virtual function from parent class
	std::pair<double,double> get_input_space(int dim) const {return fullmodel.get_input_space(dim);}

private:
	//Config const& cfg;		// ForwardModel contains cfg
	Parallel & par;				// Reference to parallel object
	ForwardModel & fullmodel;	// Reference to a FULL forward model

	// Internal sparse grid objects
	// f(x) ~= sum_i ( alpha_i * phi_i (x) )
	std::vector<sgpp::base::DataVector> 		alphas; // list of alphas. vector.size() = output_size, each alpha.size() = num_grid_points)
	// Abstract type cannot be instanciated, must use pointers
	std::unique_ptr<sgpp::base::Grid> 			grid; // Sparse grid, containing grid points (input parameters)
	std::unique_ptr<sgpp::base::OperationEval> 	eval;
	std::unique_ptr<sgpp::base::BoundingBox>	bbox;

	// maxpos grid point (gp_seq + maspos)
	std::pair<std::size_t, double> seq_maxpos;
	std::size_t impi_gpoffset = 0; //MPI_SIZE_T

private:
	void resume();

	void impi_adapt();

	std::vector<double> get_gp_coord(std::size_t seq);

	double get_gp_volume(std::size_t seq);

	sgpp::base::BoundingBox* create_boundingbox();

	void compute_hier_alphas();

	void compute_grid_points(
			std::size_t gp_offset,
			bool is_masterworker);

	void compute_gp_range(
			const std::size_t& seq_min,
			const std::size_t& seq_max);

	// SPMD refine strategy: all do its own refine
	// BUG!!! the resulting grids are not identical
	void refine_grid_all(double portion_to_refine);

	// Single refine strategy: MASTER refine adn write grid, others read grid
	void refine_grid_mpiio(double portion_to_refine);

	// Single refine strategy: MASTER refine then bcast
	void refine_grid_bcast(double portion_to_refine);

	void mpiio_read_grid();

	void mpiio_write_grid();

	void mpiio_readwrite_data(
			bool is_read,
			std::size_t seq_min,
			std::size_t seq_max,
			double* buff);

	void mpiio_readwrite_pos(
			bool is_read,
			std::size_t seq_min,
			std::size_t seq_max,
			double* buff);

	void mpispmd_get_local_range(
			const std::size_t& gmin,
			const std::size_t& gmax,
			std::size_t& lmin,
			std::size_t& lmax);

	void mpispmd_find_global_maxpos();

	void mpimw_master_compute(std::size_t gp_offset);

	void mpimw_worker_compute(std::size_t gp_offset);

	void mpimw_sync_maxpos();

	void mpimw_get_job_range(
			const std::size_t& jobid,
			const std::size_t& seq_offset,
			std::size_t& seq_min,
			std::size_t& seq_max);

	void mpimw_master_seed_workers(
			std::vector<char>& jobs,
			std::vector<char>& workers);

	void mpimw_master_prepare_adapt(
			std::vector<char>& jobs,
			std::vector<char>& workers,
			int& jobs_per_tic);

	void mpimw_master_send_todo(
			std::vector<char>& jobs,
			std::vector<char>& workers);

	void mpimw_master_recv_done(
			std::vector<char>& jobs,
			std::vector<char>& workers);

	void mpimw_worker_send_done(int jobid);

	void mpimw_master_bcast_maxpos();

	void bcast_grid(MPI_Comm comm);

	void restore_grid(std::string sg_str);

	// For debug only
	void print_workers(std::vector<char> const& workers);
	void print_jobs(std::vector<char> const& jobs);
	bool verify_grid_from_read(int joinrank, MPI_Comm intercomm);
	bool verify_grid(MPI_Comm comm, int src_rank, int dest_rank);
};
#endif /* SURROGATE_SGI_HPP_ */
