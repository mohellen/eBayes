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

#ifndef MCMC_PARALLELTEMPERING_HPP_
#define MCMC_PARALLELTEMPERING_HPP_

#include <model/ForwardModel.hpp>
#include <mcmc/MCMC.hpp>

#include <mpi.h>
#include <cmath>
#include <memory>
#include <string>


class ParallelTempering : public MCMC
{
private:
	int mpi_rank;	/// MPI rank
	int mpi_size;	/// Size of MPI_COMM_WORLD
#if (ENABLE_IMPI==1)
	int mpi_status;	/// iMPI adapt status
	std::size_t impi_gpoffset;//MPI_UNSIGNED_LONG
#endif

	double mixing_rate; /// [0,1]. How often we should mix (exchange) samples
	int num_chains;
	std::unique_ptr<double[]> inv_temps;

public:
	~ParallelTempering() {}

	ParallelTempering(
			ForwardModel* forwardmodel,
			const std::string& observed_data_file,
			double mixing_chain_rate,
			double rand_walk_size_domain_percent = 0.2);

	void run(const std::string& output_file,
				int num_samples,
				double& maxpos,
				double* maxpos_point,
				const double* init_sample_pos = nullptr);

private:
	bool is_master();

};
#endif /* MCMC_PARALLELTEMPERING_HPP_ */
