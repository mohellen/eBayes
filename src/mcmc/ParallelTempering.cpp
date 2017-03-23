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

#include <mcmc/ParallelTempering.hpp>

using namespace std;

ParallelTempering::ParallelTempering(
			ForwardModel* forwardmodel,
			const std::string& observed_data_file,
			double rand_walk_size_domain_percent)
			: MCMC(forwardmodel, observed_data_file, rand_walk_size_domain_percent)
{
	MPI_Comm_rank(MPI_COMM_WORLD, &(this->mpi_rank));
	MPI_Comm_size(MPI_COMM_WORLD, &(this->mpi_size));
#if (ENABLE_IMPI==1)
	mpi_status = -1;
	impi_gpoffset = -1;
#endif

	// Determin how many chains: min(mpi_ranks, MAX_CHAINS)
	this->num_chains = (mpi_size < MCMCPT_MAX_CHAINS) ? mpi_size : MCMCPT_MAX_CHAINS;

	// Initialize temperatures of each chain
	// For convenience purpose, the inverse of temperatures are stored
	this->inv_temps.reset(new double[num_chains]);
	for (int t=0; t < num_chains; t++)
		inv_temps[t] = pow(2.0, double(t)/-2.0);
}


void ParallelTempering::run(const std::string& output_file,
			int num_samples,
			double& maxpos,
			double* maxpos_point,
			const double* init_sample_pos)
{

}
