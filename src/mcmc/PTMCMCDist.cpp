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

#include <mcmc/PTMCMCDist.hpp>

#define DEBUG NO

PTMCMCDist::PTMCMCDist(
		MPIObject* mpi_obj,
		ForwardModel* fm,
		double rate)
		: model(fm),  // Initialize const members
		  param_size(fm->get_param_size()),
		  data_size(fm->get_data_size()),
		  posterior_sigma(fm->compute_posterior_sigma())
{
	param_space_min.reset(new double[param_size]);
	param_space_max.reset(new double[param_size]);
	random_walk_size.reset(new double[param_size]);

	for (int i=0; i < param_size; i++) {
		model->get_param_space(i, param_space_min[i], param_space_max[i]);
		random_walk_size[i] = MCMC_RANDOM_WALK_SIZE *
				(param_space_max[i] - param_space_min[i]);
	}

	mpi = mpi_obj;
	mixing_rate = rate;
	num_temps = mpi->size;

	// Generate inversed temperatures
	// T = 1, squr(2), 2, 2*squr(2), ... increment by squr(2)
	inv_temps.reset(new double[num_temps]);
	for (int i=0; i < num_temps; i++) {
		inv_temps[i] = pow(2.0, double(i)/-2.0);
	}
#if (DEBUG == YES)
	std::cout << num_temps << " temperatures are initialized." << std::endl;
	for (int i=0; i < num_temps; i++) {
		std::cout << inv_temps[i] << " ";
	}
	std::cout << std::endl;
#endif
}

void PTMCMCDist::sample(
		int num_samples,
		const double* m_init,
		double* m_maxpos)
{
	std::unique_ptr<double[]> m (new double[param_size]);
	std::unique_ptr<double[]> d (new double[data_size]);
	double pos;
	double maxpos = 0.0;

	std::unique_ptr<bool[]> swap_iter(new bool[num_samples]);
	std::unique_ptr<int[]>  swap_chain(new int[num_samples]);

    std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> unifrnd(0.0,1.0);

	// Output file, but only MASTER will write to it
	std::string ofile = std::string(OUTPUT_PATH) + "/mcmc.dat";
	std::ofstream fout;

	// A package for exchagne: [0] acc, [1] decision, [2:end-1] current state
	//		[end] current state pos
	std::unique_ptr<double[]> my  (new double[3 + param_size]);
	std::unique_ptr<double[]> nei (new double[3 + param_size]);

	// Initialize starting point...
	for (int i=0; i < param_size; i++) {
		m[i] = m_init[i];
		m_maxpos[i] = m_init[i];
	}
	// Initialize maxpos...
	model->run(&m[0], &d[0]);
	pos = model->compute_posterior(posterior_sigma, &d[0]);
	maxpos = pos;

	if (mpi->rank == MASTER) {
		// open output file for write
		fout.open(ofile.c_str(), std::fstream::out);
		if (!fout.is_open()) {
			std::cout << "Cannot open mcmc output file. Operation aborted!" << std::endl;
			exit(EXIT_FAILURE);
		}
		// To reduce MPI communication, Master pre-determines which iterations
		// should be a "potential exchange iteration", and which chains should be swapping
		for (int i=0; i < num_samples; i++) {
			if (unifrnd(gen) <= mixing_rate) {
				swap_iter[i] = true;
				swap_chain[i] = int(floor(unifrnd(gen)*(num_temps-1)));
			} else {
				swap_iter[i] = false;
				swap_chain[i] = -1;
			}
		}
	}
	MPI_Bcast(&swap_iter[0], num_samples, MPI_CXX_BOOL, MASTER, mpi->comm);
	MPI_Bcast(&swap_chain[0], num_samples, MPI_INT, MASTER, mpi->comm);

	// Draw num_samples samples...
	int dim = 0;
	for (int it=0; it < num_samples; it++) {
		// 1. Perform 1 MCMC step
		dim = it%param_size;
		tempered_metropolis_step_single_dim_update(dim, &m[0], &pos);

		// 2. Check whether this is a PEI
		if ( (swap_iter[it]) && ((mpi->rank == swap_chain[it])
				              || (mpi->rank == swap_chain[it]+1)) ) {

#if (DEBUG == YES)
			std::cout << "Rank " << mpi->rank << " prepares for swapping at iter " << it << std::endl;
#endif
			// Fix neighbor rank
			int nei_rank;
			if (mpi->rank == swap_chain[it])
				nei_rank = mpi->rank + 1; // neighbor on right
			else
				nei_rank = mpi->rank - 1; // neighbor on left

			// Compute my local acceptance ratio -> my[0]
			my[0] = pow(pos, inv_temps[nei_rank])/pow(pos, inv_temps[mpi->rank]);
			my[0] = fmin(1.0, my[0]);

			// Compute my decision -> my[1]
			my[1] = (unifrnd(gen) < my[0]) ? 1.0 : 0.0;

			// Pack my current state and pos -> my[2:end-1], my[end]
			for (int j=0; j < param_size; j++)
				my[j+2] = m[j];
			my[param_size+2] = pos;

			// Exchange with neighbor rank
			if (mpi->rank == swap_chain[it]) {
				MPI_Send(&my[0],  3 + param_size, MPI_DOUBLE, nei_rank,
						10, mpi->comm);
				MPI_Recv(&nei[0], 3 + param_size, MPI_DOUBLE, nei_rank,
						20, mpi->comm, MPI_STATUS_IGNORE);
			} else {
				MPI_Recv(&nei[0], 3 + param_size, MPI_DOUBLE, nei_rank,
						10, mpi->comm, MPI_STATUS_IGNORE);
				MPI_Send(&my[0],  3 + param_size, MPI_DOUBLE, nei_rank,
						20, mpi->comm);
			}
			// Make decision: exchange only if both me and nei accepted
			if ((my[1] == 1.0) && (nei[1] == 1.0)) {
#if (DEBUG == YES)
				std::cout << "Rank " << mpi->rank << " swapping is happening... " << std::endl;
#endif
				for (int j=0; j < param_size; j++)
					m[j] = nei[j+2];
				pos = nei[param_size+2];
			} else {
#if (DEBUG == YES)
				std::cout << "Rank " << mpi->rank << " swapping did not happen. " << std::endl;
#endif
			}
		} // end if(swap_iter)..

		// 3. MASTER write output
		if (mpi->rank == MASTER) {
			for (int i=0; i < param_size; i++)
				fout << m[i] << " ";
			fout << pos << std::endl;
		}

		// 4. Get max posterior and the corresponding sample point
		if (pos > maxpos) {
			maxpos = pos;
			for (int i=0; i < param_size; i++) {
				m_maxpos[i] = m[i];
			}
		}

		// 5. keeping track
		if (((it+1)%5 == 0) && (mpi->rank == MASTER)){
			std::cout << "\n" << "Rank " << mpi->rank << ": "
					<< it+1 << " mcmc steps completed ..." << std::endl;
			std::cout << "Current maximum posterior = " << maxpos <<
					", at " << arr_to_string(param_size, &m_maxpos[0]) << std::endl;
		}
	}
	// MASTER close file
	if (mpi->rank == MASTER) {
		fout.close();
	}
	return;
}


/***********************
 *  Private Methods
 ***********************/

void PTMCMCDist::tempered_metropolis_step_single_dim_update(
		const int& dim,		/// INPUT: the dimension being updated in this step
		double* m,		/// INPUT/OUTPUT: current state, new state
		double* pos)	/// INPUT/OUTPUT: current state posterior, new state posterior
{
	std::unique_ptr<double[]> m_tmp (new double[param_size]);
	std::unique_ptr<double[]> d_tmp (new double[data_size]);
	double pos_tmp = 0.0;

	// Initialize random generators
    std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<double> normrnd(m[dim], random_walk_size[dim]);
	std::uniform_real_distribution<double> unifrnd(0.0,1.0);

	/***********************
	 * 1. Draw a proposal
	 ***********************/
	// Initialize proposal with a single dim update:
	for (int i=0; i < param_size; i++) {
		// other dims are the same as current state
		m_tmp[i] = m[i];
	}
	// at [dim] update with a sample from N(m[dim], random_walk_size)
	m_tmp[dim] = normrnd(gen);
	// Ensure proposal is within range
	while ((m_tmp[dim] < param_space_min[dim]) ||
			(m_tmp[dim] > param_space_max[dim])) {
		m_tmp[dim] = normrnd(gen);
	}

	/*******************************
	 * 2. Compute acceptance rate
	 *******************************/
	model->run(&m_tmp[0], &d_tmp[0]);
	pos_tmp = model->compute_posterior(posterior_sigma, &d_tmp[0]);
	pos_tmp = pow(pos_tmp, inv_temps[mpi->rank]);
	double acc = fmin(1.0, pos_tmp/(*pos));

	/************************
	 * 3. Accept or reject
	 ************************/
	if (unifrnd(gen) <= acc) {
		// accept
		m[dim] = m_tmp[dim];
		*pos = pos_tmp;
	}
	// if reject, do nothing
	return;
}
