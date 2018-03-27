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

#include <mcmc/ParallelTempering.hpp>

using namespace std;


void ParallelTempering::run(
		std::size_t num_samples,
		std::vector<double> const& init_samplepos)
{
	// Cannot perform Parallel Tempering with less than 2 chains
	if (num_chains < 2) {
		cout << tools::red << "ERROR: cannot perform MCMC Parallel Tempering with less than 2 chains. Program abort."
			<< tools::reset << endl;
		exit(EXIT_FAILURE);
	}

	// Each MCMC chain is pinned to a MPI process
	// Ranks with (mpirank > num_chains) do NOT participate in MCMC computation
	if (par.mpirank >= num_chains) return;

	// Output file
	fstream fout = open_output_file();
	
	// Initialize starting point & maxpos point
	std::size_t input_size = cfg.get_input_size();
	vector<double> samplepos = initialize_samplepos(init_samplepos);
	vector<double> max_samplepos = samplepos;
	// Write initial MCMC sample
	write_samplepos(fout, samplepos);

	//=============== Parallel Tempering Stuff =================
	// Random generators
	mt19937 gen(chrono::system_clock::now().time_since_epoch().count());
	uniform_real_distribution<double> udist_r (0.0, 1.0);
	uniform_int_distribution<int> udist_i (0, num_chains-1);
	// Pre-determines exchange iterations and ranks:
	//    To reduce MPI communication, Master pre-determines which iterations (first)
	//    should be a "potential exchange iteration", and which chain (second) should be swapping
	//    The swapping chains are always c and c+1 (if c is the last chain, then c+1 is chain 0)
	vector< pair<int,int> > exchange_iter_chain (num_samples);
	if (par.is_master()) {
		double mixing_rate = stod(cfg.get_param("mcmc_chain_mixing_rate"));
		for (int i=0; i < num_samples; i++) {
			if (udist_r(gen) <= mixing_rate) {
				exchange_iter_chain[i].first = 1;
				exchange_iter_chain[i].second = udist_i(gen);
			} else {
				exchange_iter_chain[i].first = 0;
				exchange_iter_chain[i].second = -1;
			}
		}
	}
	MPI_Bcast(&exchange_iter_chain[0], num_samples, MPI_2INT, MPI_MASTER, MPI_COMM_WORLD);
	// Exchagne buffer: {sample, posterior, decision}
	//      [0,input_size-1] sample
	//		[input_size  ]   posterior
	//		[input_size+1]   decision
	vector<double> rbuf (input_size + 2);

	// Initialize temperatures of each chain
	// For convenience purpose, the inverse of temperatures are stored
	vector<double> inv_temps (num_chains);
	for (int t=0; t < num_chains; t++)
		inv_temps[t] = pow(2.0, double(t)/-2.0);
	//=========================================================

	// Run the MCMC chain
	int dim = 0;
	for (int it=0; it < num_samples; ++it) {
		// 1. Perform 1 MCMC step
		dim = it%input_size;
		one_step_single_dim(dim, samplepos);

		// 1.1 Mixing chains if needed
		if (exchange_iter_chain[it].first == 1) {
			// Engage only the mixing ranks
			int c1 = exchange_iter_chain[it].second;
			int c2 = ((c1 + 1) < num_chains) ? (c1 + 1) : 0;

			if ((par.mpirank == c1) || (par.mpirank == c2)) {
				// neighbor rank
				int nei_chain = (par.mpirank==c1) ? c2 : c1;

				// Compute exchange decision, append it to samplepos
				double acc = fmin(1.0,
						pow(samplepos.back(), inv_temps[nei_chain]) / pow(samplepos.back(), inv_temps[par.mpirank]));
				samplepos.push_back( (udist_r(gen) < acc) ? 1.0 : 0.0 ); // samplepos is temporary input_size+2 length

				// MPI communication
				if (par.mpirank == c1) {
					MPI_Send(&samplepos[0], input_size+2, MPI_DOUBLE, nei_chain,
							10, MPI_COMM_WORLD);
					MPI_Recv(&rbuf[0], input_size+2, MPI_DOUBLE, nei_chain,
							20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				} else {
					MPI_Recv(&rbuf[0], input_size+2, MPI_DOUBLE, nei_chain,
							10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Send(&samplepos[0],  input_size+2, MPI_DOUBLE, nei_chain,
							20, MPI_COMM_WORLD);
				}
				// Exchange only if both me and nei accepted
				if ((samplepos.back() > 0.5) && (rbuf.back() > 0.5)) {
					samplepos = rbuf;
					cout << tools::green << "MCMC: rank " << par.mpirank << " swapped with rank "
							<< nei_chain << " at iteration " << it << "." << tools::reset << endl;
				}
				samplepos.pop_back(); //remove the exchange decision, samplepos go back to input_size+1 length
			}
		}

		// 2. write result
		write_samplepos(fout, samplepos);

		// 3. Get max posterior and the corresponding sample point
		if (samplepos.back() > max_samplepos.back()) {
			max_samplepos = samplepos;
		}
		// 4. keeping track
#if (MCMC_OUT_PROGRESS == 1)
		print_progress(it, max_samplepos);
#endif
	}
	// Insert MAXPOS point to file
	fout << "MAXPOS ";
	write_samplepos(fout, max_samplepos);
	fout.close();
	return;
}

