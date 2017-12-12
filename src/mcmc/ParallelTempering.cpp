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

ParallelTempering::ParallelTempering(
			Parallel& par,
			ForwardModel& model,
			const std::string& observed_data_file,
			double mixing_chain_rate,
			double rand_walk_size_domain_percent)
			: MCMC(par, model, observed_data_file, rand_walk_size_domain_percent)
{
	this->mixing_rate = mixing_chain_rate;
	// Initialize temperatures of each chain
	// For convenience purpose, the inverse of temperatures are stored
	this->inv_temps.reset(new double[num_chains]);
	for (int t=0; t < num_chains; t++)
		inv_temps[t] = pow(2.0, double(t)/-2.0);
}

int ParallelTempering::get_num_chains()
{
	return this->num_chains;
}


void ParallelTempering::run(
		const string& outpath, 	/// Input
		int num_samples,					/// Input
		const vector<vector<double> >& init_sample_pos) /// Optional input
{
	if (par.mpirank >= num_chains) return;

	// Initialize rank specific output file
	string rank_output_file = outpath +
			"mcmcpt_r" + std::to_string(par.mpirank) + ".dat";

	// Initialize rank specific initial sample & pos
	unique_ptr<double[]> rank_sample_pos (
			gen_init_sample(rank_output_file, init_sample_pos));

	double pos, maxpos, acc, dec;
	unique_ptr<double[]> p (new double[input_size]);
	unique_ptr<double[]> d (new double[output_size]);
	unique_ptr<double[]> maxpos_p (new double[input_size]);

	// Open file: append if exists, or create it if not
	fstream fout (rank_output_file, fstream::in | fstream::out | fstream::app);
	if (!fout) {
		printf("Rank %d: MCMC open output file \"%s\" failed. Abort!\n", par.mpirank,
				rank_output_file.c_str());
		exit(EXIT_FAILURE);
	}

	// Initialize starting point
	for (std::size_t i=0; i < input_size; i++) {
		p[i] = rank_sample_pos[i];
	}
	pos = rank_sample_pos[input_size];
	write_sample_pos(fout, p.get(), pos);

	// Initialize maxpos point
	maxpos = pos;
	for (size_t i=0; i < input_size; i++) {
		maxpos_p[i] = p[i];
	}

	// Random generators
	mt19937 gen(chrono::system_clock::now().time_since_epoch().count());
	uniform_real_distribution<double> udist_r (0.0, 1.0);
	uniform_int_distribution<int> udist_i (0, num_chains-1);

	// Pre-determines exchange iterations and ranks:
	//    To reduce MPI communication, Master pre-determines which iterations
	//    should be a "potential exchange iteration", and which ranks should be swapping
	//    The swapping ranks are always r and r+1, if r != last rank, or
	//    r and 0, if r = last rank
	unique_ptr<bool[]> ex_iter (new bool[num_samples]);
	unique_ptr<int[]>  ex_rank (new int[num_samples]);
	if (par.is_master()) {
		for (int i=0; i < num_samples; i++) {
			if (udist_r(gen) <= mixing_rate) {
				ex_iter[i] = true;
				ex_rank[i] = udist_i(gen);
			} else {
				ex_iter[i] = false;
				ex_rank[i] = -1;
			}
		}
	}
	MPI_Bcast(ex_iter.get(), num_samples, MPI_CXX_BOOL, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(ex_rank.get(), num_samples, MPI_INT, MASTER, MPI_COMM_WORLD);

	// Exchagne buffer: {sample, pos, dec}
	//      [0,input_size-1] sample
	//		[input_size  ]   pos
	//		[input_size+1]   dec
	std::unique_ptr<double[]> my  (new double[input_size + 2]);
	std::unique_ptr<double[]> nei (new double[input_size + 2]);

	// Run the MCMC chain
	int dim = 0;
	for (int it=0; it < num_samples; it++) {

		// 1. Perform 1 MCMC step
		dim = it%input_size;
		one_step_single_dim(dim, pos, p.get(), d.get());

		// 1.1 Mixing chains if needed
		if (ex_iter[it]) {
			// Engage only the mixing ranks
			int rank1 = ex_rank[it];
			int rank2 = ((rank1 + 1) < par.mpisize) ? (rank1 + 1) : 0;

			if ((par.mpirank == rank1) || (par.mpirank == rank2)) {
				// neighbor rank
				int nei_rank = (par.mpirank==rank1) ? rank2 : rank1;

				// Pack current state (sample) and pos into exchange buffer
				for (size_t i=0; i < input_size; i++)
					my[i] = p[i];
				my[input_size] = pos;

				// Compute exchange decision @ [input_size+1]
				acc = fmin(1.0, pow(pos, inv_temps[nei_rank])/pow(pos, inv_temps[par.mpirank]));
				dec = (udist_r(gen) < acc) ? 1.0 : 0.0;
				my[input_size+1] = dec;

				// MPI communication
				if (par.mpirank == rank1) {
					MPI_Send(my.get(),  input_size+2, MPI_DOUBLE, nei_rank,
							10, MPI_COMM_WORLD);
					MPI_Recv(nei.get(), input_size+2, MPI_DOUBLE, nei_rank,
							20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				} else {
					MPI_Recv(nei.get(), input_size+2, MPI_DOUBLE, nei_rank,
							10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Send(my.get(),  input_size+2, MPI_DOUBLE, nei_rank,
							20, MPI_COMM_WORLD);
				}
				// Exchange only if both me and nei accepted
				if ((my[input_size+1] == 1.0) && (nei[input_size+1] == 1.0)) {
					printf("Rank %d: swapping with rank %d at iteration %d.\n", par.mpirank, nei_rank, it);
					for (size_t i=0; i < input_size; i++)
						p[i] = nei[i];
					pos = nei[input_size];
				}
			}
		}

		// 2. write result
		write_sample_pos(fout, p.get(), pos);

		// 3. Get max posterior and the corresponding sample point
		if (pos > maxpos) {
			maxpos = pos;
			for (int i=0; i < input_size; i++)
				maxpos_p[i] = p[i];
		}
		// 4. keeping track
#if (MCMC_OUT_PROGRESS == 1)
		if (par.is_master() && ((it+1)%MCMC_OUT_FREQ == 0)) {
			printf("\n%d mcmc steps completed.\n", it+1);
			printf("Current maxpos: %s  %f\n\n", ForwardModel::arr_to_string(p.get(), input_size).c_str(), pos);
		}
#endif
	}
	// Insert MAXPOS point to file
	fout << "MAXPOS ";
	write_sample_pos(fout, maxpos_p.get(), maxpos);

	fout.close();
	return;
}

