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

#include <mcmc/MetropolisHastings.hpp>

using namespace std;


void MetropolisHastings::run(
			std::size_t num_samples,
			std::vector<double> const& init_samplepos)
{
	// Each MCMC chain is pinned to a MPI process
	// Ranks with (mpirank > num_chains) do NOT participate in MCMC computation
	if (par.rank >= num_chains) return;

	// Output file
	fstream fout = open_output_file();
	
	// Initialize starting point & maxpos point
	std::size_t input_size = cfg.get_input_size();
	vector<double> samplepos = initialize_samplepos(init_samplepos);
	vector<double> max_samplepos = samplepos;
	// Write initial MCMC sample
	write_samplepos(fout, samplepos);

	// Run the MCMC chain
	int dim = 0;
	for (int it=0; it < num_samples; ++it) {
		// 1. Perform 1 MCMC step
		dim = it%input_size;
		one_step_single_dim(dim, samplepos);

		// 2. write result
		write_samplepos(fout, samplepos);

		// 3. Get max posterior and the corresponding sample point
		if (samplepos.back() > max_samplepos.back()) {
			max_samplepos = samplepos;
		}
		// 4. keeping track
#if (MCMC_PRINT_PROGRESS == 1)
		print_progress(it, max_samplepos);
#endif
	}
	// Insert MAXPOS point to file
	fout << "MAXPOS ";
	write_samplepos(fout, max_samplepos);
	fout.close();
	return;
}



