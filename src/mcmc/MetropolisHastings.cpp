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

#include <mcmc/MetropolisHastings.hpp>

using namespace std;

void MetropolisHastings::run(
		const std::string& output_file, /// Input
		int num_samples,				/// Input
		const vector<vector<double> >& init_sample_pos) /// Optional input
{
	// Initialize starting point
	double pos, maxpos;
	unique_ptr<double[]> p (new double[input_size]);
	unique_ptr<double[]> d (new double[output_size]);
	unique_ptr<double[]> maxpos_p (new double[input_size]);

	// Open file: append if exists, or create it if not
	fstream fout (output_file, fstream::in | fstream::out | fstream::app);
	if (!fout) {
		printf("MCMC open output file \"%s\" failed. Abort!\n", output_file.c_str());
		exit(EXIT_FAILURE);
	}

	// Initialization prioirty order:
	// 	 1. initial_point, if this is not available then
	//   2. last sample ponit from output_file, if this is not avail either then
	//   3. generate a random point
	if (!init_sample_pos.empty()) {
		// 1.
		for (std::size_t i=0; i < input_size; i++) {
			p[i] = init_sample_pos[0][i];
		}
		pos = init_sample_pos[0][input_size];
		write_sample_pos(fout, p.get(), pos);

	} else {
		if (!read_maxpos_sample(fout, p.get(), pos)) { //2.
			// 3.
			p.reset(gen_random_sample());
			model->run(p.get(), d.get());
			pos = ForwardModel::compute_posterior(observed_data.get(), d.get(), output_size, pos_sigma);
			write_sample_pos(fout, p.get(), pos);
		}
	}
	/// Initialize maxpos point
	maxpos = pos;
	for (size_t i=0; i < input_size; i++) {
		maxpos_p[i] = p[i];
	}

	// Run the MCMC chain
	int dim = 0;
	for (int it=0; it < num_samples; it++) {

		// 1. Perform 1 MCMC step
		dim = it%input_size;
		one_step_single_dim(dim, pos, p.get(), d.get());

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
		if ((it+1)%5 == 0) {
			printf("\n%d mcmc steps completed.\n", it+1);
			printf("Current maxpos: %s  %f\n", ForwardModel::arr_to_string(p.get(), input_size).c_str(), pos);
		}
#endif
	}
	// Insert MAXPOS point to file
	fout << "MAXPOS ";
	write_sample_pos(fout, maxpos_p.get(), maxpos);

	fout.close();
	return;
}



