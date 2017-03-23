// This file is part of BayeSIFSG - Bayesian Statistical Inference Framework with Sparse Grid
// Copyright (C) 2015-today Ao Mo-Hellenbrand.
//
// SIPFSG is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// SIPFSG is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License.
// If not, see <http://www.gnu.org/licenses/>.

#include <mcmc/MetropolisHastings.hpp>

using namespace std;

void MetropolisHastings::run(
		const string& output_file,		/// Input
		int num_samples,				/// Input
		double& maxpos,					/// Output
		double* maxpos_point,			/// Output
		const double* init_sample_pos)	/// Optional input
{
	// Initialize starting point
	double pos;
	unique_ptr<double[]> p (new double[input_size]);
	unique_ptr<double[]> d (new double[output_size]);

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
	if (init_sample_pos) {
		// 1.
		for (std::size_t i=0; i < input_size; i++) {
			p[i] = init_sample_pos[i];
		}
		pos = init_sample_pos[input_size];
		write_sample_pos(fout, p.get(), pos);

	} else {
		if (!read_last_sample_pos(fout, p.get(), pos)) { //2.
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
		maxpos_point[i] = p[i];
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
				maxpos_point[i] = p[i];
		}
		// 4. keeping track
#if (MCMC_OUT_PROGRESS == 1)
		if ((it+1)%5 == 0) {
			printf("\n%d mcmc steps completed.\n", it+1);
			printf("Current maxpos: %s  %f\n", ForwardModel::arr_to_string(p.get(), input_size).c_str(), pos);
		}
#endif
	}
	fout.close();
	return;
}


void MetropolisHastings::one_step_single_dim(
		int dim,
		double& pos,
		double* p,
		double* d)
{
	double pos_tmp = 0.0;
	double dmin, dmax;
	double p_dim_old = p[dim]; /// When reject a proposal, we need ot restore p[dim];

	// Initialize random generators
	mt19937 gen(chrono::system_clock::now().time_since_epoch().count());
	normal_distribution<double> ndist(p[dim], rand_walk_size[dim]);
	uniform_real_distribution<double> udist(0.0,1.0);

	// 1. Draw a proposal (we only update p[dim])
	p[dim] = ndist(gen);
	model->get_input_space(dim, dmin, dmax);
	while ((p[dim] < dmin) || (p[dim] > dmax)) /// ensure p[dim] is in range
		p[dim] = ndist(gen);

	// 2. Compute acceptance rate
	model->run(p, d);
	pos_tmp = ForwardModel::compute_posterior(observed_data.get(), d, output_size, pos_sigma);
	double acc = fmin(1.0, pos_tmp/pos);

	// 3. Accept or reject proposal
	if (udist(gen) <= acc) {
		// Case accept:
		// p is already updated, only update pos
		pos = pos_tmp;
	} else {
		// Case reject:
		// restore p[dim], pos stays the same
		p[dim] = p_dim_old;
	}
	return;
}
