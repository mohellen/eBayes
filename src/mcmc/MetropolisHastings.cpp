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
		const string& output_file,
		ForwardModel* model,
		int num_samples,
		const double* initial_point)
{
	// Problem dimensions
	std::size_t input_size = model->get_input_size();
	std::size_t output_size = model->get_output_size();

	// Initialize starting point
	unique_ptr<double[]> p (new double[input_size]);
	double pos;

	// Initialization prioirty order:
	// 	 1. initial_point, if this is not available then
	//   2. last sample ponit from output_file, if this is not avail either then
	//   3. generate a random point
	if (initial_point) {
		for (std::size_t i=0; i < input_size; i++)
			p[i] = initial_point[i];
	} else {

		double* ls = get_last_sample(output_file);



	}

	// Open file if exists, or create it
	ofstream fout (output_file, ios::app);
	if (!fout) {
		printf("MCMC Metropolis-Hastings open output file failed. Abort!");
		exit(EXIT_FAILURE);
	}

	// Get last sample from output file


	// If not, create a random

	return;
}


void MetropolisHastings::one_step_single_dim(


)
{
//	std::unique_ptr<double[]> m_tmp (new double[param_size]);
//	std::unique_ptr<double[]> d_tmp (new double[data_size]);
//	double pos_tmp = 0.0;
//
//	// Initialize random generators
//    std::random_device rd;
//	std::mt19937 gen(rd());
//	std::normal_distribution<double> normrnd(m[dim], random_walk_size[dim]);
//	std::uniform_real_distribution<double> unifrnd(0.0,1.0);
//
//	/***********************
//	 * 1. Draw a proposal
//	 ***********************/
//	// Initialize proposal with a single dim update:
//	for (int i=0; i < param_size; i++) {
//		// other dims are the same as current state
//		m_tmp[i] = m[i];
//	}
//	// at [dim] update with a sample from N(m[dim], random_walk_size)
//	m_tmp[dim] = normrnd(gen);
//	// Ensure proposal is within range
//	while ((m_tmp[dim] < param_space_min[dim]) ||
//			(m_tmp[dim] > param_space_max[dim])) {
//		m_tmp[dim] = normrnd(gen);
//	}
//
//	/*******************************
//	 * 2. Compute acceptance rate
//	 *******************************/
//	model->run(&m_tmp[0], &d_tmp[0]);
//	pos_tmp = model->compute_posterior(posterior_sigma, &d_tmp[0]);
//	double acc = fmin(1.0, pos_tmp/(*pos));
//
//	/************************
//	 * 3. Accept or reject
//	 ************************/
//	if (unifrnd(gen) <= acc) {
//		// accept
//		m[dim] = m_tmp[dim];
//		*pos = pos_tmp;
//	}
//	// if reject, do nothing
//	return;
}
