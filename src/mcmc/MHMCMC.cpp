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

#include <mcmc/MHMCMC.hpp>


MHMCMC::MHMCMC(
		ForwardModel* fm)
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
}

void MHMCMC::sample(
		int num_samples,
		const double* m_init,
		double* m_maxpos)
{
	std::unique_ptr<double[]> m (new double[param_size]);
	std::unique_ptr<double[]> d (new double[data_size]);
	double pos;
	double maxpos = 0.0;

	// Initialize starting point...
	for (int i=0; i < param_size; i++) {
		m[i] = m_init[i];
		m_maxpos[i] = m_init[i];
	}
	// Initialize maxpos...
	model->run(&m[0], &d[0]);
	pos = model->compute_posterior(posterior_sigma, &d[0]);
	maxpos = pos;

	// Prepare output file, Binary mode (if already exist, append)
	std::string ofile = std::string(OUTPUT_PATH) + "/mcmc.dat";
	std::ofstream fout(ofile.c_str(), std::fstream::out);
	if (!fout.is_open()) {
		std::cout << "Cannot open mcmc output file. Operation aborted!" << std::endl;
		exit(EXIT_FAILURE);
	}

	// Draw num_samples samples...
	int dim = 0;
	for (int it=0; it < num_samples; it++) {

		// 1. Perform 1 MCMC step
		dim = it%param_size;
		metropolis_step_single_dim_update(dim, &m[0], &pos);

		// 2. write result
		for (int i=0; i < param_size; i++)
			fout << m[i] << " ";
		fout << pos << std::endl;

		// 3. Get max posterior and the corresponding sample point
		if (pos > maxpos) {
			maxpos = pos;
			for (int i=0; i < param_size; i++) {
				m_maxpos[i] = m[i];
			}
		}

		// 4. keeping track
		if ((it+1)%5 == 0) {
			std::cout << "\n" << it+1 << " mcmc steps completed ..." << std::endl;
			std::cout << "Current maximum posterior = " << maxpos <<
					", at " << arr_to_string(param_size, &m_maxpos[0]) << std::endl;
		}
	}
	fout.close();
	return;
}


/***********************
 *  Private Methods
 ***********************/
void MHMCMC::metropolis_step_single_dim_update(
		const int dim,
		double* m,
		double* pos)
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

