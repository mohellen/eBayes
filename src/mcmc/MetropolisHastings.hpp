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

#ifndef MCMC_METROPOLISHASTINGS_HPP_
#define MCMC_METROPOLISHASTINGS_HPP_

#include <model/ForwardModel.hpp>
#include <mcmc/MCMC.hpp>
#include <string>
#include <memory>
#include <random>
#include <chrono>
#include <cmath>

class MetropolisHastings : public MCMC
{
public:
	~MetropolisHastings() {}

	MetropolisHastings(
			ForwardModel* forwardmodel,
			const std::string& observed_data_file,
			double rand_walk_size_domain_percent = 0.2)
			: MCMC(forwardmodel, observed_data_file, rand_walk_size_domain_percent) {}

	void run(
			const std::string& output_file,
			int num_samples,
			double& maxpos,
			double* maxpos_point,
			const double* init_sample_pos = nullptr);
};
#endif /* MCMC_METROPOLISHASTINGS_HPP_ */
