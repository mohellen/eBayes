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

#ifndef MCMC_MHMCMC_HPP_
#define MCMC_MHMCMC_HPP_

#include <config.h>
#include <ForwardModel.hpp>
#include <tools/io.hpp>

#include <cmath>
#include <string>
#include <random>
#include <cstdlib>
#include <utility>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>

class MHMCMC
{
private:
	// Everything related to the ForwardMdoel are constant
	ForwardModel* const model;
	const std::size_t param_size;
	const std::size_t data_size;
	const double posterior_sigma;
	std::unique_ptr<double[]> param_space_min;
	std::unique_ptr<double[]> param_space_max;
	std::unique_ptr<double[]> random_walk_size;

public:
	~MHMCMC() {}

	MHMCMC(ForwardModel* fm);

	void sample(
			int num_samples,
			const double* m_init,
			double* m_maxpos);

private:
	void metropolis_step_single_dim_update(
			const int dim,
			double* m,
			double* pos);

};


#endif /* MCMC_MHMCMC_HPP_ */
