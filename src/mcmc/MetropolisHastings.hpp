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

#ifndef MCMC_METROPOLISHASTINGS_HPP_
#define MCMC_METROPOLISHASTINGS_HPP_

#include <model/ForwardModel.hpp>
#include <mcmc/MCMC.hpp>
#include <string>
#include <memory>

class MetropolisHastings : public MCMC
{
public:
	~MetropolisHastings() {}

	MetropolisHastings(): MCMC() {}

	void run(
			const std::string& output_file,
			ForwardModel* model,
			int num_samples,
			const double* initial_point = nullptr);

private:
	void one_step_single_dim();
};
#endif /* MCMC_METROPOLISHASTINGS_HPP_ */
