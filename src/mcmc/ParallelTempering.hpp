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

#ifndef MCMC_PARALLELTEMPERING_HPP_
#define MCMC_PARALLELTEMPERING_HPP_

#include <model/ForwardModel.hpp>
#include <mcmc/MCMC.hpp>
#include <surrogate/SGI.hpp>

#include <mpi.h>
#include <cmath>
#include <memory>
#include <string>
#include <vector>


class ParallelTempering : public MCMC
{
private:
	double mixing_rate; /// [0,1]. How often we should mix (exchange) samples
	std::unique_ptr<double[]> inv_temps;

public:
	~ParallelTempering() {}

	ParallelTempering(
			ForwardModel* forwardmodel,
			const std::string& observed_data_file,
			double mixing_chain_rate,
			double rand_walk_size_domain_percent = 0.2);

	int get_num_chains();

	void run(
			const std::string& outpath,
			int num_samples,
			const std::vector<std::vector<double> >& init_sample_pos = {});
};
#endif /* MCMC_PARALLELTEMPERING_HPP_ */
