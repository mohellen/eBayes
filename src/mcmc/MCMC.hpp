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

#ifndef MCMC_MCMC_HPP_
#define MCMC_MCMC_HPP_

#include <model/ForwardModel.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iterator>
#include <random>
#include <chrono>

/******************************************
 * MCMC solver writes output data into a text file.
 * 	 - Data type: double
 * 	 - Each line consists of a sample_point & its posterior
 * 	 - A sample_point is double[input_size]
 * 	 - Therfore, each line has (input_size + 1) number of data
 ******************************************/

class MCMC {
protected:
	ForwardModel* model;
	std::size_t input_size;
	std::size_t output_size;

	std::unique_ptr<double[]> rand_walk_size;
	std::unique_ptr<double[]> observed_data;
	double observed_data_noise;
	double pos_sigma;

public:
	virtual ~MCMC() {}

	MCMC(ForwardModel* forwardmodel,
			const std::string& observed_data_file,
			double rand_walk_size_domain_percent);

	virtual void run(
			const std::string& output_file,
			int num_samples,
			double& maxpos,
			double* maxpos_point,
			const double* init_sample_pos = nullptr) = 0;

protected:
	double* gen_random_sample();

	bool read_last_sample_pos(
			std::fstream& fin,
			double* point,
			double& pos);

	void write_sample_pos(
			std::fstream& fout,
			const double* point,
			double pos);

	void one_step_single_dim(
			int dim,
			double& pos,
			double* p,
			double* d);
};
#endif /* MCMC_MCMC_HPP_ */
