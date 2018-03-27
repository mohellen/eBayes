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

#ifndef MCMC_MCMC_HPP_
#define MCMC_MCMC_HPP_

#include <model/ForwardModel.hpp>
#include <tools/Parallel.hpp>
#include <tools/Config.hpp>

#include <mpi.h>
#include <vector>
#include <string>
#include <memory>
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
	Config const& cfg;		// Const reference to config object
	Parallel & par;			// Reference to parallel object
	ForwardModel & model;	// Reference to foward model object
	// Number of parallel MCMC chains = min(mpisize, max_chains).
	// Only ranks with (mpirank < num_chains) participate in MCMC computation, others idle
	std::size_t num_chains;

public:
	virtual ~MCMC() {}

	MCMC(	
			Config const& c,
			Parallel & p,
			ForwardModel & m);

	virtual void run(
			std::size_t num_samples,
			std::vector<double> const& init_samplepos = std::vector<double>()) = 0;

protected:
	void one_step_single_dim(
			std::size_t dim,
			std::vector<double> & samplepos);

	std::vector<double> read_max_samplepos(std::fstream& fin);

	std::fstream open_output_file();

	void write_samplepos(
			std::fstream& fout,
			std::vector<double> const& samplepos);

	std::vector<double> initialize_samplepos(
			std::vector<double> const& init_samplepos = std::vector<double>()); // optional argument

	void print_progress(int iter, std::vector<double> const& max_samplepos);
};
#endif /* MCMC_MCMC_HPP_ */
