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

#ifndef MCMC_PTMCMCDIST_HPP_
#define MCMC_PTMCMCDIST_HPP_

#include <config.h>
#include <ForwardModel.hpp>
#include <mpi/MPIObject.hpp>
#include <tools/io.hpp>

#include <mpi.h>
#include <memory>
#include <cmath>
#include <string>
#include <random>
#include <cstdlib>
#include <utility>
#include <fstream>
#include <iostream>


/**
 * Parallel tempering MCMC sampler using MPI:
 *
 * This class creates number of chains equals to number of ranks contained in
 * the MPI object, with a temperatures starting at 1 and increments with
 * deltaT = sqrt(2). The inverse of temperatures are stored, for convenience.
 */
class PTMCMCDist
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

	MPIObject* mpi;		/// External object: MPI object
	double mixing_rate;	/// Probability of an iteration to be a "Potential exchange iteration" (or how often to exchange)
	int num_temps;		/// Number of temperatures
	std::unique_ptr<double[]> inv_temps; /// Array of inversed temperatures

public:
	~PTMCMCDist() {}

	PTMCMCDist(
			MPIObject* mpi_obj,
			ForwardModel* fm,
			double rate);

	void sample(
			int num_samples,
			const double* m_init,
			double* m_maxpos);

private:
	void tempered_metropolis_step_single_dim_update(
			const int& dim,
			double* m,
			double* pos);
};

#endif /* MCMC_PTMCMCDIST_HPP_ */
