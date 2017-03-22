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

#ifndef MODEL_FORWARDMODEL_HPP_
#define MODEL_FORWARDMODEL_HPP_

#include <cstddef>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <algorithm>

#define MASTER 	0

#define SGI_OUT_TIMER			1 //(1 or 0)
#define SGI_OUT_RANK_PROGRESS	1 //(1 or 0)
#define SGI_OUT_GRID_POINTS		0 //(1 or 0)
#define SGI_MPIMW_TRUNK_SIZE	10 //(integer)


class ForwardModel
{
protected:
	std::size_t input_size = 0;
	std::size_t output_size = 0;

public:
	// Define virtual destructor
	virtual ~ForwardModel() {}

	ForwardModel() {}

	ForwardModel(std::size_t lin, std::size_t lout)
			: input_size(lin), output_size(lout) {}

	virtual std::size_t get_input_size() = 0;

	virtual std::size_t get_output_size() = 0;

	virtual void get_input_space(
			int dim,
			double& min,
			double& max) = 0;

	virtual void run(const double* m, double* d) = 0;

	static
	double* get_observed_data(
			const std::string & input_file,
			std::size_t data_size,
			double& noise_in_data);

	static
	double compute_posterior_sigma(
			const double* observed_data,
			std::size_t data_size, 
			double noise_in_data);

	static
	double compute_posterior(
			const double* observed_data,
			const double* d,
			std::size_t data_size,
			double sigma);
			
	static
	double compute_l2norm(
			const double* d1, 
			const double* d2,
			std::size_t data_size);

	static
	std::string trim_white_space(const std::string & str);

	static
	std::string arr_to_string(const double* m, std::size_t len);

};
#endif /* MODEL_FORWARDMODEL_HPP_ */
