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

class ForwardModel
{
public:
	// Define virtual destructor
	virtual ~ForwardModel() {}

	ForwardModel() {}

	virtual std::size_t get_input_size() = 0;

	virtual std::size_t get_output_size() = 0;

	virtual void get_input_space(
			int dim,
			double& min,
			double& max) = 0;

	virtual double* run(const double* m) = 0;

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
	double compute_2norm(
			const double* d1, 
			const double* d2,
			std::size_t data_size);

}; //close class

#endif /* MODEL_FORWARDMODEL_HPP_ */
