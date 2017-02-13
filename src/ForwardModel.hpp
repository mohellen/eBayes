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

#ifndef FORWARDMODEL_HPP_
#define FORWARDMODEL_HPP_

#include <cstddef>

class ForwardModel
{
public:
	// Define virtual destructor
	virtual ~ForwardModel() {}

	ForwardModel() {}

	// Declare Pure Virtual method
	virtual std::size_t get_param_size() = 0;

	virtual std::size_t get_data_size() = 0;

	virtual void get_param_space(
			int dim,
			double& min,
			double& max) = 0;

	virtual void run(const double* m, double* d) = 0;

	virtual double compute_posterior_sigma() = 0;

	virtual double compute_posterior(const double sigma, const double* d) = 0;

}; //close class

class FullModel : public ForwardModel
{
public:
	virtual ~FullModel() {}

	FullModel()
		:ForwardModel()
	{}
};


class SurrogateModel : public ForwardModel
{
public:
	virtual ~SurrogateModel() {}

	SurrogateModel()
		:ForwardModel()
	{}
};


#endif /* FORWARDMODEL_HPP_ */
