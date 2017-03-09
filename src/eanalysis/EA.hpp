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

#ifndef EANALYSIS_QA_HPP_
#define EANALYSIS_QA_HPP_

#include "model/ForwardModel.hpp"

#include <memory>
#include <iostream>


class EA {
private:
	std::size_t input_size;
	std::size_t output_size;
	ForwardModel* fm;
	ForwardModel* sm;
	std::unique_ptr<double[]> test_point;
	std::unique_ptr<double[]> fm_data;

public:
	EA(ForwardModel* fullmodel, ForwardModel* surrogatemodel, const double* m);

	void set_test_point(double* m);

	double err();

	double err(const double* m);
};
#endif /* EANALYSIS_QA_HPP_ */
