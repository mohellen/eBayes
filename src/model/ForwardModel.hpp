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

#ifndef MODEL_FORWARDMODEL_HPP_
#define MODEL_FORWARDMODEL_HPP_

#include <tools/Config.hpp>
#include <cstddef>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <algorithm>


class ForwardModel
{
protected:
	Config const& cfg; //const referenct to config object

public:
	// Define virtual destructor
	virtual ~ForwardModel() {}

	ForwardModel(Config const& c) : cfg(c) {}

	virtual std::pair<double,double> get_input_space(int dim) const = 0;

	virtual std::vector<double> run(
			std::vector<double> const& m,
			bool write_vtk=false) = 0; // By default no VTK output
};
#endif /* MODEL_FORWARDMODEL_HPP_ */
