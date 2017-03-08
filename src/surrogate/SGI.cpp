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

#include "surrogate/SGI.hpp"

using namespace std;
using namespace	sgpp::base;


SGI::SGI(ForwardModel* fm)
{
#if (ENABLE_MPI==1)
	mpi_rank = -1;
	mpi_size = -1;
#if (ENABLE_IMPI==1)
	mpi_status = -1;
#endif
#endif

	this->input_size = fm->get_input_size();
	this->output_size = fm->get_output_size();
	this->maxpos_seq = 0;
	this->maxpos = 0.0;

	this->fullmodel = unique_ptr<ForwardModel>(fm);
	this->grid = nullptr;
	this->alphas = unique_ptr<DataVector[]>(new DataVector[output_size]);
}

std::size_t SGI::get_input_size()
{
	return this->input_size;
}

std::size_t SGI::get_output_size()
{
	return this->output_size;
}

void SGI::get_input_space(
			int dim,
			double& min,
			double& max)
{
	fullmodel->get_input_space(dim, min, max);
}

double* SGI::run(const double* m)
{
	double* d = new double[output_size];


	printf("\n I am awesome!! \n");

	return d;
}
