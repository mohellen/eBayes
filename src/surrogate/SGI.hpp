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

#ifndef SURROGATE_SGI_HPP_
#define SURROGATE_SGI_HPP_

#include "model/ForwardModel.hpp"
#include "model/NS.hpp"
#include "sgpp_base.hpp"

#include <mpi.h>
#include <memory>
#include <vector>

class SGI : public ForwardModel
{
private:

#if (ENABLE_MPI==1)
	int mpi_rank;	/// MPI rank
	int mpi_size;	/// Size of MPI_COMM_WORLD
#if (ENABLE_IMPI==1)
	int mpi_status;	/// iMPI adapt status
#endif
#endif
	
	std::unique_ptr<ForwardModel> 			  fullmodel;
	std::unique_ptr<sgpp::base::Grid> 		  grid;
	std::unique_ptr<sgpp::base::DataVector[]> alphas;
	
	std::size_t input_size;
	std::size_t output_size;
	
	std::size_t maxpos_seq;
	double maxpos;


public:
	~SGI(){}
	
	SGI(ForwardModel* fm);

	std::size_t get_input_size();

	std::size_t get_output_size();

	void get_input_space(
			int dim,
			double& min,
			double& max);

	double* run(const double* m);
	
private:

};
#endif /* SURROGATE_SGI_HPP_ */
