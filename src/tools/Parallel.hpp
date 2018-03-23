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

#ifndef TOOLS_PARALLEL_HPP_
#define TOOLS_PARALLEL_HPP_

#include<mpi.h>

class Parallel {
// These members are public because they are updated directly by mpi functions in other classes
public:
	int mpisize = 1;	// minimum size is 1, no matter what
	int mpirank = 0;
	int mpistatus = -1; //invalid mpi status by default

public:
	// use default constructor
	~Parallel(){}

	inline
	bool is_master() {
#if defined(IMPI)
		return ( (mpirank == 0) &&
				(mpistatus == MPI_ADAPT_STATUS_NEW || mpistatus == MPI_ADAPT_STATUS_STAYING) );
#else
		return (mpirank == 0);
#endif
	}

	void mpi_init(int argc, char* argv[]);

	void mpi_final();

	void mpi_update();
};
#endif /* TOOLS_PARALLEL_HPP_ */
