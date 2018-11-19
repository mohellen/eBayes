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

#include <mpi.h>
#include <climits>
#include <cstdint>
#include <string>
#include <fstream>
#include <memory>
#include <stddef.h>

#include <tools/Config.hpp>

// size_t is used everywhere, important to define size_t for MPI
#if (SIZE_MAX == UCHAR_MAX)
	#define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif (SIZE_MAX == USHRT_MAX)
	#define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif (SIZE_MAX == UINT_MAX)
	#define MPI_SIZE_T MPI_UNSIGNED
#elif (SIZE_MAX == ULONG_MAX)
	#define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif (SIZE_MAX == ULLONG_MAX)
	#define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
	#error "What is going on?"
#endif


class Parallel {
// These members are public because they are updated directly by mpi functions in other classes
public:
	int size = 1;	// minimum size is 1, no matter what
	int rank = 0;
	int status = -1; //invalid mpi status by default

	MPI_Datatype MPI_SEQPOS;

	const int master = 0;

public:
	// use default constructor
	~Parallel(){}

	inline
	bool is_master()
	{
#if (IMPI==1)
		return ( (rank == master) && (status == MPI_ADAPT_STATUS_NEW || status == MPI_ADAPT_STATUS_STAYING) );
#else
		return (rank == master);
#endif
	}

	void mpi_init(int argc, char** argv);

	void mpi_final();

	void mpi_update();

	void info();

private:
	MPI_Datatype create_MPI_SEQPOS();
};
#endif /* TOOLS_PARALLEL_HPP_ */


