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

#ifndef MPI_MPIOBJECT_HPP_
#define MPI_MPIOBJECT_HPP_

#include <config.h>
#include <mpi.h>

class MPIObject
{
public:
	// All member variabls should be pulic for easy access from othe classes
	MPI_Comm comm;
	int size;
	int rank;
#if (ENABLE_IMPI == YES)
	int status;
#endif

public:
	~MPIObject() {}

	MPIObject();

	// update the whole object with a new comm
	void update(MPI_Comm newcomm);
};

#endif /* MPI_MPIOBJECT_HPP_ */

