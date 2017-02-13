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

#include <mpi/MPIObject.hpp>


MPIObject::MPIObject()
{
	comm = MPI_COMM_NULL;
	size = 0;
	rank = -1;
#if (ENABLE_IMPI == YES)
	status = -1;
#endif
}

void MPIObject::update(MPI_Comm newcomm)
{
	comm = newcomm;
	if (comm == MPI_COMM_NULL) {
		size = 0;
		rank = -1;
	} else {
		MPI_Comm_size(newcomm, &size);
		MPI_Comm_rank(newcomm, &rank);
	}
	// "status" should not be touched
	// It will be updated explicitly when impi adapt occurs
}


