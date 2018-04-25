#include <tools/Parallel.hpp>


void Parallel::mpi_init(int argc, char** argv)
{
#if (IMPI==1)
	MPI_Init_adapt(&argc, &argv, &mpistatus);
#else
	MPI_Init(&argc, &argv);
#endif

	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
	return;
}

void Parallel::mpi_final() 
{
	MPI_Finalize();
	return;
}

void Parallel::mpi_update()
{
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
#if (IMPI==1)
	this->mpistatus = MPI_ADAPT_STATUS_STAYING;
#endif
	return;
}
