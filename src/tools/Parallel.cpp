#include <tools/Parallel.hpp>


void Parallel::mpi_init(int argc, char* argv[])
{
#if defined(IMPI)
	MPI_Init_adapt(&argc, &argv, &(this->mpistatus));
#else
	MPI_Init(&argc, &argv);
#endif

	MPI_Comm_size(MPI_COMM_WORLD, &(this->mpisize));
	MPI_Comm_rank(MPI_COMM_WORLD, &(this->mpirank));
	return;
}

void Parallel::mpi_final() 
{
	MPI_Finalize();
	return;
}

void Parallel::mpi_update()
{
	MPI_Comm_size(MPI_COMM_WORLD, &(this->mpisize));
	MPI_Comm_rank(MPI_COMM_WORLD, &(this->mpirank));
#if defined(IMPI)
	this->mpistatus = MPI_ADAPT_STATUS_STAYING;
#endif
	return;
}
