#include <tools/Parallel.hpp>



void Parallel::mpi_init(int argc, char** argv)
{
#if (IMPI==1)
	MPI_Init_adapt(&argc, &argv, &status);
#else
	MPI_Init(&argc, &argv);
#endif
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_POSSEQ = create_MPI_POSSEQ();
	return;
}

void Parallel::mpi_final() 
{
	MPI_Finalize();
	return;
}

void Parallel::mpi_update()
{
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#if (IMPI==1)
	this->status = MPI_ADAPT_STATUS_STAYING;
#endif
	return;
}

MPI_Datatype Parallel::create_MPI_POSSEQ()
{
	int lens[2] = {1, 1};
	MPI_Aint offs[2] = {0, sizeof(double)};
	MPI_Datatype types[2] = {MPI_DOUBLE, MPI_SIZE_T};
	MPI_Datatype newtype;
	MPI_Type_create_struct(2, lens, offs, types, &newtype);
	MPI_Type_commit(&newtype);
	return newtype;
}
