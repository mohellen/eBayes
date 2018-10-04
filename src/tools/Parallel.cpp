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
	// These only need to be called once
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

void Parallel::info()
{
	fflush(NULL);
	printf("[t%d r%d st%d] ", size, rank, status);
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


#if (0==1)
// node id is not need because only master prints progress
int Parallel::get_nodeid(Config const& c)
{
	// Get the node name in a string
	unique_ptr<char[]> host_name (new char[50]);
	int host_name_len;
	MPI_Get_processor_name(host_name.get(), &host_name_len);
	string hostname (&(host_name.get()[0]), &(host_name.get()[host_name_len]));

	// Get node file
	string hostfile = c.get_param_string("impi_node_file");
	// Open node file
	ifstream f(hostfile);
	if (!f) {
		fflush(NULL);
		printf("\nWARNING: cannot open node file. Node ID will be default to 0.\n");
		return;
	}
	// Compare node name line by line
	string s;
	int l = 0; // line number (node ID is determined by the line number at which the node name is listed)
	while (std::getline(f, s)) {
		// Compare
		if (s.compare(hostname) == 0) { // Found the node name, close file, and return node ID
			f.close();
			return l;
		}
		l++;
	}//end while
	f.close();
	fflush(NULL);
	printf("\nWARNING: cannot find node from node file. Node ID will be default to 0.\n");
	return 0;
}
#endif


