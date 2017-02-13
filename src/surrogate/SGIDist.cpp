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

#include <surrogate/SGIDist.hpp>

using namespace std;
using namespace sgpp::base;


#define TAG_WORK 		10
#define TAG_ADAPT		20
#define TAG_TERMINATE	30

/*********************************************
 *********************************************
 *       		 Public Methods
 *********************************************
 *********************************************/

SGIDist::SGIDist(
		FullModel* fm_obj,
		MPIObject* mpi_obj)
		: SurrogateModel(),
		  mpi(mpi_obj),
		  fullmodel(fm_obj), // Initialize const members
		  param_size(fm_obj->get_param_size()),
		  data_size(fm_obj->get_data_size())
{
	grid = nullptr;
	op_eval = nullptr;
	alphas.reserve(fullmodel->get_data_size());
	for (std::size_t i=0; i < data_size; i++) alphas.push_back(nullptr);
	maxpos_gp_seq = -1;
	maxpos = -1.0;

#if (ENABLE_IMPI == YES)
	CarryOver.phase = 0;
	CarryOver.gp_offset = 0;
#endif
}

std::size_t SGIDist::get_param_size()
{
	return param_size;
}

std::size_t SGIDist::get_data_size()
{
	return data_size;
}

void SGIDist::get_param_space(
		int dim,
		double& min,
		double& max)
{
	fullmodel->get_param_space(dim, min, max);
}

double SGIDist::compute_posterior_sigma()
{
	return fullmodel->compute_posterior_sigma();
}

double SGIDist::compute_posterior(
		const double sigma,
		const double* d)
{
	return fullmodel->compute_posterior(sigma, d);
}

void SGIDist::run(const double* m, double* d)
{
	// Convert m into data vector point
	DataVector point = DataVector(param_size);
	for (std::size_t i=0; i < param_size; i++) {
		point.set(i, m[i]);
	}
	// Evaluate operation check
	if (!op_eval) {
		cout << "Grid is not properly initialized or setup. "
				<< "Evaluate operation is not available. Progam abort!" << endl;
		exit(EXIT_FAILURE);
	}
	// Do evaluation
	for (std::size_t j=0; j < data_size; j++) {
		d[j] = op_eval->eval(*(this->alphas[j]), point);
	}
	return;
}

void SGIDist::initialize(
		std::size_t level,
		std::string mpi_scheme)
{
	std::size_t num_gps;

#if (ENABLE_IMPI == YES)
	if (mpi->status != MPI_ADAPT_STATUS_JOINING) {
#endif

	// 1. All: Construct grid
#if (SGI_GRID_WITH_BOUNDARY==YES)
	grid.reset(Grid::createLinearBoundaryGrid(param_size).release());	// create empty grid
#else
	grid.reset(Grid::createModLinearGrid(param_size).release());	// create empty grid
#endif
	grid->getGenerator().regular(level);			// populate grid points
	grid->setBoundingBox(*create_boundingbox());	// set up bounding box
	num_gps = grid->getSize();

	// Master:
	if (mpi->rank == MASTER) {
		// Print progress
		cout << "\n...Initializing SGI model..." << endl;
		cout << num_gps << " grid points to be added."
				<< " Total # grid points = " << num_gps << "." << endl;
		// Write grid to file
		mpi_iow_grid();
	}

#if (ENABLE_IMPI == YES)
	} else {
		num_gps = CarryOver.gp_offset;
	}
#endif

	// 2. All: Compute data at each grid point (result written to MPI IO file)
	//			and find the max posterior point
	compute_grid_points(0, mpi_scheme);
	find_maxpos();

	// 3. All: Create alphas
	create_alphas();

	// 4. Update op_eval
	op_eval = sgpp::op_factory::createOperationEval(*grid);

	// Master: print grogress
	if (mpi->rank == MASTER) {
		unique_ptr<double[]> m_maxpos (new double[param_size]);
		get_point_coord(maxpos_gp_seq, m_maxpos.get());
		cout << "Max posterior = " << maxpos <<
				", at " << arr_to_string(param_size, m_maxpos.get()) << endl;
		cout << "...Initialize SGI model successful..." << endl;
	}
	return;
}

void SGIDist::refine(
		double portion,
		std::string mpi_scheme)
{
	std::size_t num_gps, new_num_gps;

#if (ENABLE_IMPI == YES)
	if (mpi->status != MPI_ADAPT_STATUS_JOINING) {
#endif
	if (mpi->rank == MASTER)
		cout << "\n...Refining SGI model..." << endl;

	// 1. All: Refine grid
	num_gps = grid->getSize();
	portion = fmax(0.0, portion); // ensure portion is non-negative
	refine_grid(portion);
	new_num_gps = grid->getSize();

	// Master: print progress
	if (mpi->rank == MASTER)
		cout << new_num_gps-num_gps << " grid points to be added."
				<< " Total # grid points = " << new_num_gps << "." << endl;
#if (ENABLE_IMPI == YES)
	} else {
		num_gps = CarryOver.gp_offset;
	}
#endif

	// 2. All: Compute data at each grid point (result written to MPI IO file)
	//			and find the max posterior point
	compute_grid_points(num_gps, mpi_scheme);
	find_maxpos();

	// 3. All: Create alphas
	create_alphas();

	// 4. Update op_eval
	op_eval = sgpp::op_factory::createOperationEval(*grid);

	// MASTER: print grogress
	if (mpi->rank == MASTER) {
		unique_ptr<double[]> m_maxpos (new double[param_size]);
		get_point_coord(maxpos_gp_seq, m_maxpos.get());
		cout << "Max posterior = " << maxpos <<
				", at " << arr_to_string(param_size, m_maxpos.get()) << endl;
		cout << "...Refine SGI model successful..." << endl;
	}
	return;
}

void SGIDist::get_point_coord(
		std::size_t seq,
		double* m)
{
	// Get grid point coordinate
	DataVector gp_coord (param_size);
	grid->getStorage().get(seq)->getCoordsBB(gp_coord, grid->getBoundingBox());
	for (std::size_t dim = 0; dim < param_size; dim++)
		m[dim] = gp_coord.get(dim);
}

void SGIDist::get_maxpos_point(
		double* m)
{
	get_point_coord(maxpos_gp_seq, m);
	return;
}

#if (ENABLE_IMPI == YES)
void SGIDist::impi_adapt()
{
	int adapt_flag;
	MPI_Info info;
	MPI_Comm intercomm;
	MPI_Comm newcomm;
	double tic, toc;
	double tic1, toc1;

	tic = MPI_Wtime();
	MPI_Probe_adapt(&adapt_flag, &mpi->status, &info);
	toc = MPI_Wtime() - tic;
	cout << "Rank " << mpi->rank << " [STATUS " << mpi->status << "]: "
			<< "MPI_Probe_adapt " << toc << " seconds" << endl;

	if (adapt_flag == MPI_ADAPT_TRUE){
		tic1 = MPI_Wtime();
		tic = MPI_Wtime();
		MPI_Comm_adapt_begin(&intercomm, &newcomm, 0, 0);
		toc = MPI_Wtime() - tic;
		cout << "Rank " << mpi->rank << " [STATUS " << mpi->status << "]: "
				<< "MPI_Comm_adapt_begin " << toc << " seconds" << endl;
		//************************ ADAPT WINDOW ****************************

		if (mpi->status == MPI_ADAPT_STATUS_JOINING)
			mpi_ior_grid();

		MPI_Bcast(&(CarryOver.phase), 1, MPI_INT, MASTER, newcomm);
		MPI_Bcast(&(CarryOver.gp_offset), 1, MPI_UNSIGNED_LONG, MASTER, newcomm);

		//************************ ADAPT WINDOW ****************************
		tic = MPI_Wtime();
		MPI_Comm_adapt_commit(&adapt_flag);
		toc = MPI_Wtime() - tic;
		cout << "Rank " << mpi->rank << " [STATUS " << mpi->status << "]: "
				<< "MPI_Comm_adapt_commit " << toc << " seconds" << endl;

		mpi->update(MPI_COMM_WORLD);
		mpi->status = MPI_ADAPT_STATUS_STAYING;

		toc1 = MPI_Wtime() - tic1;
		cout << "Rank " << mpi->rank << " [STATUS " << mpi->status << "]: "
				<< "Total adaption time = " << toc1 << " seconds" << endl;
	}
	return;
}
#endif

/*********************************************
 *********************************************
 *       		 Private Methods
 *********************************************
 *********************************************/

BoundingBox* SGIDist::create_boundingbox()
{
	BoundingBox* bb = new BoundingBox(param_size);
	DimensionBoundary db;
	double min, max;
	for(int i=0; i < param_size; i++) {
		fullmodel->get_param_space(i, min, max);
		db.leftBoundary = min;
		db.rightBoundary = max;
		db.bDirichletLeft  = false;
		db.bDirichletRight = false;
		bb->setBoundary(i, db);
	}
	return bb;
}

void SGIDist::mpi_iow_grid()
{
	// Pack a grid
	string sg_str = grid->getStorage().serialize();
	size_t count = sg_str.size();
	unique_ptr<char[]> buff (new char[count]);
	strcpy(buff.get(), sg_str.c_str());

	// Write to file
	string ofile = string(OUTPUT_PATH) + "/grid.mpibin";
	MPI_File fh;
	// Delete existing grid file first
	if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_WRONLY, MPI_INFO_NULL, &fh)
			== MPI_SUCCESS) {
		MPI_File_delete(ofile.c_str(), MPI_INFO_NULL);
	}
	if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh)
			!= MPI_SUCCESS) {
		cout << "MPI write grid file open failed. Operation aborted! " << endl;
		exit(EXIT_FAILURE);
	}
	MPI_File_write_at(fh, 0, buff.get(), count, MPI_CHAR, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);
	return;
}

void SGIDist::mpi_ior_grid()
{
	// Open file and get file size
	string ofile = string(OUTPUT_PATH) + "/grid.mpibin";
	MPI_File fh;
	if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)
			!= MPI_SUCCESS) {
		cout << "MPI read grid file open failed. Operation aborted! " << endl;
		exit(EXIT_FAILURE);
	}
	long long int count = 0;
	MPI_File_get_size(fh, &count);
	unique_ptr<char[]> buff (new char[count]);

	// Read from file
	MPI_File_read_at(fh, 0, buff.get(), count, MPI_CHAR, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);

	// Create serialized grid string
	string sg_str(buff.get());

	// Construct new grid from grid string
#if (SGI_GRID_WITH_BOUNDARY==YES)
	grid.reset(Grid::createLinearBoundaryGrid(param_size).release());	// create empty grid
#else
	grid.reset(Grid::createModLinearGrid(param_size).release());	// create empty grid
#endif
	grid->getStorage().emptyStorage();
	grid->getStorage().unserialize_noAlgoDims(sg_str);	// restore grid from string object
	grid->setBoundingBox(*create_boundingbox()); // set up bounding box
	op_eval = sgpp::op_factory::createOperationEval(*grid);
	return;
}

void SGIDist::mpi_io_data(
		std::string which_type,
		char which_action,
		std::size_t seq_min,
		std::size_t seq_max,
		double* buff)
{
	// Do something only when seq_min <= seq_max
	if (seq_min > seq_max) return;

	string ofile;
	std::size_t vlen;
	if (which_type.compare("data") == 0) {
		ofile = string(OUTPUT_PATH) + "/data.mpibin";;
		vlen = data_size;
	} else if (which_type.compare("pos") == 0) {
		ofile = string(OUTPUT_PATH) + "/pos.mpibin";
		vlen = 1;
	} else {
		cout << "Output file does not exist. Operation aborted!" << endl;
		exit(EXIT_FAILURE);
	}
	MPI_File fh;
	if (which_action == 'w') {
		if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh)
				!= MPI_SUCCESS) {
			cout << "MPI write file open failed. Operation aborted! " << endl;
			exit(EXIT_FAILURE);
		}
		// offset is in # of bytes, and is ALWAYS calculated from beginning of file.
		MPI_File_write_at(fh, seq_min*vlen*sizeof(double), buff, (seq_max-seq_min+1)*vlen, MPI_DOUBLE, MPI_STATUS_IGNORE);

	} else if (which_action == 'r') {
		if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)
				!= MPI_SUCCESS) {
			cout << "MPI read file open failed. Operation aborted! " << endl;
			exit(EXIT_FAILURE);
		}
		// offset is in # of bytes, and is ALWAYS calculated from beginning of file.
		MPI_File_read_at(fh, seq_min*vlen*sizeof(double), buff, (seq_max-seq_min+1)*vlen, MPI_DOUBLE, MPI_STATUS_IGNORE);

	} else {
		cout << "Please specify action type: 'r' for read, or 'w' for write." << endl;
		exit(EXIT_FAILURE);
	}
	MPI_File_close(&fh);
	return;
}

void SGIDist::get_local_range(
		const std::size_t& global_min,
		const std::size_t& global_max,
		std::size_t& mymin,
		std::size_t& mymax)
{
	std::size_t num_gps = global_max - global_min + 1;
	std::size_t trunk = num_gps / mpi->size;
	std::size_t rest = num_gps % mpi->size;

	if (mpi->rank < rest) {
		mymin = global_min + mpi->rank * (trunk + 1);
		mymax = mymin + trunk;
	} else {
		mymin = global_min + mpi->rank * trunk + rest;
		mymax = mymin + trunk - 1;
	}
	return;
}

void SGIDist::find_maxpos()
{
	if (mpi->size > 1) {
		struct {
			double mymaxpos;
			int myrank;
		} in, out;
		in.mymaxpos = maxpos;
		in.myrank = mpi->rank;
		MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, mpi->comm);
		maxpos = out.mymaxpos;
		MPI_Bcast(&maxpos_gp_seq, 1, MPI_UNSIGNED_LONG, out.myrank, mpi->comm);
	}
	return;
}

void SGIDist::create_alphas()
{
	double tic = MPI_Wtime(); // start timer
	std::size_t num_gps = grid->getSize();

	// read raw data
	unique_ptr<double[]> data (new double[data_size * num_gps]);
	mpi_io_data("data", 'r', 0, num_gps-1, data.get());

	// re-allocate alphas
	for (std::size_t j=0; j < data_size; j++)
		alphas[j].reset(new DataVector(num_gps));

	// unpack raw data into alphas
	for (std::size_t i = 0; i < num_gps; i++)
		for (std::size_t j = 0; j < data_size; j++)
			alphas[j]->set(i, data[i*data_size+j]);

	// hierarchize alphas
	unique_ptr<OperationHierarchisation> hier (sgpp::op_factory::createOperationHierarchisation(*grid));
	for (std::size_t j=0; j < data_size; j++)
		hier->doHierarchisation(*(alphas[j]));

#if (SGI_ENABLE_TIMER == YES)
	if (mpi->rank == MASTER)
		cout << "Rank " << mpi->rank
				<< ": created alphas"
				<< " in " << MPI_Wtime()-tic << " seconds." << endl;
#endif
	return;
}

void SGIDist::refine_grid(double portion)
{
	double tic = MPI_Wtime(); // start timer

	std::size_t num_gps = this->grid->getSize();
	std::size_t refine_gps = num_gps * fmax(0.0, portion);
	DataVector refine_idx (num_gps);

	// Read posterior from file
	unique_ptr<double[]> pos (new double[num_gps]);
	mpi_io_data("pos", 'r', 0, num_gps-1, &pos[0]);

	// For each gp, compute the refinement index
	double data_norm;
	for (std::size_t i=0; i<num_gps; i++) {
		data_norm = 0;
		for (std::size_t j=0; j<data_size; j++) {
			data_norm += (alphas[j]->get(i) * alphas[j]->get(i));
		}
		data_norm = sqrt(data_norm);
		// refinement_index = |alpha| * posterior
		refine_idx.set(i, data_norm * pos[i]);
	}

	// refine grid
	grid->refine(refine_idx, std::size_t(ceil(double(num_gps)*portion)));

	// Master write grid to file
	if (mpi->rank == MASTER) {
		mpi_iow_grid();

#if (SGI_ENABLE_TIMER == YES)
		cout << "Rank " << mpi->rank
				<< ": refined grid"
				<< " in " << MPI_Wtime()-tic << " seconds." << endl;
#endif
	}
	return;
}

void SGIDist::compute_grid_points(
		std::size_t gp_offset,
		std::string mpi_scheme)
{
	double tic = MPI_Wtime(); // start the timer
	// NOTE: both "Master-minion" or "Naive" schemes can run under MPI & iMPI
	//		settings, but only the "Master-minion" scheme uses the iMPI features.
	if (mpi_scheme == "mm") {
		// Master-minion scheme
		if (mpi->rank == MASTER) {
			compute_gp_master(gp_offset);
		} else {
			compute_gp_minion(gp_offset);
		}
	} else {
		// Naive scheme (default)
		std::size_t num_gps = grid->getSize();
		std::size_t mymin, mymax;
		get_local_range(gp_offset, num_gps-1, mymin, mymax);
		compute_range(mymin, mymax);
	}
	MPI_Barrier(mpi->comm);

#if (SGI_ENABLE_TIMER == YES)
	if (mpi->rank == MASTER) {
		cout << mpi->size << " Ranks"
				<< ": computed " << grid->getSize()-gp_offset << " grid points ("
				<< gp_offset << " to " << grid->getSize()-1
				<< ") in " << MPI_Wtime()-tic << " seconds." << endl;
	}
#endif
	return;
}

void SGIDist::compute_range(
		const std::size_t& seq_min,
		const std::size_t& seq_max)
{
	// IMPORTANT: Ensure workload is at least 1 grid point!
	// This also prevents overriding wrong data to data files!
	if (seq_max < seq_min) return;

#if (SGI_PRINT_RANK_PROGRESS == YES)
	cout << "Rank " << mpi->rank <<
			": computing gps " << seq_min << " to " << seq_max << "." << endl;
#endif
	// Compute data & posterior
	unique_ptr<DataVector> gp_coord (new DataVector(param_size));
	GridStorage* gs = &(grid->getStorage());
	BoundingBox* bb = &(grid->getBoundingBox());

	std::size_t load = std::size_t(fmax(0, seq_max - seq_min + 1));
	unique_ptr<double[]> data (new double[load * data_size]);
	unique_ptr<double[]> pos (new double[load]);

	unique_ptr<double[]> m (new double[param_size]);
	double* d = nullptr;
	double* p = nullptr;
	double sigma = fullmodel->compute_posterior_sigma();

	std::size_t i,dim;  // loop index
	for (i=seq_min; i <= seq_max; i++) {
		// Set output pointer
		d = &data[0] + (i-seq_min) * data_size;
		p = &pos[0] + (i-seq_min);

		// Get grid point coordinate
		gs->get(i)->getCoordsBB(*gp_coord, *bb);
		for (dim = 0; dim < param_size; dim++)
			m[dim] = gp_coord->get(dim);

		// compute with full model
		fullmodel->run(m.get(), d);

		// compute posterior
		*p = compute_posterior(sigma, d);

		// Find max posterior
		if (*p > maxpos) {
			maxpos = *p;
			maxpos_gp_seq = i;
		}
#if (SGI_PRINT_GRID_POINTS == YES)
		cout << "Rank " << mpi_rank << ": gp " << i << " at "
			 << gp_coord->toString() << " completed, posterior = " << *p << endl;
#endif
	}
	// Write results to file
	mpi_io_data("data", 'w', seq_min, seq_max, data.get());
	mpi_io_data("pos", 'w', seq_min, seq_max, pos.get());
	return;
}

void SGIDist::compute_gp_master(
		const std::size_t& gp_offset)
{
	std::size_t added_num_gps;
	int num_jobs;
	int minion;
	int jobid;
	MPI_Status status;
	unique_ptr<int[]> jobs; // use array to have unique send buffer
	int scnt;
	vector<int> jobs_done;

#if (ENABLE_IMPI == YES)
	double tic, toc;
	int jobs_per_tic;
#endif

	// Determine total # jobs (compute only the newly added points)
	added_num_gps = grid->getSize() - gp_offset;
	num_jobs = (added_num_gps % SGI_MINION_TRUNK_SIZE > 0) ?
			added_num_gps/SGI_MINION_TRUNK_SIZE + 1 :
			added_num_gps/SGI_MINION_TRUNK_SIZE;

	scnt = 0;
	jobs.reset(new int[num_jobs]);
	for (int i = 0; i < num_jobs; i++)
		jobs[i] = i;
	// Reserve memory for done jobs
	jobs_done.reserve(num_jobs);

#if (ENABLE_IMPI == YES)
	toc = 0;
	tic = MPI_Wtime();
	jobs_per_tic = 0;
#endif

	// Seed minions if any
	if (mpi->size > 1)
		seed_minions(num_jobs, scnt, jobs.get());

	// As long as not all jobs are done, keep working...
	while (jobs_done.size() < num_jobs) {

		if (mpi->size > 1) {
			// #1. Receive a finished job
			MPI_Recv(&jobid, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, mpi->comm, &status);
			minion = status.MPI_SOURCE;
			jobs_done.push_back(jobid); // mark the job as done
#if (ENABLE_IMPI == YES)
			jobs_per_tic++;
#endif
			// #2. Send another job if any
			if (scnt < num_jobs) {
				MPI_Send(&jobs[scnt], 1, MPI_INT, minion, TAG_WORK, mpi->comm);
				scnt++;
			}
		} else { // if there is NO minion
			// #1. Take out job
			jobid = jobs[scnt];
			scnt++;
			// #2. Compute a job
			std::size_t seq_min, seq_max;
			jobid_to_range(jobid, gp_offset, seq_min, seq_max);
			compute_range(seq_min, seq_max);
			jobs_done.push_back(jobid);
#if (ENABLE_IMPI == YES)
			jobs_per_tic++;
#endif
		}

		// #3. Check for adaption every IMPI_ADAPT_INTERVAL seconds
#if (ENABLE_IMPI == YES)
		toc = MPI_Wtime()-tic;
		if (toc >= IMPI_ADAPT_INTERVAL) {
			// performance measure: # gps computed per second
			cout << "PERFORMANCE MEASURE: # forward simulations per second = "
				<< double(jobs_per_tic*SGI_MINION_TRUNK_SIZE)/toc
				<< endl;
			// Only when there are remaining jobs, it's worth trying to adapt
			if (scnt < num_jobs) {
				// Prepare minions for adapt (receive done jobs, then send adapt signal)
				if (mpi->size > 1)
					prepare_minions_for_adapt(jobs_done, jobs_per_tic);
				// Adapt
				MPI_Barrier(mpi->comm);
				impi_adapt();
				// Seed minions again
				if (mpi->size > 1)
					seed_minions(num_jobs, scnt, jobs.get());
			}
			// reset timer
			tic = MPI_Wtime();
			jobs_per_tic = 0;
		} // end if-toc
#endif
	} // end while

	// All jobs done
	if (mpi->size > 1) {
		for (int mi=1; mi<mpi->size; mi++)
			MPI_Send(&jobid, 1, MPI_INT, mi, TAG_TERMINATE, mpi->comm);
	}
	return;
}

void SGIDist::compute_gp_minion(
		const std::size_t& gp_offset)
{
	// Setup variables
	int job_todo, job_done; // use separate buffers for send and receive
	std::size_t seq_min, seq_max;
	MPI_Status status;

	while (true) {
		// Receive a signal from MASTER
		MPI_Recv(&job_todo, 1, MPI_INT, MASTER, MPI_ANY_TAG, mpi->comm, &status);

		if (status.MPI_TAG == TAG_TERMINATE) break;

		if (status.MPI_TAG == TAG_WORK) {
			// get the job range and compute
			jobid_to_range(job_todo, gp_offset, seq_min, seq_max);
			compute_range(seq_min, seq_max);

			// tell master the job is done
			job_done = job_todo;
			MPI_Send(&job_done, 1, MPI_INT, MASTER, job_done, mpi->comm);
		}

#if (ENABLE_IMPI == YES)
		if (status.MPI_TAG == TAG_ADAPT) {
			MPI_Barrier(mpi->comm);
			impi_adapt();
		}
#endif
	} // end while
	return;
}

void SGIDist::jobid_to_range(
		const std::size_t& jobid,
		const std::size_t& seq_offset,
		std::size_t& seq_min,
		std::size_t& seq_max)
{
	seq_min = seq_offset + jobid * SGI_MINION_TRUNK_SIZE;
	seq_max = min(seq_min+SGI_MINION_TRUNK_SIZE-1, grid->getSize()-1);
}

void SGIDist::seed_minions(
		const int& num_jobs,
		int& scnt,
		int* jobs)
{
	// the smaller of (remainning jobs) or (# minions)
	int size = int(fmin(num_jobs-scnt, mpi->size-1));
	unique_ptr<MPI_Request[]> tmp_req (new MPI_Request[size]);
	for (int i=0; i < size; i++) {
		MPI_Isend(&jobs[scnt], 1, MPI_INT, i+1, TAG_WORK, mpi->comm, &tmp_req[i]);
		scnt++;
	}
	MPI_Waitall(size, tmp_req.get(), MPI_STATUS_IGNORE);
	return;
}

void SGIDist::prepare_minions_for_adapt(
		vector<int> & jobs_done,
		int & jobs_per_tic)
{
	unique_ptr<MPI_Request[]> tmp_req (new MPI_Request[(mpi->size-1)*2]);
	unique_ptr<int[]> tmp_rbuf (new int[mpi->size-1]);
	unique_ptr<int[]> tmp_sbuf (new int[mpi->size-1]); // dummy send buffer

	for (int i=1; i < mpi->size; i++) {
		// First receive a finished job
		MPI_Irecv(&tmp_rbuf[i-1], 1, MPI_INT, MPI_ANY_SOURCE,
				MPI_ANY_TAG, mpi->comm, &tmp_req[i-1]);
		// Then send "adapt signal", send buffer is dummy
		MPI_Isend(&tmp_sbuf[i-1], 1, MPI_INT, i,
				TAG_ADAPT, mpi->comm, &tmp_req[(mpi->size-1) + i-1]);
	}
	MPI_Waitall((mpi->size-1)*2, tmp_req.get(), MPI_STATUS_IGNORE);
	for (int i=1; i < mpi->size; i++) {
		jobs_done.push_back(tmp_rbuf[i-1]);
		jobs_per_tic++;
	}
	return;
}
