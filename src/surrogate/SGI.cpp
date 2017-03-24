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

#include <surrogate/SGI.hpp>

using namespace std;
using namespace	sgpp::base;

//int MPI_Init_adapt(int *argc, char ***argv, int *local_status);
//int MPI_Probe_adapt(int *current_operation, int *local_status, MPI_Info *info);
//int MPI_Comm_adapt_begin(MPI_Comm *intercomm, MPI_Comm *new_comm_world,
//		int *staying_count, int *leaving_count, int *joining_count);
//int MPI_Comm_adapt_commit(void);



SGI::SGI(
		const string& input_file,
		const string& output_prefix,
		int resx,
		int resy)
		: ForwardModel()
{
	MPI_Comm_rank(MPI_COMM_WORLD, &(this->mpi_rank));
	MPI_Comm_size(MPI_COMM_WORLD, &(this->mpi_size));
#if (ENABLE_IMPI==1)
	mpi_status = -1;
	impi_gpoffset = -1;
#endif

	this->outprefix = output_prefix;
	this->fullmodel.reset(new NS(input_file, resx, resy));
	this->input_size = fullmodel->get_input_size();   // inherited from ForwardModel
	this->output_size = fullmodel->get_output_size(); // inherited from ForwardModel

	this->alphas.reset(new DataVector[output_size]);
	this->grid = nullptr;
	this->eval = nullptr;
	this->bbox = nullptr;

	this->maxpos_seq = 0;
	this->maxpos = 0.0;
	this->noise = 0.0;
	this->odata.reset( get_observed_data(input_file,
			this->output_size, this->noise) );
	this->sigma = compute_posterior_sigma(
			this->odata.get(), this->output_size, this->noise);
}

std::size_t SGI::get_input_size()
{
	return this->input_size;
}

std::size_t SGI::get_output_size()
{
	return this->output_size;
}

void SGI::get_input_space(
			int dim,
			double& min,
			double& max)
{
	fullmodel->get_input_space(dim, min, max);
}

void SGI::run(const double* m, double* d)
{
	// Grid check
	if (!eval) {
		cout << "SGI model is not properly built. Progam abort!" << endl;
		exit(EXIT_FAILURE);
	}
	// Convert m into data vector
	DataVector point = arr_to_vec(m, input_size);
	// Evaluate m
	for (std::size_t j=0; j < output_size; j++) {
		d[j] = eval->eval(alphas[j], point);
	}
	return;
}

void SGI::build(
		double refine_portion,
		std::size_t init_grid_level,
		bool is_masterworker)
{
	std::size_t num_points, new_num_points;
	// find out whether it's grid initialization or refinement
	bool is_init = (!this->eval) ? true : false;

#if (ENABLE_IMPI==1)
	if (mpi_status != MPI_ADAPT_STATUS_JOINING) {
#endif

	if (is_init) {
		if (is_master())
			printf("\n...Initializing SGI model...\n");
		// 1. All: Construct grid
//		grid.reset(Grid::createLinearBoundaryGrid(input_size).release()); // create empty grid
		grid.reset(Grid::createModLinearGrid(input_size).release()); // create empty grid
		grid->getGenerator().regular(init_grid_level); // populate grid points
		bbox.reset(create_boundingbox());
		grid->setBoundingBox(*bbox); // set up bounding box
		num_points = grid->getSize();
		if (is_master()) {
			printf("Grid points added: %lu\n", num_points);
			printf("Total grid points: %lu\n", num_points);
		}
	} else {
		if (is_master())
			printf("\n...Refining SGI model...\n");
		// 1. All: refine grid
		num_points = grid->getSize();
		if (!refine_grid(refine_portion)) {
			printf("Grid not refined!!");
			printf("Grid points added: 0\n");
			printf("Total grid points: %lu\n", num_points);
			return;
		}
		new_num_points = grid->getSize();
		if (is_master()) {
			printf("Grid points added: %lu\n", new_num_points-num_points);
			printf("Total grid points: %lu\n", new_num_points);
		}
	}
	mpiio_write_grid();

#if (ENABLE_IMPI==1)
	} else {
		num_points = impi_gpoffset;
	}
#endif

	// 2. All: Compute data at each grid point (result written to MPI IO file)
	//			and find the max posterior point
	if (is_init) {
		compute_grid_points(0, is_masterworker);
	} else {
		compute_grid_points(num_points, is_masterworker);
	}
	mpi_find_global_update_maxpos();

	// 3. All: Compute and hierarchize alphas
	compute_hier_alphas();

	// 4. Update op_eval
	eval.reset(sgpp::op_factory::createOperationEval(*grid).release());

	// Master: print grogress
	if (is_master()) {
		unique_ptr<double[]> m_maxpos (seg_to_coord_arr(maxpos_seq));
		printf("Max posterior = %.6f, at %s.\n",
				maxpos, arr_to_string(m_maxpos.get(), input_size).c_str());
		if (is_init)
			printf("...Initialize SGI model successful...\n");
		else
			printf("...Refine SGI model successful...\n");
	}
	return;
}

void SGI::duplicate(
		const string& gridfile,
		const string& datafile,
		const string& posfile)
{
	// Set: grid, eval, bbox
	mpiio_read_grid(gridfile);

	// Set: alphas
	compute_hier_alphas(datafile);

	// Read posterior
	std::size_t num_gps = grid->getSize();
	unique_ptr<double[]> pos (new double[num_gps]);
	mpiio_partial_posterior(true, 0, num_gps-1, pos.get(), posfile);

	// Find max pos
	for (std::size_t i=0; i < num_gps; i++) {
		if (pos[i] > maxpos) {
			maxpos = pos[i];
			maxpos_seq = i;
		}
	}
	return;
}

void SGI::impi_adapt()
{
#if (ENABLE_IMPI==1)
	int adapt_flag;
	MPI_Info info;
	MPI_Comm intercomm;
	MPI_Comm newcomm;
	int staying_count, leaving_count, joining_count;
	double tic, toc;
	double tic1, toc1;

	tic = MPI_Wtime();
	MPI_Probe_adapt(&adapt_flag, &mpi_status, &info);

	toc = MPI_Wtime() - tic;
	printf("Rank %d [STATUS %1d]: MPI_Probe_adapt %.6f seconds.\n",
			mpi_rank, mpi_status, toc);

	if (adapt_flag == MPI_ADAPT_TRUE){
		tic1 = MPI_Wtime();
		tic = MPI_Wtime();
		MPI_Comm_adapt_begin(&intercomm, &newcomm,
				&staying_count, &leaving_count, &joining_count);

		toc = MPI_Wtime() - tic;
		printf("Rank %d [STATUS %1d]: MPI_Comm_adapt_begin %.6f seconds.\n",
				mpi_rank, mpi_status, toc);

		//************************ ADAPT WINDOW ****************************
		if (mpi_status == MPI_ADAPT_STATUS_JOINING) mpiio_read_grid();

		MPI_Bcast(&impi_gpoffset, 1, MPI_UNSIGNED_LONG, MASTER, newcomm);
		//************************ ADAPT WINDOW ****************************

		tic = MPI_Wtime();
		MPI_Comm_adapt_commit();

		toc = MPI_Wtime() - tic;
		printf("Rank %d [STATUS %1d]: MPI_Comm_adapt_commit %.6f seconds.\n",
				mpi_rank, mpi_status, toc);

		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
		mpi_status = MPI_ADAPT_STATUS_STAYING;

		toc1 = MPI_Wtime() - tic1;
		printf("Rank %d [STATUS %1d]: TOTAL adaption %.6f seconds.\n",
				mpi_rank, mpi_status, toc);
	}
	return;
#endif
}

vector<vector<double> > SGI::get_top_maxpos(int num_tops, string posfile)
{
	// Read in all posterior
	size_t num_gps = grid->getSize();
	unique_ptr<double[]> pos (new double[num_gps]);
	mpiio_partial_posterior(true, 0, num_gps-1, pos.get(), posfile);

	// Initialize output
	vector<vector<double> > tops;
	tops.resize(num_tops);
	for (int k=0; k < num_tops; k++) {
		tops[k].resize(input_size + 1);
	}
	// Find the top N...
	// Initialize the tracking book
	vector<pair<double, size_t> > tracking;
	for (int k=0; k < num_tops; k++) {
		tracking[k].first = pos[k]; /// first is pos
		tracking[k].second = k;     /// second is the pos's seq number
	}
	// Search
	for (size_t i=0; i < num_gps; i++) {
		for (int k=0; k < num_tops; k++) {
			if (pos[i] > tracking[k].first) {
				tracking[k].first = pos[i];
				tracking[k].second = i;
				break; //once matched, break inner k-loop, continue with i-loop
			}
		}
	}
	// Get results
	for (int k=0; k < num_tops; k++) {
		unique_ptr<double[]> p (seg_to_coord_arr(tracking[k].second));
		for (size_t i=0; i < input_size; i++) {
			tops[k][i] = p[i];
		}
		tops[k][input_size] = tracking[k].first;
	}
	return tops;
}




/*********************************************
 *********************************************
 *       		 Private Methods
 *********************************************
 *********************************************/

/***************************
 * Grid related operations
 ***************************/
DataVector SGI::arr_to_vec(const double *& in, std::size_t size)
{
	DataVector out = DataVector(size);
	for (std::size_t i=0; i < size; i++)
		out[i] = in[i];
	return out;
}

double* SGI::vec_to_arr(DataVector& in)
{
	std::size_t size = in.getSize();
	double* out = new double[size];
	for (std::size_t i=0; i < size; i++)
		out[i] = in[i];
	return out;
}

double* SGI::seg_to_coord_arr(std::size_t seq)
{
	// Get grid point coordinate
	DataVector gp_coord (input_size);
	grid->getStorage().get(seq)->getCoordsBB(gp_coord, grid->getBoundingBox());
	return vec_to_arr(gp_coord);
}

string SGI::vec_to_str(DataVector& v)
{
	std::ostringstream oss;
	oss << "[" << std::fixed << std::setprecision(4);
	for (std::size_t i=0; i < v.getSize()-1; i++)
		oss << v[i] << ", ";
	oss << v[v.getSize()-1] << "]";
	return oss.str();
}

BoundingBox* SGI::create_boundingbox()
{
	BoundingBox* bb = new BoundingBox(input_size);
	DimensionBoundary db;
	double min, max;
	for(int i=0; i < input_size; i++) {
		get_input_space(i, min, max);
		db.leftBoundary = min;
		db.rightBoundary = max;
		db.bDirichletLeft  = false;
		db.bDirichletRight = false;
		bb->setBoundary(i, db);
	}
	return bb;
}

void SGI::compute_hier_alphas(const string& outfile)
{
#if (SGI_OUT_TIMER==1)
	double tic = MPI_Wtime();
#endif
	// read raw data
	std::size_t num_gps = grid->getSize();
	unique_ptr<double[]> data (new double[output_size * num_gps]);
	mpiio_partial_data(true, 0, num_gps-1, data.get(), outfile);

	// re-allocate alphas
	for (std::size_t j=0; j < output_size; j++)
		alphas[j].resize(num_gps);

	// unpack raw data
	for (std::size_t i=0; i < num_gps; i++)
		for (std::size_t j=0; j < output_size; j++)
			alphas[j].set(i, data[i*output_size+j]);

	// hierarchize alphas
	auto hier = sgpp::op_factory::createOperationHierarchisation(*grid);
	for (std::size_t j=0; j<output_size; j++)
		hier->doHierarchisation(alphas[j]);

#if (SGI_OUT_TIMER==1)
	if (is_master())
		printf("Rank %d: created alphas in %.5f seconds.\n",
				mpi_rank, MPI_Wtime()-tic);
#endif
	return;
}

bool SGI::refine_grid(double portion_to_refine)
{
#if (SGI_OUT_TIMER==1)
	double tic = MPI_Wtime();
#endif
	// Compute threshold number of grid points to be added
	// NOTE: to refine X points, maximum (2*dim*X) points can be added to grid
	int maxi = 10000;
	int thres = int(ceil(maxi / 2 / input_size));

	// Number of points to refine
	std::size_t num_gps = this->grid->getSize();
	int refine_gps = int(ceil(num_gps * portion_to_refine));
	refine_gps = (refine_gps > thres) ? thres : refine_gps;

	// If not points to refine, return false (meaning grid not refined)
	if (refine_gps < 1) return false;

	// Read posterior from file
	unique_ptr<double[]> pos (new double[num_gps]);
	mpiio_partial_posterior(true, 0, num_gps-1, &pos[0]);

	// For each gp, compute the refinement index
	double data_norm;
	DataVector refine_idx (num_gps);
	for (std::size_t i=0; i<num_gps; i++) {
		data_norm = 0;
		for (std::size_t j=0; j<output_size; j++) {
			data_norm += (alphas[j][i] * alphas[j][i]);
		}
		data_norm = sqrt(data_norm);
		// refinement_index = |alpha| * posterior
		refine_idx[i] = data_norm * pos[i];
	}
	// refine grid
	grid->refine(refine_idx, refine_gps);

#if (SGI_OUT_TIMER==1)
	if (is_master())
		printf("Rank %d: refined grid in %.5f seconds.\n", mpi_rank, MPI_Wtime()-tic);
#endif
	return true;
}

/***************************
 * MPI related operations
 ***************************/
bool SGI::is_master()
{
	if (mpi_rank == 0) {
#if (ENABLE_IMPI==1)
		if (mpi_status != MPI_ADAPT_STATUS_JOINING)
#endif
			return true;
	}
	return false;
}

void SGI::mpiio_write_grid(const string& outfile)
{
	// Pack grid into Char array
	string sg_str = grid->serialize();
	size_t count = sg_str.size();

	// Get local range (local protion to write)
	std::size_t lmin, lmax, load;
	mpina_get_local_range(0, count-1, lmin, lmax);
	load = lmax -lmin + 1;

	// Copy partial grid to string
	unique_ptr<char[]> buff (new char[load]);
	sg_str.copy(buff.get(), load, lmin);

	// Write to file
	string ofile = outfile;
	if (ofile == "")
		ofile = outprefix + "grid.mpibin";
	MPI_File fh;
	if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
			MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
		printf("MPI write grid file open failed. Operation aborted!\n");
		exit(EXIT_FAILURE);
	}
	MPI_File_write_at(fh, lmin, buff.get(), load, MPI_CHAR, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);
	return;
}

void SGI::mpiio_read_grid(const string& outfile)
{
	// Open file
	string ofile = outfile;
	if (ofile == "")
		ofile = outprefix + "grid.mpibin";
	MPI_File fh;
	if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)
			!= MPI_SUCCESS) {
		printf("MPI read grid file open failed. Operation aborted!\n");
		exit(EXIT_FAILURE);
	}
	// Get file size,create buffer
	long long int count;
	MPI_File_get_size(fh, &count);
	unique_ptr<char[]> buff (new char[count]);

	// Read from file
	MPI_File_read_at(fh, 0, buff.get(), count, MPI_CHAR, MPI_STATUS_IGNORE);
	MPI_File_close(&fh);

	// Create serialized grid string
	string sg_str(buff.get());

	// Construct new grid from grid string
	grid.reset(Grid::unserialize(sg_str).release()); // create empty grid
	bbox.reset(create_boundingbox());
	grid->setBoundingBox(*bbox); // set up bounding box
	eval.reset(sgpp::op_factory::createOperationEval(*grid).release());
	return;
}

void SGI::mpiio_partial_data(
		bool is_read,
		std::size_t seq_min,
		std::size_t seq_max,
		double* buff,
		const string& outfile)
{
	// Do something only when seq_min <= seq_max
	if (seq_min > seq_max) return;

	string ofile = outfile;
	if (ofile == "")
		ofile = outprefix + "data.mpibin";
	MPI_File fh;
	if (is_read) { // Read data from file
		if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)
				!= MPI_SUCCESS) {
			printf("MPI read data file open failed. Operation aborted!\n");
			exit(EXIT_FAILURE);
		}
		// offset is in # of bytes, and is ALWAYS calculated from beginning of file.
		MPI_File_read_at(fh, seq_min*output_size*sizeof(double), buff,
				(seq_max-seq_min+1)*output_size, MPI_DOUBLE, MPI_STATUS_IGNORE);

	} else { // Write data to file
		if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh)
				!= MPI_SUCCESS) {
			printf("MPI write data file open failed. Operation aborted!\n");
			exit(EXIT_FAILURE);
		}
		// offset is in # of bytes, and is ALWAYS calculated from beginning of file.
		MPI_File_write_at(fh, seq_min*output_size*sizeof(double), buff,
				(seq_max-seq_min+1)*output_size, MPI_DOUBLE, MPI_STATUS_IGNORE);
	}
	MPI_File_close(&fh);
	return;
}

void SGI::mpiio_partial_posterior(
		bool is_read,
		std::size_t seq_min,
		std::size_t seq_max,
		double* buff,
		const string& outfile)
{
	// Do something only when seq_min <= seq_max
	if (seq_min > seq_max) return;

	string ofile = outfile;
	if (ofile == "")
		ofile = outprefix + "pos.mpibin";;
	MPI_File fh;
	if (is_read) { // Read data from file
		if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)
				!= MPI_SUCCESS) {
			printf("MPI read pos file open failed. Operation aborted!\n");
			exit(EXIT_FAILURE);
		}
		// offset is in # of bytes, and is ALWAYS calculated from beginning of file.
		MPI_File_read_at(fh, seq_min*sizeof(double), buff,
				(seq_max-seq_min+1), MPI_DOUBLE, MPI_STATUS_IGNORE);

	} else { // Write data to file
		if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh)
				!= MPI_SUCCESS) {
			printf("MPI write pos file open failed. Operation aborted!\n");
			exit(EXIT_FAILURE);
		}
		// offset is in # of bytes, and is ALWAYS calculated from beginning of file.
		MPI_File_write_at(fh, seq_min*sizeof(double), buff,
				(seq_max-seq_min+1), MPI_DOUBLE, MPI_STATUS_IGNORE);
	}
	MPI_File_close(&fh);
	return;
}

void SGI::mpi_find_global_update_maxpos()
{
	if (mpi_size <= 1) return;
	struct {
		double mymaxpos;
		int myrank;
	} in, out;
	in.mymaxpos = maxpos;
	in.myrank = mpi_rank;
	MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
	maxpos = out.mymaxpos;
	MPI_Bcast(&maxpos_seq, 1, MPI_UNSIGNED_LONG, out.myrank, MPI_COMM_WORLD);
	return;
}

/***************************
 * Core computation
 ***************************/
void SGI::compute_grid_points(
		std::size_t gp_offset,
		bool is_masterworker)
{
#if (SGI_OUT_TIMER==1)
	double tic = MPI_Wtime(); // start the timer
#endif
	// NOTE: both "Master-minion" or "Naive" schemes can run under MPI & iMPI
	//		settings, but only the "Master-minion" scheme uses the iMPI features.
	if (is_masterworker) {
		// Master-worker style
		if (is_master())
			mpimw_master_compute(gp_offset);
		else
			mpimw_worker_compute(gp_offset);
	} else {
		// MPI native style (default)
		std::size_t num_gps = grid->getSize();
		std::size_t mymin, mymax;
		mpina_get_local_range(gp_offset, num_gps-1, mymin, mymax);
		mpi_compute_range(mymin, mymax);
	}
	MPI_Barrier(MPI_COMM_WORLD);

#if (SGI_OUT_TIMER==1)
	if (is_master())
		printf("%d Ranks: computed %lu grid points (%lu to %lu) in %.5f seconds.\n",
				mpi_size, grid->getSize()-gp_offset,
				gp_offset, grid->getSize()-1, MPI_Wtime()-tic);
#endif
	return;
}

void SGI::mpi_compute_range(
		const std::size_t& seq_min,
		const std::size_t& seq_max)
{
	// IMPORTANT: Ensure workload is at least 1 grid point!
	// This also prevents overriding wrong data to data files!
	if (seq_max < seq_min) return;

#if (SGI_OUT_RANK_PROGRESS==1)
	printf("Rank %d: computing %lu grid points: [%lu, %lu]\n",
			mpi_rank, (seq_max-seq_min+1), seq_min, seq_max);
#endif
	// Compute data & posterior
	DataVector gp_coord (input_size);
	GridStorage* gs = &(grid->getStorage());

	std::size_t load = std::size_t(fmax(0, seq_max - seq_min + 1));
	unique_ptr<double[]> data (new double[load * output_size]);
	unique_ptr<double[]> pos (new double[load]);

	unique_ptr<double[]> m;
	double* d = nullptr;
	double* p = nullptr;

	std::size_t i,dim;  // loop index
	for (i=seq_min; i <= seq_max; i++) {
		// Set output pointer
		d = &data[0] + (i-seq_min) * output_size;
		p = &pos[0] + (i-seq_min);
		// Get grid point coordinate
		gs->get(i)->getCoordsBB(gp_coord, *bbox);
		m.reset(vec_to_arr(gp_coord));
		// compute with full model
		fullmodel->run(m.get(), d);
		// compute posterior
		*p = compute_posterior(odata.get(), d, output_size, sigma);
		// Find max posterior
		if (*p > maxpos) {
			maxpos = *p;
			maxpos_seq = i;
		}
#if (SGI_OUT_GRID_POINTS==1)
		printf("Rank %d: grid point %lu at %s completed, pos = %.6f.\n",
				mpi_rank, i, vec_to_str(gp_coord).c_str(), *p);
#endif
	}
	// Write results to file
	mpiio_partial_data(false, seq_min, seq_max, data.get());
	mpiio_partial_posterior(false, seq_min, seq_max, pos.get());
	return;
}

void SGI::mpina_get_local_range(
		const std::size_t& gmin,
		const std::size_t& gmax,
		std::size_t& lmin,
		std::size_t& lmax)
{
	std::size_t num_gps = gmax - gmin + 1;
	std::size_t trunk = num_gps / mpi_size;
	std::size_t rest = num_gps % mpi_size;

	if (mpi_rank < rest) {
		lmin = gmin + mpi_rank * (trunk + 1);
		lmax = lmin + trunk;
	} else {
		lmin = gmin + mpi_rank * trunk + rest;
		lmax = lmin + trunk - 1;
	}
	return;
}

void SGI::mpimw_get_job_range(
		const std::size_t& jobid,
		const std::size_t& seq_offset,
		std::size_t& seq_min,
		std::size_t& seq_max)
{
	seq_min = seq_offset + jobid * MPIMW_TRUNK_SIZE;
	seq_max = min( seq_min + MPIMW_TRUNK_SIZE - 1, grid->getSize()-1 );
}

void SGI::mpimw_worker_compute(std::size_t gp_offset)
{
	// Setup variables
	int job_todo, job_done; // use separate buffers for send and receive
	std::size_t seq_min, seq_max;
	MPI_Status status;

	while (true) {
		// Receive a signal from MASTER
		MPI_Recv(&job_todo, 1, MPI_INT, MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		if (status.MPI_TAG == MPIMW_TAG_TERMINATE) break;

		if (status.MPI_TAG == MPIMW_TAG_WORK) {
			// get the job range and compute
			mpimw_get_job_range(job_todo, gp_offset, seq_min, seq_max);
			mpi_compute_range(seq_min, seq_max);
			// tell master the job is done
			job_done = job_todo;
			MPI_Send(&job_done, 1, MPI_INT, MASTER, job_done, MPI_COMM_WORLD);
		}
#if (ENABLE_IMPI==1)
		if (status.MPI_TAG == MPIMW_TAG_ADAPT) impi_adapt();
#endif
	} // end while
	return;
}

void SGI::mpimw_master_compute(std::size_t gp_offset)
{
	std::size_t added_gps;
	int num_jobs, jobid, worker, scnt=0;
	MPI_Status status;
	unique_ptr<int[]> jobs; // use array to have unique send buffer
	vector<int> jobs_done;

#if (ENABLE_IMPI==1)
	double tic, toc;
	int jobs_per_tic;
#endif

	// Determine total # jobs (compute only the newly added points)
	added_gps = grid->getSize() - gp_offset;
	num_jobs = (added_gps % MPIMW_TRUNK_SIZE > 0) ?
			added_gps/MPIMW_TRUNK_SIZE + 1 :
			added_gps/MPIMW_TRUNK_SIZE;

	// Initialize job queues
	jobs_done.reserve(num_jobs);
	jobs.reset(new int[num_jobs]);
	for (int i = 0; i < num_jobs; i++)
		jobs[i] = i;

#if (ENABLE_IMPI==1)
	tic = MPI_Wtime();
	jobs_per_tic = 0;
#endif

	// Seed workers if any
	if (mpi_size > 1)
		mpimw_seed_workers(num_jobs, scnt, jobs.get());

	// As long as not all jobs are done, keep working...
	while (jobs_done.size() < num_jobs) {
		if (mpi_size > 1) {
			// #1. Receive a finished job
			MPI_Recv(&jobid, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			worker = status.MPI_SOURCE;
			jobs_done.push_back(jobid); // mark the job as done
#if (ENABLE_IMPI==1)
			jobs_per_tic++;
#endif
			// #2. Send another job if any
			if (scnt < num_jobs) {
				MPI_Send(&jobs[scnt], 1, MPI_INT, worker, MPIMW_TAG_WORK, MPI_COMM_WORLD);
				scnt++;
			}
		} else { // if there is NO worker
			// #1. Take out job
			jobid = jobs[scnt];
			scnt++;
			// #2. Compute a job
			std::size_t seq_min, seq_max;
			mpimw_get_job_range(jobid, gp_offset, seq_min, seq_max);
			mpi_compute_range(seq_min, seq_max);
			jobs_done.push_back(jobid);
#if (ENABLE_IMPI==1)
			jobs_per_tic++;
#endif
		}

		// #3. Check for adaption every IMPI_ADAPT_FREQ seconds
#if (ENABLE_IMPI==1)
		toc = MPI_Wtime()-tic;
		if (toc >= IMPI_ADAPT_FREQ) {
			// performance measure: # gps computed per second
			printf("PERFORMANCE MEASURE: # forward simulations per second = %.6f\n",
					double(jobs_per_tic * MPIMW_TRUNK_SIZE)/toc);
			// Only when there are remaining jobs, it's worth trying to adapt
			if (scnt < num_jobs) {
				// Prepare workers for adapt (receive done jobs, then send adapt signal)
				if (mpi_size > 1)
					mpimw_adapt_preparation(jobs_done, jobs_per_tic);
				// Adapt
				impi_adapt();
				// Seed workers again
				if (mpi_size > 1)
					mpimw_seed_workers(num_jobs, scnt, jobs.get());
			}
			// reset timer
			tic = MPI_Wtime();
			jobs_per_tic = 0;
		} // end if-toc
#endif
	} // end while

	// All jobs done
	if (mpi_size > 1) {
		for (int mi=1; mi < mpi_size; mi++)
			MPI_Send(&jobid, 1, MPI_INT, mi, MPIMW_TAG_TERMINATE, MPI_COMM_WORLD);
	}
	return;
}

void SGI::mpimw_seed_workers(
		const int& num_jobs,
		int& scnt,
		int* jobs)
{
	// the smaller of (remainning jobs) or (# workers)
	int size = int(fmin(num_jobs-scnt, mpi_size-1));
	unique_ptr<MPI_Request[]> tmp_req (new MPI_Request[size]);
	for (int i=0; i < size; i++) {
		MPI_Isend(&jobs[scnt], 1, MPI_INT, i+1, MPIMW_TAG_WORK,
				MPI_COMM_WORLD, &tmp_req[i]);
		scnt++;
	}
	MPI_Waitall(size, tmp_req.get(), MPI_STATUS_IGNORE);
	return;
}

void SGI::mpimw_adapt_preparation(
		vector<int> & jobs_done,
		int & jobs_per_tic)
{
#if (ENABLE_IMPI==1)
	unique_ptr<MPI_Request[]> tmp_req (new MPI_Request[(mpi_size-1)*2]);
	unique_ptr<int[]> tmp_rbuf (new int[mpi_size-1]);
	unique_ptr<int[]> tmp_sbuf (new int[mpi_size-1]); // dummy send buffer

	for (int i=1; i < mpi_size; i++) {
		// First receive a finished job
		MPI_Irecv(&tmp_rbuf[i-1], 1, MPI_INT, MPI_ANY_SOURCE,
				MPI_ANY_TAG, MPI_COMM_WORLD, &tmp_req[i-1]);
		// Then send "adapt signal", send buffer is dummy
		MPI_Isend(&tmp_sbuf[i-1], 1, MPI_INT, i, MPIMW_TAG_ADAPT,
				MPI_COMM_WORLD, &tmp_req[(mpi_size-1) + i-1]);
	}
	MPI_Waitall((mpi_size-1)*2, tmp_req.get(), MPI_STATUS_IGNORE);
	for (int i=1; i < mpi_size; i++) {
		jobs_done.push_back(tmp_rbuf[i-1]);
		jobs_per_tic++;
	}
	return;
#endif
}
