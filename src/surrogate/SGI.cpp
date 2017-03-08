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

#include "surrogate/SGI.hpp"

using namespace std;
using namespace	sgpp::base;


SGI::SGI(
		ForwardModel* fm,
		const string& observed_data_file)
		: ForwardModel()
{
	MPI_Comm_rank(MPI_COMM_WORLD, &(this->mpi_rank));
	MPI_Comm_size(MPI_COMM_WORLD, &(this->mpi_size));
#if (ENABLE_IMPI==1)
	mpi_status = -1;
#endif
	this->input_size = fm->get_input_size();
	this->output_size = fm->get_output_size();

	this->fullmodel = unique_ptr<ForwardModel>(fm);
	this->grid = nullptr;
	this->eval = nullptr;
	this->alphas = unique_ptr<DataVector[]>(new DataVector[output_size]);

	this->maxpos_seq = 0;
	this->maxpos = 0.0;
	this->noise = 0.0;
	this->odata = unique_ptr<double[]>(
		get_observed_data(observed_data_file, this->output_size, this->noise));
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
		cout << "Grid is not properly setup or computed. Progam abort!" << endl;
		exit(EXIT_FAILURE);
	}
	// Convert m into data vector
	DataVector point = arr_to_vec(m, input_size);
	// Evaluate m
	for (std::size_t j=0; j < output_size; j++) {
		d[j] = eval->eval(this->alphas[j], point);
	}
	return;
}

void SGI::initialize(
		std::size_t level,
		bool is_masterworker)
{
	std::size_t num_points;

#if (ENABLE_IMPI==1)
	if (mpi_status == MPI_ADAPT_STATUS_JOINING) {
		num_points = CarryOver.gp_offset;
	} else {
#endif
	// 1. All: Construct grid
	grid.reset(Grid::createLinearBoundaryGrid(input_size).release());	// create empty grid
//	grid.reset(Grid::createModLinearGrid(input_size).release());	// create empty grid
	grid->getGenerator().regular(level);			// populate grid points
	grid->setBoundingBox(*create_boundingbox());	// set up bounding box
	num_points = grid->getSize();

	// Master:
	if (mpi_rank == MASTER) {
		// Print progress
		printf("\n...Initializing SGI model...\n");
		printf("%lu grid points to be added. Total # grid points = %lu.\n", num_points, num_points);
		// Write grid to file
		mpiio_write_full_grid();
	}
#if (ENABLE_IMPI==1)
	}
#endif

	// 2. All: Compute data at each grid point (result written to MPI IO file)
	//			and find the max posterior point
	compute_grid_points(0, is_masterworker);
	mpi_find_global_update_maxpos();

	// 3. All: Create alphas
	create_alphas();

	// 4. Update op_eval
	eval.reset(sgpp::op_factory::createOperationEval(*grid).get());

	// Master: print grogress
	if (mpi_rank == MASTER) {
		unique_ptr<double[]> m_maxpos (seg_to_coord_arr(maxpos_seq));
		printf("Max posterior = %.6f, at %s.\n",
				maxpos, arr_to_string(m_maxpos.get(), input_size).c_str());
		printf("...Initialize SGI model successful...\n");
	}
	return;
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

void SGI::create_alphas()
{
#if (SGI_OUT_TIMER==1)
	double tic = MPI_Wtime();
#endif
	// read raw data
	std::size_t num_gps = grid->getSize();
	unique_ptr<double[]> data (new double[output_size * num_gps]);
	mpiio_partial_data(true, 0, num_gps-1, data.get());

	// re-allocate alphas
	for (std::size_t j=0; j < output_size; j++)
		alphas[j].resize(num_gps);

	// unpack raw data into alphas
	for (std::size_t i=0; i < num_gps; i++)
		for (std::size_t j=0; j < output_size; j++)
			alphas[j].set(i, data[i*output_size+j]);

	// hierarchize alphas
	unique_ptr<OperationHierarchisation> hier (sgpp::op_factory::createOperationHierarchisation(*grid));
	for (std::size_t j=0; j<output_size; j++)
		hier->doHierarchisation(alphas[j]);

#if (SGI_OUT_TIMER==1)
	if (mpi_rank == MASTER)
		printf("Rank %d: created alphas in %.5f seconds.\n",
				mpi_rank, MPI_Wtime()-tic);
#endif
	return;
}



/***************************
 * MPI related operations
 ***************************/
void SGI::mpiio_write_full_grid()
{
	// Pack a grid
	string sg_str = grid->getStorage().serialize(1);
	size_t count = sg_str.size();
	unique_ptr<char[]> buff (new char[count]);
	strcpy(buff.get(), sg_str.c_str());

	// Write to file
	string ofile = string(OUTPATH) + "/grid.mpibin";
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

void SGI::mpiio_read_full_grid()
{}

void SGI::mpiio_partial_data(
		bool is_read,
		std::size_t seq_min,
		std::size_t seq_max,
		double* buff)
{
	// Do something only when seq_min <= seq_max
	if (seq_min > seq_max) return;

	string ofile;
	ofile = string(OUTPATH) + "/data.mpibin";;
	MPI_File fh;
	if (is_read) { // Read data from file
		if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)
				!= MPI_SUCCESS) {
			cout << "MPI read file open failed. Operation aborted! " << endl;
			exit(EXIT_FAILURE);
		}
		// offset is in # of bytes, and is ALWAYS calculated from beginning of file.
		MPI_File_read_at(fh, seq_min*output_size*sizeof(double), buff,
				(seq_max-seq_min+1)*output_size, MPI_DOUBLE, MPI_STATUS_IGNORE);

	} else { // Write data to file
		if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh)
				!= MPI_SUCCESS) {
			cout << "MPI write file open failed. Operation aborted! " << endl;
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
		double* buff)
{
	// Do something only when seq_min <= seq_max
	if (seq_min > seq_max) return;

	string ofile;
	ofile = string(OUTPATH) + "/pos.mpibin";;
	MPI_File fh;
	if (is_read) { // Read data from file
	if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)
			!= MPI_SUCCESS) {
		cout << "MPI read file open failed. Operation aborted! " << endl;
		exit(EXIT_FAILURE);
	}
	// offset is in # of bytes, and is ALWAYS calculated from beginning of file.
	MPI_File_read_at(fh, seq_min*sizeof(double), buff,
			(seq_max-seq_min+1), MPI_DOUBLE, MPI_STATUS_IGNORE);

	} else { // Write data to file
	if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh)
			!= MPI_SUCCESS) {
		cout << "MPI write file open failed. Operation aborted! " << endl;
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
//		// Master-worker style
//		if (mpi_rank == MASTER)
//			mpimw_master_compute(gp_offset);
//		else
//			mpimw_worker_compute(gp_offset);
	} else {
		// MPI native style (default)
		std::size_t num_gps = grid->getSize();
		std::size_t mymin, mymax;
		mpina_get_local_range(gp_offset, num_gps-1, mymin, mymax);
		mpina_compute_range(mymin, mymax);
	}
	MPI_Barrier(MPI_COMM_WORLD);

#if (SGI_OUT_TIMER==1)
	if (mpi_rank == MASTER)
		printf("%d Ranks: computed %lu grid points (%lu to %lu) in %.5f seconds.\n",
				mpi_size, grid->getSize()-gp_offset,
				gp_offset, grid->getSize()-1, MPI_Wtime()-tic);
#endif
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

void SGI::mpina_compute_range(
		const std::size_t& seq_min,
		const std::size_t& seq_max)
{
	// IMPORTANT: Ensure workload is at least 1 grid point!
	// This also prevents overriding wrong data to data files!
	if (seq_max < seq_min) return;

#if (SGI_OUT_RANK_PROGRESS==1)
	printf("Rank %d: computing grid points %lu to %lu.\n",
			mpi_rank, seq_min, seq_max);
#endif
	// Compute data & posterior
	DataVector gp_coord (input_size);
	GridStorage* gs = &(grid->getStorage());
	BoundingBox* bb = &(grid->getBoundingBox());

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
		gs->get(i)->getCoordsBB(gp_coord, *bb);
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







