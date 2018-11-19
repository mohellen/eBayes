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

#include <surrogate/SGI.hpp>

using namespace std;
using namespace	sgpp::base;


SGI::SGI(
		Config const& c,
		Parallel & p,
		ForwardModel & m)
		: ForwardModel(c), par(p), fullmodel(m)
{
	alphas.resize(cfg.get_output_size());
	grid = nullptr;
	eval = nullptr;
	bbox = nullptr;
	seq_maxpos = make_pair(0, 0.0);
#if (IMPI==1)
	impi_gpoffset = 0;
#endif
}

vector<double> SGI::run(
		vector<double> const& m)
{
	// Grid check
	if (!eval) {
		par.info();
		printf("ERROR: SGI::run fail because surrogate is not properly built. Program abort!\n");
		exit(EXIT_FAILURE);
	}
	// Convert m into DataVector
	DataVector point (m);
	// Evaluate m
	std::size_t output_size = cfg.get_output_size();
	vector<double> d (output_size);
	for (std::size_t j=0; j < output_size; j++) {
		d[j] = eval->eval(alphas[j], point);
	}
	return d;
}

void SGI::build()
{
	// Get config variables
	std::size_t input_size = cfg.get_input_size();
	std::size_t output_size = cfg.get_output_size();
	std::size_t init_level = cfg.get_param_sizet("sgi_init_level");
	double refine_portion = cfg.get_param_double("sgi_refine_portion");
	bool is_masterworker = cfg.get_param_bool("sgi_is_masterworker");
	bool is_resume = cfg.get_param_bool("sgi_is_resume");
	// find out whether it's grid initialization or refinement
	bool is_init = (!this->eval) ? true : false;
	std::size_t num_points;

#if (IMPI==1)
	if (par.status == MPI_ADAPT_STATUS_JOINING) {
		impi_adapt(); // JOINING ranks go to adapt directly!
	} else {
#endif
		if (is_init) {
			if (is_resume) {
				// for resuming a job, grid and data, pos are loaded from files
				// therefore, no need for build grid computation, return immediately
				resume();
				return; 
			}
			if (par.is_master()) {
				fflush(NULL);
				printf("==========================================\n");
				printf("SGI: Initializing SGI surrogate...\n");
			}
			// 1. All: Construct grid
			grid.reset(Grid::createModLinearGrid(input_size).release()); // create empty grid
			grid->getGenerator().regular(init_level); // populate grid points
			bbox.reset(create_boundingbox());
			grid->setBoundingBox(*bbox); // set up bounding box
			impi_gpoffset = 0; // NEW ranks set this variable needed by the JOINING ranks
			num_points = grid->getSize();
		} else {
			if (par.is_master()) {
				fflush(NULL);
				printf("==========================================\n");
				printf("SGI: Refining SGI surrogate...\n");
			}
			// 1. All: refine grid
			impi_gpoffset = grid->getSize(); // STAYING ranks set this variable needed by the JOINING ranks
			refine_grid_bcast(refine_portion); // MASTER refine then bcast
			num_points = grid->getSize();

#if (SGI_DEBUG==1) //Debug only: to check if bcast grid correct
			int src_rank = par.size - 1;
			bool is_grid_ok = verify_grid(MPI_COMM_WORLD, par.size-1, 0);
			if (par.is_master()) {
				par.info();
				if (is_grid_ok) {
					printf("DEBUG: bcast grid to other ranks correct!\n");
				} else {
					printf("DEBUG: bcast grid to other ranks incorrect! Abort!\n");
					exit(EXIT_FAILURE);
				}
			}
#endif
		}
		if (par.is_master()) mpiio_write_grid(); // MASTER write grid
#if (IMPI==1)
	}
#endif
	// 2. All: Compute data at each grid point (result written to MPI IO file)
	//		and find the top maxpos points
	compute_grid_points(impi_gpoffset, is_masterworker);
	// 3. All: Compute and hierarchize alphas
	compute_hier_alphas();
	// 4. Update op_eval
	eval.reset(sgpp::op_factory::createOperationEval(*grid).release());

	// Master: print grogress
	if (par.is_master()) {
		if (is_init) {
			fflush(NULL);
			printf("SGI: Initialize SGI surrogate successful.\n");
			printf("==========================================\n");
		} else {
			fflush(NULL);
			printf("SGI: Refine SGI surrogate successful.\n");
			printf("==========================================\n");
		}

		// Make a copy of data files
		string cmd = "cp " + cfg.get_grid_fname() + " " + cfg.get_grid_bak_fname();
		system(cmd.c_str());	
		cmd = "cp " + cfg.get_data_fname() + " " + cfg.get_data_bak_fname();
		system(cmd.c_str());	
		cmd = "cp " + cfg.get_pos_fname() + " " + cfg.get_pos_bak_fname();
		system(cmd.c_str());	
	}
	return;
}

vector<double> SGI::get_maxpos()
{
	vector<double> samplepos = get_gp_coord( seq_maxpos.first );
	samplepos.push_back( seq_maxpos.second );
	return samplepos;
}

void SGI::resume()
{
	if (par.is_master()) {
		fflush(NULL);
		printf("\n==========================================\n");
		printf("SGI: Building SGI surrogate from file...\n");
		// Make a copy of data files
		string cmd = "cp " + cfg.get_grid_resume_fname() + " " + cfg.get_grid_fname();
		system(cmd.c_str());	
		cmd = "cp " + cfg.get_data_resume_fname() + " " + cfg.get_data_fname();
		system(cmd.c_str());	
		cmd = "cp " + cfg.get_pos_resume_fname() + " " + cfg.get_pos_fname();
		system(cmd.c_str());	
	}
	// Set: grid, eval, bbox
	mpiio_read_grid();
	// Set: alphas
	compute_hier_alphas();
	// Read posterior
	std::size_t num_gps = grid->getSize();
	unique_ptr<double[]> pos (new double[num_gps]);
	mpiio_readwrite_pos(true, 0, num_gps-1, pos.get());

	// Find maxpos, spmd style
	std::size_t lmin, lmax;
	mpispmd_get_local_range(0, num_gps-1, lmin, lmax);
	// Each rank go through local portion to find local maxpos
	seq_maxpos.first = 0;
	seq_maxpos.second = 0.0;
	for (std::size_t i=lmin; i <= lmax; ++i) {
		if (pos[i] > seq_maxpos.second) {
			seq_maxpos.first = i;
			seq_maxpos.second = pos[i];
		}
	}
	// Find global maxpos
	mpispmd_find_global_maxpos();

	if (par.is_master()) {
		fflush(NULL);
		printf("SGI: Build SGI surrogate from file successful.\n");
		printf("==========================================\n\n");
	}
	return;
}


/*********************************************
 *       		 Private Methods
 *********************************************/
void SGI::impi_adapt()
{
#if (IMPI==1)
	int adapt_flag = MPI_ADAPT_FALSE;
	MPI_Info info;
	MPI_Comm intercomm;
	MPI_Comm newcomm;
	int staying_count, leaving_count, joining_count;
	double tic, tic1;

	// Joining ranks can omit probe_adapt, and call adapt_begin directly
	if (par.status != MPI_ADAPT_STATUS_JOINING) { 
		tic = MPI_Wtime();
		//----
		MPI_Probe_adapt(&adapt_flag, &par.status, &info);
		//----
		if (par.is_master()) {
			par.info();
			printf("iMPI: MPI_Probe_adapt() in %.6f sec\n", MPI_Wtime()-tic);
		}
	}

	// Both preexisting ranks (that has adapt_true) and joining ranks will enter adapt window
	if ((adapt_flag == MPI_ADAPT_TRUE) || (par.status == MPI_ADAPT_STATUS_JOINING)) {
		tic1 = MPI_Wtime();

		tic = MPI_Wtime();
		//----
		MPI_Comm_adapt_begin(&intercomm, &newcomm, &staying_count, &leaving_count, &joining_count);
		//----
		if (par.is_master()) {
			par.info();
			printf("iMPI: MPI_Comm_adapt_begin() in %.6f sec\n", MPI_Wtime()-tic);
		}

		//************************ ADAPT WINDOW ****************************
		if (joining_count > 0) {
			// sync grid offset
			MPI_Bcast(&impi_gpoffset, 1, MPI_SIZE_T, par.master, newcomm);
			// bcast grid to joining ranks
			bcast_grid(newcomm);

#if (SGI_DEBUG==1) //Debug only: to check if bcast grid correct
			int src_rank = staying_count + joining_count -1;
			bool is_grid_ok = verify_grid(newcomm, src_rank, 0);
			if (par.is_master()) {
				par.info();
				if (is_grid_ok) {
					printf("DEBUG: bcast grid to Joining ranks correct!\n");
				} else {
					printf("DEBUG: bcast grid to Joining ranks incorrect! Abort!\n");
					exit(EXIT_FAILURE);
				}
			}
#endif
		}
		//************************ ADAPT WINDOW ****************************

		tic = MPI_Wtime();
		//----
		MPI_Comm_adapt_commit();
		//----
		if (par.is_master()) {
			par.info();
			printf("iMPI: MPI_Comm_adapt_commit() in %.6f seconds.\n", MPI_Wtime()-tic);
		}
		// Update Comm World size, rank, status
		par.mpi_update();

		if (par.is_master()) {
			par.info();
			printf("iMPI: impi_adapt() in %.6f sec\n", MPI_Wtime()-tic1);
		}
	}
	return;
#endif
}

vector<double> SGI::get_gp_coord(std::size_t seq)
{
	std::size_t input_size = cfg.get_input_size();
	// Get grid point coordinate
	DataVector coord (input_size);
	grid->getStorage().get(seq)->getCoordsBB(coord, grid->getBoundingBox());
	// Output vector
	vector<double> gp (input_size);
	for (size_t i=0; i < input_size; ++i)
		gp[i] = coord[i];
	return gp;
}

double SGI::get_gp_volume(std::size_t seq)
{
	return pow(2.0, -grid->getStorage().get(seq)->getLevelSum());
}

BoundingBox* SGI::create_boundingbox()
{
	std::size_t input_size = cfg.get_input_size();
	BoundingBox* bb = new BoundingBox(input_size);
	DimensionBoundary db;
	for(int i=0; i < input_size; i++) {
		db.leftBoundary = fullmodel.get_input_space(i).first;
		db.rightBoundary = fullmodel.get_input_space(i).second;
		db.bDirichletLeft  = false;
		db.bDirichletRight = false;
		bb->setBoundary(i, db);
	}
	return bb;
}

void SGI::compute_hier_alphas()
{
#if (SGI_PRINT_TIMER==1)
	double tic = MPI_Wtime();
#endif
	std::size_t output_size = cfg.get_output_size();
	std::size_t num_gps = grid->getSize();
	// read raw data
	unique_ptr<double[]> data (new double[output_size * num_gps]);
	mpiio_readwrite_data(true, 0, num_gps-1, data.get());
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
#if (SGI_PRINT_TIMER==1)
	if (par.is_master()) {
		fflush(NULL);
		printf("SGI: created alphas in  %.6f sec\n", MPI_Wtime()-tic);
	}
#endif
	return;
}

void SGI::compute_grid_points(
		std::size_t gp_offset,
		bool is_masterworker)
{
#if (SGI_PRINT_TIMER==1)
	double tic = MPI_Wtime(); // start the timer
#endif
	// NOTE: "Master-minion" scheme can run under MPI & iMPI
	// 		 "Naive" (aka SIMD) scheme can run under only MPI
	if (is_masterworker) {
		// Master-worker style
		if (par.is_master()) {
			mpimw_master_compute(gp_offset);
		} else {
			mpimw_worker_compute(gp_offset);
		}
		// Bcast maxpos
		mpimw_master_bcast_maxpos(); // Includes MPI_Bcast
	} else {
		// MPI native style (default)
		std::size_t num_gps = grid->getSize();
		std::size_t mymin, mymax;
		mpispmd_get_local_range(gp_offset, num_gps-1, mymin, mymax);
		compute_gp_range(mymin, mymax);
		// Find global maxpos
		mpispmd_find_global_maxpos(); // Includes MPI_Allreduce and MPI_Bcast
	}
	//MPI_Barrier(MPI_COMM_WORLD); // no need for barrier due to MPI_Bcast

#if (SGI_PRINT_TIMER==1)
	if (par.is_master()) {
		fflush(NULL);
		printf("SGI: computed gps %lu | range %lu %lu | in.time(sec) %.6f \n",
				grid->getSize()-gp_offset, gp_offset, grid->getSize()-1, MPI_Wtime()-tic);
	}
#endif
	return;
}

void SGI::compute_gp_range(
		const std::size_t& seq_min,
		const std::size_t& seq_max)
{
	// IMPORTANT: Ensure workload is at least 1 grid point!
	// This also prevents overriding wrong data to data files!
	if (seq_max < seq_min) return;

	std::size_t output_size = cfg.get_output_size();
	std::size_t load = std::size_t(fmax(0, seq_max - seq_min + 1));
//	std::size_t num_maxpos = cfg.get_param_sizet("mcmc_max_chains");

	unique_ptr<double[]> data (new double[load * output_size]);
	unique_ptr<double[]> pos (new double[load]);
	vector<double> dvec;
	double* d = nullptr;
	double* p = nullptr;

	for (std::size_t i=seq_min; i <= seq_max; ++i) {
		// Set output pointer
		d = &data[0] + (i-seq_min) * output_size;
		p = &pos[0] + (i-seq_min);
		// compute with full model
		dvec = fullmodel.run( get_gp_coord(i) );
		std::copy(dvec.begin(), dvec.end(), d);
		// compute posterior
		*p = cfg.compute_posterior(dvec);
		// find local maxpos
		if (*p > seq_maxpos.second) {
			seq_maxpos.first = i;
			seq_maxpos.second = *p;
		}
#if (SGI_PRINT_GRIDPOINTS==1)
		par.info();
		printf("SGI: computed gp %lu at %s, pos %.6f\n",
				i, tools::sample_to_string(get_gp_coord(i)), *p);
#endif
	}
	// Write results to file
	mpiio_readwrite_data(false, seq_min, seq_max, data.get());
	mpiio_readwrite_pos(false, seq_min, seq_max, pos.get());
	return;
}

/**
 * TODO: BUG detected!!!
 * For some strange reason, each rank refine its own grid creates different grids
 * even thought num_gps and refine_gps are the same before refinement.
 * Therefore, we have to use a single-refine strategy: Master refine, others read
 */
#if (1==0)
bool SGI::refine_grid_all(double portion_to_refine)
{
#if (SGI_PRINT_TIMER==1)
	double tic = MPI_Wtime();
#endif
	// Compute threshold number of grid points to be added
	// NOTE: to refine X points, maximum (2*dim*X) points can be added to grid
	int maxi = 10000;
	int thres = int(ceil(maxi / 2 / cfg.get_input_size()));
	// Number of points to refine
	std::size_t num_gps = this->grid->getSize();
	int refine_gps = int(ceil(num_gps * portion_to_refine));
	refine_gps = (refine_gps > thres) ? thres : refine_gps;
	// If not points to refine, return false (meaning grid not refined)
	if (refine_gps < 1) return false;
	// Read posterior from file
	unique_ptr<double[]> pos (new double[num_gps]);
	mpiio_readwrite_pos(true, 0, num_gps-1, &pos[0]);
	// For each gp, compute the refinement index
	double data_norm;
	DataVector refine_idx (num_gps);
	std::size_t output_size = cfg.get_output_size();
	for (std::size_t i=0; i<num_gps; i++) {
		data_norm = 0;
		for (std::size_t j=0; j < output_size; j++) {
			data_norm += (alphas[j][i] * alphas[j][i]);
		}
		data_norm = sqrt(data_norm);
		// refinement_index = |alpha| * posterior
		refine_idx[i] = data_norm * pos[i];
	}
	// refine grid
	grid->refine(refine_idx, refine_gps);
#if (SGI_PRINT_TIMER==1)
	if (par.is_master())
		cout << "SGI: MASTER refined grid in " << MPI_Wtime()-tic << " seconds." << endl;
#endif
	return true;
}
#endif

/**
 * Single-refine scheme: only MASTER refines and write grid, others read grid
 * REASON: see function above
 */
void SGI::refine_grid_mpiio(double portion_to_refine)
{
	// Master refine and write grid
	if (par.is_master()) {
#if (SGI_PRINT_TIMER==1)
		double tic = MPI_Wtime();
#endif
		// Compute threshold number of grid points to be added
		// NOTE: to refine X points, maximum (2*dim*X) points can be added to grid
		int maxi = 10000;
		int thres = int(ceil(maxi / 2 / cfg.get_input_size()));
		// Number of points to refine
		std::size_t num_gps = this->grid->getSize();
		int refine_gps = int(ceil(num_gps * portion_to_refine));
		refine_gps = (refine_gps > thres) ? thres : refine_gps;
		// If no points to refine abort
		if (refine_gps < 1) {
			par.info();
			printf("SGI: refine grid failed due to no points to refine. Program abort!\n");
			exit(EXIT_FAILURE);	
		};
		// Read posterior from file
		unique_ptr<double[]> pos (new double[num_gps]);
		mpiio_readwrite_pos(true, 0, num_gps-1, &pos[0]);
		// For each gp, compute the refinement index
		double data_norm;
		DataVector refine_idx (num_gps);
		std::size_t output_size = cfg.get_output_size();
		for (std::size_t i=0; i<num_gps; i++) {
			data_norm = 0;
			for (std::size_t j=0; j < output_size; j++) {
				data_norm += (alphas[j][i] * alphas[j][i]);
			}
			data_norm = sqrt(data_norm);
			// refinement_index = |alpha| * posterior
			refine_idx[i] = data_norm * pos[i];
		}
		// refine grid
		grid->refine(refine_idx, refine_gps);
		mpiio_write_grid();
		std::size_t new_num_gps = grid->getSize();

		fflush(NULL);
		printf("SGI: total %zu gps, %zu added, range [%zu, %zu]\n",
				new_num_gps, new_num_gps-num_gps, num_gps, new_num_gps-1);
#if (SGI_PRINT_TIMER==1)
		printf("SGI: refined grid in %.6f seconds.\n", MPI_Wtime()-tic);
#endif
	}
	// Ensure read grid after its written completely
	MPI_Barrier(MPI_COMM_WORLD);
	// Others read grid
	if (!par.is_master()) mpiio_read_grid();
	
	return;
}

/**
 * Single-refine scheme: only MASTER refines grid, then bcast it
 * REASON: see function above
 */
void SGI::refine_grid_bcast(double portion_to_refine)
{
	// Master refine grid
	if (par.is_master()) {
#if (SGI_PRINT_TIMER==1)
		double tic = MPI_Wtime();
#endif
		// Compute threshold number of grid points to be added
		// NOTE: to refine X points, maximum (2*dim*X) points can be added to grid
		int maxi = 10000;
		int thres = int(ceil(maxi / 2 / cfg.get_input_size()));
		// Number of points to refine
		std::size_t num_gps = this->grid->getSize();
		int refine_gps = int(ceil(num_gps * portion_to_refine));
		refine_gps = (refine_gps > thres) ? thres : refine_gps;
		// If no points to refine abort
		if (refine_gps < 1) {
			par.info();
			printf("SGI: refine grid failed due to no points to refine. Program abort!\n");
			exit(EXIT_FAILURE);	
		};
		// Read posterior from file
		unique_ptr<double[]> pos (new double[num_gps]);
		mpiio_readwrite_pos(true, 0, num_gps-1, &pos[0]);
		// For each gp, compute the refinement index
		double data_norm;
		DataVector refine_idx (num_gps);
		std::size_t output_size = cfg.get_output_size();
		for (std::size_t i=0; i<num_gps; i++) {
			data_norm = 0;
			for (std::size_t j=0; j < output_size; j++) {
				data_norm += (alphas[j][i] * alphas[j][i]); //TODO: high-dim not to use l2-norm
			}
			data_norm = sqrt(data_norm);
			// refinement_index = |alpha| * V * posterior, where
			// Point volume V := 2^{-(l1+l2+...+ld)}
			refine_idx[i] = data_norm * get_gp_volume(i) * pos[i];
		}
		// refine grid
		grid->refine(refine_idx, refine_gps);
		std::size_t new_num_gps = grid->getSize();
		// print grid info
		fflush(NULL);
		printf("SGI: total %zu gps, %zu added, range [%zu, %zu]\n",
				new_num_gps, new_num_gps-num_gps, num_gps, new_num_gps-1);
#if (SGI_PRINT_TIMER==1)
		printf("SGI: refined grid in %.6f seconds.\n", MPI_Wtime()-tic);
#endif
	}
	// Bcast the grid
	bcast_grid(MPI_COMM_WORLD);
	return;
}

void SGI::mpiio_write_grid()
{
	// Pack grid into Char array
	string sg_str = grid->serialize();
	int count = static_cast<int>(sg_str.size());
	// Copy partial grid to string
	unique_ptr<char[]> buff (new char[count]);
	sg_str.copy(buff.get(), count, 0);
	// Write to file
	string ofile = cfg.get_grid_fname();
	MPI_File fh;
	if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY,
			MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
		par.info();
		printf("ERROR: fail to open %s for grid write. Program abort!\n", ofile.c_str());
		exit(EXIT_FAILURE);
	}
	if (MPI_File_write(fh, buff.get(), count, MPI_CHAR, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
		par.info();
		printf("ERROR: fail to write grid to %s. Program abort!\n", ofile.c_str());
		exit(EXIT_FAILURE);
	}
	MPI_File_close(&fh);
	return;
}

void SGI::mpiio_read_grid()
{
	// Open file
	string ofile = cfg.get_grid_fname();
	MPI_File fh;
	if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)
			!= MPI_SUCCESS) {
		par.info();
		printf("ERROR: fail to open %s for grid read. Program abort!\n", ofile.c_str());
		exit(EXIT_FAILURE);
	}
	// Get file size,create buffer
	long long int count;
	MPI_File_get_size(fh, &count);
	unique_ptr<char[]> buff (new char[count]);
	// Read from file
	if (MPI_File_read(fh, buff.get(), count, MPI_CHAR, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
		par.info();
		printf("ERROR: fail to read grid from %s. Program abort!\n", ofile.c_str());
		exit(EXIT_FAILURE);
	}
	MPI_File_close(&fh);
	// Create serialized grid string
	string sg_str(buff.get());
	// Construct new grid from grid string
	grid.reset(Grid::unserialize(sg_str).release()); // create grid
	bbox.reset(create_boundingbox());
	grid->setBoundingBox(*bbox); // set up bounding box
	eval.reset(sgpp::op_factory::createOperationEval(*grid).release());
	return;
}

void SGI::mpiio_readwrite_data(
		bool is_read,
		std::size_t seq_min,
		std::size_t seq_max,
		double* buff)
{
	// Do something only when seq_min <= seq_max
	if (seq_min > seq_max) return;
	std::size_t output_size = cfg.get_output_size();

	string ofile = cfg.get_data_fname();
	MPI_File fh;
	if (is_read) { // Read data from file
		if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)
				!= MPI_SUCCESS) {
			par.info();
			printf("ERROR: fail to open %s for data read. Program abort!\n", ofile.c_str());
			exit(EXIT_FAILURE);
		}
		// offset is in # of bytes, and is ALWAYS calculated from beginning of file.
		if (MPI_File_read_at(fh, seq_min*output_size*sizeof(double), buff,
				(seq_max-seq_min+1)*output_size, MPI_DOUBLE, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
			par.info();
			printf("ERROR: fail read data from %s. Program abort!\n", ofile.c_str());
			exit(EXIT_FAILURE);
		}
	} else { // Write data to file
		if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh)
				!= MPI_SUCCESS) {
			par.info();
			printf("ERROR: fail to open %s for data write. Program abort!\n", ofile.c_str());
			exit(EXIT_FAILURE);
		}
		// offset is in # of bytes, and is ALWAYS calculated from beginning of file.
		if (MPI_File_write_at(fh, seq_min*output_size*sizeof(double), buff,
				(seq_max-seq_min+1)*output_size, MPI_DOUBLE, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
			par.info();
			printf("ERROR: fail to write data to %s. Program abort!\n", ofile.c_str());
			exit(EXIT_FAILURE);
		}
	}
	MPI_File_close(&fh);
	return;
}

void SGI::mpiio_readwrite_pos(
		bool is_read,
		std::size_t seq_min,
		std::size_t seq_max,
		double* buff)
{
	// Do something only when seq_min <= seq_max
	if (seq_min > seq_max) return;

	string ofile = cfg.get_pos_fname();
	MPI_File fh;
	if (is_read) { // Read data from file
		if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh)
				!= MPI_SUCCESS) {
			par.info();
			printf("ERROR: fail to open %s for posterior read. Program abort!\n", ofile.c_str());
			exit(EXIT_FAILURE);
		}
		// offset is in # of bytes, and is ALWAYS calculated from beginning of file.
		if (MPI_File_read_at(fh, seq_min*sizeof(double), buff,
				(seq_max-seq_min+1), MPI_DOUBLE, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
			par.info();
			printf("ERROR: fail to read posterior from %s. Program abort!\n", ofile.c_str());
			exit(EXIT_FAILURE);
		}

	} else { // Write data to file
		if (MPI_File_open(MPI_COMM_SELF, ofile.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh)
				!= MPI_SUCCESS) {
			par.info();
			printf("ERROR: fail to open %s for posterior write. Program abort!\n", ofile.c_str());
			exit(EXIT_FAILURE);
		}
		// offset is in # of bytes, and is ALWAYS calculated from beginning of file.
		if (MPI_File_write_at(fh, seq_min*sizeof(double), buff,
				(seq_max-seq_min+1), MPI_DOUBLE, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
			par.info();
			printf("ERROR: fail to write posterior to %s. Program abort!\n", ofile.c_str());
			exit(EXIT_FAILURE);
		}
	}
	MPI_File_close(&fh);
	return;
}

void SGI::mpispmd_get_local_range(
		const std::size_t& gmin,
		const std::size_t& gmax,
		std::size_t& lmin,
		std::size_t& lmax)
{
	std::size_t num_gps = gmax - gmin + 1;
	std::size_t trunk = num_gps / par.size;
	std::size_t rest = num_gps % par.size;
	if (par.rank < rest) {
		lmin = gmin + par.rank * (trunk + 1);
		lmax = lmin + trunk;
	} else {
		lmin = gmin + par.rank * trunk + rest;
		lmax = lmin + trunk - 1;
	}
	return;
}

void SGI::mpispmd_find_global_maxpos()
{
	if (par.size <= 1) return;

	// Find out which rank has the global maxpos point
	struct {
		double maxpos;
		int	   rank;
	} in, out;
	in.maxpos = seq_maxpos.second;
	in.rank = par.rank;
	MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
	// Bcast the maxpos point
	struct {
		std::size_t seq;
		double maxpos;
	} buf;
	buf.seq = seq_maxpos.first;
	buf.maxpos = seq_maxpos.second;
	MPI_Bcast(&buf, 1, par.MPI_SEQPOS, out.rank, MPI_COMM_WORLD);
	// Every rank unpack from buffer
	seq_maxpos.first = buf.seq;
	seq_maxpos.second = buf.maxpos;
}

void SGI::mpimw_master_compute(std::size_t gp_offset)
{
	double impi_adapt_freq = cfg.get_param_double("impi_adapt_freq_sec");
	std::size_t jobsize = cfg.get_param_sizet("sgi_masterworker_jobsize");
	// Determine total # jobs (compute only the newly added points)
	std::size_t added_gps = grid->getSize() - gp_offset;
	std::size_t num_jobs = (added_gps % jobsize > 0) ? (added_gps/jobsize + 1) : (added_gps/jobsize);
	vector<char> jobs (num_jobs, JOBTODO); // JOBTODO, JOBDONE, JOBINPROG
	vector<char> workers (par.size, RANKIDLE); // RANKACTIVE, RANKIDLE
	workers[par.master] = 'x'; // exclude master rank from any search

#if (SGI_DEBUG==2) // Debug only: check jobs and workers array (set 2 to disable it permtly)
	print_jobs(jobs);
	print_workers(workers);
#endif

#if (IMPI==1)
	double phase_tic = MPI_Wtime();
	double tic = MPI_Wtime();
	double toc;
	int jobs_per_tic = 0;
#endif

	// Seed workers if any
	if (par.size > 1)
		mpimw_master_seed_workers(jobs, workers);

#if (SGI_DEBUG==2) // Debug only: check jobs and workers array (set 2 to disable this permtly.)
	print_jobs(jobs);
	print_workers(workers);
#endif

	// As long as not all jobs are done, keep working...
	while (!std::all_of(jobs.begin(), jobs.end(), [](char i){return i==JOBDONE;})) {

		if (par.size > 1) {
			// #1. Receive a finished job (only if there is any active worker)
			if (std::any_of(workers.begin(), workers.end(), [](char i){return i==RANKACTIVE;})) {
				mpimw_master_recv_done(jobs, workers);
#if (IMPI==1)
				jobs_per_tic++;
#endif
			}
			// #2. Send another job
			mpimw_master_send_todo(jobs, workers); // internally checks for todo jobs and idle workers
		} else { // if there is NO worker
			// #1. Find a todo job
			int jid = std::find(jobs.begin(), jobs.end(), JOBTODO) - jobs.begin();
			if (jid == jobs.size()) continue;
			jobs[jid] = JOBINPROG;
			// #2. Compute a job
			std::size_t seq_min, seq_max;
			mpimw_get_job_range(jid, gp_offset, seq_min, seq_max);
			compute_gp_range(seq_min, seq_max); // this also handles local maxpos list
			jobs[jid] = JOBDONE;
#if (IMPI==1)
			jobs_per_tic++;
#endif
		}

		// #3. Check for adaptation every impi_adapt_freq seconds
#if (IMPI==1)
		toc = MPI_Wtime()-tic;
		if (toc >= impi_adapt_freq) {
			// performance measure: # gps computed per second
			fflush(NULL);
			printf("SGI Performance: ranks %d | jobs %d | in.time(sec) %.6f | phase.wall.time(sec) %.6f\n",
					par.size, jobs_per_tic, toc, MPI_Wtime()-phase_tic);
			//printf("SGI Performance: %d ranks computed %d gps in %.6f sec. #gps/sec = %.6f\n",
			//		par.size, jobs_per_tic, toc, double(jobs_per_tic * jobsize)/toc);
			// Only when there are remaining jobs, it's worth trying to adapt
			if (std::any_of(jobs.begin(), jobs.end(), [](char i){return i==JOBTODO;})) {
				// Prepare workers for adapt (receive done jobs, then send adapt signal)
				if (par.size > 1)
					mpimw_master_prepare_adapt(jobs, workers, jobs_per_tic);
				// Adapt
				impi_adapt();
				workers.resize(par.size); // Update worker list
				for (auto w: workers) w = RANKIDLE; // Reset all workers to idle status
				workers[par.master] = 'x'; // Exclude master rank from any search
				// Seed workers again
				if (par.size > 1)
					mpimw_master_seed_workers(jobs, workers);
			}
			// reset timer
			tic = MPI_Wtime();
			jobs_per_tic = 0;
		} // end if-toc
#endif
	} // end while

	// All jobs done
	if (par.size > 1) {
		for (int i=1; i < par.size; ++i)
			MPI_Send(&jobs_per_tic, 1, MPI_INT, i, MPIMW_TAG_TERMINATE, MPI_COMM_WORLD);
	}
	return;
}

void SGI::mpimw_worker_compute(std::size_t gp_offset)
{
	// Setup variables
	int job_todo, job_done; // use separate buffers for send and receive
	std::size_t seq_min, seq_max;
	MPI_Status status;

	while (true) {
		// Receive a signal from MASTER
		if (MPI_Recv(&job_todo, 1, MPI_INT, par.master, MPI_ANY_TAG, MPI_COMM_WORLD, &status)
				!= MPI_SUCCESS) {
			par.info();
			printf("ERROR: fail receive from MASTER. Program abort!\n");
			exit(EXIT_FAILURE);
		}

		if (status.MPI_TAG == MPIMW_TAG_TERMINATE) break;

		if (status.MPI_TAG == MPIMW_TAG_WORK) {
			// get the job range and compute
			mpimw_get_job_range(job_todo, gp_offset, seq_min, seq_max);

#if (SGI_DEBUG==1) // Debug only: check jobid and offset match
			if (job_todo != (seq_min - gp_offset)/cfg.get_param_sizet("sgi_masterworker_jobsize")) {
				par.info();
				printf("ERROR: jobid %d, offset %lu, range [%lu, %lu] mismatch. Program abort!\n",
						job_todo, gp_offset, seq_min, seq_max);
				exit(EXIT_FAILURE);
			}
#endif
			compute_gp_range(seq_min, seq_max);
			// tell master the job is done (use a different send buffer)
			job_done = job_todo;
			mpimw_worker_send_done(job_done);
		}
#if (IMPI==1)
		if (status.MPI_TAG == MPIMW_TAG_ADAPT) impi_adapt();
#endif
	} // end while
	return;
}

void SGI::mpimw_get_job_range(
		const std::size_t& jobid,
		const std::size_t& seq_offset,
		std::size_t& seq_min,
		std::size_t& seq_max)
{
	std::size_t jobsize = cfg.get_param_sizet("sgi_masterworker_jobsize");
	seq_min = seq_offset + jobid * jobsize;
	seq_max = min(seq_min + jobsize - 1, grid->getSize()-1);
}

void SGI::mpimw_master_seed_workers(
		vector<char>& jobs,
		vector<char>& workers)
{
	vector<MPI_Request> sreq;
	sreq.reserve(par.size-1);
	vector<int> sbuf; //use unique send buffer for each Isend
	sbuf.reserve(par.size-1);
	// Seed workers avoiding MASTER itself
	for (int i=0; i < par.master; ++i) {
		sbuf.push_back( std::find(jobs.begin(), jobs.end(), JOBTODO)-jobs.begin() ); // fetch a todo job
		if (sbuf.back() >= jobs.size()) break; // no more todo jobs, stop seeding
		sreq.push_back(MPI_Request());
		MPI_Isend(&(sbuf.back()), 1, MPI_INT, i, MPIMW_TAG_WORK, MPI_COMM_WORLD, &(sreq.back()));
		jobs[sbuf.back()] = JOBINPROG; // mark job as "processing"
		workers[i] = RANKACTIVE; // mark worker as "active"
	}
	for (int i=par.master+1; i < par.size; ++i) {
		sbuf.push_back( std::find(jobs.begin(), jobs.end(), JOBTODO)-jobs.begin() ); // fetch a todo job
		if (sbuf.back() >= jobs.size()) break; // no more todo jobs, stop seeding
		sreq.push_back(MPI_Request());
		MPI_Isend(&(sbuf.back()), 1, MPI_INT, i, MPIMW_TAG_WORK, MPI_COMM_WORLD, &(sreq.back()));
		jobs[sbuf.back()] = JOBINPROG; // mark job as "processing"
		workers[i] = RANKACTIVE; // mark worker as "active"
	}
	if (sreq.size() > 0) {
		if (MPI_Waitall(sreq.size(), &sreq[0], MPI_STATUS_IGNORE) != MPI_SUCCESS) {
			par.info();
			printf("ERROR: failed to Isend for seeding workers. Program abort!\n");
			exit(EXIT_FAILURE);
		}
	}
	return;
}

void SGI::mpimw_master_prepare_adapt(
		vector<char>& jobs,
		vector<char>& workers,
		int& jobs_per_tic)
{
#if (IMPI==1)
	vector<MPI_Request> sreq;
	sreq.reserve(par.size-1);
	vector<char> sbuf; // dummy send buffer, use unique send buffer for Isend
	sbuf.reserve(par.size-1);
	// If any worker still active, receive the finished job
	while ( std::any_of(workers.begin(), workers.end(), [](char i){return i==RANKACTIVE;}) ) {
		mpimw_master_recv_done(jobs, workers);
		jobs_per_tic++;
	}
	// Send "adapt signal" to all workers (avoiding MASTER itself)
	for (int i=0; i < par.master; i++) {
		sreq.push_back(MPI_Request());
		sbuf.push_back('1');
		MPI_Isend(&(sbuf.back()), 1, MPI_CHAR, i, MPIMW_TAG_ADAPT,
				MPI_COMM_WORLD, &(sreq.back()));
	}
	for (int i=par.master+1; i < par.size; i++) {
		sreq.push_back(MPI_Request());
		sbuf.push_back('1');
		MPI_Isend(&(sbuf.back()), 1, MPI_CHAR, i, MPIMW_TAG_ADAPT,
				MPI_COMM_WORLD, &(sreq.back()));
	}
	if (sreq.size() > 0) {
		if (MPI_Waitall(sreq.size(), &sreq[0], MPI_STATUS_IGNORE) != MPI_SUCCESS) {
			par.info();
			printf("ERROR: failed to Isend adapt signal to workers. Program abort!\n");
			exit(EXIT_FAILURE);
		}
	}
	return;
#endif
}

void SGI::mpimw_master_send_todo(
		vector<char>& jobs,
		vector<char>& workers)
{
	// find a todo job
	int jid = std::find(jobs.begin(), jobs.end(), JOBTODO) - jobs.begin();
	if (jid >= jobs.size()) return; // no more todo jobs, nothing to do
	// find an idle worker
	int wid = std::find(workers.begin(), workers.end(), RANKIDLE) - workers.begin();
	if (wid >= workers.size()) return; // All workers are busy, nothing to do
 	// Send job (jid) to the idle worker (wid)
	MPI_Send(&jid, 1, MPI_INT, wid, MPIMW_TAG_WORK, MPI_COMM_WORLD);
	jobs[jid] = JOBINPROG; // Mark job as "in process"
	workers[wid] = RANKACTIVE; // Mark worker as "active"
	return;
}

void SGI::mpimw_master_recv_done(
		vector<char>& jobs,
		vector<char>& workers)
{
	// 1. Receive the finished jobid, and workerid
	int jid, wid;
	MPI_Status status;
	if (MPI_Recv(&jid, 1, MPI_INT, MPI_ANY_SOURCE, 12333, MPI_COMM_WORLD, &status) != MPI_SUCCESS) {
		par.info();
		printf("ERROR: failed to receive a done job. Program abort!\n");
		exit(EXIT_FAILURE);
	}
	wid = status.MPI_SOURCE;
	jobs[jid] = JOBDONE;
	workers[wid] = RANKIDLE;
	// 2. Receive seq_maxpos
	struct {
		std::size_t seq;
		double maxpos;
	} rbuf;
	if (MPI_Recv(&rbuf, 1, par.MPI_SEQPOS, wid, 12444, MPI_COMM_WORLD, &status) != MPI_SUCCESS) {
		par.info();
		printf("ERROR: failed to receive a done job seq_maxpos. Program abort!\n");
		exit(EXIT_FAILURE);
	}
	if (seq_maxpos.second < rbuf.maxpos) {
		seq_maxpos.first = rbuf.seq;
		seq_maxpos.second = rbuf.maxpos;
	}
#if (SGI_PRINT_RANKPROGRESS==1)
	par.info();
	printf("SGI: worker %d | job.id %d | total.jobs %lu | offset %lu\n",
			wid, jid, jobs.size(), impi_gpoffset);
#endif
	return;
}

void SGI::mpimw_master_bcast_maxpos()
{
	if (par.size <= 1) return;

	// Bcast the maxpos point
	struct {
		std::size_t seq;
		double maxpos;
	} buf;
	buf.seq = seq_maxpos.first;
	buf.maxpos = seq_maxpos.second;
	MPI_Bcast(&buf, 1, par.MPI_SEQPOS, par.master, MPI_COMM_WORLD);
	// Every rank unpack from buffer
	seq_maxpos.first = buf.seq;
	seq_maxpos.second = buf.maxpos;
}

void SGI::mpimw_worker_send_done(int jobid)
{
	//1. Send jobid
	MPI_Send(&jobid, 1, MPI_INT, par.master, 12333, MPI_COMM_WORLD);
	//2. Send seq_maxpos
	struct {
		std::size_t seq;
		double maxpos;
	} sbuf;
	sbuf.seq = seq_maxpos.first;
	sbuf.maxpos = seq_maxpos.second;
	MPI_Send(&sbuf, 1, par.MPI_SEQPOS, par.master, 12444, MPI_COMM_WORLD);
	return;
}

// For debug only
void SGI::print_workers(vector<char> const& workers)
{
	fflush(NULL);
	printf("Worker status: ");
	for (int i=0; i < workers.size()-1; ++i) {
		printf("[%d]%c ... ", i, workers[i]);
	}
	printf("[%lu]%c\n", workers.size()-1, workers[workers.size()-1]);
}

void SGI::print_jobs(vector<char> const& jobs)
{
	fflush(NULL);
	printf("Job status: ");
	for (int i=0; i < jobs.size()-1; ++i) {
		printf("[%d]%c ... ", i, jobs[i]);
	}
	printf("[%lu]%c\n", jobs.size()-1, jobs[jobs.size()-1]);
}

// Verify a joining rank's grid (read from file) against MASTER's grid
// Only MASTER gets the correct compare result
bool SGI::verify_grid_from_read(int joinrank, MPI_Comm intercomm)
{
	if ((par.status == MPI_ADAPT_STATUS_JOINING) && (par.rank == joinrank)) {
		// Pack grid into Char array
		string sg_str = grid->serialize();
		int count = static_cast<int>(sg_str.size());
		// Copy grid to string
		unique_ptr<char[]> sbuf (new char[count]);
		sg_str.copy(sbuf.get(), count, 0);
		// Send to MASTER (in the remote group)
		if (MPI_Send(sbuf.get(), count, MPI_CHAR, par.master, 713, intercomm) != MPI_SUCCESS) {
			fflush(NULL);
			printf("ERROR: Rank[%d|%d](%d) failed to send serialized grid (DEBUG: verify joining rank's grid). Program abort!\n",
					par.rank, par.size, par.status);
			exit(EXIT_FAILURE);
		}
	}
	if (par.is_master()) {
		// Pack grid into Char array
		string sg_str_own = grid->serialize();

		MPI_Status status;
		MPI_Probe(joinrank, 713, intercomm, &status);
		int count;
		MPI_Get_count(&status, MPI_CHAR, &count);
		unique_ptr<char[]> rbuf (new char[count]);
		
		if (MPI_Recv(rbuf.get(), count, MPI_CHAR, joinrank, 713, intercomm, &status) != MPI_SUCCESS) {
			fflush(NULL);
			printf("ERROR: Rank[%d|%d](%d) failed to receive serialized grid (DEBUG: verify joining rank's grid). Program abort!\n",
					par.rank, par.size, par.status);
			exit(EXIT_FAILURE);
		}
		string sg_str_joinrank(rbuf.get());
		
		// compare
		if (sg_str_own.compare(sg_str_joinrank) == 0) {
			return true;
		}
	}
	return false;
}

// The master broadcast grid to the rest of the group
// Can use with either MPI_COMM_WORLD or NEW_COMM
void SGI::bcast_grid(MPI_Comm comm)
{
	size_t size;
	string sg_str;
	int my_rank = -1; // this can exclude non-member of comm
	MPI_Comm_rank(comm, &my_rank);

	if (my_rank == 0) {
		sg_str = grid->serialize();
		size = sg_str.size();
	}
	// bcast size
	MPI_Bcast(&size, 1, MPI_SIZE_T, 0, comm);
	// others allocate string
	if (my_rank > 0) {
		sg_str = string(size, ' ');
	}
	// bcast serialized grid
	MPI_Bcast(const_cast<char*>(sg_str.c_str()), size, MPI_CHAR, 0, comm);
	// others deserialize grid
	if (my_rank > 0) {
		// deserialize grid
		restore_grid( sg_str );
	}
	return;
}


void SGI::restore_grid(string sg_str) {
	// Construct new grid from grid string
	grid.reset(Grid::unserialize(sg_str).release()); // create grid
	bbox.reset(create_boundingbox());
	grid->setBoundingBox(*bbox); // set up bounding box
	eval.reset(sgpp::op_factory::createOperationEval(*grid).release());
	return;
}

// DEBUG ONLY: only the dest rank has the correct result!!
bool SGI::verify_grid(MPI_Comm comm, int src_rank, int dest_rank)
{
	size_t size;
	string sg_str;
	int my_rank;
	MPI_Comm_rank(comm, &my_rank);

	if (my_rank == src_rank) {
		// Pack grid into string
		sg_str = grid->serialize();
		size = sg_str.size();
		// Send to dest
		MPI_Send(&size, 1, MPI_SIZE_T, dest_rank, 1941, comm);
		MPI_Send(sg_str.c_str(), size, MPI_CHAR, dest_rank, 1943, comm);
	}
	if (my_rank == dest_rank) {
		// Recv from source rank
		MPI_Recv(&size, 1, MPI_SIZE_T, src_rank, 1941, comm, MPI_STATUS_IGNORE);
		sg_str = string(size, ' ');
		MPI_Recv(const_cast<char*>(sg_str.c_str()), size, MPI_CHAR, src_rank, 1943, comm, MPI_STATUS_IGNORE);
		// Pack own grid into string
		string my_sg_str = grid->serialize();
		// Compare two grid string
		if (my_sg_str.compare(sg_str) == 0) {
			return true;
		}
	}
	return false;
}
